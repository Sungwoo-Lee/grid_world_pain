"""Tests for the MC_RAW return mode (raw returns as critic target, RAW residual advantages).

Plain-language context: recurrent PPO learns two numbers from each rollout — a *critic
target* (what total future reward the value head should have predicted) and an
*advantage* (how much better an action turned out than the critic expected). Two
independent choices go into producing that pair: WHICH ESTIMATOR builds it (Monte-Carlo
returns, or generalised advantage estimation, "GAE"), and HOW the two numbers are
RESCALED before training. "Rescaled" here means z-scored — shifted and stretched to mean
0 and standard deviation ("spread") 1.

The four older modes confound three different things under the label "matched scale":

  arm             critic target        advantage            both on one scale?  that scale
  MC, GAE_NORM    z-scored (~1)        residual target-V    yes                 ~1
  MC_FIXED, GAE   raw (~24)            separately z-scored  no                  split
  MC_RAW          raw (~24)            raw residual (~20)   yes                 ~24

"MC_RAW" is the MC_FIXED branch with EXACTLY ONE LINE DELETED — the advantage
normalisation. Nothing anywhere in the branch is rescaled. It is matched-scale like
"MC", but matched at the LARGE raw scale instead of at 1, which separates the claim
"the critic target and the advantage must share units" from the claim "the advantage
must land near spread 1". Motivating result: on the 10x10 jump-attack environment
(basic/04) at one million episodes, agents survived on average 138.8 steps under "MC"
but only 40.0 under "MC_FIXED" and 41.9 under "GAE"; the open question is whether that
is about the RELATION between the two scales or about their ABSOLUTE magnitude.

These tests pin (a) MC_RAW's targets are the RAW Monte-Carlo returns, bit-identical to
MC_FIXED's; (b) its advantages are exactly `targets - values`, with NO normalisation —
the test that would fail if anyone re-added the deleted line asserts the advantage's
spread is emphatically NOT 1 on a fixture whose raw residuals sit near 20; (c) MC_RAW is
in the MC family for bootstrapping, so it takes the single window-edge bootstrap, not
GAE's per-step one (`use_gae_bootstrap` is False); and (d) an unknown mode raises an
error naming all five legal modes, including MC_RAW.
"""
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from src.models import recurrent_ppo_trainer as rpt
from src.models.recurrent_ppo_trainer import (
    LEGAL_RETURN_MODES,
    StepInfo,
    Transition,
    compute_mc_returns,
    train_iteration,
    validate_return_mode,
)

T, N = 6, 3          # rollout steps, parallel envs
OBS_DIM, HID = 5, 4
GAMMA = 0.9


class _Cfg:
    """Static config stub carrying only the fields train_iteration reads."""
    rnn_type = "GRU"
    num_steps = T
    num_epochs = 1
    gamma = GAMMA
    gae_lambda = 0.95

    def __init__(self, return_mode):
        self.return_mode = return_mode


def _synthetic_rollout(reward_scale=30.0):
    """Deterministic fake trajectories: no env, no model, no randomness.

    `next_value` is all-zero because the MC family (MC / MC_FIXED / MC_RAW) never reads
    it — those modes bootstrap once at the window edge instead.

    `reward_scale` is deliberately LARGE (default 30) so the raw Monte-Carlo returns,
    and therefore the raw residuals `return - value`, have a spread far from 1. That is
    what makes the "advantages are not normalised" test below able to fail: if anyone
    re-adds MC_FIXED's normalisation line, the spread would snap to 1.
    """
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    rewards = (jax.random.normal(k1, (T, N)) * 3.0 + 1.0) * reward_scale
    values = jax.random.normal(k2, (T, N)) * 2.0
    # Two episode ends: a real death (reason 2) and a timeout (reason 1).
    reason = jnp.zeros((T, N), dtype=jnp.int32)
    reason = reason.at[2, 0].set(2).at[4, 1].set(1)
    dones = reason > 0
    zeros = jnp.zeros((T, N))
    step_info = StepInfo(**{
        f: zeros for f in StepInfo._fields if f not in
        ("termination_reason", "dist_per_neutral", "dist_per_predator")
    }, termination_reason=reason,
        dist_per_neutral=jnp.zeros((T, N, 1)),
        dist_per_predator=jnp.zeros((T, N, 1)))
    traj = Transition(
        obs=jax.random.normal(k3, (T, N, OBS_DIM)),
        action=jnp.zeros((T, N), dtype=jnp.int32),
        reward=rewards, done=dones,
        log_prob=jnp.zeros((T, N)), value=values, mod_info=None,
        step_info=step_info, next_value=jnp.zeros((T, N)),
    )
    h_states = jnp.zeros((T, N, HID))
    bootstrap_value = jnp.array([7.0, -3.0, 0.5])
    return traj, h_states, bootstrap_value


def _capture_batch(return_mode, monkeypatch, reward_scale=30.0):
    """Run train_iteration with collect_trajectories + update_step stubbed out, and
    return the PPOBatch that would have been handed to the optimiser."""
    traj, h_states, bootstrap_value = _synthetic_rollout(reward_scale)
    seen = {}

    def fake_collect(model, env_params, last_state, last_h_state, last_key,
                     num_steps, rnn_type="LSTM", return_mode="MC"):
        seen["collect_return_mode"] = return_mode
        return traj, h_states, last_state, last_h_state, last_key, bootstrap_value

    def fake_update(model, optimizer, batch, config):
        seen["batch"] = batch
        return jnp.float32(0.0), (0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(rpt, "collect_trajectories", fake_collect)
    monkeypatch.setattr(rpt, "update_step", fake_update)
    train_iteration(None, None, None, None, None, jax.random.PRNGKey(1), _Cfg(return_mode))
    return seen["batch"], seen["collect_return_mode"], traj, bootstrap_value


def _expected_raw_mc_returns(traj, bootstrap_value):
    """The raw Monte-Carlo returns every MC-family branch must produce."""
    terminateds = (traj.step_info.termination_reason >= 2).astype(traj.done.dtype)
    return jax.vmap(compute_mc_returns, in_axes=(1, 1, 1, 0, None), out_axes=1)(
        traj.reward, traj.done, terminateds, bootstrap_value, GAMMA)


# ---------------------------------------------------------------------------
# (a) targets are the RAW Monte-Carlo returns
# ---------------------------------------------------------------------------

def test_mc_raw_targets_are_the_raw_returns(monkeypatch):
    batch, _, traj, boot = _capture_batch("MC_RAW", monkeypatch)
    assert jnp.array_equal(batch.targets, _expected_raw_mc_returns(traj, boot))


def test_mc_raw_targets_are_not_zscored(monkeypatch):
    """Raw means raw: the targets must NOT sit at mean 0 / spread 1 the way MC's do."""
    batch, _, _, _ = _capture_batch("MC_RAW", monkeypatch)
    assert abs(float(jnp.mean(batch.targets))) > 1.0
    assert abs(float(jnp.std(batch.targets)) - 1.0) > 1.0


def test_mc_raw_targets_are_bit_identical_to_mc_fixed(monkeypatch):
    """MC_RAW and MC_FIXED share the whole estimator half — only the advantage line
    differs — so the critic targets must be exactly equal, not merely close."""
    raw_batch, _, _, _ = _capture_batch("MC_RAW", monkeypatch)
    fixed_batch, _, _, _ = _capture_batch("MC_FIXED", monkeypatch)
    assert jnp.array_equal(raw_batch.targets, fixed_batch.targets)


# ---------------------------------------------------------------------------
# (b) advantages are the RAW residual — the deleted-line tests
# ---------------------------------------------------------------------------

def test_mc_raw_advantages_are_exactly_targets_minus_values(monkeypatch):
    """Exact equality, not allclose: the branch computes this expression verbatim."""
    batch, _, traj, _ = _capture_batch("MC_RAW", monkeypatch)
    assert jnp.array_equal(batch.advantages, batch.targets - traj.value)


def test_mc_raw_advantages_are_not_normalised(monkeypatch):
    """THE test that fails if MC_FIXED's deleted normalisation line is re-added.

    The fixture's rewards are scaled up by 30, so the raw residuals `return - value`
    have a spread of order 20 — nowhere near 1. A normalised advantage would have spread
    exactly 1 and mean 0, so both assertions below would fail.
    """
    batch, _, _, _ = _capture_batch("MC_RAW", monkeypatch)
    adv_std = float(jnp.std(batch.advantages))
    adv_mean = float(jnp.mean(batch.advantages))
    assert adv_std > 5.0, f"advantage spread {adv_std} looks normalised"
    assert abs(adv_std - 1.0) > 1e-2
    assert abs(adv_mean) > 1e-2, "a normalised advantage would be mean-zero"


def test_mc_raw_advantages_differ_from_mc_fixed_advantages(monkeypatch):
    """Same targets, different advantages: MC_FIXED z-scores them, MC_RAW does not."""
    raw_batch, _, _, _ = _capture_batch("MC_RAW", monkeypatch)
    fixed_batch, _, _, _ = _capture_batch("MC_FIXED", monkeypatch)
    assert float(jnp.std(fixed_batch.advantages)) == pytest.approx(1.0, abs=1e-4)
    assert not jnp.allclose(raw_batch.advantages, fixed_batch.advantages, atol=1e-3)


def test_mc_raw_scale_is_matched_but_large(monkeypatch):
    """The design claim, expressed as a test: MC_RAW's target and advantage live on the
    SAME (large) scale, whereas MC_FIXED's live on scales that differ by a big factor."""
    raw_batch, _, _, _ = _capture_batch("MC_RAW", monkeypatch)
    fixed_batch, _, _, _ = _capture_batch("MC_FIXED", monkeypatch)
    raw_ratio = float(jnp.std(raw_batch.targets)) / float(jnp.std(raw_batch.advantages))
    fixed_ratio = float(jnp.std(fixed_batch.targets)) / float(jnp.std(fixed_batch.advantages))
    assert 0.2 < raw_ratio < 5.0, f"MC_RAW scales should be comparable, ratio {raw_ratio}"
    assert fixed_ratio > 10.0, f"MC_FIXED scales should be split, ratio {fixed_ratio}"


def test_mc_raw_differs_from_mc_on_both_halves(monkeypatch):
    """MC z-scores the target; MC_RAW does not. Neither the targets nor the advantages
    may coincide, or the two modes would be silently the same experiment."""
    raw_batch, _, _, _ = _capture_batch("MC_RAW", monkeypatch)
    mc_batch, _, _, _ = _capture_batch("MC", monkeypatch)
    assert not jnp.allclose(raw_batch.targets, mc_batch.targets, atol=1e-3)
    assert not jnp.allclose(raw_batch.advantages, mc_batch.advantages, atol=1e-3)


# ---------------------------------------------------------------------------
# (c) MC_RAW is in the MC family: window-edge bootstrap, not GAE's per-step one
# ---------------------------------------------------------------------------

def test_mc_raw_does_not_take_the_gae_bootstrap_path():
    """Unit-level check of the `use_gae_bootstrap` predicate that gates the extra
    per-step next-state value forward pass inside collect_trajectories."""
    assert validate_return_mode("MC_RAW") not in ("GAE", "GAE_NORM")


def test_mc_raw_ignores_the_per_step_next_value(monkeypatch):
    """Behavioural proof of the same thing: perturbing `next_value` cannot change
    MC_RAW's numbers, because the MC estimator never reads it."""
    base_batch, _, _, _ = _capture_batch("MC_RAW", monkeypatch)

    traj, h_states, boot = _synthetic_rollout()
    traj = traj._replace(next_value=traj.next_value + 5.0)
    seen = {}

    def fake_collect(model, env_params, last_state, last_h_state, last_key,
                     num_steps, rnn_type="LSTM", return_mode="MC"):
        return traj, h_states, last_state, last_h_state, last_key, boot

    def fake_update(model, optimizer, batch, config):
        seen["batch"] = batch
        return jnp.float32(0.0), (0.0, 0.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(rpt, "collect_trajectories", fake_collect)
    monkeypatch.setattr(rpt, "update_step", fake_update)
    train_iteration(None, None, None, None, None, jax.random.PRNGKey(1), _Cfg("MC_RAW"))
    assert jnp.array_equal(seen["batch"].targets, base_batch.targets)
    assert jnp.array_equal(seen["batch"].advantages, base_batch.advantages)


@pytest.fixture(scope="module")
def tiny_env_and_model():
    """Small real env + model, built once for the whole module (the build and the
    per-mode rollout trace are the expensive part of these integration tests)."""
    from src.environment.config_loader import load_env_config, load_env_params
    from src.environment.core import jax_reset
    from src.environment.sensor import get_observation, get_observation_breakdown
    from src.models.recurrent_ppo_network import ActorCriticRNN

    params = load_env_params(load_env_config("configs/environment/default.yaml"))
    state = jax.vmap(jax_reset, in_axes=(None, 0))(
        params, jax.random.split(jax.random.PRNGKey(0), 2))
    obs = jax.vmap(get_observation, in_axes=(0, None))(state, params)
    action_dim = 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)
    model = ActorCriticRNN(
        input_dim=int(obs.shape[-1]), action_dim=action_dim, hidden_size=8,
        rngs=nnx.Rngs(0), rnn_type="GRU", activation="relu", modulation_config=None,
        observation_breakdown=get_observation_breakdown(params),
        encoding_config={"encoding_mode": "flat", "use_layer_norm": False},
    )
    return model, params, state


@pytest.mark.integration
def test_mc_raw_collect_leaves_next_value_all_zero(tiny_env_and_model):
    """End-to-end: with return_mode="MC_RAW" the per-step next-state forward pass must
    not run, so the collected `next_value` stays all-zero — exactly like MC/MC_FIXED.
    A usable window-edge bootstrap of shape (num_envs,) still comes back."""
    model, params, state = tiny_env_and_model
    h = model.initial_state(2)
    traj, _, _, _, _, bootstrap_value = rpt.collect_trajectories(
        model, params, state, h, jax.random.PRNGKey(3), 4,
        rnn_type="GRU", return_mode="MC_RAW")
    assert not bool(jnp.any(traj.next_value != 0.0))
    assert bootstrap_value.shape == (2,)


@pytest.mark.integration
def test_mc_raw_rollout_matches_mc_fixed_rollout(tiny_env_and_model):
    """Same bootstrap path and same collection code as MC_FIXED: identical rollouts."""
    model, params, state = tiny_env_and_model
    h = model.initial_state(2)
    out = {}
    for mode in ("MC_FIXED", "MC_RAW"):
        traj, _, _, _, _, boot = rpt.collect_trajectories(
            model, params, state, h, jax.random.PRNGKey(3), 4,
            rnn_type="GRU", return_mode=mode)
        out[mode] = (traj.reward, traj.value, traj.next_value, boot)
    for a, b in zip(out["MC_FIXED"], out["MC_RAW"]):
        assert jnp.array_equal(a, b)


# ---------------------------------------------------------------------------
# (d) validation: MC_RAW is legal, typos are not
# ---------------------------------------------------------------------------

def test_legal_return_modes_are_the_five_expected():
    assert LEGAL_RETURN_MODES == ("MC", "MC_FIXED", "GAE", "GAE_NORM", "MC_RAW")


@pytest.mark.parametrize("good,canonical", [
    ("mc_raw", "MC_RAW"), ("MC_RAW", "MC_RAW"), (" mc_raw ", "MC_RAW"),
])
def test_mc_raw_accepted_case_insensitively(good, canonical):
    assert validate_return_mode(good) == canonical


@pytest.mark.parametrize("bad", ["mc-raw", "MCRAW", "MC_RAWW", "RAW", ""])
def test_invalid_return_mode_raises_naming_all_five(bad):
    with pytest.raises(ValueError) as exc:
        validate_return_mode(bad)
    msg = str(exc.value)
    for legal in ("MC", "MC_FIXED", "GAE", "GAE_NORM", "MC_RAW"):
        assert legal in msg


def test_invalid_return_mode_raises_from_train_iteration(monkeypatch):
    with pytest.raises(ValueError, match="MC_RAW"):
        _capture_batch("MC_RAWW", monkeypatch)


def test_plain_ppo_rejects_mc_raw():
    """Plain (non-recurrent) PPO does not implement MC_RAW — it must say so loudly
    rather than silently falling through to its GAE branch."""
    from src.models.ppo_trainer import train_iteration_ppo

    class _PlainCfg:
        num_steps = 4
        return_mode = "MC_RAW"

    with pytest.raises(ValueError) as exc:
        train_iteration_ppo(None, None, None, None, jax.random.PRNGKey(0), _PlainCfg())
    msg = str(exc.value)
    assert "MC_RAW" in msg
    assert "RecurrentPPO" in msg
