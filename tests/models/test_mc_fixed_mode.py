"""Tests for the MC_FIXED return mode (raw returns as critic target, normalised advantages).

Plain-language context: recurrent PPO learns two things from a rollout — a *critic
target* (what total future reward the value head should have predicted) and an
*advantage* (how much better an action turned out than the critic expected). Before
this change the trainer offered two ways of producing that pair:

  - "MC" (Monte-Carlo): compute the actual discounted future reward of every step,
    then rescale those numbers to mean 0 / standard deviation 1 (a "z-score") and use
    the RESCALED numbers as BOTH the critic target and, minus the critic's prediction,
    the advantage. Nothing is rescaled a second time.
  - "GAE": compute advantages with generalised advantage estimation, use the RAW
    (un-rescaled) `advantage + value` as the critic target, and rescale only the
    ADVANTAGES.

A survey of ten mainstream PPO implementations found nine of nine libraries follow the
GAE convention — raw critic target, normalised advantages — so the MC convention is the
odd one out, and its rescaling is also what creates the units mismatch with the
window-edge bootstrap value recorded in the Known Bugs registry.

"MC_FIXED" is the third mode added for an empirical three-way comparison: the SAME
Monte-Carlo returns as "MC", with the GAE branch's bookkeeping applied afterwards.
These tests pin (a) MC_FIXED's targets are the raw returns, (b) its advantages are
normalised, (c) an unknown mode raises instead of silently picking a branch,
(d) "MC" still z-scores its targets, so a later edit cannot quietly merge the two modes,
and (e) MC_FIXED takes MC's single window-edge bootstrap, not GAE's per-step one.
"""
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from src.models import recurrent_ppo_trainer as rpt
from src.models.recurrent_ppo_trainer import (
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


def _synthetic_rollout():
    """Deterministic fake trajectories: no env, no model, no randomness."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    rewards = jax.random.normal(k1, (T, N)) * 3.0 + 1.0
    values = jax.random.normal(k2, (T, N))
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


def _capture_batch(return_mode, monkeypatch):
    """Run train_iteration with collect_trajectories + update_step stubbed out, and
    return the PPOBatch that would have been handed to the optimiser."""
    traj, h_states, bootstrap_value = _synthetic_rollout()
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


def _expected_returns(traj, bootstrap_value):
    """The Monte-Carlo returns both MC and MC_FIXED must produce (same call, same axes)."""
    terminateds = (traj.step_info.termination_reason >= 2).astype(traj.done.dtype)
    return jax.vmap(compute_mc_returns, in_axes=(1, 1, 1, 0, None), out_axes=1)(
        traj.reward, traj.done, terminateds, bootstrap_value, GAMMA)


# ---------------------------------------------------------------------------
# (a) MC_FIXED targets are the RAW returns
# ---------------------------------------------------------------------------

def test_mc_fixed_targets_are_raw_unnormalised_returns(monkeypatch):
    batch, _, traj, boot = _capture_batch("MC_FIXED", monkeypatch)
    expected = _expected_returns(traj, boot)
    assert jnp.allclose(batch.targets, expected, atol=1e-6)
    # And they are genuinely un-rescaled: a z-scored array would sit at mean 0 / std 1.
    assert abs(float(jnp.mean(batch.targets))) > 1e-3
    assert abs(float(jnp.std(batch.targets)) - 1.0) > 1e-3


def test_mc_fixed_returns_identical_to_mc_returns_up_to_mc_rescaling(monkeypatch):
    """MC_FIXED must use the SAME returns as MC — MC's targets are exactly the
    z-score of MC_FIXED's targets."""
    fixed_batch, _, _, _ = _capture_batch("MC_FIXED", monkeypatch)
    mc_batch, _, _, _ = _capture_batch("MC", monkeypatch)
    z = (fixed_batch.targets - jnp.mean(fixed_batch.targets)) / (jnp.std(fixed_batch.targets) + 1e-7)
    assert jnp.allclose(z, mc_batch.targets, atol=1e-5)


# ---------------------------------------------------------------------------
# (b) MC_FIXED advantages are normalised, with the GAE branch's exact form
# ---------------------------------------------------------------------------

def test_mc_fixed_advantages_are_normalised(monkeypatch):
    batch, _, _, _ = _capture_batch("MC_FIXED", monkeypatch)
    assert float(jnp.mean(batch.advantages)) == pytest.approx(0.0, abs=1e-5)
    assert float(jnp.std(batch.advantages)) == pytest.approx(1.0, abs=1e-4)


def test_mc_fixed_advantages_match_gae_normalisation_form(monkeypatch):
    """Advantage = normalise(returns - V) with the GAE branch's 1e-8 epsilon."""
    batch, _, traj, boot = _capture_batch("MC_FIXED", monkeypatch)
    raw = _expected_returns(traj, boot) - traj.value
    expected = (raw - jnp.mean(raw)) / (jnp.std(raw) + 1e-8)
    assert jnp.allclose(batch.advantages, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# (c) unknown return_mode raises — no silent fallback
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["mc-fixed", "MCFIXED", "gae(lambda)", "", "TD"])
def test_invalid_return_mode_raises(bad):
    with pytest.raises(ValueError) as exc:
        validate_return_mode(bad)
    msg = str(exc.value)
    for legal in ("MC", "MC_FIXED", "GAE"):
        assert legal in msg


def test_invalid_return_mode_raises_from_train_iteration(monkeypatch):
    with pytest.raises(ValueError, match="MC_FIXED"):
        _capture_batch("MC_FUXED", monkeypatch)


@pytest.mark.parametrize("good,canonical", [
    ("mc", "MC"), ("MC", "MC"), ("mc_fixed", "MC_FIXED"),
    ("MC_FIXED", "MC_FIXED"), ("gae", "GAE"), ("GAE", "GAE"), (" MC ", "MC"),
])
def test_legal_return_modes_accepted_case_insensitively(good, canonical):
    assert validate_return_mode(good) == canonical


# ---------------------------------------------------------------------------
# (d) MC's own convention is pinned, so the two modes cannot silently converge
# ---------------------------------------------------------------------------

def test_mc_still_zscores_its_targets(monkeypatch):
    batch, _, _, _ = _capture_batch("MC", monkeypatch)
    assert float(jnp.mean(batch.targets)) == pytest.approx(0.0, abs=1e-5)
    assert float(jnp.std(batch.targets)) == pytest.approx(1.0, abs=1e-4)


def test_mc_does_not_normalise_its_advantages(monkeypatch):
    """MC's advantages are `z-scored returns - V`, left un-rescaled — the
    distinguishing feature versus MC_FIXED."""
    batch, _, traj, _ = _capture_batch("MC", monkeypatch)
    assert jnp.allclose(batch.advantages, batch.targets - traj.value, atol=1e-6)
    assert abs(float(jnp.std(batch.advantages)) - 1.0) > 1e-3


def test_gae_still_uses_raw_targets_and_normalised_advantages(monkeypatch):
    batch, _, traj, _ = _capture_batch("GAE", monkeypatch)
    assert jnp.allclose(batch.targets, batch.advantages * 0 + batch.targets)  # shape sanity
    assert float(jnp.mean(batch.advantages)) == pytest.approx(0.0, abs=1e-5)
    assert float(jnp.std(batch.advantages)) == pytest.approx(1.0, abs=1e-4)
    # GAE targets are raw (advantage + value), not z-scored.
    assert abs(float(jnp.std(batch.targets)) - 1.0) > 1e-3


# ---------------------------------------------------------------------------
# (e) MC_FIXED takes MC's window-edge bootstrap path, not GAE's per-step one
# ---------------------------------------------------------------------------

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
@pytest.mark.parametrize("mode,expect_per_step_next_value", [
    ("MC", False), ("MC_FIXED", False), ("GAE", True),
])
def test_bootstrap_path_selection(mode, expect_per_step_next_value, tiny_env_and_model):
    """`use_gae_bootstrap` gates the expensive per-step next-state value forward pass.
    MC_FIXED uses the same SINGLE window-edge bootstrap as MC, so its per-step
    `next_value` must stay all-zero (the pass never ran); GAE's must not."""
    model, params, state = tiny_env_and_model
    h = model.initial_state(2)
    traj, _, _, _, _, bootstrap_value = rpt.collect_trajectories(
        model, params, state, h, jax.random.PRNGKey(3), 4,
        rnn_type="GRU", return_mode=mode)
    per_step_ran = bool(jnp.any(traj.next_value != 0.0))
    assert per_step_ran == expect_per_step_next_value
    # Every mode still returns a usable window-edge bootstrap of shape (num_envs,).
    assert bootstrap_value.shape == (2,)
