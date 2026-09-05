"""Tests for the GAE_NORM return mode (GAE estimator, MC's normalisation scheme).

Plain-language context: recurrent PPO learns two numbers from each rollout — a *critic
target* (what total future reward the value head should have predicted) and an
*advantage* (how much better an action turned out than the critic expected). Two
independent choices go into producing that pair: WHICH ESTIMATOR builds it (Monte-Carlo
returns, or generalised advantage estimation, "GAE"), and HOW the two numbers are
RESCALED before training:

  - "MC" uses the Monte-Carlo estimator with a MATCHED scale: the target is rescaled to
    mean 0 / spread 1 (a "z-score"), the critic therefore learns on that scale, and the
    advantage is simply `target - critic prediction`, left alone.
  - "MC_FIXED" and "GAE" use a SPLIT scale: the critic target stays raw (spread ~24 in
    this project's environment) while the advantages are separately rescaled to spread 1.

On the 10x10 jump-attack environment, one million episodes and five seeds per arm, agents
survived on average 138.8 steps under "MC" but only 40.0 under "MC_FIXED" and 41.9 under
"GAE". "GAE_NORM" is the fourth arm that separates the two candidate explanations: it
applies MC's MATCHED-scale scheme to the GAE estimator, filling the missing cell of the
2x2 (estimator) x (scale) grid.

These tests pin (a) GAE_NORM's target is the z-score of `GAE advantage + value`, with
MC's 1e-7 epsilon; (b) its advantage is the un-rescaled residual `target - value`, and is
NOT unit-variance — the property that distinguishes it from "GAE", where the advantage is
rescaled; (c) GAE_NORM takes GAE's PER-STEP bootstrap path (`compute_gae` reads the
per-step next-state value), unlike MC/MC_FIXED which take a single window-edge bootstrap;
and (d) an unknown mode raises an error naming all four legal modes.
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
    compute_gae,
    train_iteration,
    validate_return_mode,
)

T, N = 6, 3          # rollout steps, parallel envs
OBS_DIM, HID = 5, 4
GAMMA, LMBDA = 0.9, 0.95


class _Cfg:
    """Static config stub carrying only the fields train_iteration reads."""
    rnn_type = "GRU"
    num_steps = T
    num_epochs = 1
    gamma = GAMMA
    gae_lambda = LMBDA

    def __init__(self, return_mode):
        self.return_mode = return_mode


def _synthetic_rollout(scale=1.0):
    """Deterministic fake trajectories: no env, no model, no randomness.

    Unlike the MC_FIXED fixture this one carries a NON-ZERO per-step `next_value`,
    because the GAE estimator (and therefore GAE_NORM) actually consumes it.

    `scale` shrinks every reward/value magnitude together. A tiny scale drives the
    target's spread down to the size of the z-score epsilon, which is what makes MC's
    1e-7 and GAE's 1e-8 epsilons distinguishable (see the epsilon test).
    """
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    rewards = (jax.random.normal(k1, (T, N)) * 3.0 + 1.0) * scale
    values = jax.random.normal(k2, (T, N)) * scale
    next_values = (jax.random.normal(k4, (T, N)) * 2.0 + 0.5) * scale
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
        step_info=step_info, next_value=next_values,
    )
    h_states = jnp.zeros((T, N, HID))
    bootstrap_value = jnp.array([7.0, -3.0, 0.5])
    return traj, h_states, bootstrap_value


def _capture_batch(return_mode, monkeypatch, scale=1.0):
    """Run train_iteration with collect_trajectories + update_step stubbed out, and
    return the PPOBatch that would have been handed to the optimiser."""
    traj, h_states, bootstrap_value = _synthetic_rollout(scale)
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


def _expected_raw_gae(traj):
    """The raw GAE(lambda) advantages both GAE and GAE_NORM must produce."""
    terminateds = (traj.step_info.termination_reason >= 2).astype(traj.done.dtype)
    return jax.vmap(compute_gae, in_axes=(1, 1, 1, 1, 1, None, None), out_axes=1)(
        traj.reward, traj.value, traj.next_value, traj.done, terminateds, GAMMA, LMBDA)


# ---------------------------------------------------------------------------
# (a) GAE_NORM's target is the z-score of (raw GAE advantage + value)
# ---------------------------------------------------------------------------

def test_gae_norm_targets_are_zscore_of_raw_gae_target(monkeypatch):
    batch, _, traj, _ = _capture_batch("GAE_NORM", monkeypatch)
    raw_target = _expected_raw_gae(traj) + traj.value
    expected = (raw_target - jnp.mean(raw_target)) / (jnp.std(raw_target) + 1e-7)
    assert jnp.allclose(batch.targets, expected, atol=1e-6)


def test_gae_norm_targets_are_unit_scale(monkeypatch):
    """The z-score puts the critic target at mean 0 / spread 1 — MC's scheme."""
    batch, _, _, _ = _capture_batch("GAE_NORM", monkeypatch)
    assert float(jnp.mean(batch.targets)) == pytest.approx(0.0, abs=1e-5)
    assert float(jnp.std(batch.targets)) == pytest.approx(1.0, abs=1e-4)


def test_gae_norm_targets_are_zscore_of_gae_targets(monkeypatch):
    """Same estimator as "GAE": GAE_NORM's target is exactly the z-score of GAE's."""
    gn_batch, _, _, _ = _capture_batch("GAE_NORM", monkeypatch)
    gae_batch, _, _, _ = _capture_batch("GAE", monkeypatch)
    z = (gae_batch.targets - jnp.mean(gae_batch.targets)) / (jnp.std(gae_batch.targets) + 1e-7)
    assert jnp.allclose(z, gn_batch.targets, atol=1e-5)


def test_gae_norm_uses_mc_epsilon_not_gae_epsilon(monkeypatch):
    """The z-score epsilon must be MC's 1e-7, not GAE's 1e-8 — copying MC's scheme
    exactly is the point of this mode.

    On ordinary data the two epsilons are indistinguishable (both are swamped by a
    spread of order 1), so this runs the branch on a rollout scaled down to 1e-8, where
    the two denominators differ by roughly 5x.
    """
    tiny_scale = 1e-8
    traj, _, _ = _synthetic_rollout(tiny_scale)
    raw_target = _expected_raw_gae(traj) + traj.value
    centred = raw_target - jnp.mean(raw_target)
    expected_mc_eps = centred / (jnp.std(raw_target) + 1e-7)
    expected_gae_eps = centred / (jnp.std(raw_target) + 1e-8)
    # Sanity: at this scale the two epsilons really are separable.
    assert not jnp.allclose(expected_mc_eps, expected_gae_eps, rtol=1e-2)

    batch, _, _, _ = _capture_batch("GAE_NORM", monkeypatch, scale=tiny_scale)
    assert jnp.allclose(batch.targets, expected_mc_eps, rtol=1e-5)
    assert not jnp.allclose(batch.targets, expected_gae_eps, rtol=1e-2)


# ---------------------------------------------------------------------------
# (b) advantages = targets - values, and NOT unit-variance
# ---------------------------------------------------------------------------

def test_gae_norm_advantages_are_target_minus_value(monkeypatch):
    batch, _, traj, _ = _capture_batch("GAE_NORM", monkeypatch)
    assert jnp.allclose(batch.advantages, batch.targets - traj.value, atol=1e-6)


def test_gae_norm_advantages_are_not_renormalised(monkeypatch):
    """THE distinguishing property versus "GAE": the advantage is the un-rescaled
    residual, so it must NOT sit at spread 1 (GAE's advantages do, by construction)."""
    batch, _, _, _ = _capture_batch("GAE_NORM", monkeypatch)
    assert abs(float(jnp.std(batch.advantages)) - 1.0) > 1e-3
    gae_batch, _, _, _ = _capture_batch("GAE", monkeypatch)
    assert float(jnp.std(gae_batch.advantages)) == pytest.approx(1.0, abs=1e-4)
    assert not jnp.allclose(batch.advantages, gae_batch.advantages, atol=1e-3)


def test_gae_norm_mirrors_mc_advantage_convention(monkeypatch):
    """MC and GAE_NORM share a scheme: advantage == target - value, un-rescaled."""
    gn_batch, _, traj, _ = _capture_batch("GAE_NORM", monkeypatch)
    mc_batch, _, mc_traj, _ = _capture_batch("MC", monkeypatch)
    assert jnp.allclose(gn_batch.advantages, gn_batch.targets - traj.value, atol=1e-6)
    assert jnp.allclose(mc_batch.advantages, mc_batch.targets - mc_traj.value, atol=1e-6)


# ---------------------------------------------------------------------------
# (c) GAE_NORM takes GAE's PER-STEP bootstrap path, not MC's window-edge one
# ---------------------------------------------------------------------------

def test_gae_norm_reads_the_per_step_next_value(monkeypatch):
    """Unit-level: GAE_NORM's numbers change when `next_value` changes — proof the
    per-step bootstrap actually feeds the estimator (MC/MC_FIXED ignore it)."""
    traj, h_states, boot = _synthetic_rollout()
    base = _expected_raw_gae(traj)
    perturbed = _expected_raw_gae(traj._replace(next_value=traj.next_value + 5.0))
    assert not jnp.allclose(base, perturbed, atol=1e-3)


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
    ("MC", False), ("MC_FIXED", False), ("GAE", True), ("GAE_NORM", True),
])
def test_bootstrap_path_selection(mode, expect_per_step_next_value, tiny_env_and_model):
    """`use_gae_bootstrap` gates the per-step next-state value forward pass. GAE_NORM
    uses the GAE ESTIMATOR, which consumes the per-step `next_value`, so that pass MUST
    run for it — unlike MC/MC_FIXED, whose per-step `next_value` stays all-zero."""
    model, params, state = tiny_env_and_model
    h = model.initial_state(2)
    traj, _, _, _, _, bootstrap_value = rpt.collect_trajectories(
        model, params, state, h, jax.random.PRNGKey(3), 4,
        rnn_type="GRU", return_mode=mode)
    per_step_ran = bool(jnp.any(traj.next_value != 0.0))
    assert per_step_ran == expect_per_step_next_value
    # Every mode still returns a usable window-edge bootstrap of shape (num_envs,).
    assert bootstrap_value.shape == (2,)


@pytest.mark.integration
def test_gae_norm_next_value_matches_gae_next_value(tiny_env_and_model):
    """Same bootstrap path means the same per-step next-state values as "GAE"."""
    model, params, state = tiny_env_and_model
    h = model.initial_state(2)
    out = {}
    for mode in ("GAE", "GAE_NORM"):
        traj, *_ = rpt.collect_trajectories(
            model, params, state, h, jax.random.PRNGKey(3), 4,
            rnn_type="GRU", return_mode=mode)
        out[mode] = traj.next_value
    assert jnp.array_equal(out["GAE"], out["GAE_NORM"])


# ---------------------------------------------------------------------------
# (d) unknown return_mode raises, naming all four legal modes
# ---------------------------------------------------------------------------

def test_legal_return_modes_contain_the_four_expected():
    """GAE_NORM's four modes must all still be legal. The tuple is open-ended by design
    (MC_RAW was appended later); this test pins membership, not the exact tuple — the
    exact tuple is pinned once, in test_mc_raw_mode.py."""
    for legal in ("MC", "MC_FIXED", "GAE", "GAE_NORM"):
        assert legal in LEGAL_RETURN_MODES


@pytest.mark.parametrize("bad", ["gae-norm", "GAENORM", "gae_normalised", "", "TD"])
def test_invalid_return_mode_raises_naming_all_four(bad):
    with pytest.raises(ValueError) as exc:
        validate_return_mode(bad)
    msg = str(exc.value)
    for legal in ("MC", "MC_FIXED", "GAE", "GAE_NORM"):
        assert legal in msg


def test_invalid_return_mode_raises_from_train_iteration(monkeypatch):
    with pytest.raises(ValueError, match="GAE_NORM"):
        _capture_batch("GAE_NURM", monkeypatch)


@pytest.mark.parametrize("good,canonical", [
    ("gae_norm", "GAE_NORM"), ("GAE_NORM", "GAE_NORM"), (" gae_norm ", "GAE_NORM"),
])
def test_gae_norm_accepted_case_insensitively(good, canonical):
    assert validate_return_mode(good) == canonical


def test_plain_ppo_rejects_gae_norm():
    """Plain (non-recurrent) PPO does not implement GAE_NORM — it must say so loudly
    rather than silently falling through to its GAE branch."""
    from src.models.ppo_trainer import train_iteration_ppo

    class _PlainCfg:
        num_steps = 4
        return_mode = "GAE_NORM"

    with pytest.raises(ValueError) as exc:
        train_iteration_ppo(None, None, None, None, jax.random.PRNGKey(0), _PlainCfg())
    msg = str(exc.value)
    assert "GAE_NORM" in msg
    assert "RecurrentPPO" in msg
