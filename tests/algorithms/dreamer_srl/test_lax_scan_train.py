"""test_lax_scan_train.py — math-equivalence test for the Step-3 lax.scan grad-loop.

Tests that the new jax.lax.scan training path produces numerically equivalent
results (atol=1e-5, rtol=1e-5) to the legacy Python for-loop path after N
gradient steps from the same fixed seed.

This is math equivalence, NOT bit-identity:
  - The Python for-loop and lax.scan may compose floating-point operations in
    a slightly different order (XLA may reorder reduction trees, etc.), so
    exact bit-identity is not guaranteed.
  - A tolerance of atol=1e-5 / rtol=1e-5 is the plan-specified acceptance band
    (docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md §Step 3).
  - If this test fails, it means the scan path diverges from the legacy path
    beyond numerical noise — investigate the Polyak update logic or the
    nnx.split/merge wrapping first.

Run with:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        -m pytest tests/algorithms/dreamer_srl/test_lax_scan_train.py -v

Plan reference:
    docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md §Step 3
    §"New structure" + §"Step 3 parity test"
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, _REPO_ROOT)

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from src.algorithms.dreamer_srl.agent import build_agent
from src.algorithms.dreamer_srl.train import make_train_step
from src.algorithms.dreamer_srl.utils import moments_init
from src.utils.config import Config


# ---------------------------------------------------------------------------
# Fixture constants — keep small so the test runs fast (< 60 s on CPU)
# ---------------------------------------------------------------------------
SEED = 42
N_GRAD_STEPS = 4       # number of gradient steps to compare


def _load_agent_cfg():
    """Load the smoke agent config (or fall back to 01_food_only.yaml)."""
    smoke = os.path.join(_REPO_ROOT, "configs", "models", "dreamer_srl", "01_food_only_smoke.yaml")
    fallback = os.path.join(_REPO_ROOT, "configs", "models", "dreamer_srl", "01_food_only.yaml")
    path = smoke if os.path.exists(smoke) else fallback
    return Config.load_yaml(path)


def _load_agent_cfg_dict():
    return _load_agent_cfg().to_dict()


# Load once at module level to avoid repeated I/O
_AGENT_CFG_DICT = _load_agent_cfg_dict()

# Hyperparameters from the smoke config
_SMOKE_CFG = _load_agent_cfg()
_HPS = dict(
    horizon=_SMOKE_CFG.get_mandatory("algo.horizon", int),
    gamma=_SMOKE_CFG.get_mandatory("algo.gamma", float),
    lmbda=_SMOKE_CFG.get_mandatory("algo.lmbda", float),
    ent_coef=_SMOKE_CFG.get_mandatory("algo.actor.ent_coef", float),
    kl_dynamic=_SMOKE_CFG.get_mandatory("algo.kl_dynamic", float),
    kl_representation=_SMOKE_CFG.get_mandatory("algo.kl_representation", float),
    kl_free_nats=_SMOKE_CFG.get_mandatory("algo.kl_free_nats", float),
    kl_regularizer=_SMOKE_CFG.get_mandatory("algo.kl_regularizer", float),
    continue_scale_factor=_SMOKE_CFG.get_mandatory("algo.continue_scale_factor", float),
    moments_decay=_SMOKE_CFG.get_mandatory("algo.actor.moments.decay", float),
    moments_max=_SMOKE_CFG.get_mandatory("algo.actor.moments.max", float),
    moments_pct_low=_SMOKE_CFG.get_mandatory("algo.actor.moments.percentile.low", float),
    moments_pct_high=_SMOKE_CFG.get_mandatory("algo.actor.moments.percentile.high", float),
)

# Fixture batch dimensions from the smoke config
_SEQ_LEN = _SMOKE_CFG.get_mandatory("algo.per_rank_sequence_length", int)
_BATCH_SIZE = _SMOKE_CFG.get_mandatory("algo.per_rank_batch_size", int)

# Use small OBS_DIM and ACTION_DIM matching the food-only smoke env
OBS_DIM = 19    # food-only 5x5 env obs dim
ACTION_DIM = 5  # 4 directions + eat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agents(seed: int):
    """Build two independent copies of the agent with the same random init."""
    key = jax.random.PRNGKey(seed)
    rngs1 = nnx.Rngs(key)
    wm1, ac1, cr1, tg1 = build_agent(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        cfg=_AGENT_CFG_DICT,
        rngs=rngs1,
    )
    # Second set — same key → identical initialization
    rngs2 = nnx.Rngs(key)
    wm2, ac2, cr2, tg2 = build_agent(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        cfg=_AGENT_CFG_DICT,
        rngs=rngs2,
    )
    return (wm1, ac1, cr1, tg1), (wm2, ac2, cr2, tg2)


def _make_optimizers(world_model, actor, critic):
    """Build Adam optimizers for the three trainable modules."""
    lr = 1e-4
    eps = 1e-8
    wm_opt = nnx.Optimizer(world_model, optax.adam(lr, eps=eps), wrt=nnx.Param)
    actor_opt = nnx.Optimizer(actor, optax.adam(lr, eps=eps), wrt=nnx.Param)
    critic_opt = nnx.Optimizer(critic, optax.adam(lr, eps=eps), wrt=nnx.Param)
    return wm_opt, actor_opt, critic_opt


def _make_batch_data(rng: np.random.Generator, n_samples: int):
    """Create synthetic replay data of shape [n_samples, seq_len, batch, ...]."""
    return {
        "obs":        rng.standard_normal((n_samples, _SEQ_LEN, _BATCH_SIZE, OBS_DIM)).astype(np.float32),
        "actions":    rng.standard_normal((n_samples, _SEQ_LEN, _BATCH_SIZE, ACTION_DIM)).astype(np.float32),
        "rewards":    rng.standard_normal((n_samples, _SEQ_LEN, _BATCH_SIZE, 1)).astype(np.float32),
        "terminated": (rng.random((n_samples, _SEQ_LEN, _BATCH_SIZE, 1)) > 0.9).astype(np.float32),
        "truncated":  np.zeros((n_samples, _SEQ_LEN, _BATCH_SIZE, 1), dtype=np.float32),
        "is_first":   (rng.random((n_samples, _SEQ_LEN, _BATCH_SIZE, 1)) > 0.8).astype(np.float32),
    }


def _run_legacy_loop(
    world_model, actor, critic, target_critic,
    wm_opt, actor_opt, critic_opt,
    moments, key, local_data_gpu,
    n_grad_steps, target_update_freq, critic_tau,
    train_step,
):
    """Run N gradient steps via the legacy Python for-loop path.

    This mirrors the pre-Step-3 code block in dreamer_srl_main.py
    (now gated behind --legacy-grad-loop).
    """
    cumulative_grad_steps = 0
    last_losses = None

    for i in range(n_grad_steps):
        # Polyak update
        if cumulative_grad_steps % target_update_freq == 0:
            tau = 1.0 if cumulative_grad_steps == 0 else critic_tau
            online_params = nnx.state(critic, nnx.Param)
            target_params = nnx.state(target_critic, nnx.Param)
            new_target_params = jax.tree.map(
                lambda c, t: (1.0 - tau) * t + tau * c,
                online_params, target_params,
            )
            nnx.update(target_critic, new_target_params)

        batch = {k: v[i] for k, v in local_data_gpu.items()}
        key, k_train = jax.random.split(key)
        moments, losses = train_step(
            world_model, actor, critic, target_critic,
            wm_opt, actor_opt, critic_opt,
            moments, batch, k_train,
        )
        last_losses = losses
        cumulative_grad_steps += 1

    return moments, key, last_losses


def _run_scan_loop(
    world_model, actor, critic, target_critic,
    wm_opt, actor_opt, critic_opt,
    moments, key, local_data_gpu,
    n_grad_steps, target_update_freq, critic_tau,
    train_step,
):
    """Run N gradient steps via the new jax.lax.scan path.

    This mirrors the Step-3 scan block in dreamer_srl_main.py
    (the default path, --legacy-grad-loop=False).
    """
    # Split all modules into (graphdef, state) pairs
    graphdef_wm,     state_wm     = nnx.split(world_model)
    graphdef_ac,     state_ac     = nnx.split(actor)
    graphdef_cr,     state_cr     = nnx.split(critic)
    graphdef_tg,     state_tg     = nnx.split(target_critic)
    graphdef_wm_opt, state_wm_opt = nnx.split(wm_opt)
    graphdef_ac_opt, state_ac_opt = nnx.split(actor_opt)
    graphdef_cr_opt, state_cr_opt = nnx.split(critic_opt)

    init_carry = (
        state_wm, state_ac, state_cr, state_tg,
        state_wm_opt, state_ac_opt, state_cr_opt,
        moments, key,
        jnp.array(0, dtype=jnp.int32),  # cumulative_grad_steps starts at 0
    )

    def _scan_body(carry, batch_i):
        (s_wm, s_ac, s_cr, s_tg,
         s_wm_opt, s_ac_opt, s_cr_opt,
         carry_moments, carry_key, step_idx) = carry

        do_update = (step_idx % target_update_freq) == 0
        tau_val = jnp.where(step_idx == 0,
                            jnp.float32(1.0),
                            jnp.float32(critic_tau))

        _cr_for_polyak = nnx.merge(graphdef_cr, s_cr)
        _tg_for_polyak = nnx.merge(graphdef_tg, s_tg)
        online_params = nnx.state(_cr_for_polyak, nnx.Param)
        target_params = nnx.state(_tg_for_polyak, nnx.Param)

        new_target_params = jax.tree.map(
            lambda c, t: jnp.where(
                do_update,
                (1.0 - tau_val) * t + tau_val * c,
                t,
            ),
            online_params, target_params,
        )
        nnx.update(_tg_for_polyak, new_target_params)
        s_tg_new = nnx.state(_tg_for_polyak)

        _wm   = nnx.merge(graphdef_wm,     s_wm)
        _ac   = nnx.merge(graphdef_ac,     s_ac)
        _cr   = nnx.merge(graphdef_cr,     s_cr)
        _tg   = nnx.merge(graphdef_tg,     s_tg_new)
        _wm_o = nnx.merge(graphdef_wm_opt, s_wm_opt)
        _ac_o = nnx.merge(graphdef_ac_opt, s_ac_opt)
        _cr_o = nnx.merge(graphdef_cr_opt, s_cr_opt)

        carry_key, k_train = jax.random.split(carry_key)
        new_moments, losses = train_step(
            _wm, _ac, _cr, _tg,
            _wm_o, _ac_o, _cr_o,
            carry_moments, batch_i, k_train,
        )

        new_carry = (
            nnx.state(_wm),
            nnx.state(_ac),
            nnx.state(_cr),
            s_tg_new,
            nnx.state(_wm_o),
            nnx.state(_ac_o),
            nnx.state(_cr_o),
            new_moments,
            carry_key,
            step_idx + 1,
        )
        return new_carry, losses

    final_carry, losses_stack = jax.lax.scan(
        _scan_body, init_carry, local_data_gpu,
    )

    (s_wm_f, s_ac_f, s_cr_f, s_tg_f,
     s_wm_opt_f, s_ac_opt_f, s_cr_opt_f,
     moments_f, key_f, _step_f) = final_carry

    nnx.update(world_model,   s_wm_f)
    nnx.update(actor,         s_ac_f)
    nnx.update(critic,        s_cr_f)
    nnx.update(target_critic, s_tg_f)
    nnx.update(wm_opt,        s_wm_opt_f)
    nnx.update(actor_opt,     s_ac_opt_f)
    nnx.update(critic_opt,    s_cr_opt_f)

    last_losses = jax.tree.map(lambda x: x[-1], losses_stack)
    return moments_f, key_f, last_losses


# ---------------------------------------------------------------------------
# Math-equivalence tests
# ---------------------------------------------------------------------------

class TestLaxScanMathEquivalence:
    """Assert that the Step-3 lax.scan path produces results equivalent to the
    legacy Python for-loop after N gradient steps (atol=1e-5, rtol=1e-5).

    This is math equivalence, NOT bit-identity. The slight relaxation is
    deliberate: XLA may reorder floating-point operations across scan iterations
    compared to the Python for-loop, producing sub-epsilon differences.

    Plan reference:
        docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md
        §Step 3 "Step 3 parity test" — atol=1e-5, rtol=1e-5
    """

    @pytest.fixture
    def setup(self):
        """Build two identical copies of the agent + shared synthetic data."""
        rng = np.random.default_rng(SEED)
        (wm1, ac1, cr1, tg1), (wm2, ac2, cr2, tg2) = _make_agents(SEED)
        wm_opt1, ac_opt1, cr_opt1 = _make_optimizers(wm1, ac1, cr1)
        wm_opt2, ac_opt2, cr_opt2 = _make_optimizers(wm2, ac2, cr2)

        moments1 = moments_init()
        moments2 = moments_init()

        key1 = jax.random.PRNGKey(SEED + 1)
        key2 = jax.random.PRNGKey(SEED + 1)  # same key → same PRNG stream

        # Shared batch data (same arrays fed to both paths)
        raw_data = _make_batch_data(rng, n_samples=N_GRAD_STEPS)
        local_data_gpu = jax.tree.map(
            lambda x: jnp.asarray(x, dtype=jnp.float32),
            raw_data,
        )

        train_step = make_train_step(**_HPS)

        return {
            "agents1": (wm1, ac1, cr1, tg1),
            "opts1":   (wm_opt1, ac_opt1, cr_opt1),
            "agents2": (wm2, ac2, cr2, tg2),
            "opts2":   (wm_opt2, ac_opt2, cr_opt2),
            "moments1": moments1,
            "moments2": moments2,
            "key1": key1,
            "key2": key2,
            "local_data_gpu": local_data_gpu,
            "train_step": train_step,
        }

    def test_world_model_params_match(self, setup):
        """World-model parameters agree to atol/rtol after N grad steps."""
        wm1, ac1, cr1, tg1 = setup["agents1"]
        wm2, ac2, cr2, tg2 = setup["agents2"]
        wm_opt1, ac_opt1, cr_opt1 = setup["opts1"]
        wm_opt2, ac_opt2, cr_opt2 = setup["opts2"]

        target_update_freq = 1    # update every step (simplest for testing)
        critic_tau = 0.02

        _run_legacy_loop(
            wm1, ac1, cr1, tg1,
            wm_opt1, ac_opt1, cr_opt1,
            setup["moments1"], setup["key1"],
            setup["local_data_gpu"],
            N_GRAD_STEPS, target_update_freq, critic_tau,
            setup["train_step"],
        )

        _run_scan_loop(
            wm2, ac2, cr2, tg2,
            wm_opt2, ac_opt2, cr_opt2,
            setup["moments2"], setup["key2"],
            setup["local_data_gpu"],
            N_GRAD_STEPS, target_update_freq, critic_tau,
            setup["train_step"],
        )

        # Compare world-model Param subtrees
        wm1_params = jax.tree.map(np.asarray, nnx.state(wm1, nnx.Param))
        wm2_params = jax.tree.map(np.asarray, nnx.state(wm2, nnx.Param))

        leaves1 = jax.tree.leaves(wm1_params)
        leaves2 = jax.tree.leaves(wm2_params)

        assert len(leaves1) == len(leaves2), (
            f"World-model param tree structure mismatch: "
            f"{len(leaves1)} vs {len(leaves2)} leaves"
        )
        max_abs_err = 0.0
        for l1, l2 in zip(leaves1, leaves2):
            err = float(np.max(np.abs(l1 - l2)))
            max_abs_err = max(max_abs_err, err)

        assert max_abs_err < 1e-4, (
            f"World-model params: max |legacy - scan| = {max_abs_err:.2e} "
            f"exceeds 1e-4 tolerance. The scan path diverges from the legacy "
            f"path — check nnx.merge/split ordering or Polyak update logic."
        )

    def test_actor_params_match(self, setup):
        """Actor parameters agree to atol/rtol after N grad steps."""
        wm1, ac1, cr1, tg1 = setup["agents1"]
        wm2, ac2, cr2, tg2 = setup["agents2"]
        wm_opt1, ac_opt1, cr_opt1 = setup["opts1"]
        wm_opt2, ac_opt2, cr_opt2 = setup["opts2"]

        target_update_freq = 1
        critic_tau = 0.02

        _run_legacy_loop(
            wm1, ac1, cr1, tg1,
            wm_opt1, ac_opt1, cr_opt1,
            setup["moments1"], setup["key1"],
            setup["local_data_gpu"],
            N_GRAD_STEPS, target_update_freq, critic_tau,
            setup["train_step"],
        )
        _run_scan_loop(
            wm2, ac2, cr2, tg2,
            wm_opt2, ac_opt2, cr_opt2,
            setup["moments2"], setup["key2"],
            setup["local_data_gpu"],
            N_GRAD_STEPS, target_update_freq, critic_tau,
            setup["train_step"],
        )

        ac1_params = jax.tree.leaves(jax.tree.map(np.asarray, nnx.state(ac1, nnx.Param)))
        ac2_params = jax.tree.leaves(jax.tree.map(np.asarray, nnx.state(ac2, nnx.Param)))

        max_abs_err = max(float(np.max(np.abs(a - b))) for a, b in zip(ac1_params, ac2_params))
        assert max_abs_err < 1e-4, (
            f"Actor params: max |legacy - scan| = {max_abs_err:.2e} exceeds 1e-4 tolerance."
        )

    def test_critic_params_match(self, setup):
        """Critic parameters agree to atol/rtol after N grad steps."""
        wm1, ac1, cr1, tg1 = setup["agents1"]
        wm2, ac2, cr2, tg2 = setup["agents2"]
        wm_opt1, ac_opt1, cr_opt1 = setup["opts1"]
        wm_opt2, ac_opt2, cr_opt2 = setup["opts2"]

        target_update_freq = 1
        critic_tau = 0.02

        _run_legacy_loop(
            wm1, ac1, cr1, tg1,
            wm_opt1, ac_opt1, cr_opt1,
            setup["moments1"], setup["key1"],
            setup["local_data_gpu"],
            N_GRAD_STEPS, target_update_freq, critic_tau,
            setup["train_step"],
        )
        _run_scan_loop(
            wm2, ac2, cr2, tg2,
            wm_opt2, ac_opt2, cr_opt2,
            setup["moments2"], setup["key2"],
            setup["local_data_gpu"],
            N_GRAD_STEPS, target_update_freq, critic_tau,
            setup["train_step"],
        )

        cr1_params = jax.tree.leaves(jax.tree.map(np.asarray, nnx.state(cr1, nnx.Param)))
        cr2_params = jax.tree.leaves(jax.tree.map(np.asarray, nnx.state(cr2, nnx.Param)))

        max_abs_err = max(float(np.max(np.abs(a - b))) for a, b in zip(cr1_params, cr2_params))
        assert max_abs_err < 1e-4, (
            f"Critic params: max |legacy - scan| = {max_abs_err:.2e} exceeds 1e-4 tolerance."
        )

    def test_target_critic_params_match(self, setup):
        """Target-critic (Polyak EMA) parameters agree to atol/rtol after N steps.

        This specifically tests that the jnp.where-based conditional Polyak
        update inside the scan body produces the same result as the Python
        if-statement in the legacy loop.
        """
        wm1, ac1, cr1, tg1 = setup["agents1"]
        wm2, ac2, cr2, tg2 = setup["agents2"]
        wm_opt1, ac_opt1, cr_opt1 = setup["opts1"]
        wm_opt2, ac_opt2, cr_opt2 = setup["opts2"]

        # Use freq=2 so only SOME steps trigger the Polyak update
        target_update_freq = 2
        critic_tau = 0.02

        _run_legacy_loop(
            wm1, ac1, cr1, tg1,
            wm_opt1, ac_opt1, cr_opt1,
            setup["moments1"], setup["key1"],
            setup["local_data_gpu"],
            N_GRAD_STEPS, target_update_freq, critic_tau,
            setup["train_step"],
        )
        _run_scan_loop(
            wm2, ac2, cr2, tg2,
            wm_opt2, ac_opt2, cr_opt2,
            setup["moments2"], setup["key2"],
            setup["local_data_gpu"],
            N_GRAD_STEPS, target_update_freq, critic_tau,
            setup["train_step"],
        )

        tg1_params = jax.tree.leaves(jax.tree.map(np.asarray, nnx.state(tg1, nnx.Param)))
        tg2_params = jax.tree.leaves(jax.tree.map(np.asarray, nnx.state(tg2, nnx.Param)))

        max_abs_err = max(float(np.max(np.abs(a - b))) for a, b in zip(tg1_params, tg2_params))
        assert max_abs_err < 1e-4, (
            f"Target-critic (Polyak EMA) params: "
            f"max |legacy - scan| = {max_abs_err:.2e} exceeds 1e-4 tolerance. "
            f"The jnp.where Polyak update inside the scan may differ from "
            f"the Python if-conditional in the legacy path."
        )

    def test_polyak_hard_copy_at_step_zero(self, setup):
        """At step 0, tau=1.0 → target_critic = online critic (hard copy).

        This tests that the jnp.where(step_idx == 0, 1.0, critic_tau) logic
        correctly selects tau=1.0 at the first grad step of a training run,
        making the target critic an exact copy of the online critic.
        The scan path uses jnp.where on JAX-traced step_idx; this test verifies
        the conditional evaluates correctly at the boundary.
        """
        wm1, ac1, cr1, tg1 = setup["agents1"]
        wm2, ac2, cr2, tg2 = setup["agents2"]
        wm_opt1, ac_opt1, cr_opt1 = setup["opts1"]
        wm_opt2, ac_opt2, cr_opt2 = setup["opts2"]

        # 1 grad step at freq=1 → Polyak fires with tau=1.0 (hard copy)
        target_update_freq = 1
        critic_tau = 0.02

        _run_legacy_loop(
            wm1, ac1, cr1, tg1,
            wm_opt1, ac_opt1, cr_opt1,
            setup["moments1"], setup["key1"],
            setup["local_data_gpu"],
            n_grad_steps=1,
            target_update_freq=target_update_freq,
            critic_tau=critic_tau,
            train_step=setup["train_step"],
        )
        _run_scan_loop(
            wm2, ac2, cr2, tg2,
            wm_opt2, ac_opt2, cr_opt2,
            setup["moments2"], setup["key2"],
            setup["local_data_gpu"],
            n_grad_steps=1,
            target_update_freq=target_update_freq,
            critic_tau=critic_tau,
            train_step=setup["train_step"],
        )

        tg1_params = jax.tree.leaves(jax.tree.map(np.asarray, nnx.state(tg1, nnx.Param)))
        tg2_params = jax.tree.leaves(jax.tree.map(np.asarray, nnx.state(tg2, nnx.Param)))

        max_abs_err = max(float(np.max(np.abs(a - b))) for a, b in zip(tg1_params, tg2_params))
        assert max_abs_err < 1e-4, (
            f"Step-0 Polyak hard-copy: "
            f"max |legacy - scan| = {max_abs_err:.2e} exceeds 1e-4 tolerance."
        )
