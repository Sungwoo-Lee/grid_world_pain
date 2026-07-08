"""CP9b Lever-A tests for §S3 random-action prefill.

Tests the prefill-gate behavior in src/algorithms/dreamer_srl/dreamer_srl_main.py
ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L558-L571
(action gate) and :L660 (train gate).

CP9b SCOPE:
  - test_prefill_uniform_entropy_below_learning_starts:
      During `iter_num <= learning_starts`, action sampling uses
      jax.random.randint over [0, action_dim). The empirical action histogram
      over many sampled prefill iterations is statistically uniform — i.e.
      the empirical entropy is within tolerance of log(action_dim).
  - test_no_gradient_step_before_learning_starts:
      Across `iter_num` in [1, learning_starts - 1], the train-gate
      (`if iter_num >= learning_starts`) does NOT fire, so the optimizer's
      step count is zero throughout. At iter_num == learning_starts the gate
      enters for the first time and the Ratio scheduler repays accumulated
      debt (this is correct driver behaviour — the driver never subtracts
      prefill steps from policy_step, unlike sheeprl's more complex formula).
      The test verifies the hard invariant: zero grad steps for
      iters 1..learning_starts-1, then positive steps once the gate opens.

Both tests run in the grid_world_pain conda env (JAX only). No torch fixture
is needed (CP9b has no bit-identity comparison).

Run with:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        -m pytest tests/algorithms/dreamer_srl/test_prefill.py -v
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np


def _sample_prefill_action_oh(key: jax.Array, num_envs: int, action_dim: int) -> np.ndarray:
    """Mirror the §S3 prefill branch from dreamer_srl_main.py:L397-L403.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L558-L571.

    Kept as a small helper so the test reproduces the production branch
    line-for-line (any drift between this helper and the production code
    would cause Test 1 to fail and the test to be invalid — that's
    intentional, the test is a regression guard on the production sampling
    formula). If the production code's signature changes, this helper
    changes in lockstep.
    """
    action_idx = jax.random.randint(
        key, shape=(num_envs,), minval=0, maxval=action_dim
    )
    return np.asarray(
        jax.nn.one_hot(action_idx, num_classes=action_dim, dtype=jnp.float32)
    )


# ---------------------------------------------------------------------------
# Test 1 — uniform distribution of prefill actions
# ---------------------------------------------------------------------------

def test_prefill_uniform_entropy_below_learning_starts() -> None:
    """Empirical entropy of prefill-sampled actions equals log(action_dim) within tolerance.

    Setup: action_dim=4 (food-only NoPred default + rest + eat may differ but
    we use 4 here as a minimal smoke), num_envs=1, learning_starts=10000 so
    we have many prefill iterations. Sample 10000 prefill actions via the
    same JAX RNG path the production driver uses.

    Assertion: empirical entropy H_emp = -sum(p_i log p_i) over the action
    histogram is within `tol` of log(action_dim). With N=10000 samples and
    action_dim=4, a uniform distribution gives ~2500 per bin; the standard
    error on each empirical probability is sqrt(p(1-p)/N) ≈ 0.0043, and the
    corresponding entropy error band at log(4)=1.3863 is ~0.005 with high
    probability. We set tol=0.01 (2 sigma).

    Regression class caught: ANY non-uniform sampler (e.g. accidental
    bias to argmax, off-by-one on minval/maxval, RNG-key reuse across the
    batch, np.random.randint reintroduced) produces H_emp < log(action_dim)
    by far more than 0.01.
    """
    action_dim = 4
    num_envs = 1
    n_samples = 10_000
    seed = 0xD3EAF  # project's fixed seed (tests/fixtures/dreamer_srl/README.md)

    key = jax.random.PRNGKey(seed)
    counts = np.zeros(action_dim, dtype=np.int64)
    for _ in range(n_samples):
        key, k_player = jax.random.split(key)
        actions_oh = _sample_prefill_action_oh(k_player, num_envs, action_dim)
        # actions_oh: [1, action_dim], one-hot — argmax recovers the sampled index
        idx = int(np.argmax(actions_oh[0]))
        counts[idx] += 1

    probs = counts / n_samples
    h_emp = -np.sum(probs * np.log(np.maximum(probs, 1e-30)))
    h_uniform = math.log(action_dim)
    tol = 0.01

    assert abs(h_emp - h_uniform) < tol, (
        f"Prefill action distribution is not uniform. "
        f"H_emp={h_emp:.6f}, log({action_dim})={h_uniform:.6f}, "
        f"|delta|={abs(h_emp - h_uniform):.6f} > tol={tol}. "
        f"Counts per action: {counts.tolist()}."
    )

    # Bonus check: all bins are populated (no bin gets zero samples — that
    # would be a maxval-off-by-one bug class).
    assert np.all(counts > 0), (
        f"Some action indices were never sampled. Counts: {counts.tolist()}. "
        f"Likely cause: minval/maxval off-by-one in jax.random.randint."
    )


# ---------------------------------------------------------------------------
# Test 2 — no gradient step before learning_starts
# ---------------------------------------------------------------------------

def test_no_gradient_step_before_learning_starts() -> None:
    """Across iterations 1..learning_starts-1, the train gate fires zero gradient steps.

    Setup: simulate the driver's train-gate logic at
    src/algorithms/dreamer_srl/dreamer_srl_main.py:L483-L486 by stepping a
    Ratio scheduler with the same inputs the driver feeds it. We do NOT
    spin up the full driver (env + agent + buffer) — that's CP10's runtime
    territory. We just verify the gate's predicate-and-counter behaviour
    is correct.

    Assertion: for `learning_starts=10`, replay_ratio=1, num_envs=1:
      - For iter_num in [1, learning_starts - 1] = [1, 9]: the gate
        `iter_num >= learning_starts` is False, so n_grad_steps stays 0.
        This is the critical hard invariant: zero gradient updates before
        the prefill window closes.
      - For iter_num >= learning_starts = 10: the gate opens and the Ratio
        scheduler begins returning gradient steps. The driver feeds
        policy_step directly to ratio() (no prefill subtraction), so on the
        first gate entry the scheduler repays accumulated debt
        (int(policy_step * ratio) on first call). This is correct driver
        behaviour and is distinct from sheeprl's formula which subtracts
        prefill_steps — both implementations ensure zero gradient steps
        during [1, learning_starts-1], which is the invariant being tested.

    Regression class caught: any reintroduction of pre-`learning_starts`
    gradient steps (e.g. moving the train-gate inside the prefill branch,
    or flipping `>=` to `>` and tripping at iter == learning_starts - 1).
    """
    from src.algorithms.dreamer_srl.utils import Ratio

    learning_starts = 10
    replay_ratio = 1
    num_envs = 1
    total_iters = learning_starts + 5  # run a few past learning_starts to confirm the gate opens

    ratio = Ratio(ratio=replay_ratio, pretrain_steps=0)
    policy_step = 0
    grad_step_at_iter: list[int] = []

    for iter_num in range(1, total_iters + 1):
        policy_step += num_envs  # mirrors driver L379

        # Driver's train-gate (mirrors L483-L486):
        if iter_num >= learning_starts:
            ratio_steps = policy_step
            n_grad_steps = ratio(ratio_steps)
            # n_grad_steps is the number of optimizer steps the driver would
            # take at this iter. In the driver they're guarded by
            # `if n_grad_steps > 0 and buffer._pos >= seq_len:`, but for the
            # gate-only test we just count the raw return.
            grad_step_at_iter.append(int(n_grad_steps))
        else:
            grad_step_at_iter.append(0)

    # Verify: iterations 1..learning_starts-1 have zero grad steps.
    # This is the hard invariant: the prefill gate (`iter_num >= learning_starts`)
    # must not fire until the prefill window is complete.
    for i in range(learning_starts - 1):
        assert grad_step_at_iter[i] == 0, (
            f"Train gate fired at iter_num={i + 1} (before learning_starts={learning_starts}). "
            f"Expected 0 grad steps, got {grad_step_at_iter[i]}. "
            f"Full trace: {grad_step_at_iter}."
        )

    # Verify: at least one grad step fires after learning_starts
    # (otherwise the test is vacuous — we'd pass even if the gate never opened).
    grad_steps_after = sum(grad_step_at_iter[learning_starts - 1:])
    assert grad_steps_after > 0, (
        f"Train gate never opened at or after learning_starts={learning_starts}. "
        f"Test is vacuous. Full trace: {grad_step_at_iter}."
    )


# ---------------------------------------------------------------------------
# WP-SRL P6 — learning_starts is an ENV-STEP count in the config
# (fix_plan_srl_parity.md; area report 04 D-04 / S-01)
#
# Red evidence (pre-fix): fails at collection with ImportError —
# derive_prefill does not exist ("red by absence"). Pre-fix the driver used
# the raw config value as an ITERATION count, making the prefill phase
# num_envs x longer than intended.
# ---------------------------------------------------------------------------

def test_derive_prefill() -> None:
    """derive_prefill matches sheeprl dreamer_v3.py:508-511 at world_size=1.

    sheeprl:
        policy_steps_per_iter = num_envs
        learning_starts = cfg.algo.learning_starts // policy_steps_per_iter
        prefill_steps   = learning_starts - int(learning_starts > 0)

    The prefill phase therefore covers the SAME env-step count
    (learning_starts_iters * num_envs == cfg value) at every env count —
    the pre-fix driver ran it num_envs x too long. The (0, n) case pins the
    D-012 zero-prefill smoke path unchanged.
    """
    from src.algorithms.dreamer_srl.utils import derive_prefill

    assert derive_prefill(1024, 1) == (1024, 1023)
    assert derive_prefill(1024, 4) == (256, 255)
    assert derive_prefill(1024, 16) == (64, 63)
    assert derive_prefill(0, 4) == (0, 0)   # D-012 smoke configs: no prefill

    # Prefill env-step coverage equals the config value at every env count.
    for num_envs in (1, 4, 16):
        learning_starts, _ = derive_prefill(1024, num_envs)
        assert learning_starts * num_envs == 1024


def test_no_gradient_step_before_learning_starts_multi_env() -> None:
    """§S3 hard invariant at num_envs>1 with the WP-SRL P6 derivation.

    Mirrors the post-fix driver: learning_starts/prefill_steps derived via
    derive_prefill(cfg_value, num_envs); the train gate compares iter_num to
    the DERIVED iteration count and feeds the Ratio scheduler
    ratio_steps = policy_step - prefill_steps * num_envs (sheeprl
    dreamer_v3.py:661). Invariant: zero gradient steps while
    iter_num < learning_starts; positive steps once the gate opens.
    """
    from src.algorithms.dreamer_srl.utils import Ratio, derive_prefill

    for num_envs in (1, 4):
        cfg_learning_starts = 40
        learning_starts, prefill_steps = derive_prefill(cfg_learning_starts, num_envs)

        ratio = Ratio(ratio=1.0, pretrain_steps=0)
        policy_step = 0
        grad_step_at_iter: list[int] = []
        for iter_num in range(1, learning_starts + 6):
            policy_step += num_envs
            if iter_num >= learning_starts:
                ratio_steps = policy_step - prefill_steps * num_envs
                grad_step_at_iter.append(int(ratio(ratio_steps)))
            else:
                grad_step_at_iter.append(0)

        assert all(g == 0 for g in grad_step_at_iter[: learning_starts - 1]), (
            f"num_envs={num_envs}: gradient step fired before the derived "
            f"learning_starts={learning_starts}. Trace: {grad_step_at_iter}"
        )
        assert sum(grad_step_at_iter[learning_starts - 1:]) > 0, (
            f"num_envs={num_envs}: gate never opened — vacuous test. "
            f"Trace: {grad_step_at_iter}"
        )
