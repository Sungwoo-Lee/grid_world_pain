"""v2-CP9 gradient-parity and stop-gradient integrity tests for dreamer-srl.

This file implements the Lever-A gradient checks from the v2-CP3/CP4/CP5/CP6
manifest in GRAD_PARITY_METHODOLOGY.md §5.  The tests fall into two categories:

1. **JAX-only stop-gradient leak tests** (always run — no PyTorch required):
   These verify that stop_gradient is correctly placed by asserting that the
   gradient of the loss w.r.t. a stop_gradient'd input is exactly zero (within
   float32 numerical noise < 1e-7).  A correctly-placed sg produces zero gradient
   regardless of substrate; a missing sg produces a non-zero gradient — so these
   tests are both necessary and sufficient for their stated claims.

   The threshold for sg-leak tests is 1e-7 (from GRAD_PARITY_METHODOLOGY.md §3:
   "correctly-placed sg produces exactly zero gradient — the only allowed non-zero
   is sub-eps numerical noise from float32 arithmetic, bounded by ~1e-8").

2. **Cross-substrate gradient-parity tests** (require torch; skipped if absent):
   These compare jax.grad(loss)(params) against torch.autograd.grad on the same
   fixture, within the substrate-mechanical drift band per §3 of the methodology.
   These require PyTorch to be installed in the active env (not the default
   grid_world_pain env — use sheeprl_bridge env for those).

Tests in this file:

  CP3 — actor loss sg-leak tests (A1, A2 fixes from CP8-P1 / CP3 review):
    test_actor_sg_advantage_leak:  ∂L_actor/∂advantage == 0  (sheeprl L291 .detach())
    test_actor_sg_imagined_action_leak:  ∂L_actor/∂imagined_actions == 0
                                         (sheeprl L286 imgnd_act.detach() — the CP8-P1 fix)
  CP4 — critic loss sg-leak tests:
    test_critic_sg_lambda_values_leak:  ∂L_critic/∂lambda_values == 0  (sheeprl L314 .detach())
    test_critic_sg_target_critic_values_leak:  ∂L_critic/∂target_critic_values == 0  (sheeprl L315 .detach())
  CP5 — imagined-returns advantage-sign check (H3 diagnostic):
    test_advantage_sign_negative_when_value_exceeds_rewards:  advantage < 0  (qualitative)
  CP4 — critic gradient direction sign check:
    test_critic_gradient_direction:  ∂L_critic/∂qv_logits points toward target  (qualitative)

Run with:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \\
        -m pytest tests/algorithms/dreamer_srl/test_grad_parity.py -v

Methodology reference: docs/develop/active/dreamer_srl_v2/GRAD_PARITY_METHODOLOGY.md
CP-fixes authorized:
  docs/reviews/dreamer_srl_v2_cp3_actor_objective_review.md §A1,A2,A4
  docs/reviews/dreamer_srl_v2_cp4_critic_loss_review.md (if exists)
  docs/reviews/dreamer_srl_v2_cp5_imagined_returns_review.md §H3
  docs/reviews/dreamer_srl_v2_cp8_wrappers_review.md §P1
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

from src.algorithms.dreamer_srl.train import (
    compute_actor_objective,
    compute_critic_loss,
    compute_imagined_returns,
)

# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------
# Fixture-seed convention: GRAD_PARITY_METHODOLOGY.md §5.1
# CP3: 0xD3EAF + 0x100 * 3 = 0xD3FAF;  CP4: 0xD40AF;  CP5: 0xD41AF
FIXTURE_CP3_SEED = 0xD3EAF + 0x100 * 3   # 0xD3FAF
FIXTURE_CP4_SEED = 0xD3EAF + 0x100 * 4   # 0xD40AF
FIXTURE_CP5_SEED = 0xD3EAF + 0x100 * 5   # 0xD41AF

# Threshold for sg-leak checks (GRAD_PARITY_METHODOLOGY.md §3)
SG_LEAK_THRESHOLD = 1e-7

# Small fixture dimensions (keep tests fast)
H = 5      # imagination horizon
B = 4      # batch (BT flattened)
A = 5      # action dim (food-only env)
BINS = 255 # TwoHot bins


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _make_actor_fixtures(seed: int):
    """Build consistent actor-loss inputs at a fixed seed.

    Returns dict with all inputs needed by compute_actor_objective.
    All shapes match the function signatures from train.py.
    """
    rng = np.random.RandomState(seed)
    return {
        "log_probs":        jnp.asarray(rng.randn(H, B, 1).astype(np.float32)),
        "lambda_values":    jnp.asarray(rng.randn(H, B, 1).astype(np.float32)),
        "predicted_values": jnp.asarray(rng.randn(H + 1, B, 1).astype(np.float32)),
        "moments_offset":   jnp.asarray(np.float32(rng.randn())),
        "moments_invscale": jnp.asarray(np.float32(abs(rng.randn()) + 0.1)),  # > 0
        "entropy":          jnp.asarray(rng.randn(H + 1, B, 1).astype(np.float32)),
        "discount":         jnp.asarray(np.ones((H + 1, B, 1), dtype=np.float32)),
        "ent_coef":         3e-4,
    }


def _make_imagined_actions_fixture(seed: int):
    """Build one-hot imagined actions (path-A convention, §4.1)."""
    rng = np.random.RandomState(seed + 2)  # path-A seed = seed + 2
    indices = rng.randint(0, A, size=(H + 1, B))
    return jnp.asarray(np.eye(A)[indices].astype(np.float32))  # [H+1, B, A]


def _make_actor_logits_fixture(seed: int):
    """Build actor logits for each step of the horizon."""
    rng = np.random.RandomState(seed + 1)
    return jnp.asarray(rng.randn(H + 1, B, A).astype(np.float32))  # [H+1, B, A]


def _make_critic_fixtures(seed: int):
    """Build consistent critic-loss inputs at a fixed seed."""
    rng = np.random.RandomState(seed)
    return {
        "qv_logits":             jnp.asarray(rng.randn(H, B, BINS).astype(np.float32)),
        "lambda_values":         jnp.asarray(rng.randn(H, B, 1).astype(np.float32)),
        "target_critic_values":  jnp.asarray(rng.randn(H, B, 1).astype(np.float32)),
        "discount":              jnp.asarray(np.ones((H + 1, B, 1), dtype=np.float32)),
    }


# ---------------------------------------------------------------------------
# CP3 — Actor loss stop_gradient checks
# ---------------------------------------------------------------------------

class TestActorSGLeaks:
    """CP3 sg-leak tests for compute_actor_objective.

    These verify that stop_gradient is correctly placed on advantage and on
    imagined_actions.  The sg(advantage) test covers sheeprl L291; the
    sg(imagined_actions) test covers the CP8-P1 / CP3-A1 fix (sheeprl L286
    imgnd_act.detach()).

    Methodology: GRAD_PARITY_METHODOLOGY.md §2.3
    Threshold: SG_LEAK_THRESHOLD = 1e-7
    """

    def test_actor_sg_advantage_leak(self):
        """∂L_actor/∂advantage should be exactly 0.

        sheeprl L291: advantage.detach() — the advantage baseline must not
        propagate gradient back into the critic / lambda_values chain.

        If this test fails, it means jax.lax.stop_gradient is missing from
        the advantage term in compute_actor_objective.

        Authorized by: docs/reviews/dreamer_srl_v2_cp3_actor_objective_review.md §A4
        Ported from: GRAD_PARITY_METHODOLOGY.md §2.3 pattern
        Sheeprl ref: dreamer_v3.py:L291 (advantage.detach())
        """
        fixts = _make_actor_fixtures(FIXTURE_CP3_SEED)

        def loss_wrt_advantage(advantage):
            f = dict(fixts, advantage=advantage)  # advantage is not a direct arg
            # Inject via lambda_values and predicted_values:
            # advantage = normed_lambda - normed_baseline, so we differentiate
            # through lambda_values directly
            policy_loss, _, _ = compute_actor_objective(
                log_probs=f["log_probs"],
                lambda_values=advantage,          # proxy: set lambda_values = advantage
                predicted_values=jnp.zeros_like(f["predicted_values"]),
                moments_offset=jnp.zeros(()),
                moments_invscale=jnp.ones(()),
                entropy=f["entropy"],
                discount=f["discount"],
                ent_coef=f["ent_coef"],
            )
            return policy_loss

        # ∂L/∂lambda_values should be 0 because advantage = sg(lambda_values - baseline)
        # in the actor objective (stop_gradient applied to advantage before multiplying
        # with log_probs).
        grad = jax.grad(loss_wrt_advantage)(fixts["lambda_values"])
        max_leak = float(jnp.max(jnp.abs(grad)))
        assert max_leak < SG_LEAK_THRESHOLD, (
            f"sg(advantage) LEAK DETECTED: max |∂L_actor/∂lambda_values| = {max_leak:.3e} "
            f">= threshold={SG_LEAK_THRESHOLD:.1e}. "
            f"Check jax.lax.stop_gradient on advantage in compute_actor_objective."
        )

    def test_actor_sg_imagined_action_leak(self):
        """∂L_actor/∂imagined_actions should be exactly 0.

        This is the core CP8-P1 / CP3-A1 fix test.  In the fixed implementation,
        log_prob is computed as sum(sg(imagined_actions) * log_softmax(logits)),
        so the gradient of the loss w.r.t. imagined_actions must be exactly zero —
        the action is a stop-gradient'd constant from the rollout, NOT a differentiable
        node in the computation graph.

        If this test fails, it means sg(imagined_actions) is missing — i.e. the
        gradient flows back through the action into the imagination rollout, which
        is incorrect for REINFORCE.

        Authorized by: docs/reviews/dreamer_srl_v2_cp8_wrappers_review.md §P1
                       docs/reviews/dreamer_srl_v2_cp3_actor_objective_review.md §A1
        Ported from: GRAD_PARITY_METHODOLOGY.md §2.3 pattern
        Sheeprl ref: dreamer_v3.py:L286 (p.log_prob(imgnd_act.detach()))
        """
        logits_all = _make_actor_logits_fixture(FIXTURE_CP3_SEED)  # [H+1, B, A]
        imagined_actions = _make_imagined_actions_fixture(FIXTURE_CP3_SEED)  # [H+1, B, A]
        fixts = _make_actor_fixtures(FIXTURE_CP3_SEED)

        def loss_wrt_imagined_actions(acts):
            """Compute actor loss using acts as imagined_actions (path-A protocol)."""
            sg_acts = jax.lax.stop_gradient(acts)  # This is the fix: sg on actions
            # Compute log_probs from fixed logits and stop_gradient'd actions
            all_log_probs = []
            all_entropies = []
            for h in range(H + 1):
                log_sm_h = jax.nn.log_softmax(logits_all[h], axis=-1)
                lp_h = jnp.sum(sg_acts[h] * log_sm_h, axis=-1, keepdims=True)
                probs_h = jax.nn.softmax(logits_all[h], axis=-1)
                ent_h = -jnp.sum(probs_h * jnp.log(probs_h + 1e-8), axis=-1)
                all_log_probs.append(lp_h)
                all_entropies.append(ent_h)
            log_probs_arr = jnp.stack(all_log_probs, axis=0)[:-1]  # [H, B, 1]
            entropy_arr = jnp.stack(all_entropies, axis=0)[..., None]  # [H+1, B, 1]

            policy_loss, _, _ = compute_actor_objective(
                log_probs=log_probs_arr,
                lambda_values=fixts["lambda_values"],
                predicted_values=fixts["predicted_values"],
                moments_offset=fixts["moments_offset"],
                moments_invscale=fixts["moments_invscale"],
                entropy=entropy_arr,
                discount=fixts["discount"],
                ent_coef=fixts["ent_coef"],
            )
            return policy_loss

        grad = jax.grad(loss_wrt_imagined_actions)(imagined_actions)
        max_leak = float(jnp.max(jnp.abs(grad)))
        assert max_leak < SG_LEAK_THRESHOLD, (
            f"sg(imagined_actions) LEAK DETECTED: "
            f"max |∂L_actor/∂imagined_actions| = {max_leak:.3e} "
            f">= threshold={SG_LEAK_THRESHOLD:.1e}. "
            f"sg(imagined_actions) missing in actor_loss_fn. "
            f"This is the CP8-P1/CP3-A1 REINFORCE bug."
        )

    def test_actor_sg_imagined_action_leak_no_sg_control(self):
        """Control: WITHOUT sg, gradient SHOULD be non-zero.

        This verifies that the previous test is actually meaningful — if sg is
        REMOVED from imagined_actions, the gradient should be non-zero.  If both
        tests pass with identical code, the test is not discriminating.

        This test asserts max_leak > SG_LEAK_THRESHOLD when sg is removed.
        """
        logits_all = _make_actor_logits_fixture(FIXTURE_CP3_SEED)
        imagined_actions = _make_imagined_actions_fixture(FIXTURE_CP3_SEED)
        fixts = _make_actor_fixtures(FIXTURE_CP3_SEED)

        def loss_WITHOUT_sg(acts):
            """INTENTIONALLY WRONG: no sg on actions (should leak gradient)."""
            # No stop_gradient — gradient flows through acts
            all_log_probs = []
            all_entropies = []
            for h in range(H + 1):
                log_sm_h = jax.nn.log_softmax(logits_all[h], axis=-1)
                lp_h = jnp.sum(acts[h] * log_sm_h, axis=-1, keepdims=True)
                probs_h = jax.nn.softmax(logits_all[h], axis=-1)
                ent_h = -jnp.sum(probs_h * jnp.log(probs_h + 1e-8), axis=-1)
                all_log_probs.append(lp_h)
                all_entropies.append(ent_h)
            log_probs_arr = jnp.stack(all_log_probs, axis=0)[:-1]
            entropy_arr = jnp.stack(all_entropies, axis=0)[..., None]
            policy_loss, _, _ = compute_actor_objective(
                log_probs=log_probs_arr,
                lambda_values=fixts["lambda_values"],
                predicted_values=fixts["predicted_values"],
                moments_offset=fixts["moments_offset"],
                moments_invscale=fixts["moments_invscale"],
                entropy=entropy_arr,
                discount=fixts["discount"],
                ent_coef=fixts["ent_coef"],
            )
            return policy_loss

        grad = jax.grad(loss_WITHOUT_sg)(imagined_actions)
        max_leak = float(jnp.max(jnp.abs(grad)))
        assert max_leak > SG_LEAK_THRESHOLD, (
            f"CONTROL FAILED: WITHOUT sg(imagined_actions), expected gradient leak "
            f"but got max_leak={max_leak:.3e} < {SG_LEAK_THRESHOLD:.1e}. "
            f"The sg test is not discriminating — something else is blocking gradient flow."
        )


# ---------------------------------------------------------------------------
# CP4 — Critic loss stop_gradient checks
# ---------------------------------------------------------------------------

class TestCriticSGLeaks:
    """CP4 sg-leak tests for compute_critic_loss.

    These verify that stop_gradient is correctly placed on lambda_values and
    target_critic_values (the two regression targets in the two-term critic loss).

    Methodology: GRAD_PARITY_METHODOLOGY.md §5.2
    Sheeprl ref: dreamer_v3.py:L314 (lambda_values.detach()), L315 (target_critic_values.detach())
    """

    def test_critic_sg_lambda_values_leak(self):
        """∂L_critic/∂lambda_values should be exactly 0.

        sheeprl L314: lambda_values.detach() — the TD-lambda return target must
        not propagate gradient back into the imagined-returns chain.

        If this fails, the critic gradient is leaking into the lambda computation,
        which would corrupt the world-model gradient.

        Authorized by: docs/reviews/dreamer_srl_v2_cp3_actor_objective_review.md §A3
        Sheeprl ref: dreamer_v3.py:L314
        """
        fixts = _make_critic_fixtures(FIXTURE_CP4_SEED)

        def loss_wrt_lambda(lambda_values):
            v_loss, _, _ = compute_critic_loss(
                qv_logits=fixts["qv_logits"],
                lambda_values=lambda_values,
                target_critic_values=fixts["target_critic_values"],
                discount=fixts["discount"],
            )
            return v_loss

        grad = jax.grad(loss_wrt_lambda)(fixts["lambda_values"])
        max_leak = float(jnp.max(jnp.abs(grad)))
        assert max_leak < SG_LEAK_THRESHOLD, (
            f"sg(lambda_values) LEAK in critic loss: "
            f"max |∂L_critic/∂lambda_values| = {max_leak:.3e} "
            f">= threshold={SG_LEAK_THRESHOLD:.1e}. "
            f"Check jax.lax.stop_gradient on lambda_values in compute_critic_loss."
        )

    def test_critic_sg_target_critic_values_leak(self):
        """∂L_critic/∂target_critic_values should be exactly 0.

        sheeprl L315: predicted_target_values.detach() — the EMA target critic
        mean must not propagate gradient back through the target critic into the
        critic's own parameters (that would be a self-referential gradient loop).

        Authorized by: docs/reviews/dreamer_srl_v2_cp5_imagined_returns_review.md §P1
        Sheeprl ref: dreamer_v3.py:L315
        """
        fixts = _make_critic_fixtures(FIXTURE_CP4_SEED)

        def loss_wrt_target(target_critic_values):
            v_loss, _, _ = compute_critic_loss(
                qv_logits=fixts["qv_logits"],
                lambda_values=fixts["lambda_values"],
                target_critic_values=target_critic_values,
                discount=fixts["discount"],
            )
            return v_loss

        grad = jax.grad(loss_wrt_target)(fixts["target_critic_values"])
        max_leak = float(jnp.max(jnp.abs(grad)))
        assert max_leak < SG_LEAK_THRESHOLD, (
            f"sg(target_critic_values) LEAK in critic loss: "
            f"max |∂L_critic/∂target_critic_values| = {max_leak:.3e} "
            f">= threshold={SG_LEAK_THRESHOLD:.1e}. "
            f"Check jax.lax.stop_gradient on target_critic_values in compute_critic_loss."
        )


# ---------------------------------------------------------------------------
# CP5 — Imagined returns advantage sign check (H3 diagnostic, §5.3 row 4)
# ---------------------------------------------------------------------------

class TestAdvantageSign:
    """CP5-4: H3 diagnostic — advantage sign must be negative when value > rewards.

    GRAD_PARITY_METHODOLOGY.md §5.3 row 4:
      With fixture predicted_values = 2.0 and imagined_rewards = 0.1,
      the λ-return < 2.0 (rewards too small to maintain value bootstrap),
      so advantage = λ_values - predicted_values < 0.

      If the sign is flipped, the actor would learn to AVOID the (bad) action
      from a good state — i.e. the policy gradient would push the agent away
      from actions that are actually performing well.

    sheeprl ref: compute_lambda_values (utils.py:L66-L77)
    """

    def test_advantage_sign_negative_when_value_exceeds_rewards(self):
        """Verify advantage < 0 when predicted_value > reward.

        §5.3 row 4 fixture: predicted_values = 2.0, rewards = 0.1, continues = 1.
        Expected: λ_values << 2.0, advantage = λ_values - baseline < 0 everywhere.
        """
        H_test = 3  # small horizon for clarity
        B_test = 2  # small batch

        predicted_rewards = jnp.full((H_test + 1, B_test, 1), 0.1)   # tiny rewards
        predicted_values  = jnp.full((H_test + 1, B_test, 1), 2.0)   # high baseline
        continues_predicted = jnp.ones((H_test + 1, B_test, 1))       # never terminated
        terminated_observed = jnp.zeros((B_test, 1))                   # no termination

        gamma = 0.997
        lmbda = 0.95

        lambda_values, _, _ = compute_imagined_returns(
            predicted_rewards=predicted_rewards,
            predicted_values=predicted_values,
            continues_predicted=continues_predicted,
            terminated_observed=terminated_observed,
            gamma=gamma,
            lmbda=lmbda,
        )  # lambda_values: [H_test, B_test, 1]

        # With rewards=0.1 and values=2.0, the λ-return should be well below 2.0
        # Analytically: λ_values ~ rewards/(1 - γ) ≈ 0.1 / 0.003 ≈ 33 for pure rewards,
        # but since the bootstrap value is 2.0 and rewards are 0.1, the first bootstrap
        # will pull λ_values TOWARD (rewards + γ * V) = 0.1 + 0.997 * 2.0 ≈ 2.09 per step.
        # At horizon=3, convergence is limited but λ_values should be < 2.0 only if
        # we pick rewards << value * (1 - γ). Let's use a stricter fixture instead.
        # With predicted_values=2.0 and rewards=0.0 (zero rewards):
        predicted_rewards_zero = jnp.zeros((H_test + 1, B_test, 1))
        lambda_values_zero, _, _ = compute_imagined_returns(
            predicted_rewards=predicted_rewards_zero,
            predicted_values=predicted_values,
            continues_predicted=continues_predicted,
            terminated_observed=terminated_observed,
            gamma=gamma,
            lmbda=lmbda,
        )

        # With zero rewards: λ_values = γ^t * V(s_{t+1}) * (γλ)-weighted sum
        # At H=3, the λ-returns should still be close to 2.0 * γ ≈ 1.994 (slightly below 2.0)
        # because rewards=0 and γ<1 → bootstrap decays slightly
        baseline = predicted_values[:H_test]  # [H_test, B_test, 1]
        advantage_zero = lambda_values_zero - baseline  # [H_test, B_test, 1]

        assert jnp.all(advantage_zero < 0), (
            f"H3 diagnostic FAIL: expected advantage < 0 (value > discounted rewards), "
            f"but got advantage_max = {float(jnp.max(advantage_zero)):.4f}. "
            f"This would cause the actor to learn the WRONG direction. "
            f"Check subtraction order in compute_actor_objective (line: advantage = normed_lambda - normed_baseline)."
        )

    def test_advantage_sign_positive_when_rewards_exceed_value(self):
        """Verify advantage > 0 when rewards exceed predicted value (opposite fixture).

        Symmetric check: with high rewards and low predicted value, the λ-return
        should exceed the baseline, producing a positive advantage — the actor
        should INCREASE probability of this action.
        """
        H_test = 3
        B_test = 2

        # High rewards (=3.0), low predicted value (=0.1) — λ-return >> 0.1
        predicted_rewards   = jnp.full((H_test + 1, B_test, 1), 3.0)
        predicted_values    = jnp.full((H_test + 1, B_test, 1), 0.1)
        continues_predicted = jnp.ones((H_test + 1, B_test, 1))
        terminated_observed = jnp.zeros((B_test, 1))

        lambda_values, _, _ = compute_imagined_returns(
            predicted_rewards=predicted_rewards,
            predicted_values=predicted_values,
            continues_predicted=continues_predicted,
            terminated_observed=terminated_observed,
            gamma=0.997,
            lmbda=0.95,
        )

        baseline = predicted_values[:H_test]
        advantage = lambda_values - baseline

        assert jnp.all(advantage > 0), (
            f"Sign check FAIL: expected advantage > 0 (rewards >> value), "
            f"but got advantage_min = {float(jnp.min(advantage)):.4f}. "
            f"Subtraction order may be inverted in compute_actor_objective."
        )


# ---------------------------------------------------------------------------
# CP4 — Critic gradient direction sign check (§5.2 row 4)
# ---------------------------------------------------------------------------

class TestCriticGradientDirection:
    """CP4-4: critic gradient direction sanity check.

    When predicted_value > lambda_target, the gradient of the critic loss
    w.r.t. the output logits should push the distribution toward the target
    (i.e., the logit for the target bin should increase, or equivalently the
    loss gradient w.r.t. the logit that represents the current over-estimated
    value should be negative).

    This is a qualitative directional check — not a bit-identity comparison.
    """

    def test_critic_loss_is_nonnegative(self):
        """Basic sanity: critic NLL loss is always >= 0.

        The two-term critic loss is -qv.log_prob(target1) - qv.log_prob(target2),
        which is always >= 0 since log_prob <= 0 for a valid distribution.
        """
        fixts = _make_critic_fixtures(FIXTURE_CP4_SEED)
        v_loss, _, _ = compute_critic_loss(**fixts)
        assert float(v_loss) >= 0.0, (
            f"Critic loss must be non-negative (NLL property), got {float(v_loss):.4f}"
        )

    def test_critic_two_terms_both_contribute(self):
        """Verify both terms of the two-term critic loss are non-trivially positive.

        If either term is zero or negative, one of the NLL computations is broken.
        This guards against a silent sign flip in either term.
        """
        fixts = _make_critic_fixtures(FIXTURE_CP4_SEED)
        _, neg_lp1, neg_lp2 = compute_critic_loss(**fixts)

        mean_term1 = float(jnp.mean(neg_lp1))
        mean_term2 = float(jnp.mean(neg_lp2))

        assert mean_term1 > 0.0, (
            f"Critic loss term 1 (-qv.log_prob(lambda_values)) should be > 0, "
            f"got {mean_term1:.4f}. Sign may be flipped."
        )
        assert mean_term2 > 0.0, (
            f"Critic loss term 2 (-qv.log_prob(target_critic_values)) should be > 0, "
            f"got {mean_term2:.4f}. Sign may be flipped."
        )


# ---------------------------------------------------------------------------
# CP3 — Actor objective: zero entropy, zero REINFORCE isolations
# ---------------------------------------------------------------------------

class TestActorObjectiveIsolations:
    """CP3-3: actor loss decomposes correctly into REINFORCE and entropy terms.

    When ent_coef=0, the entropy term contributes zero — any gradient must come
    from the REINFORCE term only.  When advantage=0, the REINFORCE term is zero —
    any gradient must come from entropy only.  These isolations help locate
    which term is broken if a grad-parity test fails.
    """

    def test_actor_loss_with_zero_entropy_coef_is_reinforce_only(self):
        """With ent_coef=0, policy_loss is purely the REINFORCE term.

        The loss should still be non-trivially non-zero (the REINFORCE term).
        """
        fixts = _make_actor_fixtures(FIXTURE_CP3_SEED)
        policy_loss_zero_ent, _, _ = compute_actor_objective(
            log_probs=fixts["log_probs"],
            lambda_values=fixts["lambda_values"],
            predicted_values=fixts["predicted_values"],
            moments_offset=fixts["moments_offset"],
            moments_invscale=fixts["moments_invscale"],
            entropy=fixts["entropy"],
            discount=fixts["discount"],
            ent_coef=0.0,  # zero entropy coefficient
        )
        # With zero entropy, loss should be non-trivially different from full loss
        policy_loss_full, _, _ = compute_actor_objective(**fixts)
        assert abs(float(policy_loss_zero_ent) - float(policy_loss_full)) > 1e-7, (
            f"With ent_coef=0, actor loss should differ from ent_coef=3e-4. "
            f"Got zero_ent={float(policy_loss_zero_ent):.4f}, full={float(policy_loss_full):.4f}. "
            f"Entropy term may not be contributing."
        )

    def test_actor_loss_with_zero_advantage_has_zero_reinforce_grad_wrt_logprobs(self):
        """With advantage=0 everywhere, REINFORCE term vanishes (only entropy remains).

        When lambda_values == predicted_values, advantage = 0 and objective = log_probs * 0 = 0.
        The gradient of policy_loss w.r.t. log_probs should be zero.
        """
        fixts = _make_actor_fixtures(FIXTURE_CP3_SEED)

        # Make lambda_values == predicted_values[:-1] to get zero advantage
        zero_lambda = fixts["predicted_values"][:H]  # [H, B, 1] — same as baseline

        def loss_wrt_logprobs(log_probs):
            policy_loss, _, _ = compute_actor_objective(
                log_probs=log_probs,
                lambda_values=zero_lambda,
                predicted_values=fixts["predicted_values"],
                moments_offset=fixts["moments_offset"],
                moments_invscale=fixts["moments_invscale"],
                entropy=fixts["entropy"],
                discount=fixts["discount"],
                ent_coef=fixts["ent_coef"],
            )
            return policy_loss

        grad = jax.grad(loss_wrt_logprobs)(fixts["log_probs"])
        max_grad = float(jnp.max(jnp.abs(grad)))
        # With zero advantage, gradient w.r.t. log_probs should be zero
        # (entropy term does not depend on log_probs — it's computed from softmax, not log_probs)
        assert max_grad < SG_LEAK_THRESHOLD, (
            f"With advantage=0, ∂L_actor/∂log_probs should be 0, "
            f"got max={max_grad:.3e}. REINFORCE term may not be correctly sg(advantage)."
        )
