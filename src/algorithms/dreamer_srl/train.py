"""train.py — training-step functions for the dreamer-srl v3 rebuild.

This module ports the core training-step logic from sheeprl@33b6366
`sheeprl/algos/dreamer_v3/dreamer_v3.py:L48-L358` (the `train` function)
to JAX/Flax-NNX pure-functional form.

Isolation rule (v2 Risks §13 / NNX_CONVENTIONS.md):
    This module does NOT import from src.models.dreamer_v3_* or any other
    file in src/models/. All components are ported from sheeprl source.

CP scope implemented here:
    CP6 — Critic loss with EMA self-regularization (cascade fix #29, §S6):
        compute_discount: §S6 discount cumprod weighting
        compute_critic_loss: two-term NLL + discount weighting

    CP7+ (not yet implemented):
        actor loss (REINFORCE + entropy + §S6 discount weighting)
        polyak_update (EMA target-critic update)
        one_train_step (full training-step orchestration)

========================================================================
§S6 — Discount weighting on actor AND critic losses
========================================================================
Sheeprl dreamer_v3.py:L259-L260 (inside `with torch.no_grad():`):
    discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma

Key invariant: discount[0] = continues[0] * gamma / gamma = continues[0].
When continues[0] = true_continue = 1.0 (§S5 splice, no termination at step 0),
discount[0] = 1.0 exactly.

Applied to critic (sheeprl L316):
    value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))
Applied to actor (sheeprl L297):
    policy_loss = -torch.mean(discount[:-1].detach() * (objective + entropy[:-1]))

The `[:-1]` slice drops the last imagined step (which has no log_prob target).
`stop_gradient` on the entire discount tensor — JAX equivalent of `torch.no_grad()`.

========================================================================
§S9 — Independent(BernoulliSafeMode, 1) wrap on continue head
========================================================================
Sheeprl dreamer_v3.py:L246:
    continues = Independent(BernoulliSafeMode(logits=world_model.continue_model(...)), 1).mode
Sheeprl dreamer_v3.py:L167:
    pc = Independent(BernoulliSafeMode(logits=world_model.continue_model(latent_states)), 1)

The `dims=1` (or equivalently `n_event_dims=1` for Independent) re-interprets the
trailing axis of size 1 as the event dimension:
    - log_prob: sums over event dim → [T, B, 1] → [T, B]
    - mode: no reduction → [T, B, 1] (the mode of the Bernoulli)

In JAX we use IndependentBernoulli (from loss.py) which wraps BernoulliSafeMode
and applies the event-dim sum in log_prob.

========================================================================
CP6 — cascade fix #29 (two-term critic loss, Hafner §3.3 slow-target reg)
========================================================================
Sheeprl dreamer_v3.py:L307-L316:
    qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
    predicted_target_values = TwoHotEncodingDistribution(
        target_critic(imagined_trajectories.detach()[:-1]), dims=1
    ).mean
    value_loss = -qv.log_prob(lambda_values.detach())
    value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
    value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))

Two key points:
  1. BOTH log_prob terms are required (the second is the EMA slow-target self-reg).
     Dropping the second term is cascade bug #29.
  2. The lambda-target passed to log_prob is the UN-normalised `lambda_values`
     (NOT the Moments-normed form used in the actor's advantage computation).
     Sheeprl L314 uses `lambda_values.detach()` directly, NOT `normed_lambda_values`.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from src.algorithms.dreamer_srl.loss import TwoHotEncoding


# ---------------------------------------------------------------------------
# CP6 — compute_discount (§S6 discount weighting)
# ---------------------------------------------------------------------------

def compute_discount(continues: jax.Array, gamma: float) -> jax.Array:
    """Compute the cumulative discount mask for actor/critic loss weighting.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L259-L260
    (inside the `with torch.no_grad():` block in the `train` function).

    sheeprl L259-L260:
        with torch.no_grad():
            discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma

    JAX equivalent: `jax.lax.stop_gradient` replaces `torch.no_grad()`.
    The entire discount tensor is detached from the computation graph.

    §S6 GOTCHA: the division by `gamma` ensures discount[0] = continues[0],
    NOT `continues[0] * gamma`. When continues[0] = 1.0 (§S5 true-continue
    splice at imagination step 0, no termination), discount[0] = 1.0 exactly.
    Without the `/ gamma`, discount[0] would be `gamma` (< 1), which would
    under-weight the first imagined step.

    §S5 true-continue splice (sheeprl L247-L248):
        true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)
        continues = torch.cat((true_continue, continues[1:]))
    The first step of `continues` is the real env continue (1 if not terminated,
    0 if terminated). The remaining H steps are imagined continues from the
    world model's continue head.

    Usage:
        # In the training loop:
        discount = compute_discount(continues, gamma)  # [H+1, BT, 1]
        # For critic and actor losses (both use [:-1] slice):
        discount_weights = discount[:-1].squeeze(-1)   # [H, BT]
        critic_loss = mean(loss_terms * discount_weights)
        actor_loss  = -mean(discount[:-1] * objective)  # sheeprl L297

    Args:
        continues: [H+1, BT, 1] float array of continue probabilities.
                   continues[0] = true_continue (§S5 splice, = 1 - terminated).
                   continues[1:H+1] = imagined continues from world-model.
        gamma: discount factor (float).

    Returns:
        discount: [H+1, BT, 1] stop_gradient'd cumulative discount mask.
                  discount[0] = continues[0] (= 1.0 when no termination at step 0).
                  discount[k] = prod(continues[0:k+1]) * gamma^(k) / gamma.

    Bit-identity test:
        tests/algorithms/dreamer_srl/test_train.py::test_discount_weighting
    """
    # sheeprl L260: discount = torch.cumprod(continues * gamma, dim=0) / gamma
    # stop_gradient = JAX equivalent of torch.no_grad()
    discount = jax.lax.stop_gradient(
        jnp.cumprod(continues * gamma, axis=0) / gamma
    )
    return discount  # [H+1, BT, 1]


# ---------------------------------------------------------------------------
# CP6 — compute_critic_loss (cascade fix #29, §S6 discount weighting)
# ---------------------------------------------------------------------------

def compute_critic_loss(
    qv_logits: jax.Array,
    lambda_values: jax.Array,
    target_critic_values: jax.Array,
    discount: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Two-term critic NLL loss with EMA self-regularization (cascade fix #29).

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L307-L316
    (critic optimization section of the `train` function).

    sheeprl L307-L316:
        qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
        predicted_target_values = TwoHotEncodingDistribution(
            target_critic(imagined_trajectories.detach()[:-1]), dims=1
        ).mean
        value_loss = -qv.log_prob(lambda_values.detach())
        value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
        value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))

    CASCADE FIX #29 — TWO-TERM LOSS:
    The standard critic loss is -qv.log_prob(lambda_values) (regression toward
    the bootstrapped lambda-return target). Hafner §3.3 adds a SECOND term:
    -qv.log_prob(target_critic_value), where target_critic_value is the EMA
    slow-target critic's value estimate. This term regularizes the critic toward
    its own slow-moving EMA copy, preventing aggressive overshooting during
    rapid early-training updates.

    The original v1 cascade implementation omitted the second term. The symptom
    was slow value-function convergence and instability. This is cascade fix #29
    in the v2 plan.

    TARGET USES UN-NORMALISED lambda_values (NOT Moments-normed):
    The actor normalizes lambda_values via Moments (§S7) for the advantage
    computation. But the CRITIC regresses against the raw (un-normalised)
    lambda_values. Sheeprl L314 uses `lambda_values.detach()` directly, before
    the Moments normalization at L276-L279. Passing Moments-normalised targets
    to the critic would regress it toward [−1,+1] scaled values instead of the
    real symlog-space value function — a silent semantic error that would make
    the critic useless as a baseline at inference time.

    DISCOUNT WEIGHTING (§S6):
    Both terms are multiplied by `discount[:-1].squeeze(-1)` before the mean.
    The `[:-1]` slice drops the last imagined step (which has no defined
    log_prob target — lambda_values has H elements, imagined_trajectories has
    H+1). The discount tensor is already stop_gradient'd by compute_discount.

    STOP_GRADIENT on targets:
    Both `lambda_values` and `target_critic_values` are stop_gradient'd before
    passing to log_prob. In the sheeprl source:
        - L314: `lambda_values.detach()` (the bootstrapped TD-lambda target)
        - L315: `predicted_target_values.detach()` (the EMA target critic mean)
    In JAX: `jax.lax.stop_gradient(...)` on both inputs.

    Args:
        qv_logits: [H, BT, 255] logits from the online critic head
                   on imagined trajectories (imagined_trajectories[:-1]).
                   NOTE: the caller applies [:-1] to imagined_trajectories
                   before calling the critic, matching sheeprl L307.
        lambda_values: [H, BT, 1] un-normalised lambda-return targets.
                   Computed by compute_lambda_values(predicted_rewards[1:], ...).
                   Must be RAW — NOT Moments-normed. In real reward space.
        target_critic_values: [H, BT, 1] EMA target-critic expected value.
                   Computed by TwoHotEncoding(target_critic_logits).mean.
                   In real reward space. stop_gradient'd by caller or here.
        discount: [H+1, BT, 1] from compute_discount(continues, gamma).
                   Already stop_gradient'd. The `[:-1]` slice is applied here.

    Returns:
        (value_loss, neg_lp1, neg_lp2)
        value_loss:   scalar — discount-weighted mean of both NLL terms.
        neg_lp1: [H, BT] — -qv.log_prob(lambda_values)  (first term)
        neg_lp2: [H, BT] — -qv.log_prob(target_critic_values)  (second term)

    Bit-identity test:
        tests/algorithms/dreamer_srl/test_train.py::test_critic_loss_two_terms
        tests/algorithms/dreamer_srl/test_train.py::test_critic_target_lambda
    """
    # Construct the critic distribution from logits — dims=1 matches sheeprl L307
    qv = TwoHotEncoding(qv_logits, dims=1)

    # sheeprl L314: value_loss = -qv.log_prob(lambda_values.detach())
    # JAX: stop_gradient on the lambda-target (do NOT let gradient flow through it)
    neg_lp1 = -qv.log_prob(jax.lax.stop_gradient(lambda_values))   # [H, BT]

    # sheeprl L315: value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
    # JAX: stop_gradient on the EMA target value (also detached)
    neg_lp2 = -qv.log_prob(jax.lax.stop_gradient(target_critic_values))  # [H, BT]

    # sheeprl L316: value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))
    # discount: [H+1, BT, 1] — slice to [H, BT, 1], then squeeze → [H, BT]
    discount_weights = discount[:-1].squeeze(-1)  # [H, BT]

    # Combine both terms and apply discount weighting
    # value_loss accumulates both: -(lp1 + lp2) weighted by discount
    value_loss = jnp.mean((neg_lp1 + neg_lp2) * discount_weights)  # scalar

    return value_loss, neg_lp1, neg_lp2
