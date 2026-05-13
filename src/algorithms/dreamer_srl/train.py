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

    CP7 — Polyak target-critic EMA update + actor REINFORCE objective (§S5/§S7):
        polyak_update: EMA blend of online → target critic weights
        compute_imagined_returns: §S5 true-continue splice + lambda-value computation
        compute_actor_objective: REINFORCE objective with §S7 advantage normalization

    CP8+ (not yet implemented):
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
CP7 — Polyak target-critic EMA update (§S7 ordering)
========================================================================
Sheeprl dreamer_v3.py:L673-L680 (inside the inner gradient-step loop):
    if cumulative_per_rank_gradient_steps % ... == 0:
        tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau
        for cp, tcp in zip(critic.module.parameters(), target_critic.parameters()):
            tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
    batch = {k: v[i].float() for k, v in local_data.items()}
    train(...)   ← train() is called AFTER the polyak update in this iteration

Key invariants:
  1. First call (step 0): tau=1.0 → hard copy: target = online (byte-identical).
  2. Subsequent calls: tau=0.02 (XS default) → EMA blend:
       target = (1 - tau) * target + tau * online
       (NOTE: sheeprl writes this as `tau * cp + (1 - tau) * tcp` which is the
       same formula with online=cp, target=tcp.)
  3. Call order: polyak fires BEFORE `one_train_step` (= sheeprl's `train()`)
     in the outer loop. This means the target critic used inside `train()` for
     value prediction is the FRESHLY UPDATED target, not the pre-step target.

JAX implementation: `polyak_update(online_params, target_params, tau)` returns
a NEW params dict (pure functional — no in-place mutation). The caller replaces
the target critic's parameter dict with the returned value.

========================================================================
CP7 — §S5 true-continue splice (compute_imagined_returns)
========================================================================
Sheeprl dreamer_v3.py:L246-L248 (inside `train()`, inside the `with ... no_grad` context):
    continues = Independent(BernoulliSafeMode(logits=world_model.continue_model(imagined_trajectories)), 1).mode
    true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)
    continues = torch.cat((true_continue, continues[1:]))

The §S5 splice replaces `continues[0]` (the first IMAGINED continue, which
could be wrong for the first step right after a real env step) with the
OBSERVED continue: `1 - terminated_observed`. This ensures the first discount
weight is grounded in reality rather than the world model's prediction.

`compute_imagined_returns` centralizes:
  1. The §S5 splice of continues.
  2. The `compute_lambda_values` call.
  3. The `compute_discount` call.
All three are consumed by both actor and critic loss.

========================================================================
CP7 — Actor REINFORCE objective with §S7 advantage normalization
========================================================================
Sheeprl dreamer_v3.py:L274-L297 (actor optimization section):
    baseline = predicted_values[:-1]
    offset, invscale = moments(lambda_values, fabric)
    normed_lambda_values = (lambda_values - offset) / invscale
    normed_baseline = (baseline - offset) / invscale
    advantage = normed_lambda_values - normed_baseline
    # For discrete actions (our case — grid-world is discrete):
    objective = (
        sum_over_action_dims(log_prob(stop_gradient(action))) * advantage.detach()
    )
    entropy = ent_coef * sum(p.entropy() for p in policies)
    policy_loss = -mean(discount[:-1].detach() * (objective + entropy[:-1]))

§S7 advantage computation (offset cancellation):
    advantage = normed_lambda - normed_baseline
              = (lambda_values - offset) / invscale - (baseline - offset) / invscale
    The `offset` cancels algebraically: advantage = (lambda - baseline) / invscale.
    However, per-term normalization matches sheeprl's code structure exactly.
    We compute BOTH terms separately (as sheeprl does), even though the algebra
    shows the offset cancels. This preserves bit-identity with sheeprl.

`stop_gradient` discipline (sheeprl L287):
    - action: `imgnd_act.detach()` → `jax.lax.stop_gradient(imagined_action)`
    - advantage: `.detach()` → `jax.lax.stop_gradient(advantage)`
    Both are stop_gradient'd in the REINFORCE log-prob-times-advantage product.

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

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from src.algorithms.dreamer_srl.loss import TwoHotEncoding, IndependentBernoulli, reconstruction_loss
from src.algorithms.dreamer_srl.utils import compute_lambda_values, moments_update, MomentsState
from src.algorithms.dreamer_srl.agent import action_shift


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


# ---------------------------------------------------------------------------
# CP7 — polyak_update (EMA target-critic update)
# ---------------------------------------------------------------------------

def polyak_update(
    online_params: Dict[str, jax.Array],
    target_params: Dict[str, jax.Array],
    tau: float,
) -> Dict[str, jax.Array]:
    """Pure-functional Polyak (EMA) update: target = (1-tau)*target + tau*online.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L678-L680
    (inside the inner gradient-step loop, BEFORE the `train()` call):

    sheeprl L678-L680:
        tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau
        for cp, tcp in zip(critic.module.parameters(), target_critic.parameters()):
            tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)

    Sheeprl writes the blend as `tau * online + (1-tau) * target` (i.e. `cp` is
    the online param, `tcp` is the target param). This is equivalent to the more
    common EMA form `target = (1-tau)*target + tau*online` — same algebra.

    CALL ORDER (§S7 ordering — sheeprl L673-L697):
        The polyak update fires BEFORE `train()` (= `one_train_step`) in sheeprl's
        inner gradient-step loop. The target critic used inside `train()` for value
        prediction is the FRESHLY UPDATED target from this iteration's polyak call.

    First call (tau=1.0 — hard copy):
        target = 1.0 * online + 0.0 * target = online  (byte-identical)
        This initializes the target network to match the online network at step 0.

    Subsequent calls (tau=0.02 — EMA blend, sheeprl XS default):
        target = 0.98 * target + 0.02 * online
        Slowly tracks the online network, providing a stable value bootstrap.

    JAX implementation (pure-functional):
        Sheeprl uses in-place mutation (`tcp.data.copy_(...)`). JAX cannot mutate
        arrays inside JIT. We instead return a NEW params dict. The caller replaces
        the target critic's parameter dict with the returned value. This is logged
        in DEVIATION_LOG.md as structural (same class as D-001: in-place mutation
        → pure-functional return).

    Args:
        online_params: flat dict {name: array} of online critic parameters.
                       Must have the same keys as target_params.
        target_params: flat dict {name: array} of target critic parameters.
                       Must have the same keys as online_params.
        tau:           EMA coefficient. tau=1.0 → hard copy. tau=0.02 → slow blend.
                       sheeprl default (XS config): tau=0.02.

    Returns:
        new_target_params: updated dict with the same keys as target_params.
                           Each array is `(1-tau) * target + tau * online`.

    Bit-identity tests:
        tests/algorithms/dreamer_srl/test_train.py::test_polyak_first_call_hard_copy
        tests/algorithms/dreamer_srl/test_train.py::test_polyak_subsequent_call_blend
    """
    return {
        k: (1.0 - tau) * target_params[k] + tau * online_params[k]
        for k in online_params
    }


# ---------------------------------------------------------------------------
# CP7 — compute_imagined_returns (§S5 true-continue splice + lambda values)
# ---------------------------------------------------------------------------

def compute_imagined_returns(
    predicted_rewards: jax.Array,
    predicted_values: jax.Array,
    continues_predicted: jax.Array,
    terminated_observed: jax.Array,
    gamma: float,
    lmbda: float,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """§S5 true-continue splice + lambda-value computation + discount weighting.

    Centralizes the three computations consumed by both actor and critic loss:
      1. §S5 true-continue splice: replace continues[0] with observed continue.
      2. compute_lambda_values: TD-lambda return estimation.
      3. compute_discount: §S6 cumprod discount mask.

    §S5 splice — ported from sheeprl@33b6366:
        dreamer_v3.py:L246-L248 (inside `train()` after imagination rollout):
            continues = Independent(BernoulliSafeMode(logits=...), 1).mode
            true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)
            continues = torch.cat((true_continue, continues[1:]))

    The splice replaces `continues[0]` — the world model's predicted continue
    for the first imagined step — with the REAL env continue from the replay
    buffer (`1 - terminated`). This grounds the first discount weight in reality:
      - If the agent was actually terminated, continues[0] = 0 → discount[0] = 0
        (the first imagined step contributes nothing, which is correct — there is
        no continuation from a terminal state).
      - If the agent was NOT terminated, continues[0] = 1 → discount[0] = 1.0
        (the first step is fully weighted).

    Lambda-values — ported from sheeprl@33b6366:
        dreamer_v3.py:L251-L256 (compute_lambda_values call):
            lambda_values = compute_lambda_values(
                predicted_rewards[1:],
                predicted_values[1:],
                continues[1:] * cfg.algo.gamma,
                lmbda=cfg.algo.lmbda,
            )

    Note: `continues[1:] * gamma` matches sheeprl's call signature where the
    `continues` argument to `compute_lambda_values` already includes the gamma
    factor. This is consistent with `utils.py:compute_lambda_values`.

    Args:
        predicted_rewards: [H+1, BT, 1] predicted rewards over the imagination
                           horizon. The `[1:]` slice (dropping step 0) is applied
                           here, matching sheeprl L253.
        predicted_values:  [H+1, BT, 1] predicted values over the imagination
                           horizon. The `[1:]` slice is applied here, matching
                           sheeprl L254.
        continues_predicted: [H+1, BT, 1] continue probabilities (mode of the
                             world model's Bernoulli continue head).
                             continues_predicted[0] is REPLACED by §S5 splice.
                             continues_predicted[1:] are used as-is.
        terminated_observed: [1, BT, 1] or [BT, 1] observed termination flag from
                             the replay buffer for this batch. Converted to
                             true_continue = 1 - terminated_observed.
        gamma: discount factor (float). Used in compute_lambda_values (via
               continues * gamma) and in compute_discount.
        lmbda: lambda mixing coefficient (float, default 0.95).

    Returns:
        (lambda_values, continues_spliced, discount)
        lambda_values:     [H, BT, 1] TD-lambda return targets (un-normalised).
                           Used by critic (raw) and actor (Moments-normalised).
        continues_spliced: [H+1, BT, 1] continues after §S5 splice.
                           continues_spliced[0] = 1 - terminated_observed.
                           continues_spliced[1:] = continues_predicted[1:].
        discount:          [H+1, BT, 1] from compute_discount(continues_spliced, gamma).
                           Already stop_gradient'd. Used by both actor and critic.

    Bit-identity test:
        Tests for lambda_values and discount are covered by the CP1 and CP6 tests
        respectively. The §S5 splice itself is a concat+slice — no numerical test
        needed beyond the structural shape assertion.
    """
    # §S5 splice: replace continues[0] with observed continue (1 - terminated)
    # sheeprl L247: true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)
    # terminated_observed may be [1, BT, 1] or [BT, 1] — ensure shape [1, BT, 1]
    true_continue = (1.0 - terminated_observed).reshape(1, continues_predicted.shape[1], 1)
    # sheeprl L248: continues = torch.cat((true_continue, continues[1:]))
    continues_spliced = jnp.concatenate(
        [true_continue, continues_predicted[1:]], axis=0
    )  # [H+1, BT, 1]

    # Lambda-value computation (sheeprl L251-L256)
    # sheeprl passes continues[1:] * gamma to compute_lambda_values
    lambda_vals = compute_lambda_values(
        predicted_rewards[1:],               # [H, BT, 1]
        predicted_values[1:],                # [H, BT, 1]
        continues_spliced[1:] * gamma,       # [H, BT, 1] — includes gamma factor
        lmbda=lmbda,
    )  # [H, BT, 1]

    # §S6 discount (sheeprl L259-L260)
    discount = compute_discount(continues_spliced, gamma)  # [H+1, BT, 1]

    return lambda_vals, continues_spliced, discount


# ---------------------------------------------------------------------------
# CP7 — compute_actor_objective (REINFORCE with §S7 advantage normalization)
# ---------------------------------------------------------------------------

def compute_actor_objective(
    log_probs: jax.Array,
    lambda_values: jax.Array,
    predicted_values: jax.Array,
    moments_offset: jax.Array,
    moments_invscale: jax.Array,
    entropy: jax.Array,
    discount: jax.Array,
    ent_coef: float,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Actor REINFORCE objective with §S7 advantage normalization.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L274-L297
    (actor optimization section of `train()`).

    sheeprl L274-L297:
        baseline = predicted_values[:-1]
        offset, invscale = moments(lambda_values, fabric)
        normed_lambda_values = (lambda_values - offset) / invscale
        normed_baseline = (baseline - offset) / invscale
        advantage = normed_lambda_values - normed_baseline
        # discrete action case (grid-world):
        objective = sum_log_prob(stop_gradient(action)) * advantage.detach()
        entropy = ent_coef * sum(p.entropy() for p in policies)
        policy_loss = -mean(discount[:-1].detach() * (objective + entropy[:-1]))

    §S7 advantage computation (offset cancellation note):
        advantage = (lambda - offset) / invscale - (baseline - offset) / invscale
                  = (lambda - baseline) / invscale    (algebra shows offset cancels)
        However, sheeprl computes BOTH normed terms SEPARATELY before subtracting.
        We match this per-term form for bit-identity (even though the offset
        cancels algebraically, floating-point evaluation of the two forms may
        produce different rounding errors in edge cases).

    STOP_GRADIENT discipline (sheeprl L287):
        - action log_probs: the `p.log_prob(imgnd_act.detach())` form —
          log_probs is computed with stop_gradient on the action.
        - advantage: `.detach()` → `jax.lax.stop_gradient(advantage)`.
          The gradient flows through log_probs (the policy distribution
          parameters), NOT through advantage.

    Args:
        log_probs: [H, BT, 1] sum of log-probs over action dimensions,
                   with stop_gradient already applied to the sampled action
                   (caller applies stop_gradient to the action before log_prob).
                   Shape: [H, BT, 1] (matching [:-1] slice of imagined_trajectories).
                   Sheeprl L284-L287: `p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]`
                   summed over action dimensions.
        lambda_values: [H, BT, 1] un-normalised lambda-return targets
                       (from compute_imagined_returns).
        predicted_values: [H+1, BT, 1] critic-predicted values on imagined
                          trajectories. baseline = predicted_values[:-1] applied here.
        moments_offset: scalar — `offset` from moments_update (low EMA).
                        Sheeprl L275: `offset, invscale = moments(lambda_values, fabric)`.
        moments_invscale: scalar — `invscale` from moments_update.
        entropy: [H+1, BT, 1] per-step policy entropy over imagined trajectories
                 (unsqueezed to [H+1, BT, 1]). The `[:-1]` slice is applied here,
                 matching sheeprl L295: `entropy.unsqueeze(dim=-1)[:-1]`.
        discount: [H+1, BT, 1] from compute_discount (already stop_gradient'd).
                  The `[:-1]` slice is applied here, matching sheeprl L296.
        ent_coef: entropy regularization coefficient (float, sheeprl XS default 3e-4).

    Returns:
        (policy_loss, objective, advantage)
        policy_loss: scalar — the final actor loss (to be minimized via gradient).
        objective:   [H, BT, 1] — log_probs * stop_gradient(advantage).
                     For continuous actions (not used in grid-world), this would
                     be just `advantage`. For discrete, it's log_probs * advantage.
        advantage:   [H, BT, 1] — normed_lambda - normed_baseline (before stop_grad).

    Bit-identity tests:
        tests/algorithms/dreamer_srl/test_train.py::test_actor_objective_advantage
        (these tests are CP7 Lever-A)
    """
    # sheeprl L274: baseline = predicted_values[:-1]
    baseline = predicted_values[:-1]  # [H, BT, 1]

    # §S7 per-term normalization (sheeprl L275-L278)
    # sheeprl L277: normed_lambda_values = (lambda_values - offset) / invscale
    normed_lambda_values = (lambda_values - moments_offset) / moments_invscale  # [H, BT, 1]
    # sheeprl L278: normed_baseline = (baseline - offset) / invscale
    normed_baseline = (baseline - moments_offset) / moments_invscale             # [H, BT, 1]

    # sheeprl L279: advantage = normed_lambda_values - normed_baseline
    # NOTE: offset cancels algebraically; we keep per-term form for bit-identity
    advantage = normed_lambda_values - normed_baseline  # [H, BT, 1]

    # sheeprl L281-L290 (discrete action case — grid-world is always discrete):
    # objective = log_prob(stop_gradient(action)) * advantage.detach()
    # log_probs already has stop_gradient on the action (caller's responsibility).
    # We apply stop_gradient on advantage here (sheeprl L290: `.detach()`).
    objective = log_probs * jax.lax.stop_gradient(advantage)  # [H, BT, 1]

    # sheeprl L295: entropy.unsqueeze(dim=-1)[:-1]  →  entropy[:-1]  (in our notation)
    entropy_term = ent_coef * entropy[:-1]  # [H, BT, 1]

    # sheeprl L296-L297:
    # discount[:-1].detach() * (objective + entropy[:-1])
    # policy_loss = -mean(...)
    # discount is already stop_gradient'd by compute_discount (= torch.no_grad())
    discount_weights = discount[:-1]  # [H, BT, 1] — stop_gradient already applied

    policy_loss = -jnp.mean(discount_weights * (objective + entropy_term))  # scalar

    return policy_loss, objective, advantage


# ---------------------------------------------------------------------------
# CP9 — one_train_step (full training-step orchestration)
# Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L48-L358
# (train() function)
# ---------------------------------------------------------------------------

def make_train_step(
    horizon: int,
    gamma: float,
    lmbda: float,
    ent_coef: float,
    kl_dynamic: float,
    kl_representation: float,
    kl_free_nats: float,
    kl_regularizer: float = 1.0,
    continue_scale_factor: float = 1.0,
    moments_decay: float = 0.99,
    moments_max: float = 1.0,
    moments_pct_low: float = 0.05,
    moments_pct_high: float = 0.95,
):
    """Factory: returns a JIT'd one_train_step with static hyperparams baked in.

    Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L48-L358.

    The `horizon` must be a static Python int (not a JAX array) because
    WorldModel.imagine uses it in a Python for-loop. Baking it into the
    closure prevents it from being traced as a JAX array.

    Usage::

        train_step = make_train_step(horizon=15, gamma=0.99, ...)
        new_moments, losses = train_step(
            world_model, actor, critic, target_critic,
            wm_opt, actor_opt, critic_opt, moments, batch, key,
        )
    """
    from src.algorithms.dreamer_srl.utils import symlog as _symlog

    H_plus_1 = horizon + 1  # captured as Python int in the closure

    @nnx.jit
    def one_train_step(
        world_model,
        actor,
        critic,
        target_critic,
        wm_opt: nnx.Optimizer,
        actor_opt: nnx.Optimizer,
        critic_opt: nnx.Optimizer,
        moments: MomentsState,
        batch: Dict[str, jax.Array],
        key: jax.Array,
    ) -> Tuple[MomentsState, Dict[str, jax.Array]]:
        """Inner JIT'd training step. See make_train_step for parameter docs.

        Ported from sheeprl@33b6366:sheeprl/algos/dreamer_v3/dreamer_v3.py:L48-L358
        (train() function, Fabric-stripped, single-process JAX/NNX form).

        Order (mirroring sheeprl L93-L317):
          1. §S1 force-set is_first[0] = 1 (sheeprl L100)
          2. §S2 action_shift(batch["actions"]) (sheeprl L104)
          3. WM forward: world_model.observe(obs, shifted_actions, is_first, key)
          4. WM losses: reconstruction_loss(decoder, reward, continue, KL terms)
          5. WM optimizer step (nnx.value_and_grad + wm_opt.update)
          6. Imagined trajectory rollout (horizon steps from posterior)
          7. §S5 true-continue splice + lambda values + discount
          8. Moments update
          9. NOTE: Polyak update fires in the main loop BEFORE one_train_step.
         10. Actor objective (compute_actor_objective, §S7 advantage normalization)
         11. Actor optimizer step
         12. Critic loss (two-term cascade fix #29, compute_critic_loss)
         13. Critic optimizer step
        """
        # -----------------------------------------------------------------------
        # §S1: force-set is_first[0] = 1 (sheeprl L100)
        # Sheeprl: data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])
        # JAX: functional index update (no in-place mutation in JIT)
        # -----------------------------------------------------------------------
        is_first = batch["is_first"]  # [T, B, 1]
        is_first = is_first.at[0].set(jnp.ones_like(is_first[0]))  # first row forced to 1.0

        # -----------------------------------------------------------------------
        # §S2: action-shift (sheeprl L104)
        # Sheeprl: batch_actions = cat(zeros[:1], data["actions"][:-1], dim=0)
        # -----------------------------------------------------------------------
        shifted_actions = action_shift(batch["actions"])  # [T, B, action_dim]

        # -----------------------------------------------------------------------
        # Sub-steps 3-5: World Model forward + losses + optimizer step
        # -----------------------------------------------------------------------
        key, k_wm = jax.random.split(key)

        def wm_loss_fn(wm):
            """WM loss (reconstruction + KL). Returns (total_loss, aux_dict)."""
            wm_outputs = wm.observe(batch["obs"], shifted_actions, is_first, k_wm)

            # Observation NLL — SymlogDistribution: Normal(symlog(pred), 1).log_prob(symlog(target))
            # sheeprl L152-L162: po[k] = SymlogDistribution(reconstructed_obs[k], dims=1)
            # dims=1 sums over the last 1 event dim (obs_dim axis)
            reconstructed_obs = wm_outputs["reconstructed_obs"]  # [T, B, obs_dim]
            obs_target = batch["obs"]                            # [T, B, obs_dim]
            obs_log_prob = -0.5 * jnp.sum(
                (_symlog(reconstructed_obs) - _symlog(obs_target)) ** 2, axis=-1
            )  # [T, B]
            obs_loss_unreduced = -obs_log_prob  # [T, B] — positive NLL

            # Reward NLL (TwoHotEncoding)
            pr = TwoHotEncoding(wm_outputs["reward_logits"], dims=1)  # [T, B, 255]
            reward_loss_unreduced = -pr.log_prob(batch["rewards"])    # [T, B]

            # Continue NLL (IndependentBernoulli — §S9)
            pc = IndependentBernoulli(wm_outputs["continue_logits"])   # logits: [T, B, 1]
            # §S10: continue target = 1 - terminated (NO gamma multiplier)
            # sheeprl L168: continues_targets = 1 - data["terminated"]
            continue_targets = 1.0 - batch["terminated"]               # [T, B, 1]
            cont_loss_unreduced = continue_scale_factor * (-pc.log_prob(continue_targets))  # [T, B]

            # KL losses (§S8: free-nats per-element BEFORE mean)
            num_cat = wm.rssm.num_categoricals
            num_cls = wm.rssm.num_classes
            post_logits = wm_outputs["posterior_logits"].reshape(
                *wm_outputs["posterior_logits"].shape[:2], num_cat, num_cls
            )  # [T, B, S, D]
            prior_logits = wm_outputs["prior_logits"].reshape(
                *wm_outputs["prior_logits"].shape[:2], num_cat, num_cls
            )  # [T, B, S, D]

            log_post = jax.nn.log_softmax(post_logits, axis=-1)   # [T, B, S, D]
            log_prior = jax.nn.log_softmax(prior_logits, axis=-1)  # [T, B, S, D]

            def _kl(lp, lq):
                """KL(p||q) summed over S categoricals and D classes → [T, B]."""
                p = jnp.exp(lp)
                return (p * (lp - lq)).sum(axis=-1).sum(axis=-1)  # [T, B]

            kl_dyn = _kl(jax.lax.stop_gradient(log_post), log_prior)  # [T, B]
            kl_repr = _kl(log_post, jax.lax.stop_gradient(log_prior))  # [T, B]
            kl_raw = kl_dyn  # for logging

            dyn_loss = kl_dynamic * jnp.maximum(kl_dyn, kl_free_nats)     # §S8 floor
            repr_loss = kl_representation * jnp.maximum(kl_repr, kl_free_nats)
            kl_loss_2d = dyn_loss + repr_loss  # [T, B]

            total = (kl_regularizer * kl_loss_2d + obs_loss_unreduced + reward_loss_unreduced + cont_loss_unreduced).mean()

            aux = {
                "wm_outputs": wm_outputs,
                "kl_mean": kl_raw.mean(),
                "kl_loss_mean": kl_loss_2d.mean(),
                "reward_loss_mean": reward_loss_unreduced.mean(),
                "obs_loss_mean": obs_loss_unreduced.mean(),
                "cont_loss_mean": cont_loss_unreduced.mean(),
            }
            return total, aux

        (wm_total_loss, wm_aux), wm_grads = nnx.value_and_grad(
            wm_loss_fn, has_aux=True
        )(world_model)
        wm_outputs = wm_aux["wm_outputs"]
        wm_opt.update(world_model, wm_grads)

        # -----------------------------------------------------------------------
        # Sub-step 6: Imagined trajectory rollout (from posterior latent)
        # sheeprl L202-L241
        # -----------------------------------------------------------------------
        posteriors = jax.lax.stop_gradient(wm_outputs["posteriors"])     # [T, B, S, D]
        recurrent_states = jax.lax.stop_gradient(wm_outputs["recurrent_states"])  # [T, B, hx]

        T_val = posteriors.shape[0]
        B_val = posteriors.shape[1]
        BT = T_val * B_val

        stoch_flat = posteriors.reshape(T_val, B_val, -1)  # [T, B, S*D]
        init_prior_flat = stoch_flat.reshape(BT, -1)        # [BT, S*D]
        init_recurrent = recurrent_states.reshape(BT, -1)   # [BT, hx]
        init_latent = jnp.concatenate([init_prior_flat, init_recurrent], axis=-1)  # [BT, latent_dim]

        key, k_imag = jax.random.split(key)
        imag_outputs = world_model.imagine(init_latent, actor, horizon, k_imag)

        imagined_latents = imag_outputs["imagined_latents"]    # [H+1, BT, latent_dim]

        # Predict rewards, values, continues on imagined trajectories (sheeprl L244-L248)
        imag_flat = imagined_latents.reshape(H_plus_1 * BT, -1)

        predicted_rewards_logits = jax.vmap(world_model.reward_model)(imag_flat)
        predicted_rewards = TwoHotEncoding(
            predicted_rewards_logits.reshape(H_plus_1, BT, -1), dims=1
        ).mean  # [H+1, BT, 1]

        predicted_values_logits = jax.vmap(target_critic)(imag_flat)
        predicted_values = TwoHotEncoding(
            predicted_values_logits.reshape(H_plus_1, BT, -1), dims=1
        ).mean  # [H+1, BT, 1]

        continues_logits = jax.vmap(world_model.continue_model)(imag_flat)
        continues_predicted = IndependentBernoulli(
            continues_logits.reshape(H_plus_1, BT, 1)
        ).mode   # [H+1, BT, 1]

        # §S5: true-continue splice + lambda values + discount
        terminated_flat = batch["terminated"].reshape(BT, 1)  # [BT, 1]

        lambda_values, continues_spliced, discount = compute_imagined_returns(
            predicted_rewards=predicted_rewards,
            predicted_values=predicted_values,
            continues_predicted=continues_predicted,
            terminated_observed=terminated_flat,
            gamma=gamma,
            lmbda=lmbda,
        )  # lambda_values: [H, BT, 1], discount: [H+1, BT, 1]

        # -----------------------------------------------------------------------
        # Sub-step 8: Moments update (sheeprl L262-L270)
        # -----------------------------------------------------------------------
        new_moments, moments_offset, moments_invscale = moments_update(
            moments,
            lambda_values,
            decay=moments_decay,
            max_=moments_max,
            percentile_low=moments_pct_low,
            percentile_high=moments_pct_high,
        )

        # -----------------------------------------------------------------------
        # Sub-step 10: Actor objective + optimizer step (sheeprl L272-L304)
        # Re-run actor on stop_gradient'd latents to get differentiable log_probs.
        # Gradient flows through actor logits; sg(action) applied inside Actor.__call__.
        # -----------------------------------------------------------------------
        actor_keys = jax.random.split(jax.random.PRNGKey(0), H_plus_1)
        sg_latents = jax.lax.stop_gradient(imagined_latents)  # [H+1, BT, latent_dim]

        def actor_loss_fn(actor_module):
            """Actor loss: REINFORCE + entropy, §S7 advantage normalization."""
            all_log_probs = []
            all_entropies = []
            for h in range(H_plus_1):  # Python loop — H_plus_1 is a static Python int
                _, lp_h, ent_h = actor_module(sg_latents[h], actor_keys[h])
                all_log_probs.append(lp_h)   # [BT, 1]
                all_entropies.append(ent_h)  # [BT]
            log_probs_arr = jnp.stack(all_log_probs, axis=0)   # [H+1, BT, 1]
            entropies_arr = jnp.stack(all_entropies, axis=0)   # [H+1, BT]

            log_probs_sliced = log_probs_arr[:-1]               # [H, BT, 1]
            entropy_for_obj = entropies_arr[..., None]          # [H+1, BT, 1]

            policy_loss, _, _ = compute_actor_objective(
                log_probs=log_probs_sliced,
                lambda_values=jax.lax.stop_gradient(lambda_values),
                predicted_values=jax.lax.stop_gradient(predicted_values),
                moments_offset=jax.lax.stop_gradient(moments_offset),
                moments_invscale=jax.lax.stop_gradient(moments_invscale),
                entropy=entropy_for_obj,
                discount=discount,
                ent_coef=ent_coef,
            )
            return policy_loss

        (actor_policy_loss, actor_grads) = nnx.value_and_grad(actor_loss_fn)(actor)
        actor_opt.update(actor, actor_grads)

        # -----------------------------------------------------------------------
        # Sub-step 12: Critic loss + optimizer step (sheeprl L306-L326)
        # sheeprl L307: qv = TwoHotEncodingDistribution(critic(imag_traj.detach()[:-1]), dims=1)
        # -----------------------------------------------------------------------
        sg_latents_h = jax.lax.stop_gradient(imagined_latents[:-1])  # [H, BT, latent_dim]

        target_critic_logits = jax.vmap(target_critic)(
            sg_latents_h.reshape(horizon * BT, -1)
        ).reshape(horizon, BT, -1)  # [H, BT, 255]
        target_critic_values = TwoHotEncoding(target_critic_logits, dims=1).mean  # [H, BT, 1]

        def critic_loss_fn(critic_module):
            """Two-term critic NLL with EMA target (cascade fix #29)."""
            critic_logits = jax.vmap(critic_module)(
                sg_latents_h.reshape(horizon * BT, -1)
            ).reshape(horizon, BT, -1)  # [H, BT, 255]

            value_loss, _, _ = compute_critic_loss(
                qv_logits=critic_logits,
                lambda_values=lambda_values,
                target_critic_values=target_critic_values,
                discount=discount,
            )
            return value_loss

        (critic_value_loss, critic_grads) = nnx.value_and_grad(critic_loss_fn)(critic)
        critic_opt.update(critic, critic_grads)

        # -----------------------------------------------------------------------
        # Return losses for logging (sheeprl L330-L346)
        # -----------------------------------------------------------------------
        losses = {
            "world_model_loss": wm_total_loss,
            "observation_loss": wm_aux["obs_loss_mean"],
            "reward_loss": wm_aux["reward_loss_mean"],
            "state_loss": wm_aux["kl_loss_mean"],
            "continue_loss": wm_aux["cont_loss_mean"],
            "value_loss": critic_value_loss,
            "policy_loss": actor_policy_loss,
            "moments_invscale": moments_invscale,
        }

        return new_moments, losses

    return one_train_step
