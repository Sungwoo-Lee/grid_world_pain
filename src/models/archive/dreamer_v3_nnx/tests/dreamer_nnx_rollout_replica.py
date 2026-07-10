"""Reference replica of DreamerTrainer._behavior_loss's ACTOR loss with the
recipe's stop-gradients applied explicitly (WP-NNX F2 / T2(b) reference).

This reproduces, step by step and on the same PRNG stream, the plain-branch
(modulation off) imagination rollout and actor-loss arithmetic of
src/models/dreamer_v3_trainer.py::_behavior_loss, but with imagined features
and sampled actions detached exactly where sheeprl detaches them
(dreamer_v3.py:219,240,273,286,307). Post-F2 the production code computes the
same gradient; pre-F2 it does not (critic-loss and dynamics gradients leak
into the actor).

MAINTENANCE: this file mirrors the production arithmetic and must be updated
in lockstep with intentional changes to _behavior_loss:
  - F4 (U2) switched the bootstrap source from target_critic to the online
    critic  -> mirrored via the `value_net` argument (callers pass whichever
    network the production code currently bootstraps from).
  - F5 (U5) sets discount-weight row 0 to the start rows' true continue ->
    mirrored via the `true_cont0` argument (None = pre-F5 behaviour, row 0 = 1).
"""
import jax
import jax.numpy as jnp
from jax import random

from src.models.archive.dreamer_v3_nnx.dreamer_v3_trainer import (
    compute_lambda_values, HORIZON, GAMMA,
)
from src.models.archive.dreamer_v3_nnx.dreamer_v3_util import from_twohot, OneHotDist

sg = jax.lax.stop_gradient


def replica_rollout(trainer, actor, rng, start_state, value_net=None):
    """Forward-replicate the plain-branch imagination rollout on the same
    PRNG stream as _behavior_loss. Detaches feats at creation and actions at
    storage (the recipe's detach sites). Returns a dict of stacked (H, N, ...)
    arrays: rews, conts, vals, feats, logits, actions, plus v_start (N,)."""
    wm = trainer.agent.wm
    if value_net is None:
        value_net = trainer.agent.ac.critic
    bins = trainer._paper_canonical_twohot_bins

    imag_batch = start_state['deter'].shape[0]
    rng_imag = random.split(rng, HORIZON * imag_batch).reshape(
        (HORIZON, imag_batch, -1))

    prev_state = start_state
    rews, conts, vals, feats, logits_seq, actions_seq = [], [], [], [], [], []
    for t in range(HORIZON):
        key = rng_imag[t]
        # sheeprl D:219,240,273 — the actor consumes DETACHED imagined feats.
        feat = sg(wm.get_feat(prev_state))
        actor_out = actor(feat)
        dist = OneHotDist(actor_out)
        action = dist.sample(key)
        prior = wm.rssm.imagine_step(prev_state, action, key)

        next_feat = wm.get_feat(prior)
        rew = from_twohot(wm.reward_head(next_feat), paper_canonical_bins=bins)
        cont = jax.nn.sigmoid(wm.continue_head(next_feat)).squeeze(-1)
        # Bootstrap value source — see `value_net` docstring (F4/U2).
        val = from_twohot(value_net(next_feat), paper_canonical_bins=bins)

        rews.append(rew); conts.append(cont); vals.append(val)
        feats.append(feat); logits_seq.append(actor_out)
        # sheeprl D:286 — log_prob consumes the DETACHED action.
        actions_seq.append(sg(action))
        prev_state = prior

    start_feat = wm.get_feat(start_state)
    # v_start from the same bootstrap source (F4/U2 — see value_net).
    v_start = from_twohot(value_net(start_feat), paper_canonical_bins=bins)

    return {
        'rews': jnp.stack(rews), 'conts': jnp.stack(conts),
        'vals': jnp.stack(vals), 'feats': jnp.stack(feats),
        'logits': jnp.stack(logits_seq), 'actions': jnp.stack(actions_seq),
        'v_start': v_start,
    }


def replica_discount_weights(conts, true_cont0=None):
    """Cumulative discount weights as in _behavior_loss.

    true_cont0=None reproduces the pre-F5 behaviour (row 0 = 1); post-F5
    (U5) callers pass the start rows' true continue (sheeprl D:247-248)."""
    if true_cont0 is None:
        row0 = jnp.ones_like(conts[:1])
    else:
        row0 = true_cont0[None] * jnp.ones_like(conts[:1])
    discount_weights = jnp.concatenate([row0, conts[:-1] * GAMMA], axis=0)
    return sg(jnp.cumprod(discount_weights, axis=0))


def replica_actor_loss(trainer, actor, rng, start_state,
                       moments_low, moments_invscale, value_net=None,
                       true_cont0=None):
    """Actor loss with explicit recipe detaches. Same PRNG stream as
    _behavior_loss. Returns the scalar loss_actor.

    value_net: network supplying the lambda-return bootstrap values
    (`val`/`v_start`). Pass whichever network the production code currently
    uses (pre-F4: trainer.target_critic; post-F4: the online critic).
    Defaults to the online critic. The advantage BASELINE always uses the
    online critic (both pre- and post-F4)."""
    ro = replica_rollout(trainer, actor, rng, start_state, value_net=value_net)
    rews, conts, vals = ro['rews'], ro['conts'], ro['vals']
    feats, logits, actions = ro['feats'], ro['logits'], ro['actions']
    critic = trainer.agent.ac.critic
    bins = trainer._paper_canonical_twohot_bins

    all_vals = jnp.concatenate([ro['v_start'][None], vals], axis=0)
    lambda_returns = compute_lambda_values(rews, all_vals, conts * GAMMA)
    norm_returns = (lambda_returns - moments_low) / moments_invscale

    discount_weights = replica_discount_weights(conts, true_cont0)

    # sheeprl D:307 — critic consumes detached feats; baseline for advantage.
    v_pred_logits = critic(sg(feats))
    baseline = from_twohot(v_pred_logits, paper_canonical_bins=bins)
    norm_baseline = (baseline - moments_low) / moments_invscale
    advantage = sg(norm_returns - norm_baseline)

    log_probs = jnp.sum(actions * jax.nn.log_softmax(logits), axis=-1)
    ES = trainer.config.get_mandatory('agent.entropy_scale', float)
    entropy = -jnp.sum(jax.nn.softmax(logits) * jax.nn.log_softmax(logits), axis=-1)

    loss_actor_step = -(log_probs * advantage + ES * entropy)
    return jnp.mean(loss_actor_step * discount_weights)
