"""Regression tests for WP-NNX F2 (U1) — missing stop-gradients in the
DreamerV3-NNX imagination phase.

Plain-language context: discrete-action DreamerV3 trains its actor by pure
REINFORCE — the reference detaches imagined latent features everywhere the
actor and critic consume them, and detaches the sampled action inside the
log-prob (sheeprl dreamer_v3.py:219,240,273,286,307). Our implementation
detached only the rollout START states, so — because action sampling uses a
straight-through estimator — a differentiable path ran from the critic's loss
and from later imagination steps back into the actor's logits. Consequences:
(i) the critic-loss gradients update the ACTOR; (ii) later-step log-prob and
entropy terms backprop through the world-model dynamics into earlier action
probabilities; (iii) the log-prob carries a spurious grad term through the ST
action sample. The fix detaches imagined feats at creation and the action at
log-prob consumption, mirroring sheeprl's detach sites.

See docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/
fix_plan_nnx_parity.md (F2) and 05_dreamer_v3_nnx_conventions.md (U1).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dreamer_nnx_fixtures import build_trainer_and_env

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from dreamer_nnx_rollout_replica import replica_actor_loss


def _setup(n_rows=6, seed=3):
    # zero_init_reward_critic=False: the live config zero-inits the critic and
    # reward output heads, which makes the U1 leak paths carry ~zero gradient
    # AT INITIALIZATION (critic final layer = 0 -> no grad through the critic;
    # returns/baseline = 0 -> advantage = 0). Random heads make the leak
    # visible, as it is on any trained network.
    trainer, params, env_state, _ = build_trainer_and_env(
        max_steps=50, cfg_overrides={'agent.zero_init_reward_critic': False})
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    fresh = trainer.agent.wm.rssm.initial(n_rows)
    # Full state dict — the imagination scan carries {deter, stoch, logits,
    # prev_action} structurally (imagine_step returns all four).
    start_state = dict(fresh)
    start_state['deter'] = jax.random.normal(k1, fresh['deter'].shape)
    start_state['stoch'] = jax.random.normal(k2, fresh['stoch'].shape)
    rng = jax.random.PRNGKey(17)
    moments_low = jnp.array(0.0)
    moments_invscale = jnp.array(1.0)
    return trainer, start_state, rng, moments_low, moments_invscale


def _grad_leaves(grads):
    return jax.tree.leaves(nnx.state(grads) if isinstance(grads, nnx.Module) else grads)


def test_critic_loss_gradient_wrt_actor_is_exactly_zero():
    """T2(a): the critic-loss component alone must produce EXACTLY zero
    gradient on every actor leaf (sheeprl detaches the features the critic
    consumes, D:307, and the actor shares no live path with the critic loss).
    Pre-fix: nonzero — the undetached rollouts['feat'] -> ST-action chain
    carries critic-loss gradients into the actor."""
    trainer, start_state, rng, low, inv = _setup()

    def critic_loss_component(actor):
        _, (metrics, _) = trainer._behavior_loss(
            actor, trainer.agent.ac.critic, rng, start_state, None, low, inv)
        return metrics['loss_critic']

    grads = nnx.grad(critic_loss_component)(trainer.agent.ac.actor)
    leaves = _grad_leaves(grads)
    assert leaves, "no gradient leaves found — test wiring broken"
    for leaf in leaves:
        assert bool(jnp.all(leaf == 0.0)), (
            "critic-loss gradients leak into the ACTOR through the undetached "
            "imagined-feature / straight-through action chain (U1 site i); "
            f"max |g| = {float(jnp.max(jnp.abs(leaf)))}"
        )


def test_actor_loss_gradient_wrt_world_model_is_exactly_zero():
    """Structural corollary (robust to later F4/F5 arithmetic changes): with
    feats detached at creation, the sampled action detached in the log-prob,
    and advantage/weights already stop-gradient-protected, the actor loss has
    NO live path into world-model parameters. Pre-fix: nonzero (dynamics
    backprop, U1 site ii)."""
    trainer, start_state, rng, low, inv = _setup()

    def actor_loss_of_trainer(tr):
        _, (metrics, _) = tr._behavior_loss(
            tr.agent.ac.actor, tr.agent.ac.critic, rng, start_state, None,
            low, inv)
        return metrics['loss_actor']

    grads = nnx.grad(actor_loss_of_trainer)(trainer)
    wm_leaves = jax.tree.leaves(nnx.state(grads)['agent']['wm'])
    actor_leaves = jax.tree.leaves(nnx.state(grads)['agent']['ac']['actor'])
    assert wm_leaves and actor_leaves

    # Sanity: the actor itself DOES get a gradient (non-vacuity).
    assert any(bool(jnp.any(l != 0.0)) for l in actor_leaves), (
        "vacuous: actor loss produced no actor gradient at all"
    )
    for leaf in wm_leaves:
        assert bool(jnp.all(leaf == 0.0)), (
            "actor-loss gradients backprop through the imagined world-model "
            "dynamics (U1 site ii); "
            f"max |g| = {float(jnp.max(jnp.abs(leaf)))}"
        )


def test_actor_gradient_matches_detached_reference():
    """T2(b): the actor-loss gradient w.r.t. actor params must equal a
    reference computation in which imagined feats and sampled actions are
    explicitly detached at sheeprl's detach sites (same PRNG stream).
    Pre-fix: differs — the audit's probe showed e.g. [-0.2486,...] vs
    [-0.2138,...] on the same key (extra ST/dynamics terms ride the
    REINFORCE gradient, U1 sites ii+iii)."""
    trainer, start_state, rng, low, inv = _setup()

    def actual_actor_loss(actor):
        _, (metrics, _) = trainer._behavior_loss(
            actor, trainer.agent.ac.critic, rng, start_state, None, low, inv)
        return metrics['loss_actor']

    def reference_actor_loss(actor):
        # NOTE: value_net must track the production bootstrap source —
        # post-F4 (U2) that is the ONLINE critic (sheeprl D:244).
        return replica_actor_loss(
            trainer, actor, rng, start_state, low, inv,
            value_net=trainer.agent.ac.critic)

    # Forward values must agree (replica reproduces the same rollout) —
    # guards against PRNG-stream drift making the grad comparison vacuous.
    actual_val = actual_actor_loss(trainer.agent.ac.actor)
    ref_val = reference_actor_loss(trainer.agent.ac.actor)
    np.testing.assert_allclose(np.asarray(actual_val), np.asarray(ref_val),
                               rtol=1e-5, atol=1e-6,
                               err_msg="replica rollout diverged from "
                                       "production forward pass")

    g_actual = nnx.state(nnx.grad(actual_actor_loss)(trainer.agent.ac.actor))
    g_ref = nnx.state(nnx.grad(reference_actor_loss)(trainer.agent.ac.actor))

    flat_a = jax.tree.leaves(g_actual)
    flat_r = jax.tree.leaves(g_ref)
    assert len(flat_a) == len(flat_r)
    for la, lr in zip(flat_a, flat_r):
        np.testing.assert_allclose(
            np.asarray(la), np.asarray(lr), rtol=1e-4, atol=1e-6,
            err_msg=("actor gradient deviates from the pure-REINFORCE "
                     "(detached) reference — undetached imagination paths "
                     "contaminate the actor gradient (U1)"))
