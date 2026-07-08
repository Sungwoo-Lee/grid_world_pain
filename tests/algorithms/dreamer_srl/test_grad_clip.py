"""WP-SRL P1 regression tests — gradient clipping on the three optimizers.

The 2026-07-06 re-audit ([[00_master_comparison]] §3 P1, area report 04 D-01)
found the port applied raw optax.adam with NO gradient clipping, while sheeprl
clips gradients on every optimizer step:
  - world model: max_norm 1000.0  (dreamer_v3.py:193-197, dreamer_v3.yaml:52)
  - actor:       max_norm 100.0   (dreamer_v3.py:300-302, dreamer_v3.yaml:127)
  - critic:      max_norm 100.0   (dreamer_v3.py:320-324, dreamer_v3.yaml:154)
Live smoke runs showed the unclipped WM loss exploding to ~1e29.

Fix: utils.make_optim_tx(lr, eps, clip_norm) = optax.chain(
         optax.clip_by_global_norm(clip_norm), optax.adam(lr, eps=eps))
with module-level recipe constants WM_CLIP_NORM / ACTOR_CLIP_NORM /
CRITIC_CLIP_NORM pinned to the sheeprl YAML values.

Red evidence (pre-fix): both tests fail at collection with ImportError —
make_optim_tx and the three constants do not exist ("red by absence" per
fix_plan_srl_parity.md test contract).

Run with:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
        -m pytest tests/algorithms/dreamer_srl/test_grad_clip.py -v
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from src.algorithms.dreamer_srl.utils import (
    ACTOR_CLIP_NORM,
    CRITIC_CLIP_NORM,
    WM_CLIP_NORM,
    make_optim_tx,
)


def _global_norm(tree) -> float:
    return float(optax.global_norm(tree))


def test_clip_norm_constants() -> None:
    """The three clip norms pin to sheeprl@33b6366 dreamer_v3.yaml:52/127/154.

    These are hardcoded recipe constants (see fix_plan_srl_parity.md P1 design
    note), exactly as fixed as the two-hot bin count. Any drift from the
    reference recipe values is a parity regression.
    """
    assert WM_CLIP_NORM == 1000.0, (
        f"WM_CLIP_NORM={WM_CLIP_NORM} != 1000.0 "
        "(sheeprl dreamer_v3.yaml:52 algo.world_model.clip_gradients)"
    )
    assert ACTOR_CLIP_NORM == 100.0, (
        f"ACTOR_CLIP_NORM={ACTOR_CLIP_NORM} != 100.0 "
        "(sheeprl dreamer_v3.yaml:127 algo.actor.clip_gradients)"
    )
    assert CRITIC_CLIP_NORM == 100.0, (
        f"CRITIC_CLIP_NORM={CRITIC_CLIP_NORM} != 100.0 "
        "(sheeprl dreamer_v3.yaml:154 algo.critic.clip_gradients)"
    )


def test_spike_grad_does_not_poison_adam() -> None:
    """A single exploding gradient must not freeze subsequent Adam updates.

    The real post-spike pathology (observed as the WM-loss ~1e29 spike): an
    unclipped gradient of global norm 1e8 inflates Adam's second moment nu to
    ~1e15-scale; nu decays at b2=0.999 while the spike's momentum decays at
    b1=0.9, so once the momentum washes out, unit-scale gradients produce
    updates of ~lr * 1/sqrt(nu_hat) ~ 1e-10 — learning freezes for thousands
    of steps.

    Discriminating fixture: step 1 applies a spike gradient (global norm 1e8),
    then 200 unit-global-norm gradient steps follow. Assert:
      - WITH the clip (make_optim_tx, clip_norm=100): the final update's
        global norm stays > 1e-5 (Adam's nu stays 100-scale; learning moves).
      - WITHOUT the clip (plain optax.adam, same lr/eps): the final update's
        global norm is < 1e-6 (frozen) — documents the pre-fix pathology and
        makes this test a regression guard if the chain ever loses the clip.
      - Step-1 post-clip updates are finite.
    """
    lr, eps, clip_norm = 1e-3, 1e-8, 100.0
    n_params = 10
    params = jnp.zeros((n_params,), dtype=jnp.float32)

    # Spike gradient: global norm exactly 1e8; then unit-global-norm gradients.
    spike_grad = jnp.full((n_params,), 1e8 / jnp.sqrt(n_params), dtype=jnp.float32)
    unit_grad = jnp.full((n_params,), 1.0 / jnp.sqrt(n_params), dtype=jnp.float32)
    assert abs(_global_norm(spike_grad) - 1e8) / 1e8 < 1e-5
    assert abs(_global_norm(unit_grad) - 1.0) < 1e-5

    def run(tx: optax.GradientTransformation) -> tuple[float, float]:
        """Return (step1_update_norm, final_update_norm) after spike + 200 unit steps."""
        state = tx.init(params)
        updates, state = tx.update(spike_grad, state, params)
        step1_norm = _global_norm(updates)
        assert bool(jnp.all(jnp.isfinite(updates))), "step-1 post-clip update not finite"
        final_norm = step1_norm
        for _ in range(200):
            updates, state = tx.update(unit_grad, state, params)
            final_norm = _global_norm(updates)
        return step1_norm, final_norm

    # WITH clip: the WP-SRL P1 factory (clip-by-global-norm -> Adam).
    _, clipped_final = run(make_optim_tx(lr, eps, clip_norm))
    # WITHOUT clip: plain Adam — the pre-fix optimizer construction.
    _, unclipped_final = run(optax.adam(lr, eps=eps))

    assert clipped_final > 1e-5, (
        f"Clipped-Adam final update norm {clipped_final:.3e} <= 1e-5: the clip "
        "is not protecting Adam's second moment from the spike (the chain may "
        "have lost optax.clip_by_global_norm)."
    )
    assert unclipped_final < 1e-6, (
        f"Unclipped-Adam final update norm {unclipped_final:.3e} >= 1e-6: the "
        "fixture no longer discriminates (expected the 1e8 spike to poison nu "
        "and freeze unit-grad updates). Re-derive the fixture."
    )
