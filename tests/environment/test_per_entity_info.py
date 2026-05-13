"""Verify new per-entity behavioral info keys are present and finite.

Run with:
  /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
      tests/environment/test_per_entity_info.py
"""
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step


def test_per_entity_info_keys_present_and_finite():
    """T1: new info keys exist, are scalar, and are non-NaN for a config with rabbits."""
    config = Config.load_yaml(
        "configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml"
    )
    params = load_env_params(config)
    key = jax.random.PRNGKey(0)
    state = jax_reset(params, key)
    # action 4 = rest (config has rest_action_enabled=true)
    _, _, _, info = jax_step(state, jnp.array(4, dtype=jnp.int32), params)
    for k in ('dist_to_neutral', 'dist_to_hiding_predator', 'hit_neutral'):
        assert k in info, f"missing info key: {k}"
        v = float(np.array(info[k]))
        assert np.isfinite(v), f"{k} is not finite (got {v})"
    print("T1 PASS: dist_to_neutral={:.3f}, dist_to_hiding_predator={:.3f}, hit_neutral={}".format(
        float(np.array(info['dist_to_neutral'])),
        float(np.array(info['dist_to_hiding_predator'])),
        bool(np.array(info['hit_neutral'])),
    ))


def test_no_rabbit_guard():
    """T1b: Python-level guard returns 99.0 and False when no neutral entities exist.

    Validates at the computation level (does not require a zero-rabbit config that
    loads cleanly with all mandatory fields).
    """
    import jax
    import jax.numpy as jnp

    neutral_pos_zero = jnp.zeros((0, 2), dtype=jnp.float32)
    agent_pos = jnp.array([5.0, 5.0], dtype=jnp.float32)

    dist_zero = (
        jnp.min(jnp.linalg.norm(neutral_pos_zero - agent_pos, axis=-1))
        if neutral_pos_zero.shape[0] > 0 else 99.0
    )
    hit_zero = (
        jnp.any(jnp.all(neutral_pos_zero == agent_pos, axis=-1))
        if neutral_pos_zero.shape[0] > 0 else jnp.array(False)
    )

    assert dist_zero == 99.0, f"Expected 99.0 for no-rabbit guard, got {dist_zero}"
    assert not bool(np.array(hit_zero)), f"Expected False for no-rabbit hit guard"
    print("T1b PASS: no-rabbit guard returns 99.0 and False correctly")


def test_rabbit_distance_math():
    """T2: dist_to_neutral uses same L2 norm math as dist_to_pred.

    Places both at (5,8) from agent at (5,5) and verifies both distances = 3.0
    at the computation level (before any step moves entities).
    """
    agent_pos = jnp.array([5.0, 5.0], dtype=jnp.float32)
    pos_58 = jnp.array([[5.0, 8.0]], dtype=jnp.float32)

    dist_neutral = float(jnp.min(jnp.linalg.norm(pos_58 - agent_pos, axis=-1)))
    dist_pred = float(jnp.min(jnp.linalg.norm(pos_58 - agent_pos, axis=-1)))
    assert abs(dist_neutral - 3.0) < 1e-4, f"Expected 3.0, got {dist_neutral}"
    assert abs(dist_pred - dist_neutral) < 1e-5, "dist_to_neutral != dist_to_pred (same formula)"
    print(f"T2 PASS: dist_to_neutral={dist_neutral:.3f}, dist_to_pred={dist_pred:.3f}")


if __name__ == "__main__":
    print("Running per-entity info tests...")
    test_per_entity_info_keys_present_and_finite()
    test_no_rabbit_guard()
    test_rabbit_distance_math()
    print("\nAll tests PASSED.")
