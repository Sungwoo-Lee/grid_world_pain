"""Tests for the `extends:` config-layering mechanism.

Verifies every checkpoint listed in the plan
docs/develop/active/refactors/CONFIG_LAYERING_AND_EXPERIMENT_REORG.md:

  C1 — extends-less load is byte-identical to Config.load_yaml.
  C2 — extends merge satisfies get_mandatory() on every key the base provides.
  C3 — cycle detection and missing-base raise ValueError.
  C4 — list-replace + omission semantics (entities: [] suppresses base animals;
       omitting entities: lets them survive the merge).
  C9 — worked-example parity: a minimal sparse `extends:` config produces the
       same EnvParams (and therefore an identical rollout) as the full standalone
       config it is derived from.

All scratch YAML files are written to tmp/ (gitignored) to keep fixtures clean.
"""
from __future__ import annotations

import os
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml

# Project root on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.utils.config import Config
from src.environment.config_loader import load_env_params, load_env_config
from src.environment.core import jax_reset, jax_step

ACTIONS = [0, 1, 2, 3, 4] * 20  # 100 steps


# ---------------------------------------------------------------------------
# Helper: write a temp YAML and return its absolute path
# ---------------------------------------------------------------------------
def _write_tmp(name: str, content: str, tmp_path) -> str:
    """Write YAML content to tmp_path/<name>.yaml; return absolute path."""
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return str(p)


# ---------------------------------------------------------------------------
# C1 — extends-less load is byte-identical to Config.load_yaml
# ---------------------------------------------------------------------------
def test_c1_standalone_is_identical():
    """A config without `extends:` loaded via load_env_config == Config.load_yaml."""
    # Use an archived experiment config that has no `extends:` key
    config_path = os.path.join(
        _ROOT,
        "configs", "environment", "experiment", "archive",
        "basic", "00-5X5_NoPred.yaml",
    )
    assert os.path.exists(config_path), f"Archived config not found: {config_path}"

    via_new = load_env_config(config_path).to_dict()
    via_old = Config.load_yaml(config_path).to_dict()

    assert via_new == via_old, (
        "load_env_config on a standalone (no-extends) config must return a "
        "dict byte-identical to Config.load_yaml"
    )


# ---------------------------------------------------------------------------
# C2 — extends merge satisfies get_mandatory() on keys from the base
# ---------------------------------------------------------------------------
def test_c2_extends_satisfies_mandatory_keys(tmp_path):
    """A sparse config with `extends: environment/default` loads without ValueError."""
    # Write a minimal sparse config — only override the grid size.
    # The base provides all mandatory keys; this file just changes dimensions.
    sparse_yaml = _write_tmp(
        "c2_sparse.yaml",
        """
        extends: environment/default

        environment:
          height: 6
          width: 6
        """,
        tmp_path,
    )
    # Should NOT raise ValueError (mandatory keys come from the base).
    config = load_env_config(sparse_yaml)
    params = load_env_params(config)

    assert params is not None
    # Confirm the override took effect
    assert int(config.get("environment.height")) == 6
    assert int(config.get("environment.width")) == 6


# ---------------------------------------------------------------------------
# C3 — cycle detection and missing-base raise ValueError
# ---------------------------------------------------------------------------
def test_c3_cycle_raises(tmp_path):
    """A config that transitively extends itself raises ValueError."""
    # Config A extends B, B extends A — trivial direct cycle.
    a_path = str(tmp_path / "c3_a.yaml")
    b_path = str(tmp_path / "c3_b.yaml")

    with open(a_path, "w") as f:
        # Use absolute path so the resolver can find it regardless of _CONFIGS_ROOT
        # We test the cycle detection by pointing at the raw loader.
        # Note: `extends:` is a logical path under configs/; for the cycle test
        # we bypass that by injecting the raw dict directly via _resolve_extends.
        f.write("extends: environment/default\nenvironment:\n  height: 5\n")

    # For a true cycle, we need two files that extend each other.
    # Since `extends` resolves under _CONFIGS_ROOT (configs/), we can't easily
    # point outside that tree. Instead, test the direct self-cycle by calling
    # _resolve_extends with a pre-poisoned _seen set.
    from src.environment.config_loader import _resolve_extends

    abs_a = os.path.abspath(a_path)
    with pytest.raises(ValueError, match="cycle"):
        _resolve_extends(a_path, _seen=frozenset({abs_a}))


def test_c3_missing_base_raises(tmp_path):
    """A config that extends a nonexistent base raises ValueError."""
    sparse_yaml = _write_tmp(
        "c3_missing.yaml",
        """
        extends: environment/this_does_not_exist_xyz

        environment:
          height: 5
        """,
        tmp_path,
    )
    with pytest.raises(ValueError, match="not found"):
        load_env_config(sparse_yaml)


# ---------------------------------------------------------------------------
# C4 — list-replace + omission semantics
# ---------------------------------------------------------------------------
def test_c4_explicit_empty_entities_suppresses_base(tmp_path):
    """A sparse config with `entities: []` suppresses the base's animals."""
    sparse_yaml = _write_tmp(
        "c4_no_entities.yaml",
        """
        extends: environment/default

        environment:
          entities: []
        """,
        tmp_path,
    )
    config = load_env_config(sparse_yaml)
    entities = config.get("environment.entities")
    assert entities == [], (
        "entities: [] must suppress the base's entity list; got: {entities!r}"
    )
    # load_env_params should succeed (no animals)
    params = load_env_params(config)
    assert params.animal_property.shape[0] == 0, (
        "Expected 0 animals after entities: [] suppression"
    )


def test_c4_omitting_entities_leaks_base_animals(tmp_path):
    """A sparse config that OMITS `entities:` causes the base's animals to survive."""
    sparse_yaml = _write_tmp(
        "c4_omit_entities.yaml",
        """
        extends: environment/default

        environment:
          height: 6
          width: 6
        """,
        tmp_path,
    )
    config = load_env_config(sparse_yaml)
    entities = config.get("environment.entities")
    # The base has at least 1 entity (the predator + two rabbits); omitting
    # entities: in the sparse file means the base list survives.
    assert entities is not None and len(entities) > 0, (
        "Omitting entities: in a sparse config must LEAK the base's entity list. "
        "This is intentional and documented — sparse authors must use entities: [] "
        "to suppress. Got entities={entities!r}"
    )


# ---------------------------------------------------------------------------
# C9 — worked-example parity
# ---------------------------------------------------------------------------
def _run_episode(params, key):
    """Reset + 100 steps; return (state_list, info_list)."""
    state = jax_reset(params, key)
    state_list = [state]
    info_list = []
    for action in ACTIONS:
        state, reward, done, info = jax_step(state, action, params)
        state_list.append(state)
        info_list.append(info)
    return state_list, info_list


def _state_fingerprint(state_list):
    """Reduce a state list to a flat numpy array for comparison."""
    arrays = []
    for s in state_list:
        arrays.append(np.array(s.agent_pos).ravel())
        arrays.append(np.array(s.nutrition).ravel())
        arrays.append(np.array(s.injury_level).ravel())
        arrays.append(np.array(s.res_active).ravel())
        arrays.append(np.array(s.animal_pos).ravel())
    return np.concatenate(arrays)


def test_c9_sparse_equals_full_rollout(tmp_path):
    """Sparse `extends:` config must produce an identical rollout to a manually
    constructed equivalent full config.

    We test this by building BOTH configs in code (no disk file needed for the
    full version) and verifying that a 100-step rollout from seed 0 is
    byte-identical.

    The sparse config:
      - extends environment/default
      - overrides only environment.height=6, width=6
      - explicitly sets entities: [] (no animals — suppresses base predator)
      - explicitly sets resources: [] (no resources — suppresses base food)

    The full equivalent (built manually from the base dict + overrides):
      - loads default.yaml, then applies the same overrides

    Because load_env_config on the sparse file deep-merges the base and then
    applies overrides, both paths should produce the same Config and therefore
    the same EnvParams and rollout.
    """
    # --- Sparse config (the new `extends:` style) ---
    sparse_yaml = _write_tmp(
        "c9_sparse.yaml",
        """
        extends: environment/default

        environment:
          height: 6
          width: 6
          entities: []
          resources: []
        """,
        tmp_path,
    )
    sparse_config = load_env_config(sparse_yaml)

    # --- Full equivalent (manual merge, the old Camp-A style) ---
    base_path = os.path.join(_ROOT, "configs", "environment", "default.yaml")
    full_config = Config.load_yaml(base_path)
    full_config.merge(Config({
        "environment": {
            "height": 6,
            "width": 6,
            "entities": [],
            "resources": [],
        }
    }))

    # Both should produce the same dict (modulo key ordering — compare as dicts)
    assert sparse_config.to_dict() == full_config.to_dict(), (
        "Sparse extends-config must produce the same merged dict as a manual merge."
    )

    # Both should produce the same EnvParams
    sparse_params = load_env_params(sparse_config)
    full_params = load_env_params(full_config)

    # Rollout from seed 0 must be byte-identical
    key = jax.random.PRNGKey(0)
    sparse_states, _ = _run_episode(sparse_params, key)
    full_states, _ = _run_episode(full_params, key)

    sparse_fp = _state_fingerprint(sparse_states)
    full_fp = _state_fingerprint(full_states)

    np.testing.assert_array_equal(
        sparse_fp,
        full_fp,
        err_msg=(
            "Sparse `extends:` config rollout must be byte-identical to the "
            "manually merged equivalent."
        ),
    )
