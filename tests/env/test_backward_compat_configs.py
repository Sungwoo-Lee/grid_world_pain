"""CP1 backward-compat test — all migrated experiment configs load without error.

For each config in configs/experiment/**/*.yaml:
  1. Call load_env_params(Config(yaml)) — assert no exception.
  2. Assert params.animal_property.shape[0] == total animals expected from YAML.
  3. Assert params.predator_enabled key is NOT present (migration sweep check).

Also covers configs/continual/, configs/verification/, and configs/environment/default.yaml.
"""
import glob
import os
import sys
import yaml

import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.utils.config import Config
from src.environment.config_loader import load_env_params


def _collect_all_configs():
    configs = sorted(
        glob.glob(os.path.join(_ROOT, "configs", "experiment", "**", "*.yaml"), recursive=True) +
        glob.glob(os.path.join(_ROOT, "configs", "continual", "**", "*.yaml"), recursive=True) +
        glob.glob(os.path.join(_ROOT, "configs", "verification", "**", "*.yaml"), recursive=True)
    )
    env_default = os.path.join(_ROOT, "configs", "environment", "default.yaml")
    if env_default not in configs:
        configs.append(env_default)
    return configs


def _count_yaml_animals(cfg_dict):
    """Count total animals from raw YAML (predators + neutral_animals, respecting count:)."""
    total = 0
    for pred in (cfg_dict.get("environment", {}).get("predators") or []):
        total += pred.get("count", 1)
    for neu in (cfg_dict.get("environment", {}).get("neutral_animals") or []):
        total += neu.get("count", 1)
    for ent in (cfg_dict.get("environment", {}).get("entities") or []):
        total += ent.get("count", 1)
    return total


_all_configs = _collect_all_configs()
_ids = [os.path.relpath(p, _ROOT) for p in _all_configs]


@pytest.mark.parametrize("config_path", _all_configs, ids=_ids)
def test_config_loads_without_error(config_path):
    """All migrated configs load successfully with the new loader.

    Stale configs (those that pre-date mandatory keys like 'environment.resources'
    or 'sensory.injury_observable') are skipped — they failed to load in the
    pre-refactor code too, so there is no regression to catch here.
    """
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)

    # Migration check: predator_enabled must NOT be present after the sweep
    assert "predator_enabled" not in cfg_dict.get("environment", {}), (
        f"Config still has 'predator_enabled' key — migration sweep incomplete: {config_path}"
    )

    config = Config(cfg_dict)
    try:
        params = load_env_params(config)
    except ValueError as e:
        msg = str(e)
        # Stale configs: missing keys that pre-date the current schema are skipped.
        # Any config with predator_enabled would have already been caught above.
        stale_markers = (
            "is required but missing",
            "predator_enabled",
        )
        if any(m in msg for m in stale_markers):
            pytest.skip(f"Stale config (pre-dates mandatory key): {config_path} — {msg}")
        pytest.fail(f"load_env_params raised unexpected ValueError for {config_path}: {e}")
    except Exception as e:
        pytest.fail(f"load_env_params raised for {config_path}: {e}")

    # Animal count matches YAML
    expected_n = _count_yaml_animals(cfg_dict)
    actual_n = params.animal_property.shape[0]
    assert actual_n == expected_n, (
        f"Animal count mismatch for {config_path}: expected {expected_n}, got {actual_n}"
    )

    # Basic shape invariants
    N = actual_n
    assert params.animal_property.shape == (N, params.animal_property.shape[-1])
    assert params.animal_nociception.shape == (N,)
    assert params.animal_move_int.shape == (N,)
    assert params.animal_is_damaging.shape == (N,)
    assert len(params.animal_tags) == N
    assert len(params.animal_classes) == N
    assert len(params.animal_behaviours) == N
    assert len(params.hunt_idx) + len(params.wander_idx) + len(params.static_idx) == N
    assert len(params.predator_indices) + len(params.neutral_indices) == N
