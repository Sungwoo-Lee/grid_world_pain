"""CP1 parity tests — unified animal entity refactor.

For each config that has a pre-refactor fixture in tests/env/fixtures/parity/,
run 100 steps from seed 0 and assert that:
  1. State fields match (animal_pos sliced by class vs old pred_pos / neutral_pos).
  2. Info-dict legacy keys match (hit_predator, hit_neutral, dist_per_predator,
     dist_per_neutral, damage_*, agent_in_bush).

Configs without a fixture are skipped (those were stale before the refactor —
they couldn't load in the old code either, so there is no reference to compare
against).
"""
import glob
import os
import re
import sys
import traceback

import numpy as np
import pytest

# Ensure project root on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp
import yaml

from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step

FIXTURE_DIR = os.path.join(_ROOT, "tests", "env", "fixtures", "parity")
ACTIONS = [0, 1, 2, 3, 4] * 20  # 100 steps


def _config_slug(config_path: str) -> str:
    """Convert config path to filesystem-safe slug (matches generate_parity_fixtures.py)."""
    rel = os.path.relpath(config_path, _ROOT)
    slug = re.sub(r'[/\\]', '__', rel)
    slug = re.sub(r'\.yaml$', '', slug)
    slug = re.sub(r'[^A-Za-z0-9_.-]', '_', slug)
    return slug


def _collect_configs():
    configs = sorted(
        glob.glob(os.path.join(_ROOT, "configs", "environment", "experiment", "**", "*.yaml"), recursive=True) +
        glob.glob(os.path.join(_ROOT, "configs", "continual", "**", "*.yaml"), recursive=True) +
        glob.glob(os.path.join(_ROOT, "configs", "verification", "**", "*.yaml"), recursive=True)
    )
    env_default = os.path.join(_ROOT, "configs", "environment", "default.yaml")
    if env_default not in configs:
        configs.append(env_default)
    return configs


def _load_fixture(config_path: str):
    """Load fixture npz, or return None if not present."""
    slug = _config_slug(config_path)
    fixture_path = os.path.join(FIXTURE_DIR, slug + ".npz")
    if not os.path.exists(fixture_path):
        return None
    return np.load(fixture_path)


def _run_episode(params, key):
    """Reset + 100 steps. Returns (state_list, info_list)."""
    state = jax_reset(params, key)
    state_list = [state]
    info_list = []
    for action in ACTIONS:
        state, reward, done, info = jax_step(state, action, params)
        state_list.append(state)
        info_list.append(info)
    return state_list, info_list


def _fixture_subset(fixture, step_i, field, legacy_key, idx, slug):
    """Fetch a per-animal-class fixture array (pred/neutral position or property_sampled).

    Two fixture schemas coexist in tests/env/fixtures/parity/: older fixtures still
    carry the legacy split keys (e.g. step000_pred_pos); fixtures regenerated after
    the unified-animal refactor only carry the unified step000_animal_{field} array,
    which must be sliced by predator_indices / neutral_indices (idx) to recover the
    same subset. Prefer the legacy key when present (untouched fixture, exact byte
    match); otherwise slice the unified array. If NEITHER is present, fail loudly —
    that is a real data gap, not something to silently skip.
    """
    legacy = fixture.get(f"step{step_i:03d}_{legacy_key}")
    if legacy is not None:
        return legacy
    unified = fixture.get(f"step{step_i:03d}_animal_{field}")
    if unified is not None:
        return unified[np.array(idx)]
    raise AssertionError(
        f"Fixture for {slug} has neither legacy '{legacy_key}' nor unified "
        f"'animal_{field}' at step {step_i} — cannot verify parity."
    )


# ── Collect test cases ────────────────────────────────────────────────────────

_all_configs = _collect_configs()
_test_params = []

for _cfg_path in _all_configs:
    _slug = _config_slug(_cfg_path)
    _fixture_path = os.path.join(FIXTURE_DIR, _slug + ".npz")
    _has_fixture = os.path.exists(_fixture_path)
    _test_params.append((_cfg_path, _slug, _has_fixture))


@pytest.mark.parametrize("config_path,slug,has_fixture", _test_params, ids=[p[1] for p in _test_params])
def test_parity(config_path, slug, has_fixture):
    """Parity gate: 100 steps from seed 0 match pre-refactor fixture."""
    if not has_fixture:
        pytest.skip(f"No pre-refactor fixture for {slug} (config was stale before refactor)")

    # Load new-code params
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    try:
        config = Config(cfg_dict)
        params = load_env_params(config)
    except Exception as e:
        pytest.fail(f"Config load failed ({config_path}): {e}")

    # Load fixture
    fixture = np.load(os.path.join(FIXTURE_DIR, slug + ".npz"))
    num_pred = int(fixture["num_pred"])
    num_neutral = int(fixture["num_neutral"])

    # Run episode
    key = jax.random.PRNGKey(0)
    try:
        state_list, info_list = _run_episode(params, key)
    except Exception as e:
        pytest.fail(f"Episode failed ({config_path}): {traceback.format_exc()}")

    # ── State field parity ────────────────────────────────────────────────────
    # N1 fix check: animal_pos[predator_indices] should match old pred_pos at reset
    # N2 fix check: animal_property_sampled[predator_indices] matches old pred_property_sampled

    pred_idx = jnp.array(list(params.predator_indices), dtype=jnp.int32) if params.predator_indices else None
    neutral_idx = jnp.array(list(params.neutral_indices), dtype=jnp.int32) if params.neutral_indices else None

    for step_i, state in enumerate(state_list):
        # Agent position
        old_agent_pos = fixture[f"step{step_i:03d}_agent_pos"]
        np.testing.assert_array_equal(
            np.array(state.agent_pos), old_agent_pos,
            err_msg=f"agent_pos mismatch at step {step_i} for {slug}"
        )

        # Body fields
        for field in ("satiation", "nutrition", "injury_level", "terminated"):
            old_val = fixture.get(f"step{step_i:03d}_{field}")
            if old_val is not None:
                new_val = np.array(getattr(state, field))
                np.testing.assert_allclose(
                    new_val, old_val, rtol=1e-5, atol=1e-5,
                    err_msg=f"{field} mismatch at step {step_i} for {slug}"
                )

        # N1: predator placement at reset (step 0) — animal_pos[pred_idx] vs fixture
        if step_i == 0 and num_pred > 0 and pred_idx is not None:
            old_pred_pos = _fixture_subset(fixture, step_i, "pos", "pred_pos", pred_idx, slug)
            new_pred_pos = np.array(state.animal_pos[pred_idx])
            np.testing.assert_array_equal(
                new_pred_pos, old_pred_pos,
                err_msg=f"N1: animal_pos[predator_indices] != fixture pred_pos at reset for {slug}"
            )

        # N1: neutral placement at reset
        if step_i == 0 and num_neutral > 0 and neutral_idx is not None:
            old_neutral_pos = _fixture_subset(fixture, step_i, "pos", "neutral_pos", neutral_idx, slug)
            new_neutral_pos = np.array(state.animal_pos[neutral_idx])
            np.testing.assert_array_equal(
                new_neutral_pos, old_neutral_pos,
                err_msg=f"N1: animal_pos[neutral_indices] != fixture neutral_pos at reset for {slug}"
            )

        # N2: property sampling at reset
        if step_i == 0 and num_pred > 0 and pred_idx is not None:
            old_pred_prop = _fixture_subset(
                fixture, step_i, "property_sampled", "pred_property_sampled", pred_idx, slug
            )
            new_pred_prop = np.array(state.animal_property_sampled[pred_idx])
            np.testing.assert_allclose(
                new_pred_prop, old_pred_prop, rtol=1e-5, atol=1e-5,
                err_msg=f"N2: animal_property_sampled[predator_indices] != fixture pred_property_sampled at reset for {slug}"
            )

        if step_i == 0 and num_neutral > 0 and neutral_idx is not None:
            old_neutral_prop = _fixture_subset(
                fixture, step_i, "property_sampled", "neutral_property_sampled", neutral_idx, slug
            )
            new_neutral_prop = np.array(state.animal_property_sampled[neutral_idx])
            np.testing.assert_allclose(
                new_neutral_prop, old_neutral_prop, rtol=1e-5, atol=1e-5,
                err_msg=f"N2: animal_property_sampled[neutral_indices] != fixture neutral_property_sampled at reset for {slug}"
            )

        # B1: per-subset PRNG — pred_pos at each step
        if num_pred > 0 and pred_idx is not None:
            old_pred_pos = _fixture_subset(fixture, step_i, "pos", "pred_pos", pred_idx, slug)
            new_pred_pos = np.array(state.animal_pos[pred_idx])
            np.testing.assert_array_equal(
                new_pred_pos, old_pred_pos,
                err_msg=f"B1: animal_pos[predator_indices] != fixture pred_pos at step {step_i} for {slug}"
            )

        # B1: wander PRNG — neutral_pos at each step
        if num_neutral > 0 and neutral_idx is not None:
            old_neutral_pos = _fixture_subset(fixture, step_i, "pos", "neutral_pos", neutral_idx, slug)
            new_neutral_pos = np.array(state.animal_pos[neutral_idx])
            np.testing.assert_array_equal(
                new_neutral_pos, old_neutral_pos,
                err_msg=f"B1: animal_pos[neutral_indices] != fixture neutral_pos at step {step_i} for {slug}"
            )

    # ── Info dict parity (B5 + legacy aliases) ───────────────────────────────
    for step_i, info in enumerate(info_list):
        for key_name in ("hit_predator", "hit_neutral", "damage_predator",
                         "damage_obstacle", "damage_hiding_predator", "agent_in_bush"):
            old_val = fixture.get(f"info{step_i:03d}_{key_name}")
            if old_val is None:
                continue
            assert key_name in info, (
                f"Legacy info key '{key_name}' missing at step {step_i} for {slug}"
            )
            new_val = np.array(info[key_name])
            np.testing.assert_allclose(
                new_val, old_val, rtol=1e-5, atol=1e-5,
                err_msg=f"info['{key_name}'] mismatch at step {step_i} for {slug}"
            )

        # Legacy distance aliases
        for dist_key in ("dist_per_predator", "dist_per_neutral"):
            old_val = fixture.get(f"info{step_i:03d}_{dist_key}")
            if old_val is None:
                continue
            assert dist_key in info, (
                f"Legacy info key '{dist_key}' missing at step {step_i} for {slug}"
            )
            new_val = np.array(info[dist_key])
            np.testing.assert_allclose(
                new_val, old_val, rtol=1e-5, atol=1e-5,
                err_msg=f"info['{dist_key}'] mismatch at step {step_i} for {slug}"
            )
