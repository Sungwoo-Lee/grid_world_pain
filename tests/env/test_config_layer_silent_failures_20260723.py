"""Regression tests for the 4 P1 config-layer silent-failure bugs fixed in
docs/develop/active/issues/FIX_CONFIG_LAYER_SILENT_FAILURES_20260723.md.

Bug 1 — legacy-scene precedence: a legacy (`predators:`/`neutral_animals:`) config
        loaded under train.py (with the default.yaml `entities:` underlay) must see
        the SAME animal scene as the same file loaded bare by eval_rollout.py.
Bug 2 — an unrecognised `perceptual_noise.modalities.<k>.mode` string must raise
        ValueError, not silently map to "noise off".
Bug 3 — an unrecognised `perceptual_noise.modalities` key must raise ValueError,
        not silently vanish from the noise arrays.
Bug 4 — `configs/environment/experiment/basic/05-sensory_noise_10x10.yaml` must
        actually keep interoception (satiation, interoceptive_nociception,
        extero_nociception) noise-free (`sigma == 0.0`) as its header claims.

All four must FAIL on pre-fix code and PASS after the fix (recorded in the
Implementation Report). A fifth guard test confirms the new strictness does not
reject a currently-valid, currently-active config.
"""
from __future__ import annotations

import os
import sys

import pytest

# Project root on sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.utils.config import Config, get_default_config
from src.environment.config_loader import (
    load_env_config,
    load_env_params,
    _parse_noise_config,
    _load_animals,
)

# Index of `animal_tags` in the flat tuple returned by `_load_animals()` — a
# scene-identifying quantity (the per-slot tag labels, in order).
_ANIMAL_TAGS_IDX = 27

_ARCHIVE_2X2 = os.path.join(
    _ROOT, "configs", "environment", "experiment", "archive", "2X2_area.yaml"
)
_NOISE_05 = os.path.join(
    _ROOT, "configs", "environment", "experiment", "basic",
    "05-sensory_noise_10x10.yaml",
)


# ---------------------------------------------------------------------------
# Bug 1 — train-path and eval-path must agree on a legacy config's scene
# ---------------------------------------------------------------------------
def test_bug1_train_eval_scene_agree():
    """Reproducer from findings_config_loader.md Finding 1.

    Train-path load: get_default_config() (carries default.yaml's `entities:`)
    then merge in the legacy 2X2_area.yaml (which has 5 legacy rabbits + 2
    legacy predators, no `entities:` key of its own).

    Eval-path load: bare load_env_config(2X2_area.yaml) — no base underlay.

    Pre-fix: train-path sees the BASE entities scene (2 predators + 2 rabbits
    slots, `has_entities=True` wins) while eval-path sees the true legacy scene
    (2 predators + 5 rabbits slots) — they DISAGREE.
    Post-fix: both see the legacy scene — they AGREE.
    """
    assert os.path.exists(_ARCHIVE_2X2), f"Fixture config not found: {_ARCHIVE_2X2}"

    # Train-path: base default.yaml underlay + legacy user config merged on top.
    # Uses `_load_animals()` directly (not the full `load_env_params()`) because
    # this archived pre-v2.0 config lacks several unrelated mandatory keys
    # (e.g. `sensory.injury_observable`) that only the base underlay supplies —
    # exercising the animal-scene precedence logic in isolation, matching the
    # plan's Finding-1 reproducer.
    train_cfg = get_default_config()
    train_cfg.merge(load_env_config(_ARCHIVE_2X2))
    train_animals = _load_animals(train_cfg)
    train_slots = int(train_animals[0].shape[0])
    train_tags = train_animals[_ANIMAL_TAGS_IDX]

    # Eval-path: bare load, no underlay (mirrors eval_rollout.py:849).
    eval_cfg = load_env_config(_ARCHIVE_2X2)
    eval_animals = _load_animals(eval_cfg)
    eval_slots = int(eval_animals[0].shape[0])
    eval_tags = eval_animals[_ANIMAL_TAGS_IDX]

    assert train_slots == eval_slots and train_tags == eval_tags, (
        f"Train-path scene ({train_slots} slots, tags={train_tags}) disagrees "
        f"with eval-path scene ({eval_slots} slots, tags={eval_tags}) for a "
        f"legacy config — the base default.yaml `entities:` underlay is "
        f"silently overriding the user's legacy predators:/neutral_animals: "
        f"sections."
    )
    # The true legacy scene is 2 predators + 5 rabbits = 7 slots (count fields
    # are all scalar `count` on this archived config, i.e. count_low==count_high).
    assert eval_slots == 7, (
        f"Sanity check: eval-path (always legacy) should see 7 animal slots "
        f"(2 predators + 5 rabbits); got {eval_slots}."
    )
    assert train_slots == 7, (
        f"Train-path must also see the legacy scene (7 animal slots) post-fix; "
        f"got {train_slots} (base entities scene leaking through)."
    )


# ---------------------------------------------------------------------------
# Bug 2 — unknown noise mode string must raise ValueError
# ---------------------------------------------------------------------------
def test_bug2_unknown_noise_mode_raises():
    """A typo'd mode string ('state-dependent' hyphen instead of
    'state_dependent') must raise ValueError, not silently disable that
    channel's noise (pre-fix: maps to mode int 0 = off, no error)."""
    cfg = get_default_config()
    cfg.merge(Config({
        "perceptual_noise": {
            "enabled": True,
            "modalities": {
                "injury": {
                    "mode": "state-dependent",  # typo: hyphen, not underscore
                },
            },
        },
    }))
    with pytest.raises(ValueError):
        _parse_noise_config(cfg)


# ---------------------------------------------------------------------------
# Bug 3 — unknown modality key must raise ValueError
# ---------------------------------------------------------------------------
def test_bug3_unknown_modality_key_raises():
    """A typo'd modality key ('olfactory' instead of the valid 'olfaction')
    must raise ValueError, not silently vanish from the noise arrays
    (pre-fix: the whole entry is dropped, no error)."""
    cfg = get_default_config()
    cfg.merge(Config({
        "perceptual_noise": {
            "enabled": True,
            "modalities": {
                "olfactory": {  # typo: valid key is "olfaction"
                    "mode": "constant",
                    "sigma": 0.1,
                },
            },
        },
    }))
    with pytest.raises(ValueError) as excinfo:
        _parse_noise_config(cfg)
    # Message should name the valid keys so an author can self-correct.
    assert "olfaction" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Bug 4 — 05-sensory_noise_10x10.yaml interoception must actually be clean
# ---------------------------------------------------------------------------
def test_bug4_interoception_clean_in_05_config():
    """The header of 05-sensory_noise_10x10.yaml claims interoception (satiation,
    interoceptive_nociception, extero_nociception) is kept CLEAN. Pre-fix, only
    `mode`/`injury_noise_scale` are overridden and `sigma` deep-merges from
    default.yaml (0.1) — a silent mismatch with the documented intent."""
    assert os.path.exists(_NOISE_05), f"Fixture config not found: {_NOISE_05}"

    params = load_env_params(load_env_config(_NOISE_05))
    order = list(params.noise_modality_order)

    for sensor_name in ("Satiation", "Interoceptive Nociception", "Extero Nociception"):
        assert sensor_name in order, f"{sensor_name!r} missing from noise_modality_order"
        idx = order.index(sensor_name)
        sigma = float(params.noise_sigmas[idx])
        assert sigma == 0.0, (
            f"{sensor_name} sigma should be 0.0 (interoception kept CLEAN per "
            f"the config header); got {sigma}."
        )


# ---------------------------------------------------------------------------
# Guard — the new strictness must not reject a currently-valid, active config
# ---------------------------------------------------------------------------
def test_valid_noise_configs_still_load():
    """05-sensory_noise_10x10.yaml exercises state_dependent/constant modes and
    every valid modality key through the new validators; it must still load
    without error both before and after the fix."""
    params = load_env_params(load_env_config(_NOISE_05))
    assert params is not None
