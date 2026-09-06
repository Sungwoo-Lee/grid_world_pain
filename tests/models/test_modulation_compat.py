"""Tests for the read-only legacy-config shim at the evaluation boundary.

Plain-language context: every training run freezes a copy of its own config into
`<run>/models/config.yaml`, and the evaluation and trajectory-collection tools
rebuild the model from that frozen copy — not from the maintained files under
`configs/`. Runs that trained before the modulation-site refactor wrote the old
flat key `modulation.temp_clip`, which the current model rejects outright. So
without a shim, re-analysing any archived neuromodulation run would fail at model
construction.

This shim translates that old key into the new
`temperature: {enabled, clip}` shape IN MEMORY when a SAVED config is loaded, and
fills in the site/mechanism settings that the pre-refactor architecture hard-wired
(observation encoder + task GRU via the update-gate bias, temperature on). It
writes nothing back to disk, and it deliberately does NOT soften the live-config
path: a config someone is editing under `configs/` that still carries `temp_clip`
still raises the migration error.

Plan: docs/develop/active/neuromodulation/MODULATION_SITE_REFACTOR.md
(the shim is a user-approved deviation from the plan's decision D16).
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
jax.config.update("jax_platform_name", "cpu")

import numpy as np
import pytest
from flax import nnx

from src.models.modulation_compat import translate_legacy_modulation_config
from src.models.recurrent_ppo_network import ActorCriticRNN

LEGACY = {
    "type": "FiLM",
    "mod_hidden_size": 8,
    "grouping_size": 1,
    "percept_bias_init": 1.0,
    "memory_bias_init": 0.0,
    "temp_clip": [0.5, 3.0],
    "memory_clip": [-2.0, 2.0],
}

MIGRATED = {
    "type": "FiLM",
    "mod_hidden_size": 8,
    "grouping_size": 1,
    "percept_bias_init": 1.0,
    "memory_bias_init": 0.0,
    "memory_clip": [-2.0, 2.0],
    "input_sensors": "all",
    "sites": {"encoder": True, "rnn": True, "actor": False, "critic": False},
    "rnn_mechanism": "gate_bias",
    "temperature": {"enabled": True, "clip": [0.5, 3.0]},
}

ENCODING_CONFIG = {"encoding_mode": "flat", "use_layer_norm": False}


def _build(modulation_config):
    return ActorCriticRNN(
        input_dim=8, action_dim=4, hidden_size=16, rngs=nnx.Rngs(0),
        rnn_type="GRU", activation="tanh",
        modulation_config=modulation_config, encoding_config=ENCODING_CONFIG,
    )


def test_live_config_with_temp_clip_still_hard_errors():
    """The shim must not soften the live path: handing the untranslated legacy
    dict straight to the model still raises the migration error."""
    with pytest.raises(ValueError, match="temp_clip"):
        _build(dict(LEGACY))


def test_translated_saved_config_builds_the_same_model():
    """A translated archived config must produce exactly the network the
    equivalent migrated config produces — same parameters, same forward pass."""
    translated = translate_legacy_modulation_config(dict(LEGACY), source="test")
    assert "temp_clip" not in translated
    assert translated["temperature"] == {"enabled": True, "clip": [0.5, 3.0]}
    assert translated["rnn_mechanism"] == "gate_bias"
    assert translated["input_sensors"] == "all"

    a = _build(translated)
    b = _build(dict(MIGRATED))
    sa = jax.tree_util.tree_leaves(nnx.state(a, nnx.Param))
    sb = jax.tree_util.tree_leaves(nnx.state(b, nnx.Param))
    assert len(sa) == len(sb) and len(sa) > 0
    for x, y in zip(sa, sb):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    obs = jax.random.normal(jax.random.PRNGKey(7), (8,))
    la, va, _, _ = a(obs, a.initial_state())
    lb, vb, _, _ = b(obs, b.initial_state())
    np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))
    np.testing.assert_array_equal(np.asarray(va), np.asarray(vb))


def test_input_is_not_mutated():
    src = dict(LEGACY)
    translate_legacy_modulation_config(src, source="test")
    assert "temp_clip" in src, "the shim must not mutate the caller's dict"


def test_already_migrated_config_passes_through_unchanged():
    out = translate_legacy_modulation_config(dict(MIGRATED), source="test")
    assert out == MIGRATED


def test_unmodulated_config_passes_through():
    assert translate_legacy_modulation_config(None, source="test") is None
    off = {"type": None}
    assert translate_legacy_modulation_config(off, source="test") is off


def test_saved_config_missing_a_mandatory_key_still_fails():
    """The shim is not a general defaulting layer: it fires only on the specific
    legacy shape. A saved config that merely omits a mandatory key still fails."""
    broken = dict(MIGRATED)
    del broken["rnn_mechanism"]
    out = translate_legacy_modulation_config(broken, source="test")
    with pytest.raises(ValueError, match="rnn_mechanism"):
        _build(out)
