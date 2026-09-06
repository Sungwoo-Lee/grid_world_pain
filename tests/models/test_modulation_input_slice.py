"""Tests for the configurable modulator INPUT slice (Part B of the site refactor).

Plain-language context: the agent carries a small second network — the
"neuromodulator" — that continuously re-tunes the main policy network. Until now
it read *every* sense the agent has. Part B lets a config restrict what it reads
to a named subset, so an experiment can ask what the modulator needs to *read*
as well as where it writes:

    agent:
      modulation:
        input_sensors: "all"                                  # every sense (the old behaviour)
        # input_sensors: ["Satiation", "Interoceptive Nociception"]   # body signals only

The names are resolved to *column positions* in the observation vector against
the run's own `get_observation_breakdown(params)` — which sensors exist, and how
wide each one is, depends on the environment config. That is what makes the
dangerous failure mode possible and is what most of this file defends against:
if a named sensor is switched OFF in the environment config, every later
sensor's columns shift, so a config that silently accepted the missing name
would feed the modulator the WRONG columns with nothing in the logs to show for
it. Naming an absent sensor must therefore be a loud error.

What these tests pin:
  * `"all"` is byte-identical to the pre-refactor network (the golden fixtures).
  * A named subset selects EXACTLY the right columns — verified column by column
    against a hand-computed breakdown, not by re-running the same resolver.
  * An unknown / gated-off sensor name raises, and the message lists what IS
    available.
  * The resolved width really reaches the modulator's GRU input dimension.

Plan: docs/develop/active/neuromodulation/MODULATION_SITE_REFACTOR.md (Part B)
"""
import dataclasses
import importlib.util
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from src.models.recurrent_ppo_network import (
    ActorCriticRNN,
    _resolve_modulator_input_indices,
)

_GEN_PATH = os.path.join(_REPO, "tests", "fixtures", "modulation", "generate_golden.py")
_spec = importlib.util.spec_from_file_location("_modulation_golden_slice", _GEN_PATH)
golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(golden)

FIXTURE_DIR = os.path.dirname(_GEN_PATH)
MODES = ["flat", "hier_ln"]

# The observation layout of the environment the input/site grid actually runs on
# (configs/environment/experiment/basic/04-jump_attack_10x10.yaml). Written out by
# hand here on purpose: a test that recomputed it from the same code it is
# checking would prove nothing. Asserted against the live breakdown below.
BASIC04_BREAKDOWN = {
    "Satiation": 1,
    "Interoceptive Nociception": 1,
    "Extero Nociception": 1,
    "Olfaction": 5,
    "Collision": 5,
    "Proprioception": 6,
    "Visual": 8,
}
BASIC04_DIM = 27

# Hand-computed column positions, counted off the table above by hand:
#   Satiation                0
#   Interoceptive Nocic.     1
#   Extero Nociception       2
#   Olfaction                3  4  5  6  7
#   Collision                8  9 10 11 12
#   Proprioception          13 14 15 16 17 18
#   Visual                  19 20 21 22 23 24 25 26
HAND_COMPUTED = {
    "interoceptive": (["Satiation", "Interoceptive Nociception"], (0, 1)),
    "exteroceptive": (["Extero Nociception", "Olfaction", "Collision", "Visual"],
                      (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                       19, 20, 21, 22, 23, 24, 25, 26)),
    "nociceptive": (["Interoceptive Nociception", "Extero Nociception"], (1, 2)),
    "single_last": (["Visual"], (19, 20, 21, 22, 23, 24, 25, 26)),
    "out_of_order": (["Visual", "Satiation"], (19, 20, 21, 22, 23, 24, 25, 26, 0)),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _mod_cfg(input_sensors="all", **over):
    """A minimal FiLM modulation dict; only `input_sensors` varies here."""
    cfg = {
        "type": "FiLM",
        "mod_hidden_size": 8,
        "grouping_size": 1,
        "percept_bias_init": 1.0,
        "percept_add_bias_init": 0.0,
        "memory_bias_init": 0.0,
        "memory_clip": [-2.0, 2.0],
        "input_sensors": input_sensors,
        "sites": {"encoder": True, "rnn": True, "actor": False, "critic": False},
        "rnn_mechanism": "gate_bias",
        "temperature": {"enabled": True, "clip": [0.5, 3.0]},
    }
    cfg.update(over)
    return cfg


def _build_basic04(input_sensors, seed=0):
    """A model whose observation layout is the real basic/04 breakdown."""
    return ActorCriticRNN(
        input_dim=BASIC04_DIM,
        action_dim=6,
        hidden_size=16,
        rngs=nnx.Rngs(seed),
        rnn_type="GRU",
        activation="relu",
        modulation_config=_mod_cfg(input_sensors),
        observation_breakdown=dict(BASIC04_BREAKDOWN),
        encoding_config={"encoding_mode": "flat", "use_layer_norm": False},
    )


def _columns_the_modulator_actually_reads(model, obs):
    """Empirically recover the set of observation columns the modulator reads.

    Perturb one column at a time and watch the modulator's own hidden state. A
    column the modulator does not receive cannot change it. This deliberately
    does NOT consult `model.mod_input_idx` — checking the resolver's output
    against the resolver would be circular; this measures the forward pass.
    """
    _, _, (_, mod_h_ref), _ = model(obs, model.initial_state())
    read = []
    for j in range(obs.shape[-1]):
        bumped = obs.at[j].add(3.0)
        _, _, (_, mod_h), _ = model(bumped, model.initial_state())
        if not np.array_equal(np.asarray(mod_h), np.asarray(mod_h_ref)):
            read.append(j)
    return tuple(read)


# --------------------------------------------------------------------------
# 1. "all" is byte-identical to the pre-Part-B network
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_all_param_tree_matches_pre_refactor_golden(mode):
    """`input_sensors: "all"` must leave the parameter tree exactly as it was
    before the refactor — same key set, same shapes, same values from the same
    seed. The fixture was captured on the pre-refactor code."""
    ref = np.load(os.path.join(FIXTURE_DIR, f"{mode}_legacy.npz"))
    data = golden.snapshot(mode, modulated=True)   # spells input_sensors: "all"
    keys = [k for k in ref.files if k.startswith("param/")]
    assert set(keys) == {k for k in data if k.startswith("param/")}
    for k in keys:
        assert np.array_equal(ref[k], data[k]), f"{k} is not bit-identical"


@pytest.mark.parametrize("mode", MODES)
def test_all_forward_matches_pre_refactor_golden(mode):
    """Same, for the forward pass: action scores, value, both hidden states and
    every signal the modulator emits."""
    ref = np.load(os.path.join(FIXTURE_DIR, f"{mode}_legacy.npz"))
    data = golden.snapshot(mode, modulated=True)
    keys = [k for k in ref.files if k.startswith("out/")]
    assert set(keys) == {k for k in data if k.startswith("out/")}
    for k in keys:
        assert np.array_equal(ref[k], data[k]), f"{k} is not bit-identical"


def test_all_reads_every_column():
    """"all" means all: the modulator's input index tuple is the full range and
    the forward pass really does depend on every observation column."""
    m = _build_basic04("all")
    assert m.mod_input_idx == tuple(range(BASIC04_DIM))
    obs = jax.random.normal(jax.random.PRNGKey(3), (BASIC04_DIM,))
    assert _columns_the_modulator_actually_reads(m, obs) == tuple(range(BASIC04_DIM))


def test_listing_every_sensor_equals_all():
    """Naming every sensor in breakdown order must produce the same network as
    "all" — same modulator input width, same parameters, same forward pass."""
    a = _build_basic04("all")
    b = _build_basic04(list(BASIC04_BREAKDOWN.keys()))
    assert a.mod_input_idx == b.mod_input_idx
    obs = jax.random.normal(jax.random.PRNGKey(11), (BASIC04_DIM,))
    la, va, _, _ = a(obs, a.initial_state())
    lb, vb, _, _ = b(obs, b.initial_state())
    np.testing.assert_array_equal(np.asarray(la), np.asarray(lb))
    np.testing.assert_array_equal(np.asarray(va), np.asarray(vb))


# --------------------------------------------------------------------------
# 2. A named subset selects EXACTLY the right columns
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(HAND_COMPUTED))
def test_resolver_matches_hand_computed_indices(case):
    names, expected = HAND_COMPUTED[case]
    got = _resolve_modulator_input_indices(names, dict(BASIC04_BREAKDOWN), BASIC04_DIM)
    assert got == expected


@pytest.mark.parametrize("case", sorted(HAND_COMPUTED))
def test_forward_pass_reads_exactly_the_hand_computed_columns(case):
    """The strong version of the check: run the model and recover, column by
    column, which parts of the observation actually reach the modulator. An
    off-by-one index build fails here even though it would still 'work'."""
    names, expected = HAND_COMPUTED[case]
    m = _build_basic04(list(names))
    obs = jax.random.normal(jax.random.PRNGKey(5), (BASIC04_DIM,))
    assert _columns_the_modulator_actually_reads(m, obs) == tuple(sorted(set(expected)))


def test_unselected_columns_still_reach_the_task_network():
    """Slicing the modulator's input must not slice the policy's. The task
    network keeps seeing the whole observation."""
    m = _build_basic04(["Satiation", "Interoceptive Nociception"])
    obs = jax.random.normal(jax.random.PRNGKey(9), (BASIC04_DIM,))
    logits_ref, _, _, _ = m(obs, m.initial_state())
    bumped = obs.at[BASIC04_DIM - 1].add(3.0)          # a Visual column: modulator-blind
    logits_bumped, _, (_, mod_h), _ = m(bumped, m.initial_state())
    _, _, (_, mod_h_ref), _ = m(obs, m.initial_state())
    np.testing.assert_array_equal(np.asarray(mod_h), np.asarray(mod_h_ref))
    assert not np.array_equal(np.asarray(logits_ref), np.asarray(logits_bumped))


# --------------------------------------------------------------------------
# 3. The resolved width reaches the modulator's GRU
# --------------------------------------------------------------------------

@pytest.mark.parametrize("names,width", [
    ("all", BASIC04_DIM),
    (["Satiation", "Interoceptive Nociception"], 2),
    (["Extero Nociception", "Olfaction", "Collision", "Visual"], 19),
    (["Visual"], 8),
])
def test_modulator_gru_input_dimension_follows_the_slice(names, width):
    """The slice is not cosmetic: the modulator's recurrent core is genuinely
    narrower, which is also why a real slice changes the checkpoint shape."""
    m = _build_basic04(names)
    assert len(m.mod_input_idx) == width
    kernel = np.asarray(m.modulator.gru.dense_i.kernel.value)   # (in_features, 3*hidden)
    assert kernel.shape[0] == width


def test_mod_input_idx_is_static_python_ints():
    """Stored as a plain tuple of Python ints so it stays static graph metadata
    and never becomes a trained or checkpointed leaf."""
    m = _build_basic04(["Satiation", "Interoceptive Nociception"])
    assert isinstance(m.mod_input_idx, tuple)
    assert all(type(i) is int for i in m.mod_input_idx)

    # The index tuple must not appear anywhere in the model's variable state —
    # if it did, it would be checkpointed (and, worse, handed to the optimiser).
    for leaf in jax.tree_util.tree_leaves(nnx.state(m)):
        arr = np.asarray(leaf)
        if arr.dtype.kind in "iu" and arr.size == len(m.mod_input_idx):
            assert tuple(arr.ravel().tolist()) != m.mod_input_idx

    # ...and the module still splits and jits, i.e. the tuple is hashable static
    # metadata rather than an unhashable array smuggled into the graphdef.
    @nnx.jit
    def _step(model, obs, h):
        return model(obs, h)

    obs = jax.random.normal(jax.random.PRNGKey(1), (BASIC04_DIM,))
    _step(m, obs, m.initial_state())


# --------------------------------------------------------------------------
# 4. Bad names fail loudly, and the message says what IS available
# --------------------------------------------------------------------------

def test_unknown_sensor_name_raises_and_lists_available_sensors():
    with pytest.raises(ValueError) as e:
        _build_basic04(["Satiation", "Nociception"])
    msg = str(e.value)
    assert "'Nociception'" in msg
    for name in BASIC04_BREAKDOWN:
        assert name in msg, f"the error must list the available sensor {name!r}"


def test_unknown_string_shorthand_raises():
    """Only the literal string "all" is a legal string; presets were rejected by
    the plan (they would hide an experimental grouping decision inside src/)."""
    with pytest.raises(ValueError, match="input_sensors"):
        _build_basic04("interoceptive")


def test_empty_list_raises():
    with pytest.raises(ValueError, match="NON-EMPTY"):
        _build_basic04([])


def test_missing_input_sensors_key_raises():
    """No-fallback-defaults contract: `input_sensors` is mandatory whenever
    modulation is on. It must not default to "all" behind the reader's back."""
    cfg = _mod_cfg()
    del cfg["input_sensors"]
    with pytest.raises(ValueError, match="input_sensors"):
        ActorCriticRNN(
            input_dim=BASIC04_DIM, action_dim=6, hidden_size=16, rngs=nnx.Rngs(0),
            rnn_type="GRU", activation="relu", modulation_config=cfg,
            observation_breakdown=dict(BASIC04_BREAKDOWN),
            encoding_config={"encoding_mode": "flat", "use_layer_norm": False},
        )


def test_null_input_sensors_raises():
    with pytest.raises(ValueError, match="input_sensors"):
        _build_basic04(None)


# --------------------------------------------------------------------------
# 5. THE important one — a sensor gated off in the ENVIRONMENT config
# --------------------------------------------------------------------------
# Silent index drift is the failure mode that would corrupt an experiment with
# nothing in the logs to show for it: switch a sensor off in the environment
# config and every later sensor's columns move. A config naming the now-absent
# sensor must fail loudly rather than quietly resolving to shifted columns.

def _live_breakdown(**overrides):
    """The breakdown the ENVIRONMENT actually produces for basic/04, optionally
    with a sensor switched off."""
    from src.utils.config import get_default_config
    from src.environment.config_loader import load_env_params, load_env_config
    from src.environment.sensor import get_observation_breakdown

    cfg = get_default_config()
    cfg.merge(load_env_config(os.path.join(
        _REPO, "configs/environment/experiment/basic/04-jump_attack_10x10.yaml")))
    params = load_env_params(cfg)
    if overrides:
        params = dataclasses.replace(params, **overrides)
    return get_observation_breakdown(params)


def test_hand_computed_breakdown_matches_the_live_environment():
    """Guards the rest of this file: if basic/04's observation layout ever
    changes, this fails first and names the drift."""
    assert _live_breakdown() == BASIC04_BREAKDOWN


def test_gated_off_sensor_makes_a_config_naming_it_fail_loudly():
    """With interoceptive nociception switched off in the environment config, a
    modulation config still asking for it must raise — NOT silently resolve to
    whatever now sits at those columns."""
    bd = _live_breakdown(interoceptive_nociception_enabled=False)
    assert "Interoceptive Nociception" not in bd
    dim = sum(bd.values())
    with pytest.raises(ValueError) as e:
        _resolve_modulator_input_indices(
            ["Satiation", "Interoceptive Nociception"], bd, dim)
    msg = str(e.value)
    assert "'Interoceptive Nociception'" in msg
    assert "Satiation" in msg and "Extero Nociception" in msg


def test_gated_off_sensor_shifts_every_later_column():
    """Shows WHY the loud failure matters: the same sensor names resolve to
    different columns under different environment configs, so an accepted stale
    name would have fed the modulator the wrong data."""
    full = _live_breakdown()
    gated = _live_breakdown(interoceptive_nociception_enabled=False)
    names = ["Extero Nociception"]
    idx_full = _resolve_modulator_input_indices(names, full, sum(full.values()))
    idx_gated = _resolve_modulator_input_indices(names, gated, sum(gated.values()))
    assert idx_full == (2,)
    assert idx_gated == (1,), "every sensor after the gated-off one moves down by its width"


def test_gated_off_sensor_fails_at_model_construction_too():
    """The same failure through the real construction path, not just the
    resolver — this is the boundary a training launch actually crosses."""
    bd = _live_breakdown(olfactory_enabled=False)
    assert "Olfaction" not in bd
    with pytest.raises(ValueError, match="Olfaction"):
        ActorCriticRNN(
            input_dim=sum(bd.values()), action_dim=6, hidden_size=16, rngs=nnx.Rngs(0),
            rnn_type="GRU", activation="relu",
            modulation_config=_mod_cfg(["Olfaction", "Visual"]),
            observation_breakdown=bd,
            encoding_config={"encoding_mode": "flat", "use_layer_norm": False},
        )
