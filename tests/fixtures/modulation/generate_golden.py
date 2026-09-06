"""Golden-fixture generator for the modulation-site refactor (Checkpoint C0a).

Plain-language purpose: this script freezes what the network *did* before the
modulation-site refactor, so that afterwards a test can prove the refactor did
not change it. It writes four small `.npz` files — the unmodulated baseline and
the "legacy-equivalent" modulated network, in each of the two observation
encoding styles the project uses (a single flat block with no LayerNorm, and
the hierarchical per-sensor encoder with LayerNorm that every real
neuromodulation config actually runs).

Each file stores every parameter of the network (flattened, by dotted name)
plus the outputs of one forward pass from a fixed seed on a fixed observation:
the action scores, the value estimate, the new hidden state, and — for the
modulated pair — every signal the modulator emitted.

CRITICAL ORDERING RULE: run this on the UNMODIFIED tree, before editing
anything under `src/`. Regenerating it afterwards turns every parity test into
a tautology that passes no matter what broke.

Usage:
    python tests/fixtures/modulation/generate_golden.py            # write all four
    python tests/fixtures/modulation/generate_golden.py --check    # compare, do not write

Deliberately lives under `tests/` rather than `scripts/` so it does not trigger
the SCRIPTS_DEPENDENCY_MAP maintenance contract for a test-only helper.
"""
import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import jax
# Fixtures are captured and compared on CPU so the parity tests never become a
# CPU/GPU numerics comparison (that is a different question, with its own test).
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import numpy as np
from flax import nnx

from src.models.recurrent_ppo_network import ActorCriticRNN
from src.models.neuromodulator import ModulatorOutput

FIXTURE_DIR = os.path.dirname(os.path.abspath(__file__))

# True once the modulation-site refactor has landed. Used ONLY to spell the
# modulation dict in whichever shape the current code accepts, so the same
# generator runs on both sides of the refactor.
NEW_MODULATION_API = "z_actor" in ModulatorOutput._fields

SEED = 0

# --- The two encoding modes -------------------------------------------------
# "flat": minimal, fast, isolates the plain path.
# "hier": hierarchical per-sensor encoder + LayerNorm — the path every real
#         recurrent_ppo_nmn_* config executes.
FLAT_SPEC = dict(
    input_dim=8,
    hidden_size=16,
    action_dim=4,
    breakdown=None,
    encoding_config={"encoding_mode": "flat", "use_layer_norm": False},
)

_HIER_BREAKDOWN = {"Satiation": 1, "Injury": 1, "Vision": 5, "Olfaction": 3}
HIER_SPEC = dict(
    input_dim=sum(_HIER_BREAKDOWN.values()),
    hidden_size=16,
    action_dim=4,
    breakdown=dict(_HIER_BREAKDOWN),
    encoding_config={
        "encoding_mode": "hierarchical",
        "use_layer_norm": True,
        "hierarchical_params": {"default_mlp": [8]},
    },
)

SPECS = {"flat": FLAT_SPEC, "hier_ln": HIER_SPEC}

# Common modulation settings shared by both spellings of the dict.
_MOD_COMMON = {
    "type": "FiLM",
    "mod_hidden_size": 8,
    "grouping_size": 1,
    "percept_bias_init": 1.0,
    "percept_add_bias_init": 0.0,
    "memory_bias_init": 0.0,
    "memory_clip": [-2.0, 2.0],
}
_TEMP_CLIP = [0.5, 3.0]


def legacy_equivalent_modulation_config():
    """The modulation dict that must reproduce pre-refactor behaviour exactly.

    Pre-refactor this is the flat `temp_clip` spelling; post-refactor it is the
    encoder+rnn sites with the gate-bias mechanism and temperature enabled at
    the same ceiling — which is, by construction, the same network.
    """
    cfg = dict(_MOD_COMMON)
    if NEW_MODULATION_API:
        cfg["sites"] = {"encoder": True, "rnn": True, "actor": False, "critic": False}
        cfg["rnn_mechanism"] = "gate_bias"
        cfg["temperature"] = {"enabled": True, "clip": list(_TEMP_CLIP)}
    else:
        cfg["temp_clip"] = list(_TEMP_CLIP)
    return cfg


def build_model(spec, modulation_config):
    return ActorCriticRNN(
        input_dim=spec["input_dim"],
        action_dim=spec["action_dim"],
        hidden_size=spec["hidden_size"],
        rngs=nnx.Rngs(SEED),
        rnn_type="GRU",
        activation="tanh",
        modulation_config=modulation_config,
        observation_breakdown=spec["breakdown"],
        encoding_config=spec["encoding_config"],
    )


def fixed_observation(spec):
    """One fixed observation, deterministic for a given spec."""
    return jax.random.normal(jax.random.PRNGKey(12345), (spec["input_dim"],))


def _flat_params(model):
    state = nnx.state(model, nnx.Param)
    leaves, _ = jax.tree_util.tree_flatten_with_path(state)
    out = {}
    for path, leaf in leaves:
        key = "param/" + ".".join(str(getattr(p, "key", getattr(p, "idx", p))) for p in path)
        out[key] = np.asarray(leaf)
    return out


def _flat_outputs(model, spec):
    obs = fixed_observation(spec)
    h = model.initial_state()
    logits, value, h_new, mod_info = model(obs, h)

    out = {
        "out/logits": np.asarray(logits),
        "out/value": np.asarray(value),
    }
    task_h, mod_h = (h_new if model.modulation_enabled else (h_new, None))
    out["out/task_h"] = np.asarray(task_h)
    if mod_h is not None:
        out["out/mod_h"] = np.asarray(mod_h)
    if mod_info is not None:
        for field in mod_info._fields:
            v = getattr(mod_info, field)
            if v is not None:
                out[f"out/mod.{field}"] = np.asarray(v)
    return out


def snapshot(mode: str, modulated: bool) -> dict:
    spec = SPECS[mode]
    mod_cfg = legacy_equivalent_modulation_config() if modulated else None
    model = build_model(spec, mod_cfg)
    data = _flat_params(model)
    data.update(_flat_outputs(model, spec))
    return data


FIXTURES = [
    ("flat_baseline.npz", "flat", False),
    ("flat_legacy.npz", "flat", True),
    ("hier_ln_baseline.npz", "hier_ln", False),
    ("hier_ln_legacy.npz", "hier_ln", True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="compare against the stored fixtures instead of writing them")
    args = ap.parse_args()

    failures = 0
    for name, mode, modulated in FIXTURES:
        path = os.path.join(FIXTURE_DIR, name)
        data = snapshot(mode, modulated)
        if args.check:
            ref = np.load(path)
            missing = set(ref.files) ^ set(data.keys())
            if missing:
                print(f"[FAIL] {name}: key-set mismatch: {sorted(missing)}")
                failures += 1
                continue
            bad = [k for k in ref.files if not np.array_equal(ref[k], data[k])]
            if bad:
                print(f"[FAIL] {name}: {len(bad)} arrays differ, e.g. {bad[:5]}")
                failures += 1
            else:
                print(f"[ok]   {name}: {len(ref.files)} arrays bit-identical")
        else:
            np.savez(path, **data)
            print(f"wrote {path}  ({len(data)} arrays)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
