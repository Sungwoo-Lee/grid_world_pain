"""Read-only compatibility shim for SAVED run configs (archived-run evaluation).

Plain-language purpose: every training run writes a frozen copy of its own config
to `<run>/models/config.yaml`, and the evaluation / trajectory-collection tools
rebuild the model from that frozen copy rather than from the maintained files
under `configs/`. Runs that trained before the modulation-site refactor wrote the
old flat `modulation.temp_clip` key, which the live model now rejects outright.
Without this shim, re-analysing any archived neuromodulation run would fail at
model construction.

What it does: when a *saved* config is loaded for evaluation, it rewrites the old
flat key into the new `temperature: {enabled, clip}` shape IN MEMORY, and fills in
the site/mechanism keys with the settings that the pre-refactor architecture
hard-wired (encoder + task-GRU, gate-bias mechanism, temperature on, and the
whole observation vector as the modulator's input). Nothing is written back to
disk.

What it deliberately does NOT do: it does not soften the live-config path. A
config someone is editing under `configs/` that still carries `temp_clip` still
raises the migration ValueError from `ActorCriticRNN.__init__` — this is a
migration shim for reading archives, not a fallback default. It is also not a
general defaulting layer: it fires only on the specific legacy shape (flat
`temp_clip` present), and a saved config that is merely missing a mandatory key
still fails loudly.

Call it at the eval boundary only. All three read-a-saved-config call sites route
through this one function:
  - evaluation.py
  - scripts/eval/eval_rollout.py           (saved-config branch only)
  - scripts/eval/traj_collect/collect_trajectories.py
"""
from typing import Optional

# The architecture every pre-refactor modulated run actually had: the modulator
# wrote to the observation encoder and to the task GRU (through the update-gate
# bias), never to the actor or critic, and the action temperature was always on.
_LEGACY_SITES = {"encoder": True, "rnn": True, "actor": False, "critic": False}
_LEGACY_RNN_MECHANISM = "gate_bias"
# ...and it read the ENTIRE observation vector; the input slice did not exist yet.
_LEGACY_INPUT_SENSORS = "all"


def translate_legacy_modulation_config(modulation_config: Optional[dict],
                                       *, source: str) -> Optional[dict]:
    """Translate a pre-refactor saved modulation dict into the current shape.

    Args:
        modulation_config: the `agent.modulation` block read from a run's SAVED
            config, or None when the run was unmodulated.
        source: human-readable provenance (a path), used in the INFO line.

    Returns:
        The dict unchanged when it needs no translation, otherwise a translated
        COPY. The input is never mutated and nothing is written to disk.
    """
    if modulation_config is None or modulation_config.get("type") is None:
        return modulation_config
    if "temp_clip" not in modulation_config:
        return modulation_config

    translated = dict(modulation_config)
    clip = translated.pop("temp_clip")
    translated.setdefault("sites", dict(_LEGACY_SITES))
    translated.setdefault("rnn_mechanism", _LEGACY_RNN_MECHANISM)
    translated.setdefault("temperature", {"enabled": True, "clip": list(clip)})
    translated.setdefault("input_sensors", _LEGACY_INPUT_SENSORS)

    print(
        f"[INFO] Saved config predates the modulation-site refactor: translated "
        f"modulation.temp_clip={list(clip)} -> temperature.{{enabled: true, clip: {list(clip)}}} "
        f"with sites={{encoder,rnn}} / rnn_mechanism='{_LEGACY_RNN_MECHANISM}' / "
        f"input_sensors='{_LEGACY_INPUT_SENSORS}' "
        f"(in memory only, {source} is not modified).",
        flush=True,
    )
    return translated
