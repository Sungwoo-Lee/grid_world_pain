# Critical Config Settings & Change Log

**Purpose:** prevent silent config drift. Two parts below — a **registry** of high-impact settings (canonical value + meaning) and a dated **change log**. Rule: change a registry value → add a change-log entry in the **same commit** (missing entry = regression). Config-system mechanics: [CONFIG_GUIDE.md](CONFIG_GUIDE.md). Ground truth for a run's value = its saved `models/config.yaml`, not the source.

## Registry

Canonical value = what a config inheriting `configs/environment/default.yaml` resolves to. Changing a value in `default.yaml` changes every config that inherits it (training **and** probes) unless that config overrides the key.

| Setting (path) | Value | Set in | Meaning |
|---|---|---|---|
| `sensory.decay_power` | **1.0** | `default.yaml` | Olfactory distance falloff: smell = `1/dist^p` (cap 2.0 on-cell; `src/environment/sensor.py`). `1.0` = gentle, long-range gradient (predictive); `2.0` = inverse-square, ~1–2 cell reach. |
| `sensory.sensor_radius` | 20 | `default.yaml` | Max cells the olfactory sensor reaches at all. |
| `sensory.vector_size` | 5 | `default.yaml` | Olfactory channel width — part of `obs_dim`; changing it breaks checkpoint compatibility. |
| `body.death_penalty` | 100 | `default.yaml` | Terminal death penalty; large vs per-step reward → risk sensitivity. |
| `sensory.olfactory_grid_range` | **0** | `default.yaml` | Radius of the olfactory sampling diamond. `0` = pre-v3.1 single sample (byte-identical). `>0` adds `(2r²+2r+1)×vector_size` dims — **changes `obs_dim`, breaking checkpoint compatibility**. |
| `sensory.visual_blur_enabled` | **false** | `default.yaml` | Anisotropic point-spread on vision. Changes observation *semantics* at identical width — in the modality fingerprint. |
| `sensory.visual_value_mode` | **sum** | `default.yaml` | `sum` (count) vs `clamp` (presence). Semantics at identical width; fingerprinted. |
| `sensory.visual_occlusion_enabled` | **false** | `default.yaml` | Line-of-sight occlusion. Aggressive on this dense scene — see the calibration table in `02_config_schema.md`. Fingerprinted. |

Grows on evidence: add a row when you identify another high-impact setting.

## Change log (newest first)

Entry = **date — `setting` old→new — reason — commit — blast radius.**

- **2026-08-21 — on-source olfactory decay: literal `2.0` → `1/(0.5^γ)`** — the constant is exactly the decay curve at half a cell, i.e. "standing on it = half a cell away", but only at γ=1. Expressing it as a floor keeps it correct at every γ. **Bit-identical at the shipped `decay_power: 1.0`**, so every config inheriting `default.yaml` is unaffected. Commit `0e8a4ef`. Blast radius: the **eleven** configs shipping `decay_power: 2.0` (`configs/continual/nmn_double_return_stages/*` ×5, `configs/verification/observability_gates_S1–S4`, `configs/verification/olfaction_parity_{neutral,predator}`) read **4.0 instead of 2.0** on-source, on ~⅓ of steps. The two `olfaction_parity_*` configs will need their stored expectations regenerated if re-run. Accepted deliberately rather than gated. Study: [[ONSOURCE_RULE_STUDY]].
- **2026-08-21 — eight new `sensory.*` keys added (v3.1/v3.2)** — `olfactory_grid_range`, `visual_blur_*` ×3, `visual_value_mode`, `visual_occlusion_*` ×3, plus per-entity `visual_mask` and `blocks_sight`. All default to pre-change behaviour; parity verified against a fixture captured from the old code on a pinned CPU backend. Commits `0e8a4ef`, `0949ddb` (rename), `9771e98`. Blast radius: 148 configs carrying a standalone `sensory:` block were swept with the parity-preserving defaults, **including archived configs still loaded by the test suite**. Occlusion's two sub-keys are conditional-mandatory (read only when enabled) to keep that surface from growing further.
- **2026-07-26 — `sensory.decay_power` 2.0→1.0** — revert to intended value (2.0 was an unannounced side change). Commit: this doc's creation commit. Blast radius: all configs inheriting `default.yaml` (current training + core avoidance probes). Caveat: pre-2026-07-26 models trained under 2.0 — pin `2.0` to eval them; re-train under 1.0 for a clean comparison.
- **2026-02-27 — `sensory.decay_power` 1.0→2.0 (silent, retro-logged)** — slipped into commit `def81c1` ("scale down to 10x10 …"), message never mentioned olfaction. The drift this doc exists to catch; superseded above.

## Protocol & required reading

- **On any registry-value change:** add a same-commit change-log entry (date, path, old→new, reason, blast radius). A schema/system change also updates [CONFIG_GUIDE.md](CONFIG_GUIDE.md) + [02_config_schema.md](02_config_schema.md).
- **Required reading** (registry + change log, before config/launch work): `env-config-reviewer`, `experiment-designer`, `training-runner`, `developer`, `senior-developer`. They enforce the protocol — a registry change without an entry is a flagged regression.
