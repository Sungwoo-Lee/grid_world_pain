# Critical Config Settings & Change Log

> **Source of truth for values**: the resolved, saved `config.yaml` inside each run's `models/` dir (what the trainer actually used) | **Config-system mechanics**: [CONFIG_GUIDE.md](CONFIG_GUIDE.md) | **Exhaustive key list**: [02_config_schema.md](02_config_schema.md) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## What this document is about

This document exists to **stop important settings from changing silently**. Every experiment in this project is a variation of one environment configured by YAML (see [CONFIG_GUIDE.md](CONFIG_GUIDE.md)). A handful of those YAML values materially change what the agent experiences — how far it can smell, how hard death is punished, how big the world is. When one of those drifts without anyone noticing, months of training and analysis can quietly run under an assumption that is no longer true.

That is not hypothetical. On **2026-02-27**, the base config's olfactory distance-discounting exponent — `sensory.decay_power`, which controls how quickly an animal's smell fades with distance — was changed from `1.0` to `2.0` inside a large "scale the world down to 10×10" commit whose message **never mentioned olfaction at all** (commit `def81c1`). For months afterward, every training run and every behavior probe used the inverse-square falloff `2.0` (smell essentially only reaching a couple of cells) while the documented/intended value was the gentler `1.0` (smell reaching noticeably farther). Nobody caught it until a value the author *remembered as 1.0* was found to be 2.0 in the saved configs.

So this doc has **two jobs**:

1. **A critical-settings registry** — the settings that most change experiment behavior, each with its **canonical value**, where it lives in the config tree, and what it means. This is the "what the important knobs are and what they should be" reference.
2. **A dated change log** — every time a registry setting changes, an entry is added here with the **date, old → new value, reason, and commit**. This is the audit trail that makes a change like the 2026-02-27 one impossible to miss.

If you are an agent or a person about to change a value in the registry below, or about to launch training, **read the registry, then follow the logging protocol at the bottom**.

## The config system in 60 seconds

Full mechanics live in [CONFIG_GUIDE.md](CONFIG_GUIDE.md); the minimum you need here:

- **One base config**: `configs/environment/default.yaml` is the canonical "standard setup". Most values (including everything in the registry below) are defined there.
- **Sparse, layered overrides**: a new experiment config writes `extends: environment/default` and then lists **only the values that differ**. At load time the base is **deep-merged** underneath, so the agent gets a complete config. A config with no `extends:` loads **standalone** (the ~91 archived "full" configs work this way, frozen).
- **Because of inheritance, changing `default.yaml` changes every config that inherits it** — training configs *and* behavior probes — unless a config explicitly overrides the key. This is exactly why a silent edit to `default.yaml` propagates everywhere.
- **Verify the value that was actually used, not the source.** The trainer writes the fully-resolved config to `<run>/models/config.yaml` (rPPO) or `<run>/models/env_config.yaml` (Dreamer). That saved file — not a fresh reload of `default.yaml` — is ground truth for what a given run experienced. (A saved key may sit under a different block than you expect; e.g. `decay_power` lives under `sensory.`, not an `olfaction.` block.)
- **No silent fallbacks**: critical keys are read with `config.get_mandatory('...')`, so a missing key raises rather than defaulting.

## Critical settings registry

The settings whose value most changes what the agent experiences. **Canonical value** is what a config inheriting `default.yaml` should resolve to. Add a row when you identify another high-impact setting; change a row's value only together with a Change-log entry below.

| Setting (config path) | Canonical value | Where set | Meaning / why it matters |
|---|---|---|---|
| `sensory.decay_power` | **1.0** | `configs/environment/default.yaml` | Olfactory distance-discounting exponent. Smell intensity from a source = `1 / distance^decay_power` (capped at 2.0 when on the source cell; formula in `src/environment/sensor.py`). `1.0` = gentle, longer-range gradient the agent can follow from afar (**more predictive/learnable**); `2.0` = inverse-square, smell only reaches ~1–2 cells. Global to the olfactory sensor (applies to every smell source). |
| `sensory.sensor_radius` | 20 | `configs/environment/default.yaml` | Max range (cells) at which the olfactory sensor registers a source at all. Pairs with `decay_power`. |
| `body.death_penalty` | 100 | `configs/environment/default.yaml` | Terminal penalty on death. Large relative to per-step rewards; changes risk sensitivity. |
| `sensory.vector_size` | 5 | `configs/environment/default.yaml` | Olfactory channel width — part of the obs fingerprint; changing it alters `obs_dim` and breaks checkpoint compatibility. |

> This registry is intentionally short and grows on evidence. If you change or add a high-impact setting, add/adjust its row **and** log the change below.

## Change log (newest first)

Each entry: **date — `setting` old → new — reason — commit — blast radius.**

- **2026-07-26 — `sensory.decay_power` 2.0 → 1.0** — Reverting to the intended value (see the historical entry below; the `2.0` was an unannounced side change). `1.0` gives the gentler, longer-range smell gradient the design intends, making the olfactory signal more predictive for the agent. **Commit**: this document's creation commit (see `git log -- docs/environment/CONFIG_CRITICAL_SETTINGS.md`). **Blast radius**: every config inheriting `configs/environment/default.yaml` — all current training configs and all core avoidance behavior probes now resolve to `1.0`. **Caveat**: models trained *before* this date used `2.0`; to evaluate an old model faithfully, pin `sensory.decay_power: 2.0` for that eval, and re-train under `1.0` for a clean 1.0-regime comparison.

- **2026-02-27 — `sensory.decay_power` 1.0 → 2.0 (silent — retro-logged)** — Changed in commit `def81c1` ("feat: scale down the grid world environment to 10x10, simplify object and predator configurations, ..."), whose message did **not** mention `decay_power`, olfaction, or smell. This is the drift this document was created to prevent; recorded here for the audit trail. Superseded by the 2026-07-26 entry above.

## Logging protocol (required)

When you change any value in the **Critical settings registry** (or add a new registry-worthy setting):

1. Make the config change.
2. In the **same commit**, add a Change-log entry at the top of the list above with: date (`date +%Y-%m-%d`), the setting path, old → new value, a one-line reason, the blast radius (which configs/runs it affects), and — since the commit hash isn't known until you commit — reference it as "this commit" (or amend the hash in afterward).
3. If the change alters the config *schema or system* (a new key, a renamed block), it must **also** update [CONFIG_GUIDE.md](CONFIG_GUIDE.md) and [02_config_schema.md](02_config_schema.md) per their Maintenance Contract.

A registry-setting change that lands **without** a Change-log entry is the exact failure this document exists to prevent — treat a missing entry as a regression.

## Required reading

This document is **required reading** — before the work each names — for the config-touching and training-launching agents:

- **`env-config-auditor`** — before any config audit or pre-flight check.
- **`experiment-designer`** — before authoring or editing any config.
- **`training-runner`** — before launching training (confirm the run's critical settings match intent).
- **`developer`** / **`senior-developer`** — before changing anything under `configs/` or `src/environment/`.

## Maintenance Contract

Any change that touches a value in the **Critical settings registry**, or the config system's handling of one, must update this document in the **same change**: adjust the registry row (if the canonical value changed) and add a **Change-log entry**. Adding a new high-impact setting means adding a registry row. This contract is what keeps the registry trustworthy and the change log complete. Config-caring agents (above) enforce it as part of their normal review/audit.
