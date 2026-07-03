# Config Audit — v3.0 training+evaluation pipeline (post `extends:` fix)

**Scope:** Pre-flight / diagnosis re-verification of the config-loading surface after the
v3.0 update wave, triggered by a real training bug: `train.py` was loading `--config` files
with a plain YAML reader that never resolved `extends:` (a mechanism that lets one config
file say "inherit settings from this other file"). Any config that relied on inheriting a
non-default parent — noise settings, random-start-position settings, harder predator
settings — silently trained with `default.yaml`'s values instead of its own intended
settings. This was fixed by routing both the single-config launch path (`train.py:381`) and
the multi-stage "continual learning" launch path (`train.py:195`) through the
extends-resolving loader (`load_env_config`). This audit independently re-verifies that fix,
plus the surrounding config/observation/noise machinery, using evidence that does **not**
just re-run the same code path that was broken.

**Files audited:** `src/environment/config_loader.py`, `src/environment/sensor.py`,
`src/environment/core.py`, `src/environment/state.py`, `train.py` (both config-loading
paths), `configs/environment/default.yaml`, `configs/environment/experiment/basic/`,
`configs/environment/experiment/basic05_variants/`, `configs/environment/experiment/basic_curriculum/`,
`configs/verification/observability_gates_S{1..4}.yaml`.

**Audited by:** env-config-auditor
**Date:** 2026-07-04

## Summary

**Pass, with two concerns worth the user's attention (neither blocks the fix itself).**
The `extends:` fix is correct and independently confirmed on **both** launch paths — not by
re-running the loader under audit, but by three independent methods: (1) reading a real,
already-launched run's saved `config.yaml` from disk (ground truth — the file the training
process itself wrote after loading its config) and confirming it carries the inherited
settings; (2) hand-writing a from-scratch YAML merge in plain Python (no project `Config`
class at all) and diffing it byte-for-byte against what the actual continual-learning
function produces; (3) replaying the *old, buggy* loader side-by-side with the *new, fixed*
loader on the same file to confirm the old one really did drop the inherited layers and the
new one doesn't. All three agree. Separately, this audit found one **pre-existing, unrelated**
issue not caused by this fix: the "static predator, no cover" curriculum stage
(`basic/00` and `basic_curriculum/00`) has the bush-hiding behavior measure turned on by
default but declares zero hiding spots, so that one metric will silently read as
"not-a-number" for that stage. See Findings.

## Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| 🟡 concern | `configs/environment/experiment/basic/00-static_predator_5x5.yaml:41-42`, `configs/environment/experiment/basic_curriculum/00-static_predator_5x5.yaml` (same body) | `environment.obstacles: []` (deliberately no bushes/rocks — the level's whole point is "no cover, static threat"), but `behavior_measures.enabled: true` is inherited from `environment/default.yaml` and never turned off. The bush-dive-rate behavior metric (M2) needs at least one obstacle with `hides_agent: true` to be meaningful; with none, `info['agent_in_bush']` is always `False` and M2 is structurally undefined (not-a-number) for every episode at this curriculum stage. Confirmed by loading all 19 active `basic*`/`basic_curriculum` configs through the real loader: this is the **only** one missing a hiding spot while behavior-measures stays on. | Either add a `blocking: false` "decorative" bush obstacle to this level, or explicitly set `behavior_measures.enabled: false` for this stage (with a comment noting M2 is undefined here), so the NaN is an intentional, documented choice rather than a silent gap. |
| 🟡 concern | `configs/verification/observability_gates_S1-S4.yaml:10` | These are fixed-seed sensor-readout gate checks (`jax.random.PRNGKey(0)`, per `scripts/verification/check_observability_gates.py`) that also set `random_start_pos: true`. Per the documented latent quirk (agent placement does not participate in the entity-occupancy mask), the agent can spawn on top of a resource or predator, firing contact effects (nociception, eating) on step 0 — which could quietly change what a "read the sensor at a known clean state" gate check is actually measuring. Reproducible under the fixed seed, so not catastrophic, but worth a second look given the script's own purpose is deterministic sensor validation. | Confirm the current gate assertions are unaffected by step-0 contact (they likely already pass and were validated against this exact behaviour); if a future gate specifically needs a "clean" no-contact reset, set `random_start_pos: false` there instead. |
| 🟢 nit (already documented, re-confirmed current) | `src/environment/config_loader.py:1450` | `perceptual_noise.enabled` is read via `config.get('perceptual_noise.enabled', False)`, not `get_mandatory` — a fallback default on a top-level feature toggle. This is a **documented, intentional** exception (see `docs/environment/02_config_schema.md`'s "Optional Keys and Defaults" table), preserving backward compatibility for the many pre-noise configs that never mention the block. Re-confirmed present and unchanged; not a new violation of the no-fallback-defaults rule. | None — already tracked as an accepted exception. Keep it documented if it's ever touched again. |
| 🟢 nit (already documented, re-confirmed current) | `src/environment/config_loader.py:1468-1477` (`_parse_noise_config`) | A `perceptual_noise.modalities` YAML key that doesn't match one of the 10 known sensor names (typo, e.g. `extro_nociception`) is **silently dropped** rather than raising an error — already flagged in `02_config_schema.md` line 1145 as a known sharp edge. Re-confirmed present. If that dropped modality is one the observation actually emits (e.g. a sensor enabled via `sensory.*_enabled`), the *separate* runtime lookup in `sensor.py`'s `apply_perceptual_noise` (`modality_map[sensor_name]`) will raise a hard `KeyError` the first time noise is applied — this is the documented FAQ §10 crash mode. Live-loaded all 19 active `basic*` configs plus all 4 `observability_gates` configs and the 2 `olfaction_parity` configs (17 total distinct files) through the real loader and diffed each one's observation breakdown against its noise-modality coverage: **no current config triggers this** — every enabled sensor has a matching (correctly spelled) modality entry. | None currently required. Worth a config-loader-side guard (raise on unknown key) as a future hardening pass, purely to convert a future typo from a training-time crash into a load-time error — out of scope for this diagnosis. |

## What was verified CORRECT (with method)

1. **`extends:` resolution — single-config path (`train.py:381`).** Confirmed via **ground
   truth**: the real saved `config.yaml` from the already-relaunched
   `rppo_basic05v03_fastmove_n109` run (`results/JAX_RecurrentPPO/20260703-154725_.../models/config.yaml`,
   launched *after* the fix) shows `random_start_injury: true`, predator
   `move_interval: [1, 1]`, `max_stamina: [30, 30]` — all values that only exist because the
   3-level chain `basic05_variants/03 → basic/05 → environment/default` was correctly
   resolved. This is the training process's own on-disk record, not a re-derivation.

2. **`extends:` resolution — continual-learning path (`train.py:195`,
   `_build_continual_schedule`).** No continual run has been launched since the fix landed
   (the last continual runs on disk predate it), so there is no saved-`config.yaml` ground
   truth for this path yet. Instead verified two independent ways:
   - **Independent manual merge.** Wrote a standalone Python script that loads
     `environment/default.yaml` and a `basic_curriculum` stage file and deep-merges them by
     hand — using plain `dict`/`yaml`, zero imports from this project's `Config` class or
     `config_loader.py`. Its output for `environment`, `perceptual_noise`, `body`, and
     `sensory` matched the real `_build_continual_schedule()` output **byte-for-byte**.
   - **No cross-stage aliasing.** Built a 5-stage schedule, mutated stage 0's config in
     place, and confirmed stages 1–4 were untouched — the per-stage `deep-copy → merge`
     sequence at `train.py:194-195` does not leak state between stages.
   - **Differential replay of the actual bug.** Loaded a config with a real 2-level
     `extends:` chain (`basic05_variants/03 → basic/05 → environment/default`) through both
     the *old* buggy loader (`Config.load_yaml`, no extends resolution) and the *new* fixed
     loader (`load_env_config`) side by side: the old one produced a 2-key dict (missing
     `perceptual_noise`, `body`, `sensory`, `behavior_measures`, and even
     `environment.height`); the new one produced the full, correctly-inherited 6-key dict.
     This confirms the fix is not a no-op and generalizes correctly to nested chains, even
     though no *currently-scheduled* continual run happens to exercise a nested (non-default)
     parent yet.
   - **Caveat honestly flagged:** every `configs-dir` currently in the repo
     (`basic_curriculum/`) only extends `environment/default` directly (one level), and
     `train.py`'s own `base_config` is *already* seeded from `environment/default.yaml`
     before the buggy code path would have run — so the pre-fix bug would not have visibly
     changed behaviour for these specific stage files even before the fix (masked, not
     absent). The mechanism is proven correct; there is no post-fix continual run yet that
     would have been *visibly* broken before the fix, unlike the single-config case where
     several real runs (basic/06, basic/07, the basic05\_variants) were confirmed to have
     trained wrong. Recommend a short continual smoke launch as a belt-and-suspenders check
     before the next real continual sweep, though the code-level evidence above is already strong.

3. **Observation ↔ noise-modality sync (Audit checklist §1).** `get_observation_breakdown()`
   (`sensor.py:350`) and `_parse_noise_config()`'s `_YAML_KEY_TO_SENSOR_NAME` map
   (`config_loader.py:1454`) cover the same 10 sensor names with matching capitalization.
   Confirmed the coupling is by **name**, not by list position — `apply_perceptual_noise`
   builds a `{name: index}` dict and looks up each active sensor by name, so a YAML author
   reordering the `perceptual_noise.modalities` block cannot silently misalign a sensor with
   the wrong noise settings. Live-loaded 17 distinct active configs (all `basic*` curriculum
   levels + variants, all 4 `observability_gates`, both `olfaction_parity` configs) and
   confirmed every sensor each config's `sensory.*_enabled` flags turn on has a matching
   noise-modality entry — zero missing mappings.

4. **Mandatory-key discipline (§2).** Spot-checked the full `load_env_params` mandatory-key
   block (`config_loader.py:1392-1437`, ~40 keys) — all read via `get_mandatory`. The one
   `.get(default)` in this region (`perceptual_noise.enabled`) is a documented, intentional
   backward-compatibility exception (Finding row above), not an oversight.

5. **Known latent-bug recurrences (§4).**
   - `body.overeating_death` bug (sets `termination_reason=3` but never `done=True`) —
     confirmed still present at `core.py:701-703` (unchanged behaviour). Confirmed **no
     active config** sets `overeating_death: true`, so it is currently inert.
   - `body.start_satiation` / `body.random_start_satiation` — confirmed still
     schema-mandatory (`get_mandatory` at `config_loader.py:1397,1414`) but never consulted at
     runtime; `jax_reset` (`core.py:1076-1098`) derives `satiation` purely from `nutrition` via
     the configured scaling curve, ignoring both fields entirely. Matches the documented FAQ
     entry; not a new regression.
   - `property` (singular) vs `properties` (plural) — `_read_properties` /
     `_read_properties_std` (`config_loader.py:269-320`) correctly accept the legacy singular
     key with a `DeprecationWarning` and prefer the plural. No active config still uses the
     legacy singular key.
   - List-replace footgun in `extends:` — confirmed correctly documented
     (`config_loader.py:88-93`) and correctly *used*: every sparse override that needs to
     change one entity in a list (e.g. `basic05_variants/03` changing only the predator's
     `move_interval`) redeclares the **full** `entities:` list rather than assuming a partial
     merge, which is the documented-required pattern for lists (dicts, like
     `perceptual_noise.modalities`, merge per-key correctly — verified in Finding-adjacent
     ground-truth check above).

6. **Static-field / recompile risk (§3).** Spot-checked the static-field table in
   `docs/environment/02_config_schema.md` against `state.py` — `noise_modality_order` and
   `perceptual_noise_enabled` are both correctly `struct.field(pytree_node=False)`. The
   jump/pounce feature's `has_attack_feature` static flag correctly gates the whole code path
   at trace time so disabled configs take the byte-identical pre-feature path. Continual
   learning's per-stage grid-size/entity-count changes are static-field changes *by design*
   (each curriculum stage is a genuinely different environment) and `train.py` already
   guards against a *worse* failure mode — a stage silently changing the observation
   dimension or action-space size — via its explicit modality-fingerprint check
   (`train.py:506-550`), run before training starts.

## Checklist

- [x] (1) Observation ↔ Noise Modality Consistency — pass (17 configs live-checked, name-keyed not position-keyed, zero mismatches)
- [x] (1.5) Behavior-measures Bush Presence — **1 concern found** (`basic/00` / `basic_curriculum/00`)
- [x] (2) Mandatory-Key Discipline — pass (one documented, intentional exception)
- [x] (3) Static-Field & JIT Recompile Risk — pass (no undocumented recompile traps; continual fingerprint guard already in place)
- [x] (4) Known Latent-Bug Recurrences — pass (all four re-checked; none newly triggered)
- [x] (5) Schema Padding & Modality-Count — pass (padded to 13, 10 in use, no 14th-sensor case exists)
- [ ] (6) Cross-Config Coherence (sweep only) — N/A, this was a pipeline-correctness diagnosis, not a sweep audit

## Conclusion

The `extends:` fix is correct on both the single-config and continual-learning launch paths,
independently confirmed by ground truth, hand-computed merge parity, and differential
old-vs-new replay — safe to keep relying on for future launches. Two 🟡 concerns (the
bushless static-predator stage's undefined M2 metric; the seeded-but-random-start
observability gates) are pre-existing and unrelated to this fix — user judgement on whether
to address before the next runs that touch them.

Audited by: env-config-auditor
