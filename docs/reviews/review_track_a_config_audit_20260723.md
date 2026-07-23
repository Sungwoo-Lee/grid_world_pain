# Config Audit — Track A config-layer fix (2026-07-23)

## Question / Verdict up front

**What is this doc about?** The project just fixed four bugs where the environment config loader would silently do the wrong thing — a mis-named noise setting, an unrecognized noise mode, and (the big one) a hardest-difficulty noise config whose header claimed the agent's "am I injured?" internal signals were noise-free when they actually still carried background noise ten times higher than intended. A separate fix made "old-style" scene descriptions (predators/rabbits listed the pre-2024 way) win over the newer format when both are present, so that training and evaluation agree on what animals are actually in the grid.

**Headline finding:** all four fixes work as intended and introduce no new risk. The noise-config fix (`05-sensory_noise_10x10.yaml`) now genuinely keeps the three "am I injured?" channels (satiation, interoceptive pain, external pain) at zero background noise while leaving the intended smell/vision noise untouched — confirmed by loading the file through the actual project loader, not by reading the YAML. The scene-precedence fix makes old-format and new-format scene descriptions agree at every load site that was checked, with zero regressions to any of the 170 currently-active environment configs, and zero unexpected recompilation risk. One small pre-existing terminology note is flagged (not a defect) about how the noise system quietly tolerates config keys for sensors that aren't turned on — already documented as safe-by-design, not new.

**Verdict: PASS.** No blockers. Safe as a config-layer change; does not by itself certify the runtime behaviour of every downstream training script (out of this audit's scope — see below).

---

**Scope:** Single-diff audit (env/config-layer bug fix), pre-commit review of uncommitted changes.
**Files audited:**
- `src/environment/config_loader.py` (dispatch-precedence + validation logic)
- `configs/environment/experiment/basic/05-sensory_noise_10x10.yaml` (the noise config with the mislabeled "clean" channels)
- `docs/environment/02_config_schema.md` (schema doc, per the Maintenance Contract)
- (Out of scope per instructions: `scripts/eval/eval_rollout.py` — another session's unrelated in-flight feature.)

**Audited by:** env-config-auditor
**Date:** 2026-07-23
**Plan under audit:** [[FIX_CONFIG_LAYER_SILENT_FAILURES_20260723]]
**Original findings:** [[findings_config_loader]] (Findings 1–3), [[findings_configs]] (P1-1)

## Summary

Pass. All three fixed pieces of code (scene-precedence dispatch, noise-mode validation, noise-modality-key validation) and the one config edit (`05-sensory_noise_10x10.yaml`) were verified by actually running the project's config loader — not by re-reading the source that produced them — against real config files, including a `git stash` before/after comparison to prove the fix leaves currently-active configs byte-identical. Every one of the four audit questions below resolves to OK. No blocking or concerning findings. One informational note (not a new finding, already covered by the diagnosis as "reviewed but clean") is included under Q1 for completeness.

## Audit Questions

### Q1 — `05-sensory_noise_10x10.yaml`: are the three interoceptive channels actually noise-off, and is the exteroceptive noise untouched? [OK]

Loaded the **merged** config (default.yaml + `04-jump_attack_10x10` + `03-random_init_10x10` + this file, following the real `extends:` chain) through `load_env_params()` and read back the actual noise arrays the running environment would use:

| Channel | mode | sigma | injury_noise_scale | Verdict |
|---|---|---|---|---|
| Satiation ("how hungry am I") | constant | **0.0** | 0.0 | clean, as documented |
| Interoceptive Nociception ("internal pain") | constant | **0.0** | 0.0 | clean, as documented |
| Extero Nociception ("external/contact pain") | constant | **0.0** | 0.0 | clean, as documented |
| Olfaction (smell — the intended "hypervigilance lever") | state-dependent | 0.15 (→0.75 at max injury) | 4.0 | unchanged, matches the file's documented design |
| Visual (on-contact ID ambiguity) | state-dependent | 0.10 (→0.30 at max injury) | 2.0 | unchanged, matches documented design |
| Collision / Proprioception / Location | constant | 0.01 / 0.05 / 0.01 | 0.0 | unchanged "tiny default" background noise, as the file's own footnote describes |

This directly contradicts the pre-fix state recorded in the diagnosis (`findings_configs.md` P1-1), where the same three interoceptive channels loaded at `sigma=0.1` (ten times the file's own "tiny noise" benchmark) despite the header's "kept CLEAN" claim. The fix is confirmed working, not just present in the diff.

**Sibling "06/07" lineage check:** no separate `06-` or `07-sensory_noise_10x10.yaml` files exist in the active config tree. `git log --follow` on the current file shows it *is* the historical "06-sensory_noise" — it was renamed to `05-` in a prior re-leveling commit (`b093023`), and there is no sibling copy left carrying the old bug. The only place the old `06-`/`07-` numbers still appear is as stale (already-flagged, unrelated) path references inside `train_command-agent.sh`'s comment/launch blocks (finding P2-7 in `findings_configs.md`) — those are shell-script text, not config files, and don't carry the sigma trap.

I additionally swept every other active config that touches `satiation:`/`interoceptive_nociception:`/`extero_nociception:` under `perceptual_noise.modalities` (the `avoidance_noise`/`avoidance_stat_noise` behavior-probe families) for the same "override mode but forget sigma" trap. None of them have it — they all set `sigma: 0.1` **explicitly** (verbatim copies of a specific trained model's noise settings, per their own header comments), so there is no silent inheritance in play there.

*Informational note, not a new defect:* the environment's full noise-index table includes three extra "always-present" slots (`Injury`, `Nutrition`, `Location`) that aren't in this config's actual observation vector (their observability flags are off). This is safe by construction — the noise-application code looks entries up by name, not position, so unused entries are simply inert — and it was already logged as "reviewed but clean" in the original diagnosis. Flagging it here only so the "obs↔noise order" checklist item isn't silently skipped.

### Q2 — Legacy-scene precedence: do active and legacy configs behave correctly at both load sites? [OK]

Loaded 3 archived legacy-format configs (`archive/2X2_area.yaml`, `archive/basic/00-5X5_NoPred.yaml`, `archive/basic/01-5X5_PredInterval3_NutGain18.yaml` — configs that describe their predators/rabbits the pre-2024 way) and 3 active modern-format configs (`basic/00-static_predator_5x5.yaml`, `basic/02-predator_and_rabbit_10x10.yaml`, `basic/04-jump_attack_10x10.yaml`), each through **both** the training-style load (shared base config underneath + merge) and the evaluation-style load (the file alone, no base underneath):

- **Legacy configs**: both load paths now agree on the same animal scene (matching tags and slot counts) — e.g. `2X2_area.yaml` yields the same 7-slot legacy scene at both call sites, with the expected migration warning firing only on the training-style load (which is the one that actually has something to warn about — an inherited-but-overridden entity list). This directly fixes the pre-existing divergence documented in the diagnosis (training used to silently substitute the shared base's animals; evaluation didn't).
- **Active (modern-format) configs**: both load paths agree, with **no** warning fired in either case, exactly as before the fix.
- **Byte-identical proof, not just reasoning**: I additionally `git stash`-reverted only the changed source file, re-ran the training-style load for all 3 active configs, restored the fix, and re-ran — the resulting animal-property arrays and pain-signal values were numerically identical in both cases. This is because, for a modern-format config with no legacy sections, the new guard condition mathematically reduces to the old one (the "has legacy sections" flag is false either way), so nothing about a modern config's behavior can change — confirmed empirically, not just by inspection.
- **Obs↔noise consistency held for both families** — every sensor name in each config's live observation breakdown is present in its noise lookup table, for all 6 spot-checked configs.

### Q3 — New strictness: does every active config still load without the new errors? [OK]

Bulk-loaded all **170** non-archived environment + verification configs (164 under `configs/environment/`, 6 under `configs/verification/`) through the real loader — matching the original diagnosis's exact count — at **both** the bare (evaluation-style) and base-underlay (training-style) call sites:

- Bare load: **170/170 OK, 0 failures.**
- Training-style load (with the shared base config merged in first): **170/170 OK, 0 failures.**

No config in active use triggers the new "unrecognized noise mode" or "unrecognized noise-channel name" errors that were added — those only fire on genuine typos, none of which exist in the currently-live config set. I also ran the plan's own regression-test file plus the pre-existing extends-layering test suite (12 tests total) and confirmed all pass, matching the developer's reported red→green results.

### Q4 — JIT/static-field recompile risk from the changed dispatch [OK — no risk for active configs; expected one-time compile for archived reruns]

The animal-scene fields this fix touches (`animal_classes`, `animal_behaviours`, `animal_tags`, and related count metadata) are all JAX static fields — meaning a change to their *values* forces a fresh compile, not a cheap reuse of a cached compiled function.

- **For every currently-active config**: values are proven byte-identical before and after the fix (see Q2's stash test), so **no new recompilation is introduced**.
- **For archived legacy configs, if someone reruns one**: the animal-scene values now correctly differ from what the buggy code used to produce (they reflect the legacy scene, not the shared base's scene) — this is the intended fix, not a defect, and it costs nothing extra: this project's main training entry point (`train.py`) does not use a persistent cross-process JAX compilation cache, so every training run already pays a fresh one-time compile at startup regardless of this change. (The one place a persistent compile cache *is* configured — the dwell-sweep evaluation tool — loads configs the "bare" way, which was already using the legacy scene both before and after this fix, so it is unaffected.)

No sweep config varies a static field unexpectedly as a result of this diff.

## Findings

| Severity | File / YAML path | Issue | Suggested fix |
|---|---|---|---|
| — | — | No blocking or concerning findings from this audit. | — |
| 🟢 nit | `docs/environment/02_config_schema.md` / noise-modality table | (informational, not a defect) the noise-index table always carries 3 slots (`Injury`, `Nutrition`, `Location`) that many configs never actually observe; this is safe by the name-based lookup design but could confuse a future reader auditing obs↔noise sync by eye. Consider a one-line note that unused slots are harmless. | Optional doc clarification only — not required before launch. |

## Checklist

- [x] (1) Observation ↔ Noise Modality Consistency — verified live for `05-sensory_noise_10x10.yaml` and 5 additional spot-checked configs; all sensors in the observation breakdown have a matching noise-table entry.
- [x] (2) Mandatory-Key Discipline — no new YAML keys or `get_mandatory` call sites introduced by this diff (confirmed by reading the diff; matches the plan's own claim).
- [x] (3) Static-Field & JIT Recompile Risk — no recompile risk for active configs (proven byte-identical pre/post-fix); expected, harmless one-time-compile difference for archived-config reruns.
- [x] (4) Known Latent-Bug Recurrences — N/A to this diff; no `overeating_death`, `random_start_pos`, legacy `property` (singular), or missing-olfactory-key patterns introduced.
- [x] (5) Schema Padding & Modality-Count Quirks — unaffected; still 13-slot padding, 10 real modalities, no new modality added.
- [x] (6) Cross-Config Coherence (sweep) — N/A, this is a single-diff audit, not a sweep.

## Manifest — exact commands run

```
# Q1 — merged noise values for 05-sensory_noise_10x10.yaml
load_env_params(get_default_config().merge(load_env_config('configs/environment/experiment/basic/05-sensory_noise_10x10.yaml')))
→ satiation/interoceptive_nociception/extero_nociception: sigma=0.0 (all three)
→ olfaction sigma=0.15/scale=4.0, visual sigma=0.10/scale=2.0 (unchanged)

# Q2 — legacy vs active scene agreement, both call sites, plus git-stash byte-identity check
_load_animals() compared bare-load vs default-underlay-load for 3 legacy + 3 active configs
git stash push -- src/environment/config_loader.py; re-run active configs; git stash pop; re-run; diff = none

# Q3 — bulk load, both call sites
170/170 configs (164 configs/environment/ + 6 configs/verification/), bare and underlay paths, 0 failures

# Q4 — reasoned from struct.field(pytree_node=False) on animal_classes/behaviours/tags in src/environment/state.py;
# no persistent JAX compile cache in train.py; dwell-sweep's persistent cache unaffected (bare-load path unchanged)

# Regression suite
pytest tests/env/test_config_layer_silent_failures_20260723.py tests/env/test_extends_layering.py -q
→ 12 passed, 1 warning (the expected, accurate DeprecationWarning)
```

## Conclusion

Safe to launch. All four audit questions resolve OK; no blockers, no concerns requiring user judgement. The fix does what the plan claims, verified against the real loader rather than re-derived from the source.

Audited by: env-config-auditor
