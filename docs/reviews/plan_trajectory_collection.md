---
title: "Plan review — Trajectory Collection Pipeline"
topic: reviews
status: active
created: 2026-08-19
last_updated: 2026-08-19  # re-review appended same day; verdict revised NOT READY → SOUND
---

# Plan Review — Trajectory Collection Pipeline

> **Reviewed plan**: [[TRAJECTORY_COLLECTION_PIPELINE]] (`docs/develop/active/behavior/TRAJECTORY_COLLECTION_PIPELINE.md`)
> **Reviewed by**: plan-reviewer
> **Date**: 2026-08-19 (initial) · 2026-08-19 (re-review after revision)

## Verdict

**SOUND** — re-review of the revised plan; the exit condition of the initial review is **met**. See **Re-review (2026-08-19)** below. The initial NOT READY review is preserved unchanged after it, as history.

## Re-review (2026-08-19) — exit condition MET, verdict revised to SOUND

The plan was revised (~240 lines added: §A11, §D14–§D16, V10, C0, expanded File Changes) in response to the two Critical findings. This re-review verified the revision's claims at source rather than taking them on trust.

**F1 (scene-reload faithfulness) — resolved.**

- The loader-precedence mechanism is confirmed verbatim at `src/environment/config_loader.py:429-435`, and the `828b77e` diff confirms the pre-fix loader dispatched entities-first, so a dual-format dump flips scene across the fix boundary exactly as §A11 states.
- The guard (§D14a) is **purely structural** — it refuses on scene-format coexistence, with no date heuristic anywhere in the mechanism. The corpus scan is load-bearing only for the *scope* claim ("historical, not live"), and this re-review **independently reproduced it**: 334 saved configs, 153 entities-only / 151 legacy-only / **12 dual-format** / 18 empty-scene, the 12 matching the plan's list (2026-05-29 → 06-11, ten `logcheck_*`).
- The guard runs before params, `env_fp`, and manifest creation (File Changes flow), so an ambiguous run cannot create a store directory; V10 asserts exactly that, **against a real dual-format run**, and additionally asserts the guard cannot warn-instead-of-raise or attempt to resolve. C0 orders the check first among implementation checkpoints.
- Refuse-rather-than-resolve is the right call: nothing in a run directory records which branch its trainer took, so any resolution would be a guess; the cost is 12 historical runs (ten throwaway), and `--allow-ambiguous-scene` + `scene_ambiguous: true` covers independently-adjudicated cases. The tool loses nothing it could legitimately handle.

**F2 (red reset-parity gate) — downgrade to Low is justified; the durable residual is retained.**

- All three cited commits verified: fixtures from `3d20aab` (2026-05-28); `84014e4` (2026-07-04) changed **exactly** the four `observability_gates_S{1..4}` configs (confirmed: `start_pos: [5,5]`, `random_start_pos: false` now in the config; the failure's actual `[4,4]` is the 0-indexed internal form of the 1-indexed `[5,5]`, against the stale fixture's `[2,2]`). The failure set equals exactly the four touched configs with zero unexplained residue — a deliberate config change post-dating its fixtures, not code drift.
- The durable residual survives in the plan: §D15's "red test nobody triages" lesson, the schema doc's code-drift caveat, `collection_git_sha` + `run_dir_name` + `train_config_mtime` in the manifest, fixture regeneration as a Phase 0 exit condition, and the train-time-git-SHA follow-up named to `senior-developer` outside scope.

**F3–F6 spot-checked, all addressed as claimed**: strict key+shape restore assertion in the `load_policy` seam with C0 testing that it *fires*; schema-doc tables generated from `STEP_COLUMNS` with a matching-output test, plus V9's bare-pyarrow read closing the code↔store↔doc loop and C3 honestly relabelled circular; §D16 phased rollout with per-phase failable exit conditions and V1/V3/V4 as acceptance gates on every production store; `validate_store_draws` whole-store bounds / non-degeneracy (correctly conditioned on `low < high`) / activation-count checks. **F7–F10 confirmed fixed. Open assumption 4 (rename atomicity on the NAS)**: the fsync-before-rename + directory-fsync design, V6-on-NAS requirement, and the explicit statement that `SIGKILL` cannot reach the node-crash case (client page cache survives a killed process; only a killed node loses unflushed writes) are sound and honestly bounded — and the driver's full-read validation (§D10) is a second net that would catch a truncated shard post hoc.

### New findings from the revision (all Low; none blocks implementation)

| # | Sev | Location | Issue | Suggested fix | Owner |
|---|---|---|---|---|---|
| N1 | 🟢 | §D14a vs `config_loader.py:429` | The guard predicates entities on **truthiness** (`bool(...)`) while today's loader uses **`is not None`** — they diverge on a config carrying an explicit *empty* `entities:` list alongside legacy blocks (precedence flipped across `828b77e` for that shape too, since the old loader dispatched on presence). Verified against the corpus: **zero** of 334 saved configs have that shape, and post-fix runs cannot recreate the ambiguity, so this is currently harmless. | Cheapest hardening: also refuse when `entities` is present-but-empty alongside a non-empty legacy block; at minimum, a comment in `assert_scene_unambiguous` recording that the predicate is deliberately truthiness-based and the empty+legacy shape was verified absent (scan 2026-08-19), so a future maintainer neither "fixes" it into a mismatch nor trusts it blindly. | `developer` |
| N2 | 🟢 | `tests/test_trajectory_collection.py` (V10 host) | V10-as-pytest depends on 12 specific run directories under gitignored `results/` on the NAS — the test silently loses its subject if those runs are cleaned, and errors on any machine without the mount. | Copy one dual-format `config.yaml` (a small YAML, not run data) into `tests/fixtures/` as the deterministic guard-fires case; `skipif`-guard the against-real-corpus cases on path existence. | `developer` |
| N3 | 🟢 | `write_shard_atomic` (File Changes) | Directory fsync over CIFS may be unsupported (opening a directory fd for fsync can raise on network filesystems). If so it fails loudly on the first shard — V6-on-NAS will surface it immediately, so nothing silent — but the fallback should be an explicit recorded decision, not an ad-hoc `try/except: pass` added under time pressure. | Decide and document the CIFS-rejects-dir-fsync behaviour when implementing `write_shard_atomic`; V6 already runs on the NAS and will exercise it. | `developer` |
| N4 | 🟢 | `docs/develop/active/issues/KNOWN_BUGS.md:73` | The registry row for the parity failure still reads "OPEN — needs triage, twice-confirmed, owner unassigned". §D15's verified diagnosis (stale fixtures vs `84014e4`, not code drift) supersedes it; the registry should say so, and point at §D15 and the Phase 0 regeneration task. | Registry update. | `bug-curator` |

No new Critical or Moderate findings. Ordering of the added steps was checked (guard before params before manifest; Phase 0 before any collection; C0 first among checkpoints) and no added check is unfailable; no contradiction was found between the added sections and the pre-existing ones.

### Cost of being wrong (re-review)

With the guard in place, the residual failure surface is implementation-level (the guards existing in the plan but not firing in the code) — which C0, V5(b), V7, and V10 each test as *fires*, not as passes-on-clean-input. If the re-review is wrong anyway, the cost reverts to the original: a store describing the wrong world with no internal signal. The plan now carries four independent layers against that (structural guard, provenance manifest, applicability boundary, guard-fires tests), which is as much as a plan can do; the rest is `senior-developer`'s post-implementation adherence check.

---

## Initial review (2026-08-19) — NOT READY (superseded by the re-review above; preserved as history)

**NOT READY** — two Critical findings, both cheap to fix before implementation.

The plan proposes a tool that puts a trained agent back into "the exact world it was trained in" and records a million episodes per training run. The review found that the plan **assumes, and never verifies, that rebuilding the world from the run's saved config actually reproduces the training world**. There is a documented, already-bitten mechanism by which that assumption fails silently: before a config-loader fix landed on 2026-07-23 (commit `828b77e`), the trainer silently used a different scene than the one the config's author wrote. A saved config from such a run can contain *both* scene descriptions, and today's loader — reloading that file — picks the one the trainer *discarded*. The collector would then faithfully record a million episodes of the agent in the wrong world, and nothing in the plan's ten checkpoints or nine verification steps can detect it, because every check builds the environment through the same loading step.

Second, the repository's reset-behaviour parity gate (`tests/env/test_unified_parity.py`) is **currently failing**, twice-confirmed on a clean tree, unowned — and the plan never mentions it. That test is standing evidence that environment reset behaviour has drifted at least once without anyone noticing, in exactly the reset-time code whose outputs this pipeline records.

Everything else in the plan is unusually strong: the seeding design, the anti-overwrite guards, the resume design, and the verification plan genuinely close three of the five known-bug hazards checked (fixed-seed episode repetition, stale-data blending, derived-measure smuggling).

**What flips the verdict**: (1) a collector-side guard that hard-fails on a saved config containing both scene formats, plus a documented applicability boundary for pre-`828b77e` runs; (2) triage of the red parity test (or an explicit, manifest-recorded code-drift caveat).

**Severity legend**: 🔴 Critical = fix before going further · 🟡 Moderate = likely costs a re-run · 🟢 Low = cosmetic · ❓ Open = an assumption nobody has verified yet.

## Findings

| # | Sev | Location | Issue | Suggested fix | Owner |
|---|---|---|---|---|---|
| F1 | 🔴 | §A10; File Changes `collect_trajectories.py` flow | The saved resolved config is declared "the correct source of truth", but the collector re-loads it through **today's** `load_env_params`, which applies legacy-scene precedence (`src/environment/config_loader.py:429-435`: a non-empty `environment.predators`/`neutral_animals` block wins over `environment.entities`). Pre-`828b77e` runs (KNOWN_BUGS.md:160) trained on the *merged* scene while their dump can carry both blocks — reload flips precedence and rebuilds the world the trainer discarded. V1 and V2 both consume the same `params` object, so both pass with the wrong world: circular w.r.t. config loading. `train.py` records no training-time git SHA, so no recorded fact identifies affected runs. | (a) Collector hard-fails (`ValueError`) when the saved config contains both a non-empty legacy scene block and an `entities:` block — that coexistence is exactly the fingerprint of an ambiguous pre-fix dump; explicit override flag + manifest flag if someone knowingly proceeds. (b) State the applicability boundary (runs trained after 2026-07-23) in §A10 and in `TRAJECTORY_STORE_SCHEMA.md`. (c) Record the run directory's creation date in `_manifest.json` as the provenance proxy. | `senior-developer` (plan), then `developer` |
| F2 | 🔴 | plan-wide; §D4 state sources | `tests/env/test_unified_parity.py` is red — agent-position mismatch **at step 0**, twice-confirmed pre-existing, owner unassigned (KNOWN_BUGS.md:73). The plan neither depends on it nor notices it is broken. It is the gate that would catch a reset-time regression in the very state fields this pipeline records, and its failure is standing evidence that env reset behaviour drifted at least once (stale fixtures) without regeneration. It also exposes the plan's unstated assumption that env code is unchanged between training and collection — the manifest records only the *collection-time* git SHA. | Triage the red test before the first production collection (likely fixture regeneration; if it is a live reset bug, everything downstream is suspect). Add one sentence to the plan acknowledging the gate and gating collection on its triage; add the code-drift caveat (training-time code version is unrecorded) to the schema doc's Known-caveats section. | `bug-curator` (assign), `developer` (triage) |
| F3 | 🟡 | File Changes, `load_policy` seam; V2 | No check that the rebuilt model structurally matches the checkpoint. Known bugs: a missing/misspelled modulation block silently builds an unmodulated agent; model-size CLI flags possibly not persisted to the saved config. V2 is circular here — both the fast and legacy paths rebuild from the same saved config, so both agree with the same wrong agent. | Assert the orbax-restored tree's keys **and shapes** exactly match the built model's tree — hard-fail on any missing or unexpected key. One check covers both known bugs. | `developer` |
| F4 | 🟡 | C3; `TRAJECTORY_STORE_SCHEMA.md` §3 | Circular schema verification: C3 validates the store by reading it back through `open_store`, which shares `build_step_schema` with the writer — a schema-definition bug echoes itself back as proof. Separately, nothing verifies the schema **doc** (half the deliverable) matches `STEP_COLUMNS`/`EPISODE_COLUMNS`; hand-transcribed tables rot. | Generate the doc's key-list tables from the code (or add a test that parses the doc's tables and compares names/order/types to the code). Add one check that reads a shard with **bare pyarrow, no `trajectory_store` import**, and compares against the doc's table. | `developer` |
| F5 | 🟡 | Verification Plan; §D10 driver validation | No named staged rollout. C8 (5k-episode paired measurement) and V8 (one block on a node) are block-scale, but nothing sequences "pilot store → run V1–V5 against it → then scale to 10 runs", and the driver's final validation is structural only (blocks present, no duplicate seeds, row counts). V1/V3/V4 are never stated to run against the **production** store. | Add an explicit phase list: (i) pilot ~25k episodes of one run on one node, run V1–V5 + D10 validation on that store; (ii) one full run; (iii) remaining runs. Run the cheap sampled checks (V1, V3, V4) on the final production store as an acceptance gate. | `senior-developer` |
| F6 | 🟡 | §D10 driver validation | The realised draws — the whole point of the store — get a 200-sample association check (V1) but no whole-store sanity check. A degenerate sampler, or a field wired to a constant that V1's sample happens to miss per-column, survives. | Add to the driver's final validation: every realised-draw column lies within the manifest's sampling bounds, and has > 1 distinct value wherever `low < high`. Cheap (columnar min/max/nunique) and catches constants, defaults, and out-of-range wiring at full scale. | `developer` |
| F7 | 🟢 | §D7 device table | GPU `batch_size = 8192` is claimed to need "~1.8 GB scan buffer"; §D5's own rate (609 B/env-step) gives 8192 × 500 × 609 B ≈ **2.4 GB**. Conclusion (fits an 11 GB 2080 Ti) unchanged; fix the number. | Correct the figure. | author |
| F8 | 🟢 | §D4.1 column 13 | `damage` cites "see **Open Question 1**", but the Open Questions section's sole item 1 is now observation precision — the damage question was decided (Decisions Taken #1). Stale pointer; could mislead the implementer into thinking `damage` is unsettled. | Point column 13 at Decisions Taken #1. | author |
| F9 | 🟢 | V1 | Exact float equality between vmapped and unbatched `jax_reset` is asserted "since it is the same computation". If a benign bitwise difference ever trips it, an implementer will quietly weaken to `allclose` — pre-state the policy (tolerance and its justification) now so the check cannot be softened ad hoc. | One sentence in V1. | author |
| F10 | 🟢 | `collect_worker.sh`; schema doc §5 | (a) The worker script should state the explicit interpreter path `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python` per the project conda rule (the `sweep_worker.sh` precedent presumably complies — say so). (b) The worked "per-episode bush-dwell **fraction**" reader snippet must label its output a 0–1 fraction, given the retracted `bush_dwell` units claim. | One line each. | author |

## Known-bug cross-check (five hard blockers)

| Known bug | Plan status |
|---|---|
| 1. Pre-`828b77e` scene-precedence bug → saved-config reload unfaithful | **NOT addressed** → F1 (Critical) |
| 2. Eval harness holds one fixed seed per checkpoint → silent episode repetition | **Addressed.** Seeds are `seed_base + episode_index` by construction (§D10); V1 replays 200 randomly-sampled recorded episodes from their *claimed* seed and asserts the recorded draws match — silent repetition makes recorded content disagree with the claimed seed, so V1 fails. This is a content check, not a "calls a different function" assertion. Residual strengthened by F6. |
| 3. Auto-aggregating roots blend stale data | **Addressed.** `env_fp` path partitioning + manifest hard-guard + atomic shards (§D3), V7 is a direct regression test for the 2026-07-04 incident, and a redone block is bit-identical by seed-purity, so even an accidental double-write cannot blend. |
| 4. `bush_dwell` units error / undecided episode-end event rule → derived measures smuggled into schema | **Addressed.** Store records raw per-step `agent_in_bush` occupancy only; no dwell fraction, no event aggregation in the schema; `reward_sum` is explicitly labelled data-not-metric. Minor residual: F10(b). |
| 5. `test_unified_parity.py` currently red | **NOT addressed** → F2 (Critical) |

Lower-severity trio: stale-params-inside-jit is squarely addressed (§A5, C1, V2); silent-unmodulated-agent and unpersisted model-size flags are **not** → F3.

## Unverified assumptions (❓)

1. **Reloading a run's saved resolved config through today's loader reconstructs the training world** (F1 — the conclusion of every future analysis rests on this).
2. **Environment code is unchanged between a run's training and its collection** (F2 — training-time git SHA is unrecorded anywhere).
3. **The restored checkpoint structurally matches the rebuilt model** (F3).
4. **`os.replace` is atomic and durable on this NAS mount.** The whole "a shard on disk is always complete" guarantee rides on it. V6 should be run on the NAS filesystem, not local disk; consider flush+fsync before rename (a node crash — not just a process SIGKILL — can leave a renamed-but-unflushed file that only the driver's full-read validation would catch).
5. **The 14.3 eps/s CPU throughput measured on one process holds with 16 workers per node writing to a shared NAS** (aggregate write bandwidth is trivial at ~5 MB/s, so low risk; V8 partially covers).

## Passes skipped

Pass 7 (empirical-claim soundness) — not an analysis verdict. Pass 6 controls/power/confounds — structurally inapplicable to a collection tool with no treatment arms; the applicable items (GPU feasibility, obs↔noise sync untouched, no budget wiring) were checked. Mechanical YAML validation of the spec template is `env-config-reviewer`'s gate at launch time, not duplicated here.

## Cost of being wrong

If F1 or F2 bites, the store faithfully records a million episodes per run of the agent in a world it was not trained in, the manifest fingerprints that wrong world as if it were truth, and every downstream "what did this training change do?" conclusion is wrong with no internal signal. Re-collection compute is cheap (~1.5 h wall for all ten runs); the unrecoverable cost is wrong conclusions silently entering analyses and, eventually, a paper.

---

*Reviewed by: plan-reviewer*
