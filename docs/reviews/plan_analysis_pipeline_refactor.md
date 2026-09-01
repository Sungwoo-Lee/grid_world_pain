---
title: Plan review — analysis-pipeline refactor (core/ + study manifests)
topic: reviews
status: active
created: 2026-09-01
last_updated: 2026-09-01
---

# Plan review: the analysis-pipeline refactor

## Verdict

**NOT READY.** The plan proposes to pull the shared trajectory-store sweep out of three analysis
programs into one `core/` library, and to prove the port safe by requiring every migrated output to
be byte-for-byte identical to the current outputs. The direction is right and matches what the user
asked for (stop re-reading the data-loading code for every new analysis). But the plan's two
load-bearing sections contradict each other: the redesigned "declarative accumulator" sweep
necessarily changes the order in which floating-point numbers are added up, and a changed addition
order changes the last digits of the results — so the byte-identical test the plan calls "the whole
safety argument" is guaranteed to fail for a *correct* port of the redesign. I verified this
empirically: NumPy's three summation primitives (`reduceat`, `bincount`, `add.at`), which the three
existing scripts use interchangeably for the same logical operation, give bitwise-different sums on
identical data. As written, the plan has no verification protocol that can pass, and the likely
failure branch — quietly loosening the comparison tolerance until it passes — is exactly the kind of
renegotiated check that lets a real porting bug through into numbers that back a published analysis.

Secondary problems: the motivating "12 of 21 guards missing" table is wrong in at least two cells
(regex-generated, as the author suspected); the day-long cost estimate ignores the other two script
folders and the verification re-runs; and the design has no seed dimension even though the obvious
next study is a 2–3-seed replication.

**Exit conditions to flip the verdict** are listed at the end — all are plan edits, no code needed.

## Findings

Severity legend: 🔴 Critical = fix before going further · 🟡 Moderate = likely costs a re-run ·
🟢 Low = cosmetic · ❓ Open = an assumption nobody has verified yet.

| # | Sev | Location | Issue | Suggested fix |
|---|-----|----------|-------|---------------|
| 1 | 🔴 | Plan §Migration step 2 vs §"The load-bearing idea" | Byte-identity and the declarative-accumulator redesign are mutually exclusive. `build_arm_data.py` accumulates per-episode float sums with `np.add.reduceat` (`build_arm_data.py:141-143`) while `hiding_drivers.py` accumulates the same logical quantities with `np.bincount` (`hiding_drivers.py:210-217`); any unified `scan.PerEpisode(reduce="sum")` must pick one primitive, and the three primitives are bitwise-inequivalent on float data (verified empirically on this env's NumPy). Non-dyadic columns (`injury_level`, `nutrition`, the nociception convolution) will differ in the last ulp; JSON/CSV/npz bytes follow. A gate that fails on every correct port gets renegotiated ad hoc — on the very numbers it guards. | State a criterion that is both achievable and meaningful: (a) **bit-exact** for every integer-valued accumulator (counts, 0/1 sums — order-independent, so exactness is free and any mismatch is a real bug); (b) for real-valued accumulators, `np.load`/`json.load` then element-wise agreement at a pre-registered tight tolerance (e.g. `rtol=1e-12`), with the tolerance justified in the plan, not chosen after a failure; (c) figures verified on the *plotted data*, not PNG bytes, wherever plotting moves into `core/plot.py`. Alternatively: two-phase port — transliteration first (byte-exact achievable; verified npz and PNG are reproducible byte-for-byte on this env), redesign second with criterion (a)/(b). Either way the plan must say which. |
| 2 | 🟡 | Plan §"The problem, in one table" | The guard table is wrong in ≥2 of 21 cells, in both directions. `hiding_drivers.py` **does** exclude the t=0 row from all rates (`m = t >= 1`, `hiding_drivers.py:208`) and **does** filter absent animals (NaN from `mean_over` + `np.isfinite` masks in every fit, `hiding_drivers.py:265,286`; `& act[gi]` on proximity, `:205`; inf→NaN for spawn distance, `:250`) — both marked "missing". The "absent animals" cell for `build_time_course.py` is structurally inapplicable (it reads no animal columns at all), not missing. Conversely "seeds contiguous, no duplicates" is marked "yes" for `hiding_drivers`, whose check (`hiding_drivers.py:136`) can be fooled by a duplicate+gap pair — `build_arm_data.py:75` additionally checks uniqueness. Honest count is ~8–9 missing of ~20 applicable, and the claim "every bug this analysis hit lives in an empty cell" no longer holds cell-by-cell (the NaN-binning bug lived in `build_arm_data`, whose cell now reads "yes"). | Rebuild the table by reading the code (this review's per-cell audit can be lifted directly). The refactor's motivation survives — the duplication claims all verified true, and `find_store` has genuinely drifted (single-root in `figures/_common.py:66` vs multi-root in `hiding_drivers.py:80`) — so the fix is honesty, not abandonment. |
| 3 | 🟡 | Plan §"The load-bearing idea" | The declarative API as sketched cannot express most of what the three sweeps actually do without an escape hatch: 2-D crossed bins (distance × injury grids, `build_arm_data.py:106-117`), per-episode×bin matrices (`hiding_drivers.py:215-217`), composite geometry ("rabbit near AND NOT predator near", `hiding_drivers.py:206`; t=0-row bush/predator distances, `:218-227`), data-dependent quantile edges (`build_arm_data.py:183-185`), the early-window mask (`:177`), and above all the within-episode nociception convolution (`build_time_course.py:48-62`), which is sequential over steps with a reset-boundary guard and fits no per-row accumulator. Either the DSL grows ~10 primitives (a new language to learn — the indirection cost the plan itself flags) or ports go through a raw-callable hatch that becomes the normal path. | Downgrade the deliverable honestly: a shared **sweep driver** that owns shard iteration, the seed/alignment/step-count asserts, `estart`, the t=0 mask and the previous-row index — and hands those precomputed to a per-study accumulate callback — satisfies the user's actual request ("never re-read the loading and basic pipeline") with none of the DSL risk. Ship `PerEpisode`/`Binned` as optional conveniences on top, plus `perceived_nociception` as a named `core/` function carrying its boundary-condition regression test (the `src > estart` guard was a real, published-impact bug — Known Bugs, "hiding-drivers analysis reconstruction", 2026-08-25). |
| 4 | 🟡 | Plan §Proposed layout + §Migration (omission) | Scope of the other consumers is undecided. `scripts/analysis/figures/` (9 scripts) and `scripts/analysis/supplementary/` (19 scripts, not the 9 the dependency map still claims) hold the third copies of `slot_layout`/`smell_channels`/`listcol`/`nociception_kernel` — the plan's "three divergent copies" collapse to two, not one, if they are out of scope, and the day estimate is fiction if they are in. They are referenced from `docs/experiments/active/trajectory_factors/`. Note `hiding_drivers.py` is not only the old study's tool: `lad06`/`lad07` read its CSVs (SCRIPTS_DEPENDENCY_MAP §`scripts/analysis/ladder/*.py` row), so porting it sits on the ladder's critical path. | Decide and write it down. Reasonable call: port `hiding_drivers` (live producer), leave `figures/` and `supplementary/` frozen as archived reproductions with a README pointer at `core/` — but then stop claiming all three copies die. |
| 5 | 🟡 | Plan (omission) | The maintenance contract is not mentioned. Moving `build_*.py` + 15 figure scripts into `studies/sensor_ladder/` and adding `core/` adds/moves ~25 files under `scripts/` — SCRIPTS_DEPENDENCY_MAP.md must be updated in the same change (its §0 repo-root-depth hazard is also exactly the trap `from analysis.core import scan` walks into: the new package layout needs a stated import mechanism — package `__init__.py`s + a documented `sys.path`/cwd convention). The map has already rotted once in this area (says "9 files" for a 19-file `supplementary/`). | Add a File Changes section naming the map update (and the `docs/experiments/active/sensor_ladder/` path references to the moved scripts) as same-commit obligations. |
| 6 | 🟡 | Plan §Migration steps 2–3 | The golden files are not protected. The test compares new output against the existing `results/analysis/ladder/*.json` — but the ported code, run naively, writes to the same paths and destroys the golden before the comparison; `results/` is gitignored on a NAS with no second copy. Regeneration is only sweep-hours, not the 3-hour collection, but the plan's own safety protocol should not depend on remembering this. Also unstated: whether the *current* code still reproduces those files (the store gained a second collection pass; the honest baseline is old-code-fresh-run vs new-code-fresh-run, same day). | Write ported output to a scratch path (or `cp -a results/analysis/ladder /tmp/golden-$(date +%s)` first), and regenerate the golden with the old code at port time. |
| 7 | 🟡 | Plan §Estimated cost | "Roughly a day" budgets zero hours for the verification re-runs the plan itself mandates: three sweeps × 14 arms × full store, twice (golden + port) for build_arm_data (~minutes-to-tens-of-minutes per arm over the 452 GB store), plus 15 figure ports each gated on a comparison, plus the dependency-map and doc-path updates of finding 5. Two to four days is the honest number even with `figures/`/`supplementary/` out of scope. | Re-estimate, or explicitly stage it (core + build_arm_data port day 1; the rest as it's touched). |
| 8 | ❓ | Plan (omission) | No seed dimension. The manifest maps each arm to exactly one run (`RUNS` glob, `arm_runs()` in `_ladder.py:85`), and the obvious next experiment is a 2–3-seed replication that would be `core/`'s first real client. Designing store discovery and accumulators around `(unit, seed)` with the ladder as the 1-seed case costs little now and a rework later if skipped. Timing cuts both ways: the replication is also the argument for doing the refactor *now*, before a third copy of the sweep gets written for it — the plan should claim that, and design for it. | State the seed model in the manifest section; sequence the refactor to land before the replication analysis. (Whether to spend the days at all is `pi`'s call, not this review's.) |
| 9 | 🟢 | Plan table row 7 | "Population matches the other products" — the only guard missing everywhere — is never defined. What does `store.open_run` actually assert: episode count vs the manifest? seed range vs the sibling product's JSON? | One sentence defining the check. |

## Assumption register

- **Unverified**: that re-running the *current* code reproduces the existing golden files byte-for-byte (finding 6). Must be established before step 2 means anything.
- **Unverified**: that the replication study will consume these accumulators rather than needing new ones (finding 8) — this is what decides whether the DSL earns its complexity.
- **Verified by this review**: the duplication counts (`slot_layout` ×3, `smell_channels` ×3, `listcol` ×3, `nociception_kernel` ×2, `find_store(s)` ×2, drifted) are all real.
- **Verified by this review**: npz (`np.savez_compressed`), matplotlib PNG (same version, 3.10.8), and `json.dump` outputs are byte-reproducible across runs on this env — so byte-identity *is* achievable for a pure transliteration; it is only the redesign that breaks it.

## Prior-art check

Known Bugs registry (grepped for analysis/ladder/store rows): no collision; two rows are directly
relevant and should be cited by the plan — the fixed reconstruction-fidelity bug (the `src > estart`
nociception guard, found by an earlier adversarial review; its regression test belongs in `core/`
next to the function, finding 3) and the still-LATENT `supplementary/timectrl.py` row (bins by the
contemporaneous body-state value; if `supplementary/` is ever ported, that bug ports with it). No
prior or competing plan exists under `docs/develop/` for this refactor.

Passes skipped: experiment-plan specifics and empirical-claim soundness — this is an engineering
plan, not an experiment design or analysis verdict.

## Cost of being wrong

If the plan proceeds as written: two to four days instead of one, a verification gate that fails on
correct code and then gets loosened under deadline pressure on the exact aggregates that back the
published ladder analysis, and a core API the very next study cannot use without rework. No
unrecoverable data is at risk (finding 6's golden files regenerate in sweep-hours).

## Exit conditions (what flips this to SOUND)

1. Replace the byte-identity criterion per finding 1 (or commit to the two-phase port) — the one 🔴.
2. Correct the motivating table (finding 2) and decide the `figures/`/`supplementary/` scope (finding 4).
3. Name the SCRIPTS_DEPENDENCY_MAP + doc-path updates as same-commit obligations (finding 5).
4. Add the golden-protection step (finding 6) and state the seed model (finding 8).

Reviewed by: plan-reviewer
