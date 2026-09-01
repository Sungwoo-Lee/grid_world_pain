---
title: Plan review — analysis-pipeline refactor (core/ + study manifests)
topic: reviews
status: active
created: 2026-09-01
last_updated: 2026-09-01
---

# Plan review: the analysis-pipeline refactor

This file now holds **two review rounds**. Round 2 (current, top) reviews revision 2 of the plan;
round 1 (bottom, superseded) reviewed revision 1 and returned NOT READY on a verification gate that
could not pass. Revision 2 genuinely fixed that and six of the seven secondary findings — but the
redesign that fixed them introduced two new load-bearing contradictions of its own.

---

## Round 2 — Verdict

**NOT READY** — but closer, and for different reasons than last time.

The plan proposes one shared scan of the trajectory store (a "driver" that owns shard order,
episode boundaries, and the alignment checks, handing each study a validated frame) to replace the
three hand-written scans the ladder analysis currently runs on. Revision 2 resolved round 1's
Critical honestly: the impossible byte-identity gate became a four-tier comparison criterion, and
this review verified **empirically, on a real shard of the ladder store**, that the criterion is
achievable with four orders of magnitude of margin — the fear that float tolerance would get
renegotiated is now unfounded.

The two new Criticals are both self-contradictions inside revision 2's own design:

1. **The driver's guaranteed conventions and the golden-comparison gate cannot both hold.** The
   plan promises every study "gets" the previous-row shift and the t=0 exclusion — but the gate
   demands ported output match the old scans, and two of the three old scans *don't follow those
   conventions* (the plan's own guard table says so, in its "no" cells). A port that takes the
   guarantee changes published numbers and fails the gate; a port that passes the gate reproduces
   the old behaviour bug-for-bug and falsifies the guarantee. The plan never says which yields.
2. **The population contract deadlocks the workflow it was invented for.** Opening the store
   asserts its episode count matches every derived product already on disk — so after the next
   collection pass (the ladder has already had two), every product is stale and the rebuild that
   would refresh them can no longer open the store. The check belongs at *read* time, on
   population-stamped products — the pattern `_ladder.load_time_course` already implements.

Both are plan edits, not code. Exit conditions at the end of this round's section.

Severity legend: 🔴 Critical = fix before going further · 🟡 Moderate = likely costs a re-run ·
🟢 Low = cosmetic · ❓ Open = an assumption nobody has verified yet.

## Round 2 — Findings

| # | Sev | Location | Issue | Suggested fix |
|---|-----|----------|-------|---------------|
| R2-1 | 🔴 | Plan §"A scan driver with a callback" ("The driver guarantees, for every study, forever … the t=0 exclusion, the previous-row shift") vs §Verification + §Migration 3–4 | The guarantee and the golden gate contradict each other in exactly the guard table's "no" cells. Concrete case: `build_time_course.py`'s dose-response pairs the outcome row's bush state with the **same row's** perceived-nociception value (`build_time_course.py:332-334`) — contemporaneous, not the previous-row signal the agent actually decided on. That is the same class as the **LATENT** `supplementary/timectrl.py` row in Known Bugs ("bins by the contemporaneous body-state value"), and `dose_bush` feeds the published lad09 figure. A port that receives the guaranteed previous-row shift changes those numbers → fails the tier gate; a port that reproduces the old indexing passes the gate → "a check every study gets, including the ones already written" is false, and the refactor silently blesses a known bug class in a published product. Same conflict, milder, for `hiding_drivers`' contemporaneous within-episode bins (`hiding_drivers.py:214` — though there it is arguably by design for an episode-level factor analysis). | Add a **divergence register** to the plan: for every "no" cell of the guard table, an explicit decision — *reproduce bug-for-bug* (gate on the reproduction; file the divergence as a follow-up change with a pre-computed expected diff, adjudicated separately) or *deliberately deviate* (gate waived for that product, diff documented before porting). And reword the guarantee: the driver **asserts** data properties (shard order, boundaries, contiguity, population) and **provides** conventions (`is_step`, `prev`, masks) that a callback must opt into — it cannot enforce their use (see R2-4). |
| R2-2 | 🔴 | Plan §"The population contract" | As specified, `store.open_run()` asserts the store's episode count "matches every other derived product already on disk". After any store growth every product is stale, so the rebuild that would refresh them cannot open the store — regeneration requires deleting all products first, after which the check protects nothing. "Regenerate one product only" is impossible by construction. Also undefined: which files count as "derived products" (a registry? which field of a CSV that records no episode count?), first-run behaviour (no products yet — vacuous pass?), and concurrent regeneration. The predictable resolution under pressure is deleting the assert — the flagship guard, invented to catch the stale-time-course incident. | Split the contract. `open_run()` asserts **store-internal** properties only (contiguity, uniqueness, one-shard episodes, step count vs episode table). `provenance.py` stamps every written product with its population (count + seed range). The **cross-product** check runs at *read* time in `core`'s loaders — the pattern `_ladder.load_time_course` (`_ladder.py:305-340`) already implements correctly today, including the stale-mix error message. Name the product registry and the single-product-regeneration flow in the plan. |
| R2-3 | 🟡 | Plan §"The evidence" ("Nine of seventeen applicable cells are empty") | The headline count contradicts the printed table *again*. The table has 6 rows × 3 = 18 cells, 1 n/a → 17 applicable; non-"yes" cells number 6 (plus 1 "partial") — not nine. The population-contract row was evidently dropped from the table after the count was computed: the prose still refers to "the missing bottom-row check" that the table no longer contains, and 6 + the dropped row's 3 = the claimed 9 (of 20, not 17). Round 1's finding 2 recurs in miniature — the correction blockquote says the table was read from the code, but the count wasn't read from the table. | Restore the population row (no / no / no — it is the strongest motivator, and R2-2 needs it in view) and restate: 9 of 20 applicable. All other cells were re-verified independently this round and are right, except row 5 "time course: yes" — see R2-1; that cell is at best "partial" (the per-t curves keyed to the exogenous start injury need no shift; the dose-response does and lacks it). |
| R2-4 | 🟡 | Plan §"A scan driver with a callback" (the `collect(frame, acc)` sketch) | The frame contract shown (`step`, `prev`, `is_step`, `present`) cannot port two of the three sweeps. `hiding_drivers.aggregate` needs the **global episode index** and a **t=0-row view** (`gidx`, `st` — reset-row reads at `hiding_drivers.py:200,219-227`) and **per-slot per-row** active masks (`& act[gi]`, `:205`) — not an episode-level `present()`. `build_time_course` needs `estart` for the convolution and the cross-shard `inj0` fill. Both need `nep`/`seed0` to size accumulators before the scan — and the sketch is ambiguous about who allocates `acc` (it appears as both a parameter and a return value). Exposing all of this is fine and necessary; unstated, it gets discovered ad hoc mid-port. | Add a frame-contract audit table to the plan: field × which of the three sweeps needs it. And make convention breaches *auditable* since they cannot be structural: column access via `frame.step('col')` / `frame.prev('col')` accessors, raw per-row arrays only behind an explicit `frame.raw('col')` documented as unguarded — then every deviation from the guaranteed conventions is one grep away. |
| R2-5 | 🟡 | Plan §Scope + §Migration steps 5–7 | Step 5 deletes `hiding_drivers.py`, which breaks the freeze rationale of step 7: `supplementary/README.md` names `hiding_drivers.py` as the producer of the a01 factor ranking, so "kept to reproduce that study as published" fails the day its producer is deleted. Separately, the ported version must keep writing `results/analysis/lad/<arm>/multivariate.csv` **exactly**, because `lad06`/`lad07` filter arms by CSV existence (`lad06_world_factor_map.py:41`) — a changed output path does not error, it silently thins the published figure. | State in the plan: (a) output paths of all ported producers are preserved verbatim; (b) `hiding_drivers.py` is either frozen-copied alongside `supplementary/` or its port stays run-agnostic and CLI-compatible, with the a01 doc and `supplementary/README.md` updated in the same change. |
| R2-6 | 🟡 | Plan §Verification ("`results/analysis/ladder/` is copied aside") | Golden protection and fresh-baseline regeneration name only `results/analysis/ladder/`. The `hiding_drivers` goldens live elsewhere: `results/analysis/lad/<arm>/` (the ladder invocation `lad06` reads) and `results/analysis/hiding_drivers/<tag>/` (the script's default). Step 2's "regenerate the goldens" must enumerate all three roots or the step-4 comparison has no protected baseline. | Name all golden roots explicitly in steps 2–3. |
| R2-7 | 🟡 | Plan §Verification, tier 1 ("integer-valued accumulators … bit-exact") | The tier is value-defined, not dtype-defined (right call — `agent_in_bush` is stored bool, but 0/1 sums land in float64 accumulators), yet nobody is named to classify each output field into a tier, or when. Classification done at diff time is the renegotiation risk in miniature. | Enumerate the tier per output field (per npz key / JSON field) in the plan or the comparison harness's manifest, and have the harness assert `x == np.round(x)` on every tier-1 field — a misclassified field then fails loudly instead of getting argued. |
| R2-8 | 🟢 | Plan §"The evidence"; §Scope; sketch | Nits, once: the store-finder duplication is actually ×3, not ×2 — `_ladder.arm_stores` (`_ladder.py:97`) is a third, differently-globbed implementation (motivation understated, harmlessly). The freeze README should cite the LATENT `timectrl.py` Known-Bugs row so the frozen bug isn't rediscovered as new. `store.open_run()` as sketched lacks the checkpoint selector `hiding_drivers` has. | One line each. |

## Round 2 — What revision 1's findings became (scorecard)

| Round-1 finding | Status in revision 2 |
|---|---|
| 1 🔴 byte-identity vs redesign | **Fixed, verified.** DSL → driver+callback (the review's own suggested downgrade); byte gate → four tiers, pre-registered. Empirically checked this round (below): achievable with margin. Residue: tier classification ownership (R2-7). |
| 2 🟡 wrong guard table | **Fixed in the cells, broken in the count.** All six rows re-audited independently against the code this round; every cell confirmed (`hiding_drivers` uniqueness "partial" ✓, time-course absent-animals "n/a" ✓ — its column list reads no animal columns) **except** row 5's time-course "yes" (R2-1) — and the "nine of seventeen" headline doesn't match the printed table (R2-3). |
| 3 🟡 DSL can't express the sweeps | **Fixed.** Driver+callback per the review's suggestion; `perceived_nociception` moves to `core/env.py` with the boundary regression test. Residue: frame contract under-specified (R2-4). |
| 4 🟡 undecided scope | **Fixed.** `figures/` + `supplementary/` frozen with README notes; the two-copies consequence stated honestly; `supplementary/` count corrected to 19 (verified: 19 `.py` files on disk; dependency map still says 9). Residue: deleting `hiding_drivers.py` contradicts the freeze rationale (R2-5). |
| 5 🟡 dependency map unmentioned | **Fixed.** Named as mandatory same-commit update, with the §0 depth hazard and the stale `supplementary/` row called out; `sensor_ladder.md` + `artifact_template.html` path references named too. |
| 6 🟡 goldens unprotected | **Mostly fixed.** Scratch path / copy-aside + fresh old-code-today baseline are in. Residue: only one of three golden roots named (R2-6). |
| 7 🟡 cost | **Fixed.** 2–4 days, with the revision-1 omission named. |
| 8 ❓ no seed dimension | **Fixed.** Manifest keyed `(unit, seed)` with the ladder as the one-seed case; the PI framing (build before the replication writes a fourth scan) is claimed as the plan suggests. |
| 9 🟢 population check undefined | **Defined — and now over-defined into a deadlock** (R2-2). |

## Round 2 — Empirical verification of the tier criterion

Run this round on a real shard (`trajectories_lad/…_lad_A_baseline_…/steps_00000.parquet`,
870,172 rows, 5,000 episodes, this env's NumPy):

- **Per-episode sums of `injury_level`, `nutrition`, `agent_in_bush` are bit-exact across all
  three primitives** (`np.add.reduceat`, `np.bincount`, `np.add.at`) **and across per-segment
  `np.sum`** (pairwise order). Reason: the store's float columns are float32-origin values summed
  in float64 — every partial sum over an episode (≤ 501 rows, values ≤ 100) is exactly
  representable, so accumulation order cannot matter. Round 1's "bitwise-inequivalent" result was
  obtained on synthetic float64 data; on the real store the primitives agree exactly.
- **The most order-sensitive quantity in the pipeline** — the nociception convolution with its
  non-dyadic kernel weights, recomputed with the j-loop reversed — differs by at most
  **6.4e-16 relative**; per-episode sums of it by 6.9e-16.

Consequence: `rtol = 1e-12` is not another number that will get renegotiated — it has ~4 orders of
magnitude of headroom, and most tier-2 fields will in practice come out bit-exact. If anything the
plan could tighten tier 2 to "expected bit-exact, `rtol=1e-12` accepted with a note". The only
gate failures to expect are **real semantic differences** — which is exactly what R2-1's
divergence register exists to pre-declare.

## Round 2 — Assumption register

- **Verified this round**: the tier criterion is achievable (above); `supplementary/` = 19 `.py`,
  `figures/` = 9, 15 `lad*` figure scripts; `lad06`/`lad07` read `results/analysis/lad/<arm>/multivariate.csv`
  and silently drop arms whose CSV is missing; the dependency map's `supplementary/` row is stale
  as the plan claims.
- **Unverified (carried)**: that the seed-replication study will consume these accumulators — now
  less load-bearing (the callback replaces the DSL) but still what decides the `studies/` layout.
- **Unverified (new)**: that all 14 arms' stores pass the driver's *stricter* asserts (e.g. the
  uniqueness check `hiding_drivers` lacks). `build_arm_data` already enforces uniqueness on the
  same stores, so likely yes — but confirm in step 2, before any port, so an assert firing is a
  data finding rather than a port blocker.

## Round 2 — Prior-art check

Known Bugs registry re-grepped this round: the **LATENT** `supplementary/timectrl.py` row
(contemporaneous-value binning) is now directly implicated by R2-1 — the same class exists in
`build_time_course.py`'s dose-response, which is in scope and feeds a published figure. That
connection is new information the registry does not record; **`bug-curator` should be spawned** to
add a row (or extend the timectrl row) for `build_time_course.py:332-334` once the plan's owner
confirms the reading. The fixed `src > estart` row is correctly cited by the plan. No competing
plan docs found.

Passes skipped: experiment-plan specifics (pass 6) and empirical-claim soundness (pass 7) — this
is an engineering plan.

## Round 2 — Cost of being wrong

If implemented as written: at step 4 the time-course port hits the guarantee-vs-gate contradiction
and someone resolves it ad hoc on numbers behind the published ladder page — the exact
renegotiation-under-pressure this refactor exists to prevent — with the extra hazard that a
known-LATENT bug class gets blessed into `core/` as verified-correct behaviour; and the first
store top-up after migration bricks the rebuild path until someone deletes the population assert,
the flagship guard. No unrecoverable data at risk; the cost is days, plus one silently wrong
published dose-response product.

## Round 2 — Exit conditions (what flips this to SOUND)

1. **R2-1**: add the per-"no"-cell divergence register (reproduce-then-fix, with pre-declared
   expected diffs) and reword the driver's promise as asserts-plus-provisions.
2. **R2-2**: move the cross-product population check to read time on population-stamped products;
   define the product registry, first-run, and single-product-regeneration semantics.

The 🟡s (R2-3–R2-7) should be folded into the same edit; none of them alone blocks.

Reviewed by: plan-reviewer (round 2, 2026-09-01)

---
---

# Round 1 review (superseded — reviewed revision 1 of the plan)

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

*(Round-2 correction: the bitwise-different result was obtained on synthetic float64 data; on the
real store's float32-origin columns the three primitives agree bit-exactly. The finding's
conclusion — byte-identity was the wrong gate for a redesign — stands; the mechanism was
overstated. See Round 2 §Empirical verification.)*

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
