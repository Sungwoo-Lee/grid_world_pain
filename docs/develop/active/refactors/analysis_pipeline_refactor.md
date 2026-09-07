---
title: A reusable analysis pipeline — one guarded scan instead of three
topic: refactors
status: active
created: 2026-09-01
last_updated: 2026-09-04
revision: 4 — resumed by the user 2026-09-04; round-3 review folded in; implementation under way
---

# A reusable analysis pipeline

**Shareable page:** https://claude.ai/code/artifact/db00bfca-be2d-436e-bdb9-b19c894f663d
&mdash; the visualization of this plan, source at [`scripts/analysis/pipeline_layout.html`](../../../../scripts/analysis/pipeline_layout.html).
**Republish:** publish that file to the URL above. From a session that did not publish it, read
the URL first and pass it as `url`; publishing without it makes a separate artifact.

---

## Status: IN PROGRESS, resumed 2026-09-04 by the user

**Resumed by explicit user decision on 2026-09-04**, not by one of the triggers below. The user was
told nothing was built and asked for it to be implemented, step by step, with each step gated on
reproducing every current published number. The triggers stay recorded because they are still the
right test for *starting* this work; they simply were not what happened.

Round 3 of review returned **SOUND WITH CONCERNS** — both round-2 Criticals verified genuinely
closed, with no new Critical. Its findings are folded in below and its full text is in the Feedback
section.

It is parked because the case for doing it *now* got weaker while the case for writing it down got
stronger. The refactor produces no new scientific result. The study it would have protected is
finished and published. And all three bugs it was justified by have since been fixed in the existing
scripts — the absent-animal binning, the stale-population rebuild, and the contemporaneous
dose-response pairing (`c20ff5db`), which this very plan's review is what surfaced. The duplication
it targets is real and has already drifted, but drift costs nothing until someone writes against it.

**What should wake it up**, in rough order of how strongly:

1. **A fourth program needs to scan the store.** This is the real trigger. Writing a fourth
   hand-rolled scan means choosing which of the seven guards to remember, which is exactly how the
   first three ended up disagreeing.
2. **A fifth instance of a guard-class bug**, or a first instance in a *new* study. The class is
   recurrence-prone by demonstration, not by argument.
3. **Someone needs to re-run a published figure and cannot**, because the store finder's three
   implementations have drifted further apart.

**What should NOT wake it up**: that the plan exists and is nearly ready. It has been through two
reviews and would need a third; resuming on momentum is how a no-output refactor eats a week.

**If it is resumed**, start by re-reviewing revision 3 rather than implementing it. Revisions 1 and 2
were both rejected, and revision 3's own guard-count table was wrong three times before it was
generated instead of typed — this document has not earned the benefit of the doubt.

## What this proposes, and why

Every study in this project has written its own scan of the trajectory store: open the shards in
order, work out where each episode begins, line up predictors with outcomes, filter, accumulate.
Three such scans now exist — `hiding_drivers.py`, `ladder/build_arm_data.py`,
`ladder/build_time_course.py` — written months apart. This proposes replacing them with **one scan
that every study calls**, plus a small per-study folder holding only that study's facts.

The request behind it: *"I don't want to re-read the data loading and basic pipeline for every
analysis."*

## The evidence

**Duplication, verified by reading the code.** `slot_layout` is implemented **three** times,
`smell_channels` three times, `listcol` three times, `nociception_kernel` twice, and the store finder
**three** times (`figures/_common.find_store`, `hiding_drivers.find_stores`, `_ladder.arm_stores`) —
and those **have already drifted**: one handles a single root, the others several, which is why the
ladder's second collection pass needed a new implementation rather than reusing what existed.

**Divergent safety checks.** Each scan performs whichever checks its author had in mind. Read from
the code, not pattern-matched:

| check | `build_arm_data` | `build_time_course` | `hiding_drivers` |
|---|---|---|---|
| seeds contiguous, no duplicates | yes | no | partial (no uniqueness check) |
| episodes never split across shards | yes | no | yes |
| step count matches the episode table | yes | no | no |
| the `t=0` row is not a step | yes | no | yes (`m = t >= 1`, line 208) |
| predictors read from the previous row | yes | **was no — fixed 2026-09-01** | no |
| absent animals excluded | yes | n/a (reads no animal columns) | yes |
| population matches the other derived products | no | no | no |

**Eleven of twenty applicable cells were unsatisfied** when this plan was written, counting `partial`
and the row fixed today. That count is now *generated* from an explicit cell list rather than typed:
revision 1 claimed twelve of twenty-one, revision 2 claimed nine of seventeen, and both were wrong —
the second because the population row was dropped from the table after the count was taken. Three
wrong counts of one small table is itself an argument for generating the numbers that go into
documents. (The second review independently arrived at nine of twenty; recounting cell by cell gives
eleven — six `no` plus one `partial` in the printed six-row table, plus three for the restored
population row, plus the time-course previous-row cell that the review itself showed was mislabelled
`yes`. With that cell now fixed in code, **ten** remain.) Two of the three bugs this analysis hit sit in an
empty cell: the time course was rebuilt from stale data because nothing cross-checked its population
against the other products, and episodes containing no predator became the comparison group because
nothing filtered absent animals.

> **Correction.** Revision 1 of this document claimed twelve of twenty-one cells were empty, and that
> *every* bug sat in one. That table was generated by pattern-matching the source and was wrong in
> both directions — `hiding_drivers.py` does exclude the `t=0` row and does filter absent animals; the
> patterns missed both because they use different spellings. The table above was read from the code.
> The duplication claims were all verified true, so the motivation stands, but it was overstated.

**How much of a figure script is its own analysis.** By line count, a typical one (`lad05`, 87 lines)
is about 10% analysis; the rest is docstring, plotting, provenance and boilerplate. And splitting
`_ladder.py` by hand, **8 of its 36 exports are study-agnostic** and 28 are ladder facts. That ratio is
the shape of the fix: a thin study manifest on a thick shared core.

## Design

```
scripts/analysis/
  core/                     # study-agnostic. Nothing here knows what an "arm" is.
    store.py                # shard discovery across collection passes; the population contract
    scan.py                 # THE scan: one loop, every check, always
    env.py                  # anything derived from a run's saved config
    stats.py                # rates, pooled contrasts, curves, the quasi-binomial fit
    provenance.py           # record_samples, population, the figure<->script contract
    plot.py                 # house style, group colouring, label-fit assertion
    publish.py              # the artifact builder and its build-time assertions
  studies/
    sensor_ladder/
      study.py              # the manifest: runs, unit names, bin edges, groupings. FACTS.
      collect.py            # the callback: given one validated frame, add to my accumulators
      figNN_*.py            # one per figure, as now
```

### A scan driver with a callback — not a mini-language

Revision 1 proposed a declarative accumulator API: a study would *describe* what to count and never
write a loop. The review showed that will not survive contact with what the existing sweeps actually
do — crossed two-dimensional grids, per-episode × bin matrices, composite geometry conditions
("rabbit near AND NOT predator near"), quantile-derived bin edges, and above all
`perceived_nociception`, a sequential within-episode convolution with a reset boundary that no
per-row primitive expresses. Either the language grows a dozen constructs, or the escape hatch
becomes the normal path — and an abstraction everybody bypasses is worse than none.

So `scan.sweep()` is a **driver**. It owns the parts that have gone wrong before and hands the study
a frame it can trust:

```python
from analysis.core import store, scan

st = store.open_run(run_dir, roots=[...])        # asserts the population contract

def collect(frame, acc):
    """frame is already validated. frame.step is the outcome row; frame.prev is the row the
    action was chosen on; frame.is_step masks out t=0; frame.present('predator') masks
    episodes with no predator. What happens here is ordinary NumPy, in the study's folder."""
    ...

acc = scan.sweep(st, columns=[...], on_shard=collect)
```

The driver **asserts** data properties (shard order and alignment, seed contiguity and uniqueness,
the step-count cross-check) and **provides** conventions the callback must use (`frame.is_step`,
`frame.prev`, `frame.present(...)`). Revision 2 blurred that distinction: an assert cannot be
bypassed, a provision can — a callback is free to index raw arrays. The guarantee is therefore
**auditable, not structural**. Raw arrays sit behind an explicit `frame.raw()`, so a breach is one
`grep` away, and a frame-contract audit table records which accessor each ported sweep uses.

> **A contradiction the second review found, and how it was resolved.** Revision 2 claimed the driver
> would give every study the previous-row convention "including the ones already written". But
> `build_time_course` binned the perceptual dose-response **contemporaneously** — pairing the
> nociception at row *t* with the bush state at row *t*, when the action producing row *t* was chosen
> while feeling row *t−1*. Porting it with the guaranteed shift would change a **published** number
> and fail the gate; porting it bug-for-bug would pass the gate while making the guarantee false.
> Rather than carry that into the refactor, **the bug was fixed at source on 2026-09-01**, before any
> porting. Measured impact, after rebuilding all fourteen arms: the published panel-C range moves
> from 12.6–17.8% bush hiding in the lowest quarter of felt nociception and 31.3–49.6% in the highest,
> to 12.6–17.9% and 31.7–49.5%. The largest single change is 0.4 percentage points. It is small
> because the felt signal is a twelve-step convolution and adjacent rows barely differ — but the
> convention was wrong, and the smallness was not knowable before measuring.
>
> Any *remaining* cell where porting a guard would move a published number goes into a **divergence
> register**: reproduce the old behaviour, gate on the reproduction, then fix it afterwards as an
> adjudicated change with a pre-declared expected diff. What must not happen is the gate being
> renegotiated at porting time on numbers behind a published page.

**The divergence register, populated.** Revision 3 defined this register and then left it empty,
which is the same as not having one. Round 3 of review established that the register can only ever
contain *provision* cells, and that there are exactly two — the asserts (contiguity, uniqueness,
one-shard episodes, step count) either pass or fire on the data, so adopting one cannot move a
number and cannot fight the gate. That is what actually confines the R2-1 contradiction rather than
relocating it.

| cell | why it diverges | decision |
|---|---|---|
| `build_time_course`, "the `t=0` row is not a step" | The t=0 point is **plotted**, and it carries finding 5's headline — the injury gap is largest at step 0, in 14 of 14 arms. Excluding it would delete the observation the figure exists to make. | **Not a divergence — by design.** Maps to `frame.initial`, which is a first-class accessor, not an escape. No register row needed beyond this one. |
| `hiding_drivers.py:214`, per-step injury bins | Bins contemporaneously, and feeds the injury cross-tabs in `summary.json` that are published in the a01 document. | **Reproduce bug-for-bug, gate on the reproduction.** Adjudicate afterwards as its own change with a pre-declared expected diff — the same treatment the dose-response bug got, except that one was fixed before porting because nothing else depended on its old value. |

**What the frame must actually expose.** Revision 2 named three accessors, which is not enough for
two of the three sweeps — the review listed what is missing, and each is a real need in existing
code:

| need | why it exists | accessor |
|---|---|---|
| the outcome row | the row whose behaviour is being counted | `frame.step` |
| the row the action was chosen on | every predictor in this analysis | `frame.prev` |
| mask out `t=0` | the world as handed to the agent is not a step | `frame.is_step` |
| the `t=0` row itself | `build_arm_data` reads the starting world configuration | `frame.initial` |
| episode start index | `perceived_nociception` must not convolve across a reset | `frame.estart` |
| global episode index | `hiding_drivers` writes one CSV row per episode across shards | `frame.episode_id` |
| per-entity presence | a third of episodes have no predator | `frame.present('predator')` |
| per-slot active mask | multi-slot entity columns are ragged | `frame.active(slot)` |
| accumulator allocation | shard callbacks accumulate into study-owned arrays | `scan.sweep(..., init=)` |
| everything else | the escape hatch, deliberately conspicuous | `frame.raw(col)` |

Each ported sweep gets a row in a **frame-contract audit table** naming which accessors it uses and,
if it calls `frame.raw()`, why. That is what makes "auditable" mean something more than "we intend
to".

### The audit table, first entry

`studies/sensor_ladder/collect_arm_data.py`, ported 2026-09-04. Counts generated from the source
rather than typed.

| accessor | uses | what for |
|---|---|---|
| `raw` | 8 | the plain columns — bush, injury, nutrition, damage, ate_food, agent row/col |
| `is_step` | 3 | excluding the reset row from every per-step accumulation |
| `list_raw` | 2 | the ragged animal row/col columns |
| `at_initial` | 2 | the starting wound and hunger, read off the reset row |
| `per_episode_sum_with_initial` | 2 | damage and eating, which legitimately include the reset row |
| `per_episode_sum` | 1 | bush steps, which legitimately do not |
| `prev` | 1 | the row the action was chosen on — every predictor in this analysis |
| `episode_of_step`, `episode_id`, `episodes` | 3 | mapping rows and shards back to global episode indices |
| `estart`, `n`, `steps_per_episode` | 3 | the early-window mask and the step-count cross-check |

**Ten `raw`/`list_raw` calls, and none of them is a breach.** The escape hatch exists to fetch a
column the driver has no opinion about; it becomes a breach only when it is used to *sidestep a
convention* — reading a predictor without the previous-row shift, or counting a reset row as a step.
Every call here fetches a column and then applies `is_step`, `prev` or one of the per-episode sums
to it. That is the contract working as intended, not being avoided.

The count is worth keeping visible precisely because it is the number that would grow if a later
port started bypassing the driver. A rising `raw` count with a falling `is_step` count is the shape
of the failure this table exists to catch.
`perceived_nociception` moves into `core/env.py` *with a regression test for its reset boundary* —
the `src > estart` guard was a real bug with published impact (Known Bugs, 2026-08-25).

### The population contract — split in two, because one version deadlocks

Revision 2 had `store.open_run()` assert that the store matched every derived product on disk. That
deadlocks its own workflow: after a collection top-up — the ladder has had two — every product is
stale by definition, so the rebuild that would refresh them cannot open the store; regenerating one
product alone becomes impossible; first-run is undefined. The predictable resolution under deadline
pressure is to delete the assert, which is the flagship guard.

Split by when each check can be true:

- **At open (`store.open_run`)** — store-internal properties only, always checkable: seeds contiguous
  with no duplicates across collection passes, every episode's rows contained in one shard, summed
  step count equal to the episode table's `length`.
- **At write (`provenance`)** — every derived product is stamped with the population it was built
  from: episode count, seed range, store fingerprints.
- **At read (the figure scripts)** — a product whose stamp disagrees with its siblings, or with the
  store, is refused. This is the pattern `_ladder.load_time_course` already implements, and it is what
  caught the half-regenerated time course.

Regenerating one product alone is then normal: it is written with a current stamp, and the next read
that mixes it with a stale sibling fails loudly.

**One tension this creates, settled before step 3.** Stamping a population into each product changes
that product's bytes, which collides head-on with a gate built on byte-identity — and the predictable
resolutions are both bad: drop the stamp (losing R2-2's flagship guard) or loosen the gate at the
moment a port is trying to pass it. Neither happens. **During migration the stamp is written to a
sidecar `<product>.provenance.json`, not into the product**, so the gated bytes are unchanged and the
population check still runs at read time. Folding the stamp into the products themselves, if it is
ever wanted, becomes its own adjudicated change with its own expected diff — after the port is
proven, not during it.

### The manifest is keyed `(unit, seed)`

The next study is a two-or-three-seed replication of the ladder. If the manifest maps one unit to one
run, that study forces this design open again immediately. So it is `(unit, seed)` from the start,
with the current ladder as the one-seed case.

```python
RUNS   = "results/JAX_RecurrentPPO/*_lad_*/"
ROOTS  = ["results/trajectories_lad", "results/trajectories_lad2"]
UNITS  = ["A_baseline", "B_olf_only", ...]
SEEDS  = [42]                                  # the replication adds 7, 1337
LABELS = {...}
BIN_EDGES = {"injury": [25, 50, 75], "distance": [1, 2, 3, 4, 5, 6, 7, 8]}
GROUPS    = {"resolves_identity": lambda cfg: ...}
```

## Scope

**In:** `hiding_drivers.py`, `ladder/` (22 files). `hiding_drivers.py` is on the ladder's critical
path — figures 6 and 7 read its CSVs — so it cannot be left behind.

**A fourth scanner the earlier revisions missed.** `scripts/analysis/context_dependence.py` (236
lines, written 2026-08-25) reads the store directly and writes `results/analysis/lad/<arm>_ctx.json`.
Every revision of this plan has said "three programs"; there are four. It has no dependency-map row
either. It is **out of scope for the port** — its outputs were built on the pass-1 population only
and nothing in the ladder study reads them — but it is named here because it is the modulator
study's tool and therefore the single likeliest first client of `core/`. Which makes it the honest
test of whether this refactor was worth doing: if the next scanner is written against `core/` and
gets all seven guards for free, the case is proven; if it is written by copying one of the existing
four again, it was not.

**Out, deliberately:** `scripts/analysis/figures/` (9 files) and `scripts/analysis/supplementary/`
(19 files, not the 9 the dependency map claims — a separate correction that map owes). Both back the
earlier trajectory-factors study, are hand-run reproductions of a published analysis, and porting
them would double the work for no new capability. They are **frozen**: a note goes in each folder's
README saying they predate `core/` and are kept to reproduce that study as published. The honest
consequence is that two copies of the duplicated helpers survive, not one.

## Verification, and why byte-identity is the wrong gate

Revision 1 made byte-identical output the whole safety argument; revision 2 rejected it because the
three scans' summation primitives are not bitwise equivalent on floats. **The second review measured
that on the real store, and my claim was too strong.** On 870,000 rows of the `A_baseline` store,
per-episode sums of `injury_level`, `nutrition` and `agent_in_bush` are **bit-exact** across
`add.reduceat`, `bincount`, `add.at` and a per-segment `np.sum` — the columns hold float32-origin
values summed in float64, so every partial sum is exactly representable and order cannot matter. Only
the nociception convolution is genuinely order-sensitive, and reversing its inner loop moves it by
6.4 × 10⁻¹⁶ relative, four orders of magnitude inside the pre-registered tolerance.

The tiered criterion therefore is not a retreat from rigour, and the tolerance will not need
renegotiating. It stands because order-sensitivity is a property of the data as much as the
primitive, and a gate that depends on today's columns being conveniently exact is one awkward column
away from failing a correct port.

The tiered criterion instead, fixed **before** any porting starts:

| product | criterion |
|---|---|
| integer-valued accumulators (0/1 sums, counts) | **bit-exact.** Order-independent, so exactness is free and any mismatch is a real bug. Tier membership is declared per output field in `core/provenance.py`, and the harness asserts `x == round(x)` on every tier-1 field, so a misclassification fails loudly instead of silently relaxing the gate |
| float accumulators | `rtol = 1e-12`, pre-registered here, not chosen after seeing the diff |
| figures | **PNG md5.** Revision 3 said "equality of the plotted arrays", but no extraction mechanism for that exists in the code, and it is *weaker* than the page tier already demands: the page inlines the PNGs, so byte-identity of the page already requires byte-identity of every figure. The 15 PNGs are git-tracked, so the golden is free. A plotted-array diff is the diagnostic when a PNG differs, not the gate |
| tables and the page | byte-identical (they are generated from the arrays) |
| regression CSVs | **not gated numerically.** They carry p-values like `2.26e-212`, where a one-ulp change in z moves the value by ~10⁻⁹ relative, so no tolerance is meaningful. Gate `aggregate.npz` bit-exact instead; the CSVs follow from it deterministically, which was verified — identical npz gives byte-identical CSVs |

**The baseline must be fresh.** ~~The goldens currently on disk were produced before the second
collection pass~~ — that was true when revision 3 was written and is **false now**: all fourteen arm
JSONs stamp both store roots, 1,000,000 episodes, seeds 1,000,000–1,999,999. The requirement stands
in the form that matters, and has been met: the comparison is old-code-run-today against
new-code-run-today on the same store, verified by re-running `build_arm_data.py` and getting an
md5-identical product. And the goldens must be protected: ported code writes to a scratch path, and **all three golden roots** are copied aside first —
`results/analysis/ladder/`, `results/analysis/lad/` and `results/analysis/hiding_drivers/<tag>/` — it is gitignored, and cheap to regenerate, but the
protocol should not rely on remembering that.

## What is at risk, and what is not

**The collected stores are not at risk.** They are read-only inputs; no analysis script has a write
path pointing at them. Re-running every derived product costs about 1.5 hours of CPU — 25 minutes for
the aggregates, 14 for the time course, ~50 for the regressions.

The risk is that **a wrong number is silent**. This session produced three cases where a job reported
success and the numbers were wrong anyway: a time course clobbered by concurrent writers, a NaN
binning that manufactured a result, and a figure whose value labels sat on the wrong bars. None cost
data; all three nearly cost a published claim. That is why the migration needs a mechanical check
rather than a careful reading.

## Migration order

1. Build `core/` alongside the existing code, changing nothing that runs.
1b. **The gate must exist and be shown to fail before anything is ported.** The order previously
   assumed a comparison harness without ever creating one. `scripts/analysis/core/golden.py` is
   that harness; it is negative-controlled in six directions (identical input reproduces; a count
   off by one fails; a float outside `rtol` fails; the same float inside it passes; a count that
   acquires a fractional part is a TIER BREAK; a manifest matching nothing refuses to run rather
   than passing vacuously). *Done 2026-09-04, commit `0362a20b`.*
1c. **A scratch output root must exist before a port can run.** `_ladder.OUT_ROOT` was a hardcoded
   constant, so a port importing it would overwrite the live products and then compare the result
   against what it had just written — a self-comparison, which passes for the same reason every
   self-comparison passes. This was the single silent-failure mode in the whole order. `OUT_ROOT`
   now reads `$LADDER_OUT_ROOT`, defaulting to the current path so nothing existing moves.
2. **Capture and verify the goldens fresh** — not "regenerate", which contradicts the golden
   directory's own README. The requirement is that today's code reproduce them, and that was
   checked: `build_arm_data.py` re-run on `A_baseline` gives an md5-identical product, which also
   establishes the sweep is deterministic. The 37 `.npz` — the actual gate objects for steps 3 and
   4 — are hashed in place rather than copied, because they are 2.2 GB.
2b. **Thirty-two golden entries are excluded from the gate**, because no current code reproduces
   them and gating on them would manufacture 32 failures — which is exactly the noise one genuine
   failure hides in. The 18 a01 `hiding_drivers/*.csv` predate commit `01fe5701` (a run today is a
   strict row superset with every shared array bit-exact); the 14 `lad/<arm>_ctx.json` were built
   on the pass-1 population only. The split is enforced by `--gate-manifest`, not merely
   documented.
3. Port `build_arm_data` → `core.scan`; compare against the golden under the tiered criterion.
4. Port `build_time_course`, then `hiding_drivers`; same gate each time. **`hiding_drivers.py`
   short-circuits its entire scan whenever `aggregate.npz` already exists** (line 359), so a gate
   run using the default cache path loads the old cache and verifies nothing about the scan. Every
   run in this migration passes a scratch `--out` *and* a scratch `--cache`.
5. Only once all three pass, retire the old implementations — but `hiding_drivers.py` is **kept**,
   not deleted: `supplementary/README.md` names it as the producer for the earlier study, so
   "reproduce as published" fails without it. Its port must also keep writing
   `results/analysis/lad/<arm>/multivariate.csv` at that exact path, because
   `lad06_world_factor_map.py` filters arms by whether that CSV exists — a moved output path would
   silently thin the published figure instead of erroring.
6. Port the fifteen figure scripts, comparing plotted arrays.
7. Freeze `figures/` and `supplementary/` with a README note.

## File changes

- **`docs/environment/SCRIPTS_DEPENDENCY_MAP.md`** — mandatory same-commit update: ~25 files move or
  are added under `scripts/`. Its §0 repo-root/`sys.path` depth hazard applies directly to
  `from analysis.core import scan`, and the map's `supplementary/` row is already stale (says 9 files,
  there are 19).
- **`docs/experiments/active/sensor_ladder/sensor_ladder.md`** and `artifact_template.html` — both
  name script paths that will move.
- **`scripts/analysis/figures/README.md`**, **`scripts/analysis/supplementary/README.md`** — frozen notice.

## Cost

**Two to four days**, revised up from the "roughly a day" in revision 1, which budgeted nothing for
the verification sweeps (three scans × fourteen arms, run twice) or the fifteen gated figure ports.

## What this does not fix

- It does not make analyses correct. It makes the *scan* correct once, so a study's mistakes live in
  its own accumulators where they are visible.
- It would not have caught the nociception error, which was a conceptual mistake about what the agent
  senses, not a plumbing failure.
- It adds an indirection: reading one figure script will mean opening `core/scan.py` to see how a bin
  was filled. That is the cost, worth paying only because the alternative is three divergent copies.

## Open question for the PI, not for engineering

The seed replication is the natural first client of `core/`, which argues for building it now, before
a fourth scan is written for it. It is also two to four days not spent on the replication. That trade
is a portfolio call.

## Feedback from plan-reviewer

**Verdict: NOT READY** (2026-09-01). Full review with per-cell audit, empirical tests, and exit
conditions: [plan_analysis_pipeline_refactor](../../../../reviews/plan_analysis_pipeline_refactor.md)
(`docs/reviews/plan_analysis_pipeline_refactor.md`).

The direction is right and matches the actual request (stop re-reading the loading/alignment code
for every new analysis). Four things must change before implementation:

1. 🔴 **Step 2's byte-identical gate and the declarative-accumulator redesign are mutually
   exclusive.** The three scripts use three different NumPy summation primitives (`reduceat`,
   `bincount`, `add.at`) for the same logical operation, and these are bitwise-inequivalent on
   float data (verified empirically). A unified sweep changes float accumulation order, so a
   *correct* port of the redesign fails the byte test on non-dyadic columns (`injury_level`,
   `nutrition`, the nociception convolution). Either restate the criterion (bit-exact for
   integer-valued accumulators — free, since 0/1 sums are order-independent; pre-registered tight
   `rtol` on loaded values for float ones; plotted-data equality for figures) or commit to a
   two-phase port: transliteration under byte-identity first, redesign second.
2. 🟡 The motivating guard table is regex-generated and wrong in ≥2 of 21 cells (`hiding_drivers`
   *does* exclude t=0 — `m = t >= 1` — and *does* filter absent animals via NaN + `isfinite`; the
   time-course "absent animals" cell is inapplicable, not missing). The duplication claims all
   verified true; fix the table, keep the motivation.
3. 🟡 Undecided scope (`figures/`, 19-file `supplementary/`), the unmentioned SCRIPTS_DEPENDENCY_MAP
   same-commit obligation, unprotected golden files in step 2, and a day estimate that budgets zero
   hours for the mandated verification sweeps (honest: 2–4 days).
4. ❓ No seed dimension in the manifest, though the next study is a 2–3-seed replication — design
   `(unit, seed)` in from the start, and sequence this to land before that analysis.

— plan-reviewer, 2026-09-01

### Second review — revision 2 (2026-09-01)

**Verdict: NOT READY** — for new reasons, not the old ones. Full round-2 findings, the
per-cell re-audit, and the empirical tolerance test:
[plan_analysis_pipeline_refactor](../../../../reviews/plan_analysis_pipeline_refactor.md)
(`docs/reviews/plan_analysis_pipeline_refactor.md`, Round 2 section).

Revision 2 genuinely fixed round 1: the tiered criterion replaces an impossible gate with an
achievable one (verified on a real shard — the three summation primitives are **bit-exact** on this
store's float columns, and the most order-sensitive quantity, the nociception convolution
reordered, differs by ~6e-16, four orders inside the pre-registered `rtol=1e-12`; the tolerance
will not need renegotiating). Scope, the dependency-map obligation, golden freshness, `(unit,
seed)`, and the 2–4-day estimate are all genuinely addressed. Two new Criticals live in the
redesign itself:

1. 🔴 **The driver's guarantee and the golden gate contradict each other in exactly the guard
   table's "no" cells.** `build_time_course`'s dose-response pairs the outcome row's bush state
   with the *same row's* perceived-nociception value (`build_time_course.py:332-334`) — the
   contemporaneous-binning class the Known Bugs registry already carries as LATENT for
   `timectrl.py`, feeding the published lad09 figure. A port that takes the guaranteed
   previous-row shift changes those numbers and fails the gate; a port that passes the gate
   reproduces the bug and falsifies "a check every study gets, including the ones already
   written". Fix: a per-"no"-cell **divergence register** (reproduce bug-for-bug, gate on the
   reproduction, fix as a follow-up with a pre-declared expected diff), and reword the guarantee
   as asserts (data properties, enforced) vs provisions (`is_step`/`prev`/masks, opt-in).
2. 🔴 **The population contract deadlocks its own workflow.** Asserting at `open_run` that the
   store matches every derived product on disk means that after the next collection pass every
   product is stale and the rebuild that would refresh them cannot open the store; regenerating
   one product alone is impossible by construction. Fix: `open_run` asserts store-internal
   properties only; `provenance.py` stamps each product with its population; the cross-product
   check moves to **read** time in core's loaders — the pattern `_ladder.load_time_course`
   already implements correctly.
3. 🟡 The "nine of seventeen empty cells" headline contradicts the printed table (6 "no" + 1
   "partial" of 17) — the population row was dropped from the table after the count was taken,
   and the prose still cites "the missing bottom-row check". Restore the row; the honest count is
   9 of 20. Also row 5's time-course "yes" is wrong per the point above. Other cells re-verified
   correct. Plus: the frame contract is under-specified for two of the three sweeps (global
   episode index, t=0-row view, per-slot active masks, `estart`, accumulator allocation);
   deleting `hiding_drivers.py` in step 5 breaks the frozen folders' reproduce-as-published
   rationale and the ported version must keep writing `results/analysis/lad/<arm>/*.csv` verbatim
   (`lad06` silently drops arms whose CSV is missing); golden protection must also cover
   `results/analysis/lad/` and `results/analysis/hiding_drivers/`; and each output field's
   comparison tier should be enumerated up front with tier-1 integer-valuedness asserted by the
   harness.

Exit conditions are the two Criticals; both are plan edits.

— plan-reviewer (round 2), 2026-09-01

### Third review — revision 3 (2026-09-04, pre-implementation)

**Verdict: SOUND WITH CONCERNS.** No Critical finding, so no round-3 section was added to
`docs/reviews/plan_analysis_pipeline_refactor.md`; the full findings were returned inline to the
implementing session. Both round-2 Criticals are genuinely closed: the previous-row fix
(`c20ff5db`) dissolves the concrete instance of the guarantee-vs-gate contradiction, and the
time-course goldens on disk were verified to carry the post-fix numbers (lowest/highest felt-pain
quarter bush hiding 12.6–17.9% / 31.7–49.5%). The asserts-vs-provisions split confines the
contradiction class: an assert either passes or fires on data and cannot move a number, so only the
two remaining *provision* cells can — and both are already known (time-course t=0 is by design;
`hiding_drivers.py:214` per-step injury bins are contemporaneous). The population contract's
three-way split matches what `_ladder.load_time_course` already does.

Concerns to resolve **before step 3** (all plan edits or one-off baseline work, none a rerun):

1. The captured golden holds stale, unreproducible entries: the 18 a01 `hiding_drivers` CSVs
   predate commit `01fe5701` (eight factors added the same evening) — a fresh run is a strict
   row-superset with every shared array bit-exact — and the 14 `lad/<arm>_ctx.json` were built on
   the pass-1 population only (single store root). Mark both sets non-gate or regenerate.
2. The golden omits the 37 npz files (2.2 GB), which are the actual gate objects for steps 3–4: the
   CSV tier is only decidable through a bit-exact `aggregate.npz` (p-values near 1e-212 make any
   rtol meaningless), and the live npz are unprotected against a port that reuses the hardcoded
   `_ladder.OUT_ROOT`. Add their md5s (or copy them) now.
3. Stamping products at write time changes their bytes, which conflicts with the byte-identity
   tiers; decide sidecar vs. subset-compare in one sentence.
4. Insert a harness step (negative-controlled) and a scratch-output-root step before step 3; run
   ported `hiding_drivers` with a scratch `--cache`, or the cache short-circuit gates nothing.
5. Figure tier: gate on PNG bytes (git-tracked, verified reproducible) with plotted-array diff as
   the diagnostic; the page tier already implies it.
6. Populate the divergence register with its two known rows; refresh the stale prose (parked
   status, "goldens predate pass 2", replication framing, artifact header, step-2 wording) and name
   `scripts/analysis/context_dependence.py` — a fourth store scanner already on disk — in scope.

— plan-reviewer (round 3), 2026-09-04
