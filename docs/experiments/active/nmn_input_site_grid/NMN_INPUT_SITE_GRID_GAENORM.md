---
title: "The estimator-swapped twin — the same 16-cell modulation grid, run under GAE_NORM instead of MC"
topic: nmn_input_site_grid
status: active
created: 2026-09-07
last_updated: 2026-09-07
wandb_group: nmn_input_site_grid_gaenorm
wandb_tag: "rppo_nmngaenorm_t{1,2,3,4,5,16}<slug>_{ALL,I,X}_s42"
develop_link: docs/develop/active/neuromodulation/MODULATION_SITE_REFACTOR.md
---

# The estimator-swapped twin — the same 16-cell modulation grid, run under GAE_NORM

> **Status**: TRAINED — all 16 cells plus the control launched and finished 2026-09-07 at the full
> 10M-episode budget. Million-episode evaluation trajectories were collected 2026-09-08 into
> `results/trajectories_nmngae/`; analysis not yet written.
> **Date**: 2026-09-07 (pre-registered), status updated 2026-09-08
> **Author**: `experiment-designer`
> **Mode**: fully pre-registered. Every threshold and failure-mode ruling below is fixed **before**
> any run of this grid launches and may not be adjusted afterwards.
> **Twin of**: [[NMN_INPUT_SITE_GRID]] — the sixteen runs launched 2026-09-07 and training now.
> That document is the **parent**: it owns the science, the sixteen cells, the confound register,
> the behavioural measures and the failure-mode catalog. **This document owns only what differs.**
> Everything not restated here is inherited from it verbatim.
> **Related**: [[TRAINING_HEALTH_AUDIT]] (the parent grid's engineering audit, whose three
> corrections are carried forward in §6 below) · [[return_mode_cmp_10M]] (source of the estimator
> choice and of every dispersion number used here) · [[MODULATION_SITE_REFACTOR]] (the code both
> grids consume) · [[project_plan]]

---

## 1. Question

### 1.1 In plain language

The agent in this project can carry a small second network — the **neuromodulator** — that reads
the agent's senses and continuously re-tunes the main policy network, the way a brain chemical such
as acetylcholine re-tunes cortex. A sixteen-run screening grid is training right now that crosses
two things about that arrangement: **what** the modulator reads (all 27 sensory numbers, or the two
internal-body channels only, or the nineteen outside-world channels only) and **which** parts of
the main network it re-tunes (the sensory front-end, the memory cell, the action head, the value
head, or all four at once), plus one run with no modulator at all as the control. That grid, its
reasoning and its analysis plan are the parent document, [[NMN_INPUT_SITE_GRID]].

**This document adds a second copy of that same sixteen-run grid, changed in exactly one setting.**
The setting is the recipe the trainer uses to turn a rollout of experience into the two numbers that
PPO actually learns from — the target the value head is trained towards, and the "advantage" that
tells the policy how much better an action turned out than expected. The running grid computes
those two numbers one way (called **`MC`**, for Monte-Carlo: add up the actual rewards that
followed). The new grid computes them a different way (called **`GAE_NORM`**: blend the value
head's own predictions into the estimate, which trades a little bias for much less noise), and
changes nothing else whatsoever. Not one other key in any of the sixteen files.

**The question this pair answers.** The parent grid will produce a ranking — which place to attach
the modulator looks most promising, which sensory diet looks most promising. The obvious worry
about that ranking is that it might be an artifact of how the trainer happens to compute its
learning signal, rather than a fact about modulation. Running the identical grid under a second,
independently-motivated learning signal is the cheapest test of that worry: **any conclusion that
survives both grids is not a property of the learning signal, and any conclusion that flips between
them was never safe to draw.**

**Why this particular second recipe, and not the more obvious one.** There is a plainer alternative
called `GAE`, and it was rejected on purpose. `MC` and `GAE` differ in **two** things at once — the
estimator (how the future is estimated) *and* the scale convention (whether the value head's target
and the policy's advantage are put on the same numerical scale, or on scales that differ by roughly
a factor of twenty-four). `GAE_NORM` is the GAE estimator carrying `MC`'s matched-scale convention.
Swapping `MC` → `GAE_NORM` therefore changes **only the estimator**, so a difference between the
two grids is attributable to that alone. Under plain `GAE` the two grids would differ in two things
and would not be comparable on the estimator at all — which is the entire point of building the
twin.

**`GAE_NORM` is also the better-behaved of the two arms, which matters more than it sounds.** In
the five-seed, five-arm comparison recorded in [[return_mode_cmp_10M]], `GAE_NORM` survived longer
than `MC` — about **170 steps against about 165**, a gap of roughly **4.5 steps** whose 95%
confidence interval excludes zero — and it did so with the **tightest seed-to-seed spread of any
arm in that study**: its five seeds finished within **1.3 steps of each other**, against 4.5 for
`MC` and 23 to 101 for the three arms that train badly. So all sixteen arms of this twin are
expected to train in the same healthy regime the parent grid has now been **empirically confirmed**
to be in — every one of its sixteen runs is learning, none has produced a not-a-number or an
infinity, and every modulator is receiving real training signal ([[TRAINING_HEALTH_AUDIT]]).

### 1.2 What is being varied, and what is not

| | Parent grid ([[NMN_INPUT_SITE_GRID]]) | This grid |
|---|---|---|
| Cells | 1 unmodulated control + 5 write targets × 3 input slices = **16** | **identical** |
| Write targets | encoder / memory cell / action head / value head / all four | **identical** |
| Input slices | `"all"` (27) / `["Satiation", "Interoceptive Nociception"]` (2) / `["Extero Nociception", "Olfaction", "Collision", "Visual"]` (19) | **identical** |
| Environment | `basic/04` (10×10 jump-attack) | **identical** |
| Modulator hyperparameters | FiLM, `grouping_size: 1`, `mod_hidden_size: 16`, temperature **off**, `rnn_mechanism: activation` | **identical** |
| Learning rates | `lr_actor` = `lr_critic` = `0.0005` | **identical** |
| Budget / seed | 10,000,000 episodes, seed **42**, one seed per cell | **identical** |
| **`return_mode`** | **`MC`** | **`GAE_NORM`** |

That table is not a summary — it is the whole design. The one-key claim in its last row is enforced
mechanically rather than asserted: see §3.2.

### 1.3 What this experiment is, and is not

**It is** a robustness replicate of a screening grid under a second learning signal, and — as a
by-product — the first measurement of whether the ~4.5-step `GAE_NORM` advantage measured on
*unmodulated* agents survives when a modulator is attached.

**It is not** a test of any hypothesis about neuromodulation on its own. Every hypothesis in the
parent's §1.4 is scored inside each grid separately, at one seed per cell, exactly as the parent
registers. This grid adds a *consistency* axis, not statistical power. §2.3 below is explicit about
why those are different things, and it is the section most likely to be misread.

---

## 2. Design — only what differs from the parent

### 2.1 What is inherited verbatim

Independent variables (§2.1), controlled variables (§2.2), the confound register C1–C14 (§2.4), the
pre-registered predictions H1–H5 with their refutation criteria (§2.5), the behavioural measures and
their bin edges (§4.2), the diagnostic modulator measures (§4.3), and the failure-mode catalog (§5)
are **taken from the parent unchanged** and are not restated here. Read the parent for all of them.

Four inherited items are worth naming because a reader of *this* document could otherwise miss
them:

- **Proprioception is in neither restricted slice.** It is classically neither interoceptive nor
  exteroceptive, so it appears only under `ALL`. Consequence (parent confound C2): `ALL` and `X`
  differ by **eight** dimensions, not two, so **no `ALL`-vs-`X` difference may ever be attributed
  to interoception**. The clean interoceptive contrast is `I` vs `X`, and only that one.
- **Input width is not matched across the input factor** (27 / 2 / 19), so any `I`-vs-`X`
  difference confounds *which* information the modulator reads with *how much* (parent C3).
- **The budget is in episodes, and better agents run longer episodes**, so every headline
  comparison is made at **matched environment steps**, never at matched episodes (parent C8).
- **Nothing in either grid is confirmatory.** Every cell is n = 1.

### 2.2 The one changed variable

`agent.return_mode`: `MC` → `GAE_NORM`, in all sixteen files including the unmodulated control.

The two settings, in the trainer's terms:

| Setting | Critic target | Advantage | Scale convention |
|---|---|---|---|
| `MC` | Monte-Carlo return, z-scored | `target − V`, not rescaled | **matched** |
| `GAE_NORM` | λ-return (GAE, λ = 0.95), z-scored | `target − V`, not rescaled | **matched** |
| *(`GAE`, rejected)* | *λ-return, raw* | *separately rescaled* | *split* |

The rejected row is why `GAE_NORM` and not `GAE`: it moves the scale convention as well as the
estimator, and the five-seed study measured that scale convention to be worth far more than the
estimator — the two split-scale arms finished at roughly 69–98 survival steps against 157–159 for
the two matched-scale arms at matched experience. A `GAE` twin would have differed from the parent
grid by the dominant variable and the variable of interest simultaneously.

### 2.3 What a difference between the two grids would, and would not, license

This is the section that governs how the pair may be read. It is pre-registered.

**First, the property that makes the pair unusually clean.** Both grids run at seed 42 on the same
environment with the same architecture, so a matched pair of cells starts from **bit-identical
initial weights and the same environment seeding stream**. The two runs of a pair diverge only
because their gradient updates differ from the first update onwards. Nothing else — not the
initialisation, not the world — is resampled between them.

**Second, and immediately, the price of that property.** Because the initialisation is *not*
resampled, **cross-grid agreement is a weaker replication than a second seed would be, not a
stronger one.** A finding that is an artifact of this particular initialisation will reproduce in
both grids and look confirmed. Agreement across the twin therefore **does not** substitute for the
five-seed confirmation the parent's §4.4 requires before any candidate becomes a claim. The pair
tests robustness to the *learning rule*; it does not test robustness to the *seed*, and those are
different failure modes.

**Third: both grids are single-seed, so a cell-by-cell comparison between them is one draw against
one draw.** The parent already carries the rule that no cell can confirm a behavioural hypothesis
alone. Comparing two single-seed grids cell by cell compounds it: each twin difference is a
difference of two single draws, and there are sixteen of them, so some will be large by chance
alone. Nothing in §4.2 below treats a single large twin difference as a finding.

**Licensed conclusions** — what the pair may be used to say:

| # | Licensed | Why |
|---|---|---|
| L1 | "This ranking of write targets (or of input slices) **does not survive** a change of return estimator, and is therefore demoted from the follow-up shortlist." | A necessary condition failed. Demotion is conservative and costs only a follow-up slot. |
| L2 | "This ranking **is preserved** under a change of return estimator, which removes one specific alternative explanation (that it is an artifact of the MC learning signal)." Stated as a removed alternative, never as confirmation. | Agreement rules out one confound. It rules out nothing about seeds — see above. |
| L3 | "The ~4.5-step `GAE_NORM`-over-`MC` advantage measured on unmodulated agents **does / does not** persist when a modulator is attached at site S." Reported as a candidate at n = 1, and only for the control and for cells whose twin difference clears the band in §4.2. | The control pair has a five-seed anchor behind it; the modulated pairs do not. |
| L4 | "Cell `t5crt` (the value-head site) behaves differently across the two grids." — interpretable **as a candidate estimator interaction**, because `return_mode` *is* the recipe that produces the value head's training target, so this is the one cell where an interaction is predicted a priori rather than fished for. | Pre-registered here, before the data exists, precisely so it cannot be claimed post hoc. |

**Unlicensed conclusions** — pre-registered as forbidden:

| # | Forbidden | Why |
|---|---|---|
| F1 | "The return estimator interacts with site S" for any S other than `t5crt`. | One draw against one draw, sixteen times over. `t5crt` is the sole a-priori exception, and even it is a candidate, not a claim. |
| F2 | Pooling the two grids to treat any cell as n = 2. | They are not two seeds. They share an initialisation and differ in a training hyperparameter. Pooling them would be averaging over a variable that is under test. |
| F3 | Any interoceptive claim from an `ALL`-vs-`X` contrast, in either grid or across them. | Parent confound C2. Unchanged by the twin. |
| F4 | Reading a twin difference as evidence about the *estimator* in general. | Both grids are one environment, one architecture, one seed. |
| F5 | Adjusting any threshold in this document, or in the parent, after seeing either grid. | Pre-registration. |

### 2.4 Thresholds

**Within-grid, this grid uses the parent's bands unchanged: below 15 steps is no evidence, 15–30 is
flagged and unresolved, above 30 is a candidate real effect promoted to a five-seed confirmation.**

That is a deliberate choice and it is conservative rather than convenient. The parent derived those
bands from the `MC` arm's own seed dispersion (a 9.4-step five-seed range at matched experience).
`GAE_NORM`'s unmodulated dispersion is **tighter** — a 2.7-step range at matched experience, 1.3 at
end-of-budget — so bands derived from it would be *narrower*, and this grid would then declare
effects the parent grid would call noise at the same numerical size. Keeping one set of bands is
what makes the two grids' verdicts commensurable cell by cell, which is the entire purpose of the
pair. The cost is stated: **this grid is under-powered relative to its own noise floor, on purpose**,
and a real 5-to-10-step effect in it will be reported as "no evidence".

**Between-grid (the twin difference).** Define, for each of the sixteen cells,

$$
\Delta_{\text{twin}}(\text{cell}) \;=\; \text{survival}_{\text{GAE\_NORM}}(\text{cell}) \;-\; \text{survival}_{\text{MC}}(\text{cell})
$$

both read at matched environment steps. The reference value is `Δ_twin(control)`, which has a
five-seed anchor behind it (**+4.7 steps** at 1,488 M environment steps, from [[return_mode_cmp_10M]]
§4.3). A cell is remarkable only if its `Δ_twin` departs from `Δ_twin(control)` by more than the
band below.

| Departure of `Δ_twin(cell)` from `Δ_twin(control)` | Ruling |
|---|---|
| Below **21** steps | **No evidence.** The cell behaves as the control does under the estimator swap. |
| **21 to 42** steps | **Flagged, unresolved.** Candidate only; may not be described as an interaction. |
| Above **42** steps | **Candidate estimator interaction.** Promoted to a five-seed confirmation of *both* arms before any claim. |

21 and 42 are the parent's 15 and 30 inflated by √2, because a difference of two independent single
draws has about √2 the dispersion of one. That inflation is **conservative on purpose**: the two
runs of a pair share an initialisation, so their outcomes are probably positively correlated, which
would make the true dispersion of `Δ_twin` *smaller* than √2 — but the correlation has not been
measured, so the design assumes independence and pays for it in resolution.

**One pre-registered opportunity to tighten these two numbers, with a deadline.** The five existing
unmodulated `MC` seeds and the five existing unmodulated `GAE_NORM` seeds are **seed-matched**
(s42–s46 in both arms), so the five values of `Δ_twin` they define are a direct empirical estimate of
this statistic's dispersion under exactly this pairing, on runs that already exist and cost nothing
to analyse. If that computation is done and recorded **before any arm of this grid finishes
training**, the bands may be replaced by ones derived from it. If it is not done by then, **21 and
42 stand and may not be revisited.** This is registered as a request in §7.

### 2.5 Compute

Sixteen runs at 10M episodes. The parent estimates 11–15 hours per run on a lab GPU; `GAE_NORM`
consumed marginally *more* environment steps than `MC` at the same episode budget in the prior study
(1,563 M against 1,516 M, because it survives longer per episode), so the same 11–15 hour estimate
applies with a small upward lean. Sixteen runs at one per GPU is one wave across nodes 106–114.

**Placement is `training-runner`'s call against live GPU state, and it must account for the parent
grid.** Ten of the parent's sixteen runs were still training when this design was written. Nothing
here should be launched onto a GPU the parent grid is using.

---

## 3. Launch Manifest

System-of-record for every run of this grid. `experiment-designer` owns the planned columns;
`training-runner` fills Node / GPU / Launched at / WandB run ID / Log path in place at launch;
`experiment-analyzer` reads the table to find the runs. **The runner does not invent tags for runs
in this manifest** — the values below are authoritative.

**Tag scheme**: `rppo_nmngaenorm_<target-slug>_<input-code>_s<seed>`. **wandb-group**:
`nmn_input_site_grid_gaenorm`. Tag and wandb-name are identical on every row.

**Why this stem and not an appended suffix.** The parent grid's tags are `rppo_nmnsite_*`, and
anything formed by *appending* to them (`rppo_nmnsite_..._gaenorm`) would leave the parent's stem a
strict prefix of this grid's — so `grep rppo_nmnsite_` would sweep up both grids and no grep could
select the parent alone. The distinguishing token is therefore placed **before** the varying part.
The two stems `rppo_nmnsite_` and `rppo_nmngaenorm_` are prefix-free in both directions, and
`rppo_nmngaenorm` was checked against all **405** existing run directories under
`results/JAX_RecurrentPPO/` — zero matches, and no existing tag is a prefix of any tag below. (This
is the same check, and the same reasoning, that rejected `rppo_nmn_t*` for the parent grid because
it collides with the existing `rppo_nmn_tempceil*` family.)

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Code SHA | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|----------|------|-----|-------------|--------------|----------|
| 1 | planned | `T1_none` | `rppo_nmngaenorm_t1none_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 2 | planned | `T2_enc_ALL` | `rppo_nmngaenorm_t2enc_ALL_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 3 | planned | `T2_enc_I` | `rppo_nmngaenorm_t2enc_I_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 4 | planned | `T2_enc_X` | `rppo_nmngaenorm_t2enc_X_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 5 | planned | `T3_rnn_ALL` | `rppo_nmngaenorm_t3rnn_ALL_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 6 | planned | `T3_rnn_I` | `rppo_nmngaenorm_t3rnn_I_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 7 | planned | `T3_rnn_X` | `rppo_nmngaenorm_t3rnn_X_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 8 | planned | `T4_act_ALL` | `rppo_nmngaenorm_t4act_ALL_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 9 | planned | `T4_act_I` | `rppo_nmngaenorm_t4act_I_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 10 | planned | `T4_act_X` | `rppo_nmngaenorm_t4act_X_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 11 | planned | `T5_crt_ALL` | `rppo_nmngaenorm_t5crt_ALL_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 12 | planned | `T5_crt_I` | `rppo_nmngaenorm_t5crt_I_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 13 | planned | `T5_crt_X` | `rppo_nmngaenorm_t5crt_X_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 14 | planned | `T16_quad_ALL` | `rppo_nmngaenorm_t16quad_ALL_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 15 | planned | `T16_quad_I` | `rppo_nmngaenorm_t16quad_I_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |
| 16 | planned | `T16_quad_X` | `rppo_nmngaenorm_t16quad_X_s42` | nmn_input_site_grid_gaenorm | pilot | 42 | — | — | — | — | — | — |

The Cell names deliberately match the parent's manifest exactly, so a cell name identifies a *pair*
and the twin difference of §2.4 is a table join rather than a manual lookup.

`wandb-job-type` is **`pilot`** on every row — the honest label for a single-seed screen, and it
keeps these runs out of production multi-seed queries.

**On the `Code SHA` column and its gate — amended from the parent, deliberately.** The parent
requires all sixteen rows to show the same commit **with `git_dirty: false`**. That gate is
**unsatisfiable as written** and this grid does not repeat it; see §6.3. What this grid requires
instead: all sixteen rows show the **same** commit; that commit has `e1aab726` (Part B of the
modulation-site refactor) as an ancestor; `git diff e1aab726 <SHA> -- src/` is empty; and each run's
own saved config and startup banner report the sites and input sensors its cell specifies. The last
of those is the strongest check available, because it inspects what the system *produced* rather
than what the source ought to produce.

### 3.1 Configs — written and verified (2026-09-07)

All 16 runs share one environment config, unmodified, identical to the parent's. Each run gets its
own agent config, because agent configs in this repo do not support `extends:` and are
self-contained.

**In plain language, what exists now.** All sixteen agent configuration files have been written by
the **same generator that wrote the parent grid's**, extended to take the grid as a parameter rather
than forked into a second copy — two generators is how two grids that are supposed to differ in one
key quietly come to differ in two. Every one of the sixteen was then loaded the way the trainer
loads it and used to build a real network, and every one was diffed key-by-key against its
counterpart in the parent grid.

| Run | Config (env) | Config (agent) — written, verified |
|-----|--------------|--------------------------------|
| 1 | `configs/environment/experiment/basic/04-jump_attack_10x10.yaml` | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t1none.yaml` |
| 2 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t2enc_ALL.yaml` |
| 3 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t2enc_I.yaml` |
| 4 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t2enc_X.yaml` |
| 5 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t3rnn_ALL.yaml` |
| 6 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t3rnn_I.yaml` |
| 7 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t3rnn_X.yaml` |
| 8 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t4act_ALL.yaml` |
| 9 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t4act_I.yaml` |
| 10 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t4act_X.yaml` |
| 11 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t5crt_ALL.yaml` |
| 12 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t5crt_I.yaml` |
| 13 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t5crt_X.yaml` |
| 14 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t16quad_ALL.yaml` |
| 15 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t16quad_I.yaml` |
| 16 | same | `configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t16quad_X.yaml` |

The generator is
`configs/models/recurrent_ppo/nmn_input_site_grid/generate_site_grid_arms.py` — one file, now
serving both grids, selected with `--grid mc|gaenorm|all`. It reads each grid's shared agent body
from that grid's base config (`recurrent_ppo_cmp_mc.yaml` for the parent,
`recurrent_ppo_cmp_gaenorm.yaml` here — two files that themselves differ in exactly
`agent.return_mode`, checked), and changes exactly one key relative to it: `lr_critic`
0.0001 → 0.0005, in all sixteen files including the control, exactly as the parent does and for the
same reason (parent confound C5; the key is read by no code path today, so it is inert for these
runs and exists only so that a later rerun from a saved config cannot silently train one arm's
critic at a fifth of the rate the rest used).

**The parent grid's sixteen files were not touched.** The generator refactor was verified
byte-for-byte against them *before* anything was regenerated: `--check --grid mc` passes on the
refactored generator with the runs still training.

### 3.2 What was verified, and how

Produced by `generate_site_grid_arms.py --verify`, which loads every config through the same merge
order the trainer uses and then **constructs the model**. A file that merely parses as YAML proves
nothing here: the point of the refactor's mandatory keys is that a wrong config is refused loudly,
so the refusal is collected now rather than at launch. Result: **597 checks, 0 failures**, across
both grids.

- **All thirty-two configs load, construct and take a step.** Every arm builds a model and returns
  finite action scores and a finite value estimate on a forward pass.
- **The modulator's input width was read off the built network**, not computed by hand: 27 numbers
  under `ALL`, 2 under `I`, 19 under `X` — confirmed three ways per arm (the resolved index tuple,
  the modulator GRU's declared input width, and the actual row count of its input weight matrix).
- **The four site switches were read off both the network and the modulator inside it**, so a
  copy-paste that left two sites on would have been caught.
- **The environment is the same for all thirty-two.** Each arm's observation layout was recomputed
  and compared: `Satiation 1, Interoceptive Nociception 1, Extero Nociception 1, Olfaction 5,
  Collision 5, Proprioception 6, Visual 8 = 27` in every case.
- **The control's lineage is exact.** Flattened key-by-key, this grid's control differs from
  `recurrent_ppo_cmp_gaenorm.yaml` in `agent.lr_critic` and nothing else — the same relation the
  parent's control has to `recurrent_ppo_cmp_mc.yaml`.
- **The twin diff — the check that makes this a matched pair rather than two loosely similar
  studies.** Every one of the sixteen configs was flattened and compared key-by-key against its
  counterpart in the parent grid. **All sixteen differ in exactly one key, `agent.return_mode`**,
  in the direction `MC` → `GAE_NORM`. No extra keys, no missing keys, no other changed value.
- **The mechanism controls hold**: uniform FiLM operator, `rnn_mechanism: activation`, temperature
  **off**, `grouping_size: 1`, `mod_hidden_size: 16`, and the **same plain GRU cell the unmodulated
  control uses** rather than the legacy modulated cell.
- **`return_mode` resolves to `GAE_NORM` after the full merge**, not merely in the file — checked
  on the merged config each arm's model is built from, so a later-merged layer could not have
  overridden it.
- **Negative controls: the loud failures really are loud.** A deliberately misspelled sensor name, a
  missing site key, and all-sites-off-with-temperature-off each raise an error at construction
  rather than building a plausible-looking wrong model.

Re-run at any time (CPU only; it never competes with training for a GPU):

```
JAX_PLATFORMS=cpu /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
  configs/models/recurrent_ppo/nmn_input_site_grid/generate_site_grid_arms.py --verify
```

**No schema change.** Every key used here is already read by the loader and was already used by the
parent grid. Neither `return_mode` nor `lr_critic` appears in
[[CONFIG_CRITICAL_SETTINGS]], so no registry change-log entry is required; nothing in
[[CONFIG_GUIDE]] describes these directories, so no guide update is triggered.

#### Still outstanding before launch

- **`env-config-reviewer` pre-flight** on all sixteen files. Points to put in front of it: the
  `return_mode: GAE_NORM` value itself; the learning-rate deviation (parent C5); the sensor-name
  spellings against `get_observation_breakdown`; the sites / `rnn_mechanism` / temperature block;
  the deliberate absence of `temperature.clip` when temperature is disabled; the deliberate absence
  of `lr_modulator` (no such key exists in the code — writing it would be inventing schema); and
  the twin relation to the parent grid's sixteen files.
- **A trajectory-collection spec**, if the behavioural measures of the parent's §4.2 are to be
  computed for this grid. None has been written, for the same reason the parent's arm spec is not
  runnable: every entry needs a run **directory** name, and those are stamped with the launch
  timestamp. If written, it must pin the same population as the parent's —
  **300,000 episodes, `seed_base` 1000000**, `obs_precision: float32`, final checkpoint,
  deterministic policy, one collector process per node. **`seed_base` must not drift**: it is what
  makes every arm-to-arm and grid-to-grid behavioural comparison paired, and collecting under a
  different base degrades the comparison silently, with no error and no warning.

### 3.3 Launch command

One invocation per run, through `run_command.py` onto the assigned node. Shown for Run 11
(`T5_crt_ALL`); the other 15 differ only in `--agent_config`, `--tag`, `--wandb-name` and
`--device`.

```
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/nmngaenorm_t5crt_ALL.yaml \
  --episodes 10000000 \
  --device cuda:0 \
  --log-interval 10 \
  --tag "rppo_nmngaenorm_t5crt_ALL_s42" \
  --wandb-name "rppo_nmngaenorm_t5crt_ALL_s42" \
  --wandb-group "nmn_input_site_grid_gaenorm" \
  --wandb-job-type "pilot"
```

`--seed`, `--num-envs` and `--checkpoint-frequency` are **not** passed: they are config-owned
(`configs/train/default.yaml`, overridden by `configs/train/recurrent_ppo.yaml`) per the project's
config-owns-values convention.

---

## 4. Analysis plan (pre-specified)

### 4.1 Within this grid

**Identical to the parent's §4.1–§4.3, with one substitution.** Survival steps, never cumulative
reward; two readings (matched environment experience as the primary, end-of-episode-budget reported
alongside and labelled confounded); one number per cell with **no confidence interval across seeds,
because there is one seed**; mandatory temporal evolution at 50 M / 100 M / 200 M / 400 M / 800 M /
common maximum, plus time-to-100-steps and late-training slope.

**The substitution is the control gate.** The parent's C7 gate scores its control against the five
existing unmodulated `MC` seeds. This grid's control is scored against the five existing unmodulated
**`GAE_NORM`** seeds, same study, same environment, same budget, same seed set:

| | Parent grid | This grid |
|---|---|---|
| Reference runs | `…rppo_cmp10m_mc_s42…s46` | `…rppo_cmp10m_gaenorm_s42…s46` |
| Five-seed end-of-budget values | 162.4, 164.2, 165.7, 166.8, 166.9 | 169.9, 170.0, 170.1, 170.6, 171.2 |
| **Level check — control must finish inside** | **162.4 – 166.9** | **169.9 – 171.2** |

Same procedure as the parent: trajectory overlay first, level check second, and **if the control
falls outside that range the grid is reported as void pending diagnosis**, not as results with a
caveat — the control is the reference for every comparison in the grid. Checked at the earliest
checkpoint at which both curves exist, not only at the end.

Note the asymmetry the tighter reference creates, and accept it: `GAE_NORM`'s five-seed range is
**1.3 steps wide** against `MC`'s 4.5, so this grid's control gate is materially harder to pass than
the parent's. A control that lands at, say, 168.5 would fail this gate while an equivalently-placed
`MC` control would pass. That is the correct behaviour — the gate asks whether the training path
reproduces a known result, and the known result here is known more precisely.

### 4.2 Across the two grids

Run **after** both grids' controls have passed their respective gates. If either control fails, the
cross-grid analysis is not attempted at all.

1. **Twin-difference table.** All sixteen `Δ_twin(cell)` values at matched environment steps, joined
   on Cell name, with `Δ_twin(control)` on its own row as the reference. Scored against the 21 / 42
   bands of §2.4. Every row carries an explicit n = 1 label.
2. **Rank-agreement.** The five write targets ranked within each grid (at fixed input slice), and
   the three input slices ranked within each grid (at fixed write target), then compared. Reported
   as the rank correlation between grids **and** as the raw ranking tables, because a correlation
   over five items is not itself informative and the tables are what a reader needs. Feeds L1 and L2
   of §2.3; feeds no other conclusion.
3. **The `t5crt` pre-registered look** (L4). The value-head cells' twin differences, reported
   separately and labelled as the one a-priori-predicted interaction. Reported whatever the result,
   including a null.
4. **Does the estimator advantage survive modulation?** `Δ_twin(control)` against the mean of the
   fifteen modulated `Δ_twin` values. Descriptive only, at n = 1 per cell.
5. **Behavioural measures, only if collected.** If a trajectory-collection spec is run for this
   grid, the parent's §4.2 measures are computed on the identical episode population
   (`seed_base` 1000000) and compared across grids **paired by episode**. Every behavioural
   threshold is stated against the **five-seed band**, never against the within-run interval — the
   two differ by about a factor of six, and an arm can clear its own error bar comfortably while
   sitting inside the null band. If no spec is run, this analysis is simply absent; it is not
   approximated.

### 4.3 What must be re-run with more seeds before anyone believes it

Unchanged from the parent's §4.4, with one addition: **a candidate estimator interaction (a cell
clearing the 42-step band, or `t5crt` under L4) requires a five-seed confirmation of *both* arms of
that cell — ten runs — not five.** A five-seed confirmation of one arm cannot establish a
difference between two arms.

---

## 5. Failure-mode catalog (pre-decided)

Inherited from the parent's §5 in full. Three additions specific to the pair:

| # | Situation | Pre-decided ruling |
|---|---|---|
| P1 | **This grid's control fails its level gate (169.9–171.2) while the parent's control passed its own.** | The grid is **void pending diagnosis**, and the *pair* analysis is abandoned rather than run on a broken half. The parent grid stands on its own. Route the discrepancy to `senior-developer`: a control that reproduces under one `return_mode` and not the other localises the problem to the return-computation path. |
| P2 | **A cell trains badly in one grid and normally in the other** (fragile by the parent's §5 criteria in one arm only). | The pair for that cell is **unresolved**, not an interaction. It is not counted as evidence for L1 (failure to survive the estimator swap), because a fragile arm at n = 1 carries essentially no information — the parent's §2.3 lesson 2 measured 20–100 step dispersion in that regime. |
| P3 | **Rankings agree across grids, and a reader proposes to treat that as confirmation.** | Refused, in advance, by §2.3: the two runs of a pair share an initialisation, so agreement removes one confound (the learning signal) and cannot remove the seed confound. The five-seed gate of §4.3 still applies in full. |

---

## 6. Corrections carried forward from the parent grid's audit

Three findings from [[TRAINING_HEALTH_AUDIT]] apply to this grid as well. They are carried forward
here so they are not rediscovered, and are **not** re-argued — the audit owns them.

### 6.1 The metrics reference is stale for FiLM-type runs — do not apply its sigmoid correction

`docs/develop/active/neuromodulation/NMN_METRICS_REFERENCE.md` states that the gain numbers logged
for the sensory front-end (`gamma_uni_*`, `gamma_multi_*`) are "pre-sigmoid" values that must be
squashed through a sigmoid before being read as an actual gain, and gives a healthy band of 1.0–3.0
for them. **Under `modulation.type: FiLM` — which every modulated arm of both grids uses — that is
no longer true.** The front-end now applies the same plain multiply-and-add as the other three
sites, so the logged number **is** the gain and initialises at 1.0. A reader who applies the
documented correction to these runs will misread every front-end number, and the documented healthy
band would mark a perfectly healthy run as sub-healthy. This is a documentation defect, not a code
defect; the audit flagged it for maintenance and neither the audit nor this design edits the
reference.

### 6.2 "Enabling a site is a no-op at step 0" holds on average, not per unit

Both designs rely on the reasoning that switching a write site on cannot hurt at the very start of
training, because the site begins at gain 1 and offset 0. **That holds as an average across the
layer's 128 units, not for each unit.** The modulator layer producing those numbers has
randomly-initialised weights, so at the first update each individual unit's gain is 1 **plus a
random deviation of roughly ±0.3**, and each unit's offset is 0 plus a similar deviation. Nothing in
either design is invalidated by this, and the audit found no run harmed by it — but the argument
should be stated in expectation rather than asserted literally, and any cell-1-versus-cell-2
reasoning that leans on exact step-0 identity is leaning on something that is not exactly true.

### 6.3 The `git_dirty: false` launch gate is unsatisfiable and is not repeated here

The parent's manifest required all sixteen rows to record the same commit **with
`git_dirty: false`**. In practice fifteen of the parent's sixteen runs recorded `git_dirty:
"unknown"` and one recorded `true`, so **no row met the gate**. The cause is mechanical, not
scientific: the provenance recorder shells out to `git status --porcelain` with a **10-second
timeout** and records the string `"unknown"` on timeout, and sixteen training processes starting
within eleven seconds of each other all run `git status` against the same repository on the NAS,
where that command takes the index lock and is documented as slow. The cheap `git rev-parse` calls
succeeded in all sixteen, which is what lock contention looks like.

**This grid does not repeat an unsatisfiable gate.** §3's amended requirement replaces it with
checks that can actually be met and are stronger anyway: same commit on all sixteen rows, that
commit containing `e1aab726`, an empty `git diff … -- src/` against it, and — the decisive one —
each run's own saved config and startup banner reporting the sites and input sensors its cell
specifies, which inspects what the system produced rather than what the source ought to produce.
The underlying engineering defect (provenance becomes unreliable exactly when it matters most, at a
simultaneous multi-run launch) is the audit's §8 item 2 and remains open; it is a code change and is
not made by this design.

---

## 7. Metrics and tooling requested

**No new logged metrics are required.** Everything §4 needs is already logged, and the parent's own
four metric requests (audit §7) are enhancements, not blockers, for either grid.

**One analysis task is requested, and it has a deadline.** Compute `Δ_twin` over the five existing
seed-matched unmodulated pairs — `rppo_cmp10m_mc_s42…s46` against `rppo_cmp10m_gaenorm_s42…s46`, at
matched environment steps — and record its five values and their range. These ten runs already
exist; the work is a WandB query, not compute. **Why now**: it is the only direct empirical estimate
of how much a twin difference varies for reasons other than the treatment, and §2.4's bands of 21
and 42 steps are currently derived from a conservative √2 assumption rather than measured. **If it
is recorded before any arm of this grid finishes training**, the bands may be replaced by measured
ones; **after that point they are frozen**, because tightening a threshold once grid data exists is
not pre-registration. Owner: `experiment-analyzer`. Cost: cheap.

---

## 8. Results

*(To be filled after training. Nothing here until both grids' controls have been scored against
their gates.)*

---

## 9. Analysis

*(To be filled after training.)*

---

## 10. Conclusions

*(To be filled after training.)*

---

## Appendix

### A. Where each fixed number in this document came from

| Number | Value | Source |
|---|---|---|
| `GAE_NORM` unmodulated, end of 10M episodes, five seeds | 169.9, 170.0, 170.1, 170.6, 171.2 (mean 170.36 ± 0.52) | [[NMN_INPUT_SITE_GRID]] §2.3 (per-seed) · [[return_mode_cmp_10M]] §4.1 (arm mean) |
| `MC` unmodulated, end of 10M episodes, five seeds | 162.4, 164.2, 165.7, 166.8, 166.9 | same |
| `GAE_NORM` − `MC` at matched experience (1,488 M env steps) | **+4.7 steps**; complete seed separation, exact Mann-Whitney *p* = 0.0079 | [[return_mode_cmp_10M]] §4.3 |
| `GAE_NORM` − `MC` under greedy evaluation, 2,000 episodes/seed | **+4.47 steps**, 95% CI **[+0.75, +8.19]**, *p* = 0.024 (170.04 vs 165.57) | [[return_mode_cmp_10M]] §4.9 |
| `GAE_NORM` five-seed spread — tightest of any arm | **1.3 steps** at end of budget, **2.7** at matched 349 M | [[NMN_INPUT_SITE_GRID]] §2.3 (spread table) · [[return_mode_cmp_10M]] §4.2 (per-seed at 349 M) |
| Split-scale arms' matched-experience survival (why plain `GAE` was rejected) | `GAE` 98.3, `MC_FIXED` 69.3 against `MC` 156.8, `GAE_NORM` 158.9 at 349 M env steps | [[return_mode_cmp_10M]] §4.2 |
| Environment steps consumed at 10M episodes | `GAE_NORM` 1,563 M, `MC` 1,516 M | [[return_mode_cmp_10M]] §2.3 confound C1 |
| Within-grid bands 15 / 30 steps | inherited unchanged from [[NMN_INPUT_SITE_GRID]] §2.3 | parent |
| Twin-difference bands 21 / 42 steps | 15 and 30 × √2 (difference of two single draws, assuming independence) | this document §2.4 |
| Modulator input widths 27 / 2 / 19 | read off the constructed networks by `--verify`, not computed by hand | §3.2 |
| 405 run directories checked for tag collision | `ls results/JAX_RecurrentPPO/` | §3 |
| Wall clock 11–15 h per run | parent §2.6, plus `GAE_NORM`'s 3% higher step consumption | parent |

**One note on provenance of the framing numbers.** The reasoning that motivated this grid cited the
`GAE_NORM`-over-`MC` gap as roughly +4.5 steps with a 95% interval of [+0.75, +8.19] and the
tightest seed spread of any arm. Those are the greedy-evaluation figures of §4.9 of the source
study (170.04 vs 165.57 survival steps), and the spread claim is the five-seed table in
[[NMN_INPUT_SITE_GRID]] §2.3. The
matched-experience training-log figures are +4.7 (169.5 vs 164.8) and are what §2.4 and §4 of this
document use, since every comparison here is at matched experience. The two estimates agree to
within half a step, which is why the choice between them changes nothing in the design.

### B. Relationship to the parent document

This document is **subordinate**. Where the two disagree, the parent wins, with exactly three
recorded exceptions, each of which is an intentional amendment argued in place:

1. The control's level gate uses the `GAE_NORM` five-seed range (§4.1), not the `MC` one.
2. The `git_dirty: false` launch gate is replaced with satisfiable checks (§3, §6.3).
3. Sections §2.3, §2.4 (between-grid part), §4.2, §5 (P1–P3) and §7 have no counterpart in the
   parent; they exist only because the pair exists.

The parent carries a back-link to this document in its own §3.1 handoff area.

### C. Changelog

| Date | Change |
|---|---|
| 2026-09-07 | Created. Configs generated and verified (597 checks, 0 failures, both grids). Not launched. |
