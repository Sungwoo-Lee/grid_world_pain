---
title: Adversarial review of the sensor-ladder analysis verdict
topic: sensor_ladder
status: active
created: 2026-08-30
last_updated: 2026-09-01
---

# Review: does the sensor-ladder report's evidence support its conclusions?

## Verdict

**CONCLUSION SUPPORTED WITH CAVEATS — three Critical corrections required before any of it is
cited.** The report ([[sensor_ladder]], `docs/experiments/active/sensor_ladder/sensor_ladder.md`)
analyses fourteen agents that differ only in what they can sense, and asks what each sense buys and
what an unearned injury does to behaviour. Its two structural findings — that the fourteen agents
split cleanly into a group that stops hiding from harmless rabbits and a group that does not, and
that better senses buy *selective* rather than *more* defence — are real patterns that I reproduced
independently from the raw data. But three specific claims do not survive scrutiny: one published
table row attributes a survival cost to a single sensor change when the compared agents actually
differ in **two** settings; the report's explanation of why the injury effect disappears over a
whole episode is **quantitatively backwards** (I verified the opposite of its stated mechanism in
all fourteen arms); and its strongest statistical argument — "fourteen out of fourteen agents moved
the same way, which cannot be chance" — treats the fourteen as independent witnesses when they
watched literally the *same* 300,000 worlds (bit-identical, which I verified), were trained from the
same random seed, and in nine cases share an identical network input shape. The last finding is a
statement about the **argument**, not a claim that the opposite is true: the effects are probably
real in the measured window; the certainty language is what is unearned.

Severity legend: 🔴 Critical = fix before going further · 🟡 Moderate = likely costs a re-run ·
🟢 Low = cosmetic · ❓ Open = an assumption nobody has verified yet.

## 🔴 Critical findings

### C1 — Table 3's "visual range" row is a two-variable contrast published as single-variable

`R1_range1` has **blur disabled** (verified from its saved sensory summary in
`results/analysis/ladder/R1_range1.json`); its reference in the committed `ARM_REFERENCE`
(`scripts/analysis/ladder/_ladder.py:64`) was `V4_blur05`, which has blur 0.5. So the published
"visual range 1 instead of 2 → **−33.5 steps**" (sensor_ladder.md §1, Table 3) confounds the range
drop with the blur change. The correct single-variable reference is `V5_sharp` (range 2, blur off):
**−7.9 steps** (230.5 − 238.4), +1.9 pp dwell — a factor-of-four smaller effect. Compounding it, the
report claims the pairing "is checked against the saved configs" (§Fig 2 Method) — **no such check
existed** in the committed code; the claim described a verification that was never written, and this
mispairing is exactly what it would have caught.

*Status:* the author's parallel session independently caught this during my review and has an
uncommitted fix in the working tree (corrected pairing + a real `check_single_variable_pairs`
assertion). *Exit condition:* commit the fix, re-run `lad02` + `make_report_tables.py`, regenerate
Figure 2 and Table 3, and rewrite the §1 prose around the range effect. Owner: `experiment-analyzer`
(doc), author session (code, in flight).

### C2 — §7's Panel B explanation is refuted by its own data; headline finding 5 omits its window

The report explains the whole-episode reversal of the injury effect as a *dilution* artefact:
lightly wounded agents live longer (270 vs 255 steps), so "its whole-episode average is diluted by
more late, low-hiding steps". I decomposed this in every arm. The composition effect runs the
**other way** (shorter heavily-wounded episodes weight their high early dwell *more*, which would
make the whole-episode effect *more* positive), and the reversal is instead driven by a genuine
**negative late-window association**: past step 25, bush dwell *falls* with the assigned wound in
all fourteen arms (−0.67 to −1.15 pp; A_baseline −2.06). Recomputing the dose-response at other
windows: `V4_blur05` is +4.21 pp at w=25 but +0.79 at w=50 and −0.12 at w=100; `O1_occl_rock`
+6.78 → +2.16 → +0.64. The wound effect does not merely "wash out" — it **reverses sign
mid-episode**, and the chosen 25-step window sits near the maximum of the curve. The honest claim is
"a randomised wound causes more hiding for roughly the first 25 steps and slightly *less*
thereafter; the whole-episode net is ≈ 0" (consistent with the report's own M1 regression, where
start injury on whole-episode dwell has p = 0.20). Headline finding 5 states "+2.8 to +6.8 pp" with
no window qualifier. *Exit condition:* rewrite the Panel B mechanism paragraph to match the
decomposition, add the window to finding 5, and ideally plot slope-vs-window. Owner:
`experiment-analyzer`.

### C3 — The "14/14 cannot be a fluctuation" argument treats correlated agents as independent

The report's caveat section argues that findings resting on "a pattern across many arms" survive
the one-seed problem because "fourteen independently trained agents … moving the same direction …
could not [happen by chance]". Two problems. **(a) The fourteen are not independent.** They replayed
*bit-identical* worlds (I verified `inj0`, `nut0`, animal counts and odour draws are exactly equal
across all fourteen stores), share training seed 42, and the nine "identity" arms share an identical
observation shape — hence plausibly identical initial weights and identical training-data RNG
streams. World-draw noise and seed-42 quirks push all arms the same way, so sign-unanimity cannot be
multiplied like fourteen coin flips. **(b) The unanimity is window-contingent.** Recomputing the
hypervigilance gap (rabbit-smell amplification minus predator-smell amplification) at other windows:
it stays positive at w=10 in all five arms I tested, but at w=50 it flips negative in three of five
(`V4_blur05` −0.41, `O2_occl_veg` −0.54, `A_baseline` −0.57). And the marginal arm at w=25
(`A_baseline`, gap +0.03 pp, both amplifications *negative*) is inside noise yet counted toward
"14/14". The finding is defensible as an early-window, exploratory pattern; "could not be chance" is
not. *Exit condition:* drop the independence language, state the window-contingency, and quantify
uncertainty properly — because worlds are shared, the right tool is an episode-level (world-level)
bootstrap that resamples the 300,000 episodes once and recomputes all fourteen arms per draw, asking
how often unanimity survives. Owner: `experiment-analyzer`.

## 🟡 Moderate findings

| # | Location | Issue | Suggested fix |
|---|---|---|---|
| M1 | lad05/lad08 `resolves_identity`; §4 | Grouping rule (`range ≥ 2 AND channels > 1`) is post-hoc (whole analysis landed in one commit; no design doc, no pre-registration anywhere) and conflates **reach** with **identity**: `R1_range1` carries all 8 identity channels yet is grouped "cannot resolve"; `V5_sharp`'s identity-resolving sight is, by the report's own §3, unreliable. "Four independent measures agree" overstates — they are four correlated views of one behaviour stream per agent (limitation 5 partially admits this; the §4 prose does not). | Reword to "consistent", name the rule as descriptive of the split rather than derived; note R1 shows identity is necessary-but-not-sufficient. |
| M2 | §4, Q1/Q2 as "decisive cases" | Collapsing 8 channels to 1 also shrinks the visual input from 104 to 13 dims — a different network input width and (same seed) different init. Identity information is confounded with input width/capacity. | Acknowledge; an arm with 8 duplicate channels would separate them. |
| M3 | Finding 6 / §5 prose | Internal inconsistencies: "response only *grows* in the nine arms" is contradicted by its own ranges (V2 +0.18 counted as growth vs B_olf +0.26 and Q2 +0.19 counted as flat — the groups overlap); "effect largest in the three occlusion arms" holds for the raw rabbit amplification but **not** for the gap measure that defines the finding (V5_sharp +2.58 and Q2 +2.20 lead); V5 is "unreliable sight" in §3 but implicitly "reliable sight to fall back on" in finding 6. | Rewrite finding 6 to one consistent measure and drop the overlapping-range categorical claim. |
| M4 | Table 4, last column | "one more rabbit in the world (pp)" actually reports the GLM's `dpp_per_sd`, and SD(n_rabbits) = 0.816 — per-rabbit values are ~22% larger (A_baseline +4.43, not +3.61). Signs and pattern unaffected. | Relabel, or switch `make_report_tables.py` to `dpp_per_unit`. |
| M5 | §5 hypervigilance control | The predator-smell "control" channel is contaminated by the intensity↔discriminability anticorrelation the report itself documents (Fig 9 dip: loudest predators are least predator-like). A suppressed predator-slope amplification could reflect that confound rather than a criterion shift, inflating the rabbit-minus-predator gap. | Note it; or control on the difference channel. |

## 🟢 Low

- `build_arm_data.py`: `E["dmg"]` and `n_ate` include the t=0 row (unlike `bush_steps`); harmless
  iff those columns are zero at reset — assert it. Dead variable `ib_ep` with a stale comment.
- Table 4's "rabbit odour slope" column is early-window (first 25 steps) while the neighbouring
  proximity column is all-steps; not stated in the doc.

## ❓ Open assumptions

- No pre-registered hypotheses or refutation criteria exist for this study anywhere in
  `docs/experiments/` — every grouping and window was chosen on the same data that tests it. All
  confirmatory language should read as exploratory.
- "A rabbit's smell is a false alarm by construction": a *loud* rabbit's odour may be genuinely less
  discriminable from a predator's (both channels near ceiling squeeze the difference), making the
  response partially rational under ambiguity rather than pure waste. Unchecked.
- `ate_food`/`damage` guaranteed zero on the reset row — assumed, unverified.

## What checked out (verified, not assumed)

- All six published tables regenerate **bit-identically** from `make_report_tables.py`.
- The presence-filtering fix is **complete**: every accumulator conditioning on animal existence
  (`pd/rd/pdc/rdc`, both odour grids) carries the `has_p`/`has_r` mask; quantile edges are
  NaN-aware; the `fi == 0` guard only caches; shard misalignment crashes loudly; step counts are
  cross-checked against the episode table.
- Identical-worlds pairing verified empirically across all fourteen stores (bit-identical world
  variables), despite collection on heterogeneous GPU classes.
- §3's sensor-code reading is accurate (`sense_visual`: blur off = exact cell match; blur on =
  mass-normalised PSF), and its behavioural signatures reproduce (V5's near-distance dip 37.2 vs
  51.3, raised far baseline 10.5 vs 7.0).
- Prose numbers reproduce: 25.4/34.1, r = −0.62/−0.70, 270-vs-255, n ≈ 33.5k, +15.0 → +9.3,
  gap range +0.03…+2.58; my independent reimplementation of the w=25 hypervigilance numbers matches
  Table 5 exactly. Training budgets are step-parity across arms (10,000,005–10,000,081).

## Cost of being wrong

Nothing here risks data loss or a wasted training run — the risk is **publication-grade**: the
R1 row (C1) would put a 4×-inflated causal number into any downstream citation; C2/C3 would turn a
transient, window-specific injury effect into a general "wounds cause hiding / hypervigilance"
claim in a paper. A two-to-three-seed replication of the ladder is the only thing that can convert
the pattern findings from exploratory to confirmatory.

Reviewed by: plan-reviewer

---

## Follow-up review, 2026-09-01 — negative percentages, and the perceived-nociception correction

Two questions were re-examined adversarially, from the code and the stores, after the main review
above. Severity legend as above.

**Verdicts.** (1) The negative numbers on the figures: **SOUND — no arithmetic defect.** No figure
plots a negative bush-dwell *rate*; every negative value is a difference of two rates, is labelled
as one on the axis itself, and none exceeds 38.3 pp in magnitude (bound for a difference of rates
is 100). (2) The perceived-nociception reading: **CONFIRMED — Critical (C4) against the report's
current §5 narrative.** The agent cannot sense its injury level; what it receives is a lagged,
reset-zeroed trace, and the report's timing story is written as if behaviour follows the wound when
the data show it follows the trace.

### Question 1 — negative percentages (cleared)

- Every accumulator in `build_arm_data.py` and `build_time_course.py` adds `agent_in_bush` ∈ {0,1}
  to the numerator and 1.0 to the denominator at identical index sets, so 0 ≤ rate ≤ 100 always;
  verified empirically over all arms: no raw rate below 0 or above 100 anywhere in the JSONs.
- `L.rate` / `L.dist_curve` NaN out cells with < 1000 steps of support; `L.proximity_effect` pools
  counts (not rates) over distance bins 1–2 vs 6+ and over the chosen wound quarters before the
  single division, and requires ≥ 1000 steps on *each* side. All as documented.
- Figures plotting differences: 2, 5, 8 (right panel only), 9 (panels B/D), 10, 11, and the Fig 14
  annotations. **Figure 12 plots raw rates only** and carries no negative numbers — the inventory
  that listed it among the difference figures is wrong on that one point. Fig 8's *left* panel is
  raw rates.
- Reader-confusion audit: every difference axis prints "a DIFFERENCE, in percentage points" plus
  the explicit minuend/subtrahend. Weakest labels: old lad09 panel B ("extra bush dwell") lacks the
  "below zero = hides less" gloss the other figures carry, and Table 9's column header "wound: bush
  dwell span (pp)" says *span* (reads as a magnitude) for a signed difference that is −2.34 in
  `A_baseline`. 🟢 Both worth one-line fixes; neither is an error.

### Question 2 — the agent cannot sense its injury level (confirmed, 🔴 C4 for the report text)

Verified from source and from ground truth the system produced:

- **Buffer/kernel semantics as claimed.** `core.py:1112` zeroes the buffer at reset — the reset
  row's injury is *never* written into it; `core.py:115` rolls and writes the post-step injury at
  slot 0; kernel slot 0 is exactly zero. The strict `src > estart` guard in
  `build_time_course.perceived_nociception` is therefore correct, not conservative.
- **Reconstruction is exact.** The trajectory stores record the observation vector (`obs_true`);
  perceptual noise is disabled, so channel 1 *is* the policy input. Over 1,358,558 rows of a real
  `V4_blur05` shard, max |reconstruction − recorded obs| = 2.2 × 10⁻⁷ (float32 rounding). This is
  a non-circular check: the comparison target was produced by the environment at collection time.
- **Time course as claimed.** Cumulative kernel weight: **0 at t=0 and also t=1**, 9.0% at t=2,
  35.7% at t=4, 100% only at t=12. An agent that wakes with injury ~95 feels 0.000 for two steps,
  0.085 at t=2, peaks at 0.927 at t=12 — while the physical wound is already healing.
- **Behaviour tracks the trace, not the wound.** In all five regenerated arms the perceived-signal
  gap (Q4−Q1) peaks at **t=12** and extra hiding peaks at **t=14–16** (2–4 steps behind the
  feeling); the injury gap peaks at **t=0**, where extra hiding is ≈ 0 — the physical-wound reading
  cannot explain a zero response to a maximal wound, the perceived reading predicts it exactly.
  (`A_baseline`, whose whole effect is +1.05 pp, is the noise-level exception.)

**What C4 kills or qualifies in the current report** (`sensor_ladder.md` §5, Figure 9, finding 5):
the panel title "The response follows it — extra hiding fades as the wound does", the reading
"falls away on roughly the wound's own schedule … an effect that tracks its cause through time",
finding 5's "lasts about as long as the wound does", and "25 steps … is approximately the lifetime
of the dose" (it is the lifetime of the *felt* dose — the perceived gap is still ~30 injury-units
at t=24 when the physical gap is 15). The **causal claims survive**: the perceived signal is a
deterministic function of the assigned wound, so binning by the randomised `inj0` remains a valid
causal contrast; what was wrong is the mechanistic timing narrative. The rewritten
`lad09_injury_time_course.py` (four panels, body-vs-feeling) states this correctly.

### Other places an unsensed quantity is used

- 🟡 **Nutrition is also unobservable** (`nutrition_observable: false`) — the sensed channel is
  satiation, S = maxS·(N/maxN)^k, an *instantaneous monotone* transform, so binning by nutrition is
  perceptually valid (no analogue of the injury lag). But the report makes the unobservability
  point only for injury; finding 7's "hunger outweighs the wound by 1.6–4.5×" compares a zero-lag
  percept against a lagged, attenuated one over a 25-step window, so part of that ratio is sensor
  dynamics, not drive weighting. Qualify, don't retract.
- 🟢 Carried injury (Fig 14 C, Table 7) is unsensed directly but is already framed as the mistake
  exhibit, not a percept. Note for the rewrite: a *mid-episode* wound arrives with a phasic
  extero-nociception contact signal, whereas the reset wound arrives with none — one more reason
  panel C differs, and the reason "the interoceptive channel is the only route" is exactly true
  for the assigned wound.
- ❓ Reward used the **true** injury during training (drive term (inj/100)²), so the agent had a
  gradient incentive to infer its wound faster than the percept allows; empirically it did not.
- ❓ Figure 9 comparisons at t ≳ 40 condition on survival, and Table 8 shows the heavy-wound
  quarter dies sooner — the late-time convergence is measured on differently-selected populations.

### Process finding

- 🟡 `_ladder.load_time_course`'s docstring claims it "asserts that every arm's file was produced
  by the SAME episode population as the aggregates" — the code only checks the file *exists*. At
  review time 9 of 14 time-course files lack the new `noci` key (regeneration in flight), which the
  advertised assertion would have surfaced. Implement the check (episode totals in the file vs the
  arm JSON) or delete the claim.

### Cost of being wrong (this follow-up)

The negative-percentage worry costs nothing — the arithmetic is right. C4 is publication-grade: the
uncorrected §5 narrative would put "behaviour tracks the wound's time course" into a paper when the
data show behaviour tracking a perceptual trace the wound merely drives — the exact distinction an
interoception paper exists to make. Owner: `experiment-analyzer` (report rewrite);
`developer` if the `load_time_course` guard is to be implemented.

Reviewed by: plan-reviewer
