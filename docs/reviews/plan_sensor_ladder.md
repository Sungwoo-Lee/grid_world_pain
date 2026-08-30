---
title: Adversarial review of the sensor-ladder analysis verdict
topic: sensor_ladder
status: active
created: 2026-08-30
last_updated: 2026-08-30
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
