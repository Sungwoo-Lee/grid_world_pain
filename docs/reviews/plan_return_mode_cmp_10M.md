---
title: Adversarial review of the 10M-episode return-mode comparison verdict
topic: return_mode_cmp
status: active
created: 2026-09-05
last_updated: 2026-09-05
---

# Review: does the 10M return-mode analysis's evidence support its conclusions?

## Verdict

**SUPPORTED WITH CAVEATS for the core result; three headline sentences are NOT SUPPORTED BY THE
EVIDENCE SHOWN as written.** The document
([[return_mode_cmp_10M]], `docs/experiments/active/return_mode_cmp/return_mode_cmp_10M.md`)
compares five ways of scaling a reinforcement-learning agent's critic target and advantage, five
seeds each, for ten million episodes. Its central empirical result — that the two "textbook
convention" settings which looked permanently broken at one million episodes were merely *slow*,
escaped the sit-still-and-starve trap given more budget, and are roughly an order of magnitude
less experience-efficient than the project's own scheme — is real, reproduced in the raw
per-seed data, and correctly replaces the earlier "stall" reading. The run inventory is complete
(all 25 runs, none dropped), the confound of unequal experience per episode is handled properly
by re-plotting against environment steps, and the two pre-registered predictions are scored
honestly, including the one that was wrong.

What does not hold up is three things the document *adds on top* of that result. First, it says
the slowdown factor is "constant" and shows "no sign of a lower ceiling"; its own working table
shows the factor rising steadily with survival level in the best-measured arm, and the published
table omits the row that makes this plainest. Second, it says the training-time advantage of the
`GAE_NORM` setting over the project default "disappears entirely" under greedy evaluation; the
greedy evaluation, at five seeds and 200 episodes each, cannot detect a difference smaller than
about eight steps, and the effect in question is five. The two measurements do not disagree —
one of them is simply too coarse to see the effect, and no conclusion about exploration should be
built on a discrepancy that may not exist. Third, it declares a mechanism the project has asserted
("the value loss dominates the shared network body and starves the actor") *refuted* and installs
a replacement ("entropy collapse"). The refutation rests on tests that cannot distinguish "a
constant handicap that slows learning ten-fold" from "no handicap", on a contrast that never
relieved the supposed dominance (99.99% → 99.96%), on a statistical null with a minimal
detectable difference of ~50 steps, and on loss-value shares that are not a proxy for gradient
shares at all. The replacement is a plausible hypothesis supported by one arm's observational
data and no intervention — a weaker standard than the one on which the original was rejected.
The last finding is a statement about the **argument**: the mechanism is *unmeasured*, and this
document should say so rather than say it is wrong.

Severity legend: 🔴 Critical = fix before going further · 🟡 Moderate = likely costs a re-run ·
🟢 Low = cosmetic · ❓ Open = an assumption nobody has verified yet.

## 🔴 Critical findings

### C1 — §5.6 does not refute the shared-trunk mechanism, and the replacement is adopted on weaker evidence

Five independent problems, any one of which would be enough to downgrade "refuted" to "untested".

**(a) The claim under test is mischaracterised.** Confound row C6 (`return_mode_cmp_10M.md:158`)
says the loss-share test is "decisive *as stated*, because the project's own claim was about the
summed loss". The 1M document's Finding 3 (`return_mode_cmp_1M.md:341-349`) says the opposite in
its own words: "this analysis does not make [the learning-rate argument]. The argument is about
**direction**... the gradient Adam sees is the *sum* of a value term and a policy term". Loss
shares were the *evidence offered*; the *claim* was about gradient direction in the shared
parameters. §5.6 then concedes the gradient version "cannot be tested from these logs at all"
(§7, first row). A refutation of a gradient claim from loss values, after admitting gradients are
not logged, is the thing the review brief warned about.

**(b) Test A cannot distinguish "constant handicap" from "no handicap".** Test A shows survival
doubled while the value share stayed at 99.99%. The inference "learning proceeded under value
dominance, so value dominance cannot be what prevented it" refutes only the *stall* form of the
mechanism (prevented learning). The document's own headline — a roughly constant ~10x slowdown —
is exactly what a constant, unrelieved handicap would produce. A mechanism that predicts
"ten times slower" is *confirmed*, not refuted, by a constant handicap and ten-times-slower
learning.

**(c) Test B never moved the variable out of the regime where the mechanism predicts nothing.**
`MC_RAW` raised the policy term's share from 0.0017% to 0.0348%. The value term still holds
99.96%. If the mechanism is "the shared body is shaped almost entirely by value regression", it
predicts no benefit from 99.99% → 99.96%. Test B is also confounded by the very thing §5.6 later
names as `MC_RAW`'s failure route: entropy collapse. The document uses the same null result both
as evidence *against* the gradient mechanism and as evidence *for* entropy collapse; if entropy
collapse explains why `MC_RAW` did not improve, then `MC_RAW`'s failure to improve says nothing
about the gradient mechanism.

**(d) The null is unpowered.** At the matched 349 M-step budget the `MC_RAW` sd is 39.4 and
`MC_FIXED`'s 11.7, so the standard error of the difference is ~18 steps and the minimal detectable
difference at 80% power is ~50 steps. Under greedy evaluation (sd 58 vs 33) it is ~87 steps
(bootstrap 95% CI on `MC_RAW` − `MC_FIXED`: [−61, +43]). "A twenty-fold increase in the policy's
share bought nothing measurable" (§5.6, §6.1) is a null that could not have measured a 40-step
gain. The document itself says `MC_RAW`'s "mean describes no seed" and "should be reported as a
split, never as a mean" (§5.1) — and then rests two conclusions on the mean-based Welch *p* = 0.95.
Per seed at 349 M steps the two escaped `MC_RAW` seeds (107.7, 118.6) are above every `MC_FIXED`
seed (max 81.1) and level with `GAE`; the three stuck seeds are below every `MC_FIXED` seed. That
is "high-variance", not "identical".

**(e) Loss-value shares are not a proxy for gradient shares.** The PPO policy surrogate is
`mean(ratio × A)`. With per-batch z-scored advantages `A` has mean zero and `ratio` ≈ 1, so the
*value* of the policy term is ≈ 0 in expectation regardless of how large its *gradient* is. A
policy share of 0.0017% therefore measures nothing about the gradient — and `MC_RAW`'s larger
|policy| value is partly just a non-zero-mean raw residual (`return − V` with a biased critic).
This cuts both ways: it also means the 1M document's loss-share evidence never supported the
mechanism. The honest state is *unmeasured in both directions*.

**The replacement mechanism** (policy-vs-entropy ratio shift of ~120x → entropy collapse →
exploration starvation) is stated as a result (§5.6 bold: "It failed because the actor was
starved of exploration"; §6.1 bullet; both pre-registration Outcome sections). Its support is
one arm, five seeds, observational, no intervention. Reverse causation is not excluded: a policy
that has settled into resting becomes deterministic *because* resting is stable under large
advantages — the fine trace for `mcraw_s45` (`tmp/20260905_return_mode_cmp_10M.md:359-378`) shows
entropy already at 0.037 nats by 2 M episodes while the agent was still a wanderer eating
0.67 food/episode; the switch to resting came at ~3 M. Direction is plausible; it is not shown.
The document already names the right test (§6.3 row 1, entropy coefficient × 20) — until it runs,
this is a hypothesis.

**Exit condition.** Retitle §5.6 to "unmeasured; the loss-share evidence is uninformative either
way". Replace "the project should stop asserting this mechanism" with "should stop asserting it
*as established*". Fix C6. Label entropy collapse "hypothesis, consistent with the data, test =
§6.3 row 1" in §1 item 2, §6.1, §6.2(d), and in the Outcome sections of both pre-registration
files. Revert the 1M document's status header (`return_mode_cmp_1M.md:16-19`, "failed two direct
tests... should not be cited") to "untested; the loss-share evidence cited in Finding 3 does not
bear on it". Owner: `experiment-analyzer`.

### C2 — §5.2 / §4.7 / §1 item 3: "the advantage disappears entirely" is read off a measurement that cannot see a 5-step effect

Greedy-evaluation per-seed means (`tmp/cmp10m/eval_summary.csv`): `MC` 176.1, 180.3, 172.5,
179.0, 180.0 (sd 3.31); `GAE_NORM` 177.2, 178.2, 184.7, 169.9, 177.7 (sd 5.26). Standard error of
the difference = 2.78; minimal detectable difference at 80% power ≈ **8 steps**; 95% CI on
`GAE_NORM` − `MC` ≈ **[−6.4, +6.4]** (bootstrap [−4.9, +4.9]; paired-by-world across the 200
shared worlds gives the same: −0.04 ± 3.65). The training-log effect is **+4.7 to +5.2**. The
interval *contains* the effect. So "177.55 versus 177.59 — a dead tie", "disappears entirely",
"this materially changes how the result should be described" (§4.7 item 2, §5.2, §1 item 3) are
not supported; the supported statement is "the greedy evaluation is too coarse to confirm or
refute a five-step gap". The document's own item 1 in §4.7 says the proxy and the greedy number
are "not interchangeable at the ±10-step level" — which is precisely the resolution at which
item 2 then reads a tie.

Consequences: (i) the "exploration-cost reconciliation" in §5.2 explains a discrepancy that the
data do not establish exists; (ii) §6.3 row 2 is built on the same non-discrepancy; (iii) the
answer to the caller's question is **no, the earlier statement does not have to be withdrawn** —
`GAE_NORM` above `MC` in training-time survival, all five seeds above all five, is intact. What
must not be passed on is "and the effect vanishes under greedy evaluation".

**Exit condition.** Reword §1 item 3, §4.7 item 2, §5.2 and §6.1 to "not resolvable by the greedy
evaluation at this size". Optionally (evaluation only, no training): re-run the 25 evaluations
at ~2,000 episodes per seed, which brings the between-arm resolution to ~3 steps. Owner:
`experiment-analyzer`.

### C3 — §1 item 1 / §5.4: "roughly constant... does not grow with level... no sign of a lower ceiling" is contradicted by the document's own working table

The survivorship-controlled dilation table (`tmp/20260905_return_mode_cmp_10M.md:222-258`) reads:

| level | `GAE` (3 seeds) | `MC_FIXED` (2 seeds) | `MC_RAW` (2 seeds) |
|---|---|---|---|
| 50 | **6.4x** | **5.9x** | 12.9x |
| 60 | 7.8x | 10.2x | 10.6x |
| 80 | 9.1x | 12.0x | 10.2x |
| 100 | 10.4x | 13.8x | 11.0x |
| 110 | 9.8x | 14.1x | 10.2x |
| 120 | 10.0x | 12.6x | 8.2x |
| 130 | **11.6x** | — | — |

The published table (§5.4) starts at level 60, omitting the level-50 row (6.4x / 5.9x) that
sits *outside* the "measured range 8x to 14x" quoted in §1. In the best-powered arm (`GAE`, 3
seeds, all 5 seeds reaching 120) the factor rises monotonically from 6.4x to 11.6x — an
80% increase across the range. `MC_FIXED` rises 5.9x → 14.1x. Only `MC_RAW` (2 seeds) falls. A
time-to-level ratio that grows with level is the early signature of a lower asymptote; on 2-3
seeds it is also compatible with noise. What is *not* compatible with this table is "a pure
constant-factor slowdown fits the observed range; a diverging one does not" (§5.4) or "there is
no sign in this data of a lower ceiling" (§1). This is the same over-reading of a trend that
produced the 1M "stall" call, in the opposite direction.

The catch-up extrapolation (§5.4, 9-12 billion steps) *is* correctly labelled untested — but with
a rising factor it is a lower bound, and should say so.

**Exit condition.** Restore the level-50 row; replace "constant / does not grow" with "8-14x,
drifting upward with level in two of the three arms (`GAE` 6.4x → 11.6x, `MC_FIXED` 5.9x →
14.1x); whether that drift is the onset of a ceiling or seed noise at n = 2-3 is undetermined";
delete the "diverging one does not fit" sentence; make the billion-step figure a lower bound. The
*delay-not-stall* conclusion itself (every arm escaped, cap-rate > 0, starvation turned over)
stands and needs no change. Owner: `experiment-analyzer`.

## 🟡 Moderate findings

| # | Location | Issue | Suggested fix |
|---|---|---|---|
| M1 | §5.8, §6.1 (H₃ refuted / H₄ supported), `mc_raw_prereg.md` Outcome | The pre-registered decision rule returns "between 60 and 100 → cannot rank"; the document replaces it post hoc with "tie against `MC_FIXED` ⟹ relational refuted", and that tie is the unpowered null of C1(d). The *strong* relational reading ("magnitude irrelevant") is genuinely refuted — all five `MC_RAW` seeds sit below `MC`'s worst seed at every budget — but "bought nothing" relative to `MC_FIXED` is not shown. | Record the pre-registered outcome ("cannot rank") first, then the post-hoc reasoning labelled as such; state H₃-strong refuted on the all-seeds-below-`MC` fact, not on *p* = 0.95. |
| M2 | §5.3 table row `s45`, §5.3(a), §6.1 | "Collapsed into the trap *after* a working foraging policy... catastrophic forgetting of an already-learned behaviour". The fine trace shows 0.43-0.70 food/episode at 35-37 survival steps, rest ≈ 0, entropy 0.037 nats by 2 M. That is a wanderer with incidental eating at the do-nothing survival level — not a forager. | "A low-entropy wandering policy that ate incidentally switched to resting at ~3 M and stopped eating". Drop "catastrophic forgetting". |
| M3 | §5.3 opening, §6.1 | "Four of five seeds are out of the trap or on their way out". Supported: `s44`/`s46` left the exact-100 zero-food state (greedy eval confirms 3.6 / 2.3 food, one 500-step episode). Extrapolated: that they would complete the ramp — `s44`'s departure occupies only the last ~0.5-1 M episodes and both finish at 45 survival steps. By the document's own §4.5 criterion 2/5 escaped. | Give both counts: 2/5 crossed 50 steps; 4/5 left the zero-food state; "on their way out" by analogy with `s42`'s ramp, labelled as such. |
| M4 | §5.3(d), §6.1 bullet | Body says "cannot say" but §6.1 asserts "No logged optimisation signal predicts which seed escapes". At n = 4 the smallest attainable *p* is 0.083; at n = 15 power to detect ρ ≈ 0.5 is ~50%. Also the "clean pre-divergence window 3-4 M" is not clean for `s45`, which entered the trap during it (rest 0.002 → 29 → 32 across 2/3/4 M). | "None detected at n = 4-15, which is roughly the power to see a moderate correlation half the time"; note the `s45` window caveat. §6.3 row 5 already says this — lift it into §6.1. |
| M5 | §5.7 | Interaction is real and not one-seed-driven (`GAE` 81.6-113.4 vs `MC_FIXED` 55.0-81.1 at 349 M do not overlap; dropping any seed leaves ≥ +21). But "+29 steps" is a snapshot on the steep part of two rising curves: the split-row estimator gap is −0.6 at 100 M steps, +8.6 at 200 M, +29 at 349 M, +32 at 10 M episodes. The two-way ANOVA is run on cells whose variances differ by >100x (1.1² vs 14²); its *p*-values are not trustworthy at n = 5/cell, and the episode-axis interaction is *p* = 0.044. | Report the split-row Welch *p* = 0.008 as the statistic; express the estimator effect as a dilation ratio (`GAE` reaches each level ~1.2-1.4x sooner than `MC_FIXED`: 172 vs 226 M at level 60, 306 vs 406 M at 100) rather than a step count at one budget. |
| M6 | §4.7 item 1 | "+8.2 steps greedy bias, Wilcoxon *p* = 4×10⁻⁵" treats 25 evaluations as independent, but all 25 share the same 200 worlds, so the world-sampling offset (per-episode sd ≈ 210 → SE ≈ 15 steps for the high arms) is common to every one of them. The bias direction is plausible; its size and *p* are not established. Likewise *r* = 0.992 is driven by the between-arm range (34 → 185); the within-arm proxy validity — the only thing at issue for the 5-step question — is untested. | State the proxy as validated for *arm-level ranking*; drop the *p*-value on the bias or compute it with the world offset modelled. |
| M7 | 1M doc header (`return_mode_cmp_1M.md:16-19`), both prereg Outcome sections | Already propagate "failed two direct tests... should not be cited" and "entropy collapse" as results. | Update in step with C1. |

## 🟢 Low

- `MC_RAW` end-of-training entropy is 0.095 in the §4.6 table and 0.086 in the table beneath it.
- Three escape-time definitions coexist (`escape.py` threshold 60; the §4.5 table and
  `pooled_precursor.py` threshold 50; `precursor2.py` ramp-onset 5.0 / 6.6 / 9.0 M). Rank order is
  preserved so no test changes; the document should name one.
- `gae_norm_prereg.md` names its decision metric as "mean survival over the final *evaluation*
  window"; no evaluation window existed, the training series was used. Worth a one-line note.

## ❓ Open assumptions (never verified in the document)

1. The offline evaluation loaded the *source* environment YAML (`basic/04` → extends
   `03-random_init` → default) through the eval loader, not the run's saved environment config.
   `04`'s modification time (2026-07-04) predates the runs; the base files were not checked.
   Identical across all 25 runs, so between-arm comparisons are safe; the absolute level is what
   would drift.
2. The evaluation fell through `eval_rollout.py`'s "behavior_measures block absent → defaults"
   branch (`scripts/eval/eval_rollout.py:960-984`): seeds 0-199, greedy, training-time noise. Fine
   for comparability; note that the seeds were a tooling default, not a chosen list.
3. "Return spread ≈ 24" — every "scale ≈ 24" label in the design table is the 1M-era measurement
   the document itself calls "certainly stale" (§7 row 2).
4. Same git commit in all 25 runs' provenance (C7) — asserted, not shown.
5. `MC_RAW` ran on a mix of GPU classes including an 11 GB 2080 Ti (node 105); no systematic effect
   expected, none checked.

## Project-rule check

- **Survival steps only** — satisfied; `Episode/Reward` appears once, as one of fourteen precursor
  diagnostics, and carries no conclusion.
- **No fallback defaults** — not applicable to an analysis; the tooling fallback in open item 2 is
  noted, not a violation.
- **Plain-language entry point** — §1 passes the 200-word check; formal hypotheses follow the
  translations.
- **Pre-registrations** — the 1M "stall" prediction is recorded as wrong (§5.8); the `MC_RAW`
  personal prediction is recorded as "right for the wrong reason"; the `GAE_NORM` predictions are
  scored. Honest, with the post-hoc substitution in M1 to be labelled.
- **Run inventory** — all 25 rows of the manifest are analysed; none dropped.
- **Known Bugs** — `grep` of the registry for `return_mode`, `eval_rollout`, `Episode/Steps` and
  `shared-trunk` returns the 2026-09-04 stale-seed-copy row (read-hazard only; the document reads
  the correct top-level keys) and three FIXED eval-tooling rows. Nothing this analysis collides
  with. No new registry row proposed.

## Cost of being wrong

No data is at risk and no run is wasted. The cost is entirely in what gets believed: as worded,
the document would put "constant 10x penalty, no ceiling in sight", "`GAE_NORM`'s edge is an
exploration artefact" and "shared-trunk mechanism refuted, entropy collapse is the real cause"
into the project's write-up. The first would be contradicted by any longer run if the upward
drift is real; the second would withdraw a reproducible five-seed result on the strength of a
null that could not see it, and would spend the next evaluation experiment chasing a discrepancy
that may not exist; the third repeats the exact pattern this investigation has been retracting
— a mechanism asserted on a measurement that does not bear on it — and would steer the §7
metrics request and the follow-up priorities toward the wrong quantity.

---

Reviewed by: plan-reviewer · 2026-09-05
