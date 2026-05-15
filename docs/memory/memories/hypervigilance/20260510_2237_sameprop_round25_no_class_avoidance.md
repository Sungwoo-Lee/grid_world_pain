---
id: 20260510_2237_sameprop_round25_no_class_avoidance
date: 2026-05-10
time: "22:37"
folder: hypervigilance
tags: [hypervigilance, refutation, learned_lesson, decision]
summary: "Round 2.5 (10M ep, n=1 each cell) refutes both pre-registered hypotheses: per-tag Δ_TL = +0.004 cells in Cell A1 (75× inside the H₀ band) confirms location-conditional corner-camping, not class recognition; Cell C aggregated Δ = −0.53 cells (sign-flipped vs R1's +0.63) with bilateral rabbit avoidance lands in §4.3 Inverted. The original sameProp survey effect (+0.6 cells, RPPO appears to keep predators farther than rabbits) decomposes into two confounds working in concert — food/quadrant overlap and spatial-avoidance camouflage — with no genuine class-conditional avoidance under matched olfactory smells. Provisional pending Round 2.6 seed 44 for Cell C; Cell A1's verdict is unconditional."
related: ["20260508_1444_sameprop_round1_finding_and_confound", "20260509_1532_sameprop_round2_truncated_verdict", "20260509_1533_tag_based_distance_supersedes_quadrant"]
session_origin: claude_code
session_label: "hypervigilance Round 2.5 launch + analysis"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/89124f20-6e84-466c-ab10-50a9651c68c8.jsonl
raw_completeness: full
---

# Round 2.5 — class-conditional avoidance under sameProp refuted; survey effect was a two-confound stack

## Key conclusion

When the patrolling predator and the neutral rabbits carry identical olfactory `properties`, the agent does **not** learn class-conditional avoidance. Round 2.5 ran the two pre-registered Round 2 cells to the full 10-million-episode budget on n106 single-seed each. **Cell A1 (predator passive, restricted to TL, rabbits at TL+BR)** produced per-tag Δ_TL = +0.004 cells — the agent's distance to the TL predator and the TL rabbit is statistically indistinguishable, with the agent surviving 486/500 steps in the BR corner. That is the smoking-gun signature of **location-conditional corner-camping**, 75× inside the §4.3 H₀ band ceiling. **Cell C (food decoupled to TR+BL, rabbits at TL+BR, predator full-grid)** produced aggregated Δ = −0.53 cells — rabbits *farther* than the predator — with bilateral rabbit avoidance (both rabbit tags 1.48–1.76 cells past predator distance). Round 1's apparent +0.63-cell rabbit-closer effect therefore was substantially **food/quadrant coupling**, not class recognition. Combining the two cells: the post-hoc survey finding that opened this study (a +0.6-cell gap interpreted as "RPPO discriminates predators under sameProp") decomposes into **two confounds working in concert** — neither of which is class-conditional avoidance. The result is methodologically clean: the per-tag distance metric shipped this week directly exposed the corner-camping signature that the aggregated metrics could not, and the §5 row-2 failure-mode catalog row written speculatively at design time fired exactly as written.

## Evidence, measurements, facts

- **Cell A1**: WandB `nm8gn7y2`. Final episode 10,000,003. Last-10% window (9.0–10.0M ep), mean across 128 envs of ~1880 history records:
  - Aggregated `Episode/MeanDistRabbit` ≈ 2.58, `Episode/MeanDistPredator_TL` ≈ 2.58 → per-tag **Δ_TL = +0.004 cells**.
  - `Episode/Steps` = 485.7 / 500 (camping signature; agent dies almost never).
  - `Episode/Term_MaxSteps` = 0.924 (92% of episodes end on the step cap, not by death).
  - §4.3 verdict: **H₀(A1) confirmed at 75× margin** — at 1.3% of the 0.3-cell H₀-band ceiling, 0.4% of the 1.0-cell H₁-band floor.
  - §4.4 stability: per-window σ across the last 3 sub-windows is three orders of magnitude smaller than the 1.0-cell threshold band. Stable.
  - Mid-training transient (windows 4–7, ~3M–7M ep): the agent briefly de-saturated from the camping basin and re-entered. The verdict is robust across the transient — windows 8–10 are flat. Documented in design doc §10.5 as a curiosity.
  - §6 designer's prior: 70% on H₀(A1) confirmed (matches).
- **Cell C**: WandB `bdnfc0lu`. Final episode 10,000,022. Last-10% window:
  - Aggregated **Δ = MeanDistPredator − MeanDistRabbit = −0.532 cells** (rabbits farther than predator). ΔH = −2.39/ep.
  - Per-tag cross-check: `Episode/MeanDistRabbit_TL` and `Episode/MeanDistRabbit_BR` are both 1.48–1.76 cells past `Episode/MeanDistPredator_full` — bilateral rabbit avoidance, not a single-quadrant artefact.
  - §4.2 verdict: lands in the **Inverted band** (Δ ≤ −0.3 cells). H₁(C) refuted; H₀(C) supported in spirit (food-coupling was load-bearing for R1) but the agent's response went past neutral into active rabbit avoidance.
  - §6 designer's prior: 60% on the stabilised-inverted branch (matches).
  - Headline number is at 1.6–1.8× the §4.2 threshold — just shy of the §5 row-8 2× single-seed elevation bar. **Provisional**; Round 2.6 seed 44 is queued.
- **Round 1 reference values** (the comparator at 7.2M ep): MeanDistRabbit ≈ 3.77, MeanDistPredator ≈ 4.40, Δ = +0.63, ΔH = +3.1/ep. Round 2.5 Cell C reverses both signs.
- **Random-policy baseline** on 10×10: ≈ 4.68 cells. Both Cell A1 and Cell C distance metrics sit below this — the agent is doing *something* spatial; it just isn't class-conditional.
- **Per-tag tooling earned its keep**: without `Episode/MeanDistRabbit_TL` vs `Episode/MeanDistPredator_TL`, Cell A1's aggregated `Episode/MeanDistPredator − Episode/MeanDistRabbit` was ~+3.86 cells (~6× R1's gap, ~10× the H₁ confirmation margin). That single number, in isolation, would have been published as "class avoidance dramatically confirmed." The per-tag pair flipped the verdict by surfacing that `MeanDistRabbit_TL ≈ MeanDistPredator_TL` — the same-corner pair was treated identically. Pre-registered §5 row-2 ("agent never visits TL → MeanDist looks high purely by spatial separation") fired exactly as written.
- Wall clock: ~18h (Cell C) and ~22h (Cell A1) on n106 RTX 3090.
- Run on `v1.3` branch, post-implementation of per-tag metrics (commits `0a73613` + `6d3d382`).

## Decisions and actions

- Frontmatter on `docs/experiments/active/hypervigilance/sameprop_round25_design.md` updated to `status: analyzed`, `last_updated: 2026-05-10`. §§9–11 written. Headline-finding banner added below status block. Commit `82ae039`.
- Diary `training-done` rows logged for both tags at 21:46 on 2026-05-09 (the runs finished within the 2026-05-09 → 2026-05-10 wrap; the analyzer used the open training-start row's date for the lookup, which is correct per `diary_append.py` semantics — `training-done` matches by tag, not by date).
- **Round 2.6 queued**: re-launch Cell C with seed 44 only (Cell A1 does not need seed 45 — its verdict is 75× past threshold). Same config (`02-sameProp_R2_decoupleFood.yaml`), same node 106, ~22h. Cell-C verdict is at 1.6–1.8× threshold and the §6 prior assigned 40% to the alternative branch, so seed-locking is worth the GPU-day before the inverted finding is treated as final.
- Wider arc: after Cell C is seed-locked, the next study question for the hypervigilance research arc closes the spatial-avoidance loophole entirely. Likely Round 3 design: food in all four quadrants (no quadrant is uniformly safe), so any future channel-attribution work runs on a clean baseline where the agent cannot trivially camp.
- Original sameProp survey ("RPPO appears to discriminate predators under matched smells") and Round 1 relog finding (+0.63 cells at 7.2M ep, n=2) are **not** standalone class-conditional avoidance results. They should be cited as "post-hoc + replicated effect that two pre-registered follow-ups attributed to confounds."
- The companion study summary `docs/experiments/summaries/20260509_1552_sameprop_rabbit_avoidance_study.md` was written before Round 2.5 finished and reports R2.5 as "running." A re-summary (new dated file in `summaries/`, append-only) is now appropriate; the old summary stays as a historical snapshot of what was known on 2026-05-09 15:52.

## Open questions and follow-ups

- **Round 2.6 (Cell C only, seed 44)** — does the inverted Δ persist at a second seed? The aggregated Cell-C verdict is 1.6–1.8× threshold; designer's prior gave 40% to the alternative branch. Schedule on next n106 rotation.
- **Why does the agent in Cell C actively avoid rabbits (not just go neutral)?** The agent reached a stable equilibrium 0.5 cells below predator distance. Possible explanation: with food and rabbits in disjoint quadrants, the food-eat trajectory takes the agent through the food quadrants only; rabbits sit in quadrants with nothing else interesting. This is a "post-hoc plausible" explanation — not tested. Could be checked by adding a "fraction of episode in rabbit quadrants" metric in Round 3.
- **Cell A1 mid-training transient** (windows 4–7, agent briefly de-saturated from camping basin). The verdict is robust across the transient, but the dynamics are unusual — what triggered the brief excursion, and why did the agent return to camping? Worth a future ablation if the mid-training metric becomes important elsewhere.
- **Generalisation to NMN agents**: Round 2.5 used plain RPPO. Whether NMN/FiLM agents under the same sameProp + Cell A1 setup also corner-camp or whether the modulator changes the policy's relationship to the safe corner is an open hypervigilance question. Out of scope for sameProp resolution but a natural sequel.
- **Random-start position**: both cells used `random_start_pos: true`. The corner-camping equilibrium is reached *despite* the agent occasionally spawning in TL. The §5 row-2 failure mode pre-registered the *converged* behaviour, not the *trajectory* into it; future analyses might benefit from "fraction of episode-1 steps near predator-tag entity" as an early-training signal.

## References

- Design doc: `docs/experiments/active/hypervigilance/sameprop_round25_design.md` (status `analyzed`; headline banner; §§9–11 with the verdict + cross-round contrast + Round-2.6 escalation; Appendix C escalation policy fired for Cell C).
- Predecessor design + truncated verdict: `docs/experiments/active/hypervigilance/sameprop_round2_design.md` §9 (the partial Round 2 readings that pointed at this verdict before R2.5 confirmed it).
- Round 1 baseline analysis: `docs/experiments/active/hypervigilance/round1_relog_baseline_analysis.md` (the +0.63-cell, two-seed comparator that Round 2.5 Cell C inverts).
- Per-tag metric design rationale: insight `20260509_1533_tag_based_distance_supersedes_quadrant` (the design decision that this verdict empirically validates).
- Per-tag implementation plan: `docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md` (`status: implemented`, `verification_status: pass`).
- Round 2 truncated verdict: insight `20260509_1532_sameprop_round2_truncated_verdict` (the SIGINT'd partial run that pointed at the same verdict at ~5% of the budget).
- Sibling insight: `20260508_1444_sameprop_round1_finding_and_confound` (the Round 1 verdict + confound flag that motivated Round 2 / 2.5).
- Sibling insight: `20260508_1445_sameprop_discriminating_channels` (Phase-1 channel-ranking memo; the channel-level mechanism Round 2 / 2.5 was testing).
- Diary: `docs/diary/2026-05-09.md` (training-done rows for both R2.5 cells at 21:46 on 2026-05-10).
- Study summary (snapshot, pre-R2.5-finish): `docs/experiments/summaries/20260509_1552_sameprop_rabbit_avoidance_study.md` — needs a re-summary dated post-R2.5 to incorporate this verdict.
- Implementation commits: `0a73613` (per-tag feature), `6d3d382` (RPPO Site 1 fix), `82ae039` (R2.5 analysis).
- WandB runs: `nm8gn7y2` (Cell A1, https://wandb.ai/sungwoolee/grid_world_pain/runs/nm8gn7y2), `bdnfc0lu` (Cell C, https://wandb.ai/sungwoolee/grid_world_pain/runs/bdnfc0lu).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 89124f20-6e84-466c-ab10-50a9651c68c8` (re-enter the session) or `python scripts/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/89124f20-6e84-466c-ab10-50a9651c68c8.jsonl /tmp/20260510_2237.md` (one-shot markdown view).
