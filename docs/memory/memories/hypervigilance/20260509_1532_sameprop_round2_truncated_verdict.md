---
id: 20260509_1532_sameprop_round2_truncated_verdict
date: 2026-05-09
time: "15:32"
folder: hypervigilance
tags: [hypervigilance, refutation, learned_lesson]
summary: "Round 2 SIGINT'd at 0.4M ep / 3.5h: H1(C) provisionally refuted (Δ=−0.43 sign-flipped vs R1's +0.63); Cell A1's striking Δ=+3.86 is structurally uninterpretable because the agent survives 482/500 steps in the BR corner — §5 row-2 failure mode is active and cannot be discounted without per-tag distance metrics."
related: ["20260508_1444_sameprop_round1_finding_and_confound", "20260508_1445_sameprop_discriminating_channels"]
session_origin: claude_code
session_label: "hypervigilance Round 2 partial-verdict + per-tag metrics ship"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/89124f20-6e84-466c-ab10-50a9651c68c8.jsonl
raw_completeness: full
---

# SameProp Round 2 truncated-data partial verdict

## Key conclusion

Both Round 2 cells (`0u266oj5` Cell C decoupleFood seed 42; `27svrmhv` Cell A1 passivePredator seed 43) were SIGINT'd at ~3.5h / ~0.4M episodes — about 4–5% of the 10M-episode pre-registered budget. Per-cell readouts at the truncated end:

- **Cell C**: Δ = MeanDistPredator − MeanDistRabbit = **−0.43** (sign-flipped vs R1's converged +0.63), ΔH = −1.87/ep, §4.4 stability **failed** (still moving). Provisional verdict: **H₁(C) refuted in spirit** — the Round-1 effect is plausibly food-quadrant collinearity, but the sign-flip suggests the agent is now actively avoiding the rabbit quadrants rather than going neutral. Needs re-launch to confirm.
- **Cell A1**: Δ = **+3.86** (~6× R1's +0.63), ΔH = +17.0/ep, §4.4 stability **passed**. Statistically would confirm H₁(A1) by 10× margin, BUT operationally indistinguishable from §5 row-2 failure mode (agent never visits TL — survives 482/500 steps in BR rabbit/food corner). **Verdict UNINTERPRETABLE** without per-quadrant or per-tag distance evidence.

## Evidence, measurements, facts

- Authoritative WandB runs (DO NOT junk): `0u266oj5` (Cell C, 0.49M ep), `27svrmhv` (Cell A1, 0.39M ep).
- Junk WandB runs (CIFS-cache duplicate): `f5d933ro` (a second Cell C launched on cuda:0 by mistake — see `20260508_1446_cifs_race_node112_recurrence`).
- Last log lines (`logs/20260508_141750.log`, `logs/20260508_142015.log`): "Shutdown signal received (signal 2)" — i.e., SIGINT, not SIGTERM. Manual interrupt by user.
- Cell C metrics at 0.49M ep:
  - MeanDistRabbit ≈ 4.39, MeanDistPredator ≈ 3.96, → Δ = −0.43 (rabbits now FURTHER than predators)
  - RabbitHits ≈ 1.87/ep, PredatorHits ≈ 3.74/ep → ΔH = −1.87/ep
  - Survival ≈ 240 steps (vs R1 ceiling ~340)
- Cell A1 metrics at 0.39M ep:
  - MeanDistRabbit ≈ 3.84, MeanDistPredator ≈ 7.70, → Δ = +3.86
  - RabbitHits ≈ 17.4/ep, PredatorHits ≈ 0.4/ep → ΔH = +17.0/ep
  - Survival ≈ 482/500 steps — agent essentially never dies; consistent with safe-corner camping.
- Random-policy baseline on 10×10: MeanDist ≈ 4.684 (uniform-random reference).
- R1 reference values at 7.2M ep: MeanDistRabbit 3.77 / MeanDistPredator 4.40 / Δ = +0.63 / ΔH = +3.1/ep.
- Pre-registered §4 thresholds were keyed to the last-10% window of 10M episodes (≈9.0M–10.0M); the actual data falls in the early-learning regime, NOT the converged regime. Round 1 metrics did not stabilize until ~5M episodes — these readouts are 10× too early to apply §4 thresholds directly.

## Decisions and actions

- User chose option 2: "Analyze truncated data, document partial verdict" rather than re-launching at 10M budget.
- §9 partial-analysis section appended to the design doc (`docs/experiments/active/hypervigilance/sameprop_round2_design.md`) with bold truncation banner; frontmatter `status: active` → `status: partial`.
- Diary `training-done` rows logged at 20:03 on 2026-05-08 with one-line truncation results.
- **Re-launch is gated on the §7 metrics feature** (per-tag distance, captured in sibling insight `20260509_1533_tag_based_distance_supersedes_quadrant`). Re-running Round 2 at 10M episodes without per-tag metrics will not resolve Cell A1's quadrant-vs-class ambiguity — it would just produce the same uninterpretable Δ=+3.86 at convergence.

## Open questions and follow-ups

- Is Cell C's sign-flip (Δ=−0.43) a transient early-learning artefact, or a genuine "rabbits now systematically further than predators" effect that would persist to convergence? Only re-launch at 10M episodes resolves this.
- Cell A1's quadrant-camping behavior (482/500 in BR) suggests the agent learned that BR is uniformly safe rather than discriminating predator class. Per-tag metrics will distinguish the two; design doc §5 row-2 failure-mode catalog covers this.
- Did the SIGINT come from another session needing the GPUs on n112, or from the user manually killing them after seeing the design's 10M-episode horizon as too long? Not investigated. Either way, the operational lesson is: future Round-2-class designs should plan checkpoint cadence around the 5M-ep stabilization shoulder so partial-data verdicts are at least possible.

## References

- Design doc: `docs/experiments/active/hypervigilance/sameprop_round2_design.md` (§9 partial analysis added in commit `42cc049`).
- Round 1 baseline: `docs/experiments/active/hypervigilance/round1_relog_baseline_analysis.md`.
- Channels memo: `docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`.
- Sibling insight (this session): `20260509_1533_tag_based_distance_supersedes_quadrant` — the metric that gates re-launch.
- Sibling insight (this session): `20260509_1534_synthetic_smoke_masks_dict_assembly_bugs` — the wiring bug that almost shipped silently.
- Diary rows (`docs/diary/2026-05-08.md` lines ~68–69): training-done at 20:03 with truncation summaries.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 89124f20-6e84-466c-ab10-50a9651c68c8` (re-enter the session) or `python scripts/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/89124f20-6e84-466c-ab10-50a9651c68c8.jsonl /tmp/20260509_1532.md` (one-shot markdown view).
