---
id: 20260622_1748_basic_curriculum_overtraining_collapse_and_intervals
date: 2026-06-22
time: "17:48"
folder: curriculum_learning
tags: [learned_lesson, decision]
summary: "From-scratch rPPO survival-step curves on the basic curriculum: the EASY levels (static-forage L0, slow-predator L1) learn fast (~0.2M episodes) but then CATASTROPHICALLY COLLAPSE from over-training (L0 ~3.8M, L1 ~9.0M) to a degenerate single-action 'spam one direction -> starve' policy with entropy ~0 (rPPO has no entropy floor / early stop). Stable levels converge L2 ~1.4M, L3 ~1.2M, L4 ~3.6M. The 5-stage continual curriculum budgets [1M,1M,2M,2M,4M] were set by capping each easy stage BELOW its measured collapse onset, so the curriculum's short early stages dodge the instability by design."
related: ["20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin"]
session_origin: claude_code
session_label: "basic curriculum reframe -> continual-learning schedule (data-driven intervals)"
importance: high
status: settled
valid_until: null
confidence: medium
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/96e71c7b-dc03-44c9-a98c-1c2acc86e0d9.jsonl
raw_completeness: full
---

# Basic curriculum: easy levels over-train-collapse; curriculum intervals set below collapse onset

## Key conclusion
Measuring per-level **survival steps** (project rule: never reward) on the 5-level basic curriculum trained from scratch with RecurrentPPO (10M episodes each, num_envs=128), the EASY worlds learn fastest but are UNSTABLE: L0 (static/forage) and L1 (slow predator) hit a strong policy by ~0.2M episodes, then **catastrophically collapse from over-training** — L0 degrades to ~92 survival steps around ~3.8M, L1 to ~129 around ~9.0M, ending in a degenerate single-action policy (L0 spammed one direction 91/101 actions; entropy → 0). This is an RecurrentPPO over-training instability: there is no entropy-coefficient floor or eval-based early stop, so an easy world trained too long degenerates. The harder, richer worlds are stable: L2 (fast, ~1.4M, plateau ~418), L3 (+rabbit, ~1.2M, ~428 — healthiest), L4 (far-sight, ~3.6M, ~261 — slowest/lowest). This drove the continual-curriculum stage budgets: each easy stage is budgeted BELOW its collapse onset, so a curriculum's short early stages sidestep the instability — a positive reason curriculum > long single-stage here.

## Evidence, measurements, facts
- **Per-level converged-by (≥90% plateau) and plateau (survival steps)**: L0 ~0.2M → 448 then COLLAPSE→92 @~3.8M; L1 ~0.2M → 361 then COLLAPSE→129 @~9.0M; L2 ~1.4M → 418 (stable); L3 ~1.2M → 428 (stable, healthiest); L4 ~3.6M → 261 (stable, hardest).
- **Metric source**: WandB local files were pruned, so curves were reconstructed from local eval recordings (50 checkpoints/run × 3 greedy eval episodes; survival = len(actions)); recording dir name = training episode, so x-axis is episodes directly. Full write-up: `docs/experiments/active/basic_curriculum/basic_curriculum_convergence.md`.
- **Curriculum schedule chosen**: cumulative `episode_boundaries = [1000000, 2000000, 4000000, 6000000, 10000000]` (per-stage 1M/1M/2M/2M/4M). Easy stages 1M each are well below their 3.8M/9.0M collapse onsets; L2/L3 (~1.2–1.4M) get 2M; L4 (~3.6M) gets 4M final. Committed in `4a42738` (`configs/continual/basic_curriculum_schedule.yaml`).
- **Caveat**: single seed per level — collapse onset is likely seed-sensitive; eval signal noisy at 3 episodes/checkpoint, so budgets carry margin.

## Decisions and actions
- **Methodology (reusable)**: set continual-curriculum stage budgets from measured *from-scratch* survival-step convergence per level, then CAP each stage below any over-training collapse onset, with margin for curriculum weight carry-forward. Don't just give every stage the same large budget.
- **Flagged separate fix**: the L0/L1 over-training collapse is a genuine RecurrentPPO instability (no entropy floor / no eval-based early stop). The curriculum sidesteps it, but any easy world trained long needs an entropy-coefficient floor or early stop — a candidate bug-fix, not patched here.

## Open questions and follow-ups
- Does curriculum weight carry-forward let later stages converge FASTER than their from-scratch numbers? Measurable on the running curriculum (`mpql5i25`, node 106) via the `stage/transition` markers.
- Collapse onset under multiple seeds (single-seed here).
- Distinct from cross-stage catastrophic FORGETTING seen in the NMN continual probe ([[20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin]]) — that is forgetting across a passive/active schedule; this is within-level over-training collapse on a single static world.

## References
- **Why a new folder**: `curriculum_learning` holds curriculum/continual training findings and stage-budget methodology. Closest existing is `cluster_ops` (lab ops/env mgmt) or `dreamer_diagnosis` (Dreamer-specific) — but this is an algorithm-agnostic training-dynamics + curriculum-design finding that will grow as the curriculum program continues, so it earns its own discoverable folder rather than hiding in ops or a model-specific folder.
- Analysis: [`basic_curriculum_convergence`](../../../experiments/active/basic_curriculum/basic_curriculum_convergence.md). Curriculum design: [`basic_curriculum`](../../../experiments/active/basic_curriculum/basic_curriculum.md).
- Schedule + intervals: commit `4a42738`. Related continual-forgetting insight: [[20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 96e71c7b-dc03-44c9-a98c-1c2acc86e0d9` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260622_1748_basic_curriculum_overtraining_collapse_and_intervals.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
