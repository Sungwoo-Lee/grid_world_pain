---
id: 20260624_0516_curriculum_underperformed_baseline_negative_transfer
date: 2026-06-24
time: "05:16"
folder: curriculum_learning
tags: [learned_lesson, refutation, decision]
summary: "The 5-stage continual curriculum (RecurrentPPO, weights carried forward, recurrent state hard-reset at each boundary) UNDERPERFORMED from-scratch single-task training: no stage learned faster, the fast-predator stage showed negative transfer (~221 vs ~418 from-scratch), warm-starting ACCELERATED the easy-stage entropy collapse (stage 1 degenerated to entropy~0 inside its 1M cap, far earlier than the ~9M from-scratch onset), and end-of-curriculum far-sight (~200-300) tied the from-scratch baseline (~261) despite 6M extra episodes. The curriculum's transfer bet was net-NEGATIVE."
related: ["20260622_1748_basic_curriculum_overtraining_collapse_and_intervals", "20260624_0517_continual_failure_is_plasticity_loss_not_budget"]
session_origin: claude_code
session_label: "continual curriculum result + plasticity-vs-budget diagnosis + field-evolution references"
importance: high
status: settled
valid_until: null
confidence: medium
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/96e71c7b-dc03-44c9-a98c-1c2acc86e0d9.jsonl
raw_completeness: full
---

# Continual curriculum underperformed the from-scratch baseline (negative transfer)

## Key conclusion
A 5-stage continual curriculum on the basic difficulty ladder (RecurrentPPO, network weights carried forward across stages, but the recurrent hidden state HARD-RESET and the env rebuilt at every boundary) **lost to from-scratch single-task training** on the target. The curriculum bet — that knowing an easier world makes the next, harder world learn faster — failed on every axis: no stage learned faster than from scratch; the fast-predator stage suffered **negative transfer**; warm-starting *accelerated* the easy-stage policy-entropy collapse; and the final hardest world ended no better than a fresh agent despite 6M episodes of "head start." Net transfer was negative, not positive. This **refines / partly refutes** [[20260622_1748_basic_curriculum_overtraining_collapse_and_intervals]], whose design prediction ("short early stages dodge the collapse by design") was wrong — carry-forward made the collapse worse, not better. (Its from-scratch convergence numbers still hold; only the dodge prediction failed.)

## Evidence, measurements, facts
- **Per-stage end-of-stage survival (curriculum) vs from-scratch plateau**: static 473 vs 448 (ok); slow **94 (collapsed)** vs 361; fast **221** vs 418 (negative transfer); rabbit 383 vs 428 (ok, healthiest); far-sight ~200-300 vs **261** (equal-to-worse).
- **Warm-start accelerated collapse**: stage 1 reached entropy~0 / near-deterministic (one action ~55/75 steps) WITHIN its 1M-episode cap — earlier than its ~9M from-scratch collapse onset. Primacy-bias-style early over-fit, accelerated by the warm start.
- **Transition cost**: every boundary dropped survival ≥100 steps (recurrent-state reset + env rebuild); hard stages took 0.5–1M episodes to recover; the stage-0→1 dip never recovered.
- **Data**: WandB `mpql5i25` (curriculum, 10M, completed); survival reconstructed from the 128-env training-mean Episode/Steps AND the 60-checkpoint greedy-eval recordings — the two sources agree on direction. From-scratch baselines = the 5 standalone runs (far-sight = `c8cd77ft`). Single seed (42) — direction robust across two sources, magnitude uncertain. Metric = survival steps, never reward.
- Full analysis: `docs/experiments/active/basic_curriculum/basic_curriculum_continual_result.md`.

## Decisions and actions
- Verdict: the curriculum as scheduled "did not work" — only stages 0 and 3 trained cleanly.
- Two root causes isolated: (a) **no entropy floor** → easy-stage collapse; (b) **hard recurrent-state reset** at boundaries → transition cost + part of the negative transfer.
- Follow-ups launched: long-L4 unmodulated (`oq2vvh8g`, ~994M on far-sight) and neuromodulated FiLM long-L4 (`b81gyq0y`) — the modulator's temperature head as a candidate adaptive-entropy controller.
- The literature-grounded diagnosis (why, and whether more training fixes it) is [[20260624_0517_continual_failure_is_plasticity_loss_not_budget]].

## Open questions and follow-ups
- Single seed — re-run with ≥2 seeds for magnitudes.
- Does the FiLM modulator (`b81gyq0y`) mitigate the collapse vs unmod (`oq2vvh8g`)?
- Does carry-forward ever help once an entropy floor + no-recurrent-reset are applied?

## References
- Refines [[20260622_1748_basic_curriculum_overtraining_collapse_and_intervals]] (its from-scratch numbers stand; its dodge-collapse prediction is refuted here).
- Analysis: [`basic_curriculum_continual_result`](../../../experiments/active/basic_curriculum/basic_curriculum_continual_result.md). Runs: curriculum `mpql5i25`; baseline far-sight `c8cd77ft`.
- Diagnosis + literature: [[20260624_0517_continual_failure_is_plasticity_loss_not_budget]].
- R

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260624_0517_continual_failure_is_plasticity_loss_not_budget]] (curriculum_learning, 2026-06-24) — Per current RL literature, the continual curriculum's deficit is primarily LOSS 
<!-- END BACKLINKS -->
