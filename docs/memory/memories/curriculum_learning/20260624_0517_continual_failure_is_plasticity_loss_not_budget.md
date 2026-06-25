---
id: 20260624_0517_continual_failure_is_plasticity_loss_not_budget
date: 2026-06-24
time: "05:17"
folder: curriculum_learning
tags: [learned_lesson, decision]
summary: "Per current RL literature, the continual curriculum's deficit is primarily LOSS OF PLASTICITY (a damaged-network problem), not under-training — so ~100x more target-level training is unlikely to beat the baseline on its own. The field has shifted from catastrophic forgetting (backward/stability failure) to loss of plasticity (forward/plasticity failure: the network loses the ability to learn new tasks). Fixes target trainability not memory: entropy floor / ReDo / stop recurrent-reset / shrink-and-perturb. Long-L4 run is best read as a diagnostic."
related: ["20260622_1748_basic_curriculum_overtraining_collapse_and_intervals", "20260624_0516_curriculum_underperformed_baseline_negative_transfer"]
session_origin: claude_code
session_label: "continual curriculum result + plasticity-vs-budget diagnosis + field-evolution references"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/96e71c7b-dc03-44c9-a98c-1c2acc86e0d9.jsonl
raw_completeness: full
---

# The continual failure is loss of plasticity, not a training-budget problem

## Key conclusion
Grounded in current RL literature, the curriculum's under-performance is primarily **loss of plasticity** — a partially "frozen"/damaged network that has lost the capacity to keep learning — **not** an under-training problem. So the launched ~100×-more-episodes-on-far-sight run is **unlikely to beat the from-scratch baseline on its own**; most of the extra compute will be wasted unless a plasticity-restoring intervention is added. The deeper reframe: the continual-learning field has moved from treating **catastrophic forgetting** (the *backward / stability* failure — overwriting old tasks) as the only problem, to recognizing **loss of plasticity** (the *forward / plasticity* failure — the network progressively loses the ability to learn *new* tasks). Our far-sight stage's signature — *stochastic policy + plenty of data + still can't climb* — is the textbook plasticity-loss signature, not the under-training signature.

## Evidence, measurements, facts
- **Field chronology**: catastrophic forgetting (McCloskey & Cohen 1989; deep-CL wave: EWC Kirkpatrick 2017, SI, Progressive Nets, GEM) → cracks appear (Ash & Adams 2020 warm-start hurts; Kumar 2021 rank collapse; Lyle 2022 capacity loss; Nikishin 2022 primacy bias + resets) → loss-of-plasticity as a first-class problem (Abbas 2023 continual deep RL; Sokar 2023 dormant neurons / ReDo; Dohare 2024 *Nature* + continual backprop).
- **Verdict reasoning**: from-scratch far-sight plateaued ~261 by ~3.6M, so episodes 4M→994M are a flat marginal curve run on a *degrading* substrate; Dohare/Abbas/Lyle characterize exactly this regime as one where more gradient steps do NOT recover performance.
- **Fixes (ranked)**: (1) entropy floor / adaptive entropy coefficient — cheapest, attacks the collapse that sank stages 0/1; (2) stop hard-resetting the recurrent state (carry/burn-in belief state, Caccia 2022); (3) ReDo / dormant-neuron recycling (directly targets the stuck far-sight plasticity failure); (4) shrink-and-perturb at boundaries; (5) CReLU activation.
- Critique memo: `docs/project/critiques/curriculum_underperformed_baseline_plasticity_vs_budget.md`. Field-evolution references primer (22 web-verified refs): `docs/project/references/continual_learning/continual_learning_field_evolution.md`.
- Single-seed caveat: magnitudes uncertain, but the qualitative signature + literature mapping are robust.

## Decisions and actions
- Do NOT expect the long-L4 run to beat the baseline; treat it as a **diagnostic** (a flat/eroding far-sight curve out to hundreds of M = plasticity confirmed). If it runs anyway, add ReDo to make it a plasticity-restoration test.
- **Minimum viable fix = entropy floor.** The NMN FiLM **temperature head is itself a candidate adaptive-entropy controller** — overlap with the project's neuromodulation/continual direction; the neuromodulated FiLM curriculum (`b81gyq0y`) probes exactly this.
- Built the `continual_learning` references library for the upcoming literature review.

## Open questions and follow-ups
- Which fix moves the needle most (entropy floor vs ReDo vs no-recurrent-reset)?
- The essential control to settle budget-vs-plasticity: a from-scratch far-sight run extended to the same ~994M budget (if flat, the budget hypothesis is dead).
- Will the FiLM temperature head act as a de-facto entropy floor and let `b81gyq0y` beat unmod `oq2vvh8g`?

## References
- Explains the result in [[20260624_0516_curriculum_underperformed_baseline_negative_transfer]]; builds on [[20260622_1748_basic_curriculum_overtraining_collapse_and_intervals]].
- Critique: [`curriculum_underperformed_baseline_plasticity_vs_budget`](../../../project/critiques/curriculum_underperformed_baseline_plasticity_vs_budget.md). References primer: [`continual_learning_field_evolution`](../../../project/references/continual_learning/continual_learning_field_evolution.md).
- R

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260624_0516_curriculum_underperformed_baseline_negative_transfer]] (curriculum_learning, 2026-06-24) — The 5-stage continual curriculum (RecurrentPPO, weights carried forward, recurre
<!-- END BACKLINKS -->
