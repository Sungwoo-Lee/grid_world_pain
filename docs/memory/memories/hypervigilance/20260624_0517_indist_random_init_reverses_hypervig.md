---
id: 20260624_0517_indist_random_init_reverses_hypervig
date: 2026-06-24
time: "05:17"
folder: hypervigilance
tags: [hypervigilance, refutation, learned_lesson]
summary: "Re-running the probes on the IN-DISTRIBUTION model (random-init hg10, trained on random nutrition[10,100]/injury[0,80], decay 2.0, 10M) REVERSES the hypervigilance finding: the agent eats at every nutrition level, ignores both harmless rabbits, and only the real predator kills it. The earlier 'hypervigilance/eat-failure' was an off-distribution artifact of evaluating a STATIC-init model on depleted/injured starts it never trained on."
related: ["20260622_1744_hypervig_probe_hg10_overgeneralizes_threat", "20260622_1745_frozen_probe_eval_match_sensory_renderer", "20260623_1623_nutrition_runway_confounds_hypervig_read"]
session_origin: claude_code
session_label: "conflict + hypervigilance behavior-probe build & eval (v3.0)"
importance: high
status: settled
valid_until: 2026-08-31
confidence: medium
supersedes: ["20260622_1744_hypervig_probe_hg10_overgeneralizes_threat"]
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4efbe660-28c2-4643-b231-d3c6d2635b5a.jsonl
raw_completeness: full
---

# In-distribution (random-init) model reverses the hg10 "hypervigilance" read

## Key conclusion
The earlier "hg10 is hypervigilant / fails to eat" conclusion was an **off-distribution artifact of the model**, not a behavioral property. The model used before (the dp1 hg10, and chasingRabbit) trained **always from full nutrition + zero injury** (random_start flags false), so the hungry/injured/depleted *starts* used in the probes were states it never saw. Re-running the same nutrition sweep on the **in-distribution** model — the random-init hg10 (run 20260619-024734_rppo_hg10_s0.5_sig0.4_s42, trained with random_start nutrition [10,100] / injury [0,80], decay_power 2.0, 10M, ckpt 10000063) — **reverses the result**: the agent eats the adjacent food at **every** nutrition level (incl. nutr10), **ignores both** the chasing and wandering harmless rabbits (eats and survives near them), and **only the actual damaging predator** kills it (and even then it reaches the food first). So: **no hypervigilance, correct discrimination.** Methodology lesson: before reading behavior off a frozen-checkpoint probe, verify the eval **start-state lies within the model's random_start training ranges** — a static-init model produces off-distribution wandering/starving that masquerades as avoidance.

## Evidence, measurements, facts
- Eat-vs-flee map, random-init model (injury 0, EAT/501 = ate & survived full episode):
  - no-animal: EAT/501 at nutr 10/30/50/70/100 (eats even at nutr10 — the static model STARVED here, FLEE/11).
  - chasing rabbit: EAT/501 at every level. wander rabbit: EAT/501 at every level.
  - predator: ate/18-21 (reaches food and eats, predator then kills it; survives slightly longer at nutr100).
- Both models are hg10 s0.5/sig0.4; they differ in random_start (false->true) and decay_power (1.0->2.0). The random-init model's training ranges (nutr[10,100], inj[0,80]) cover all probe start states.
- Configs: configs/environment/experiment/behavior_probes/nutrition_sweep_d2/ (decay 2.0). Eval videos: results/eval/conflict_probe_preview/random_init/ (paired with static_init/).
- Caveats: one seed, one episode per cell; 7x7 single-corner-food geometry is still mildly off-distribution vs the 10x10 quadrant-food training (but the agent eats fine, so it isn't breaking).

## Decisions and actions
- Supersedes [[20260622_1744_hypervig_probe_hg10_overgeneralizes_threat]] — its "unconditional hypervigilance" claim is overturned.
- Reframes [[20260623_1623_nutrition_runway_confounds_hypervig_read]]: the wander-away/runway behavior it documented was the static model's off-distribution response, not a general agent property; on the in-distribution model it disappears.
- For all initial-state probes, use a random-init model and match decay_power to it (here 2.0). Reorganized eval outputs into static_init/ vs random_init/ with a paired naming framework animal_injII_nutrNNN_bushDD.

## Open questions and follow-ups
- Re-run on the decay-1.0 + random-init model (the olfactory_ambiguity_lindecay sweep, still training) for the exact olfaction+init match.
- Multiple seeds / full eval-seed set; quantify rather than 1 episode/cell.
- Predator kills the agent at ~18 steps regardless of nutrition — under-vigilance or 7x7 probe over-lethality? Test a 10x10 quadrant-food (training-like) geometry to remove the residual geometry off-distribution.

## References
- Supersedes [[20260622_1744_hypervig_probe_hg10_overgeneralizes_threat]]; reframes [[20260623_1623_nutrition_runway_confounds_hypervig_read]]; extends the eval-matching lesson in [[20260622_1745_frozen_probe_eval_match_sensory_renderer]] (match not just sensory params but the body-state training distribution).
- Raw conversation: synced via ./sync-agent-data.sh claude push (UUID 4efbe660-28c2-4643-b231-d3c6d2635b5a).

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
