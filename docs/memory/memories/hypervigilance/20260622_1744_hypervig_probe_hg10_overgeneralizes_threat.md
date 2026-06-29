---
id: 20260622_1744_hypervig_probe_hg10_overgeneralizes_threat
date: 2026-06-22
time: "17:44"
folder: hypervigilance
tags: [hypervigilance, learned_lesson, decision]
summary: "New conflict/hypervigilance behavior-probe: the hg10 rPPO checkpoint (~7.2M, partial) flees a harmless chasing rabbit identically to a damaging predator and starves avoiding food, while eating fine with no animal — a hypervigilance signature (caveat: mid-training, one seed)."
related: ["20260609_1720_chasing_rabbit_avoidance_damage_driven", "20260616_1514_experimental_env_as_behavior_platform", "20260623_1623_nutrition_runway_confounds_hypervig_read"]
session_origin: claude_code
session_label: "conflict + hypervigilance behavior-probe build & eval (v3.0)"
importance: high
status: superseded
superseded_by: ["20260624_0517_indist_random_init_reverses_hypervig"]
valid_until: 2026-08-31
confidence: medium
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4efbe660-28c2-4643-b231-d3c6d2635b5a.jsonl
raw_completeness: full
---

# Hypervigilance probe: hg10 over-generalizes threat to a harmless chasing rabbit

## Key conclusion
A new "conflict-needs" behavior probe (hungry+injured agent next to food, an animal approaching, the only cover/bush swept near/mid/far) plus its hypervigilance variant (replace the predator with a harmless chasing rabbit — identical hunt motion, only class/damage/smell differ) cleanly separates a discriminating agent from a hypervigilant one. The hunger-gated rPPO checkpoint **hg10** (run 20260620-180026_rppo_hg10_s0.5_sig0.4_dp1_s42, step 7200007, ~7.2M of ~10M, one seed) lands as **hypervigilant**: it flees the harmless rabbit essentially identically to the damaging predator and starves itself avoiding the food, yet eats and survives indefinitely when no animal is present. **Caveat**: hg10 is mid-training and single-seed, so this is "hypervigilant OR not-yet-learned discrimination" — not a settled scientific claim. **Confound caveat (added 2026-06-23):** a follow-up nutrition sweep shows this reading is partly confounded — at the start state used here (injury 70, nutrition 20) the agent starves during its wander-to-forage policy even with NO animal present, so much of the "flees the rabbit and starves" was a nutrition-runway artifact, not deliberate avoidance. See [[20260623_1623_nutrition_runway_confounds_hypervig_read]].

## Evidence, measurements, facts
- Three-way eval (all hg10, all decay-matched), 7×7 grid, start sat/nutr 20, injury 70:
  - **No animal** (control): survived 501 (full), reached food and ate, final nutrition 50.
  - **Harmless chasing rabbit**: survived 21, never reached food (fled), starved (nutrition 0); injury stayed 0.69 (rabbit did zero damage — confirmed harmless).
  - **Predator**: survived 16, never reached food (fled), killed (injury → 1.0).
- Trajectories for predator vs rabbit are near-identical at every bush distance (near/mid/far); at "far" both flee to the bush (min dist 1). Agent never reaches food in any animal-present condition (min food dist 1 = immediate flight).
- Configs: configs/environment/experiment/behavior_probes/explore/hypervigilance/{hv_pred,hv_rabbit}_{near,mid,far}.yaml + hv_ctrl_noanimal.yaml.
- Pipeline: eval_rollout.py --record (chasingRabbit/hg10 checkpoint) → .rec.gz → render_recordings.py (canonical renderer.py). Videos under results/eval/conflict_probe_preview/.

## Decisions and actions
- Built and committed the conflict probe (5 configs) and the hg10-matched hypervigilance set (6 + no-animal control). The predator-vs-harmless-chasing-rabbit contrast (same geometry+motion, only identity differs) is the hypervigilance assay.
- Probe verified end-to-end with a real frozen agent; result is interpretable and the controls behave as designed.

## Open questions and follow-ups
- Re-eval hg10 at the FINAL 10M checkpoint (is the hypervigilance a training-stage artifact?).
- Compare against the s=0 sweep member (fully separable smells): if s=0 discriminates but s=0.5 does not, it is a perceptual-separability effect, not pure hypervigilance.
- Multiple seeds; quantify across the full eval-seed set rather than 1-2 episodes.
- Re-run at adequate nutrition (70–100) to remove the starvation-runway confound (see [[20260623_1623_nutrition_runway_confounds_hypervig_read]]) before claiming animal-avoidance.

## References
- Design: docs/experiments/active/behavior_measures/experiment_environment_designs_v1.md §4 (Conflict probe).
- Contrast with [[20260609_1720_chasing_rabbit_avoidance_damage_driven]] — the MATURE chasing-rabbit agent (10M) showed the OPPOSITE (damage-driven discrimination, no flight from harmless rabbits); hg10 differs, likely due to training stage and/or the s=0.5 smell separability. This probe is the platform built in [[20260616_1514_experimental_env_as_behavior_platform]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 4efbe660-28c2-4643-b231-d3c6d2635b5a` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260622_1745_frozen_probe_eval_match_sensory_renderer]] (hypervigilance, 2026-06-22) — Frozen-checkpoint probe eval: the env config's sensory params (esp. decay_power)
- [[20260623_1623_nutrition_runway_confounds_hypervig_read]] (hypervigilance, 2026-06-23) — A nutrition sweep (injury 0, nutr 10->100) shows hg10's eat-failure is driven by
- [[20260624_0517_indist_random_init_reverses_hypervig]] (hypervigilance, 2026-06-24) — Re-running the probes on the IN-DISTRIBUTION model (random-init hg10, trained on
<!-- END BACKLINKS -->
