---
id: 20260810_1753_bushrefuge_injury_suppresses_bush_use_heal_by_rest
date: 2026-08-10
time: "17:53"
folder: behavior_measures
tags: [behavior_measures, refutation, learned_lesson, design, decision]
summary: "Bush-refuge rPPO agents DO hide heavily (predator 34-42%, rabbit-chase 26-32%, rabbitwander 11-14% of episode; bush_dwell is a 0-1 FRACTION, not steps). Robust anti-hypervigilant effect: injury REDUCES cover use in the harmless wandering-rabbit conditions (inj00>inj70 by ~4.5-8.6 pp, block-bootstrap CI excludes 0 in BOTH b03+b04). Cause confirmed at per-step level: while injured the agent rests ~86% in the OPEN to heal (heal-only-by-rest + homeostatic reward), collapsing bush use to 1-3%; the whole-episode gap is made entirely in the injured window. Corrects/supersedes 20260805_0117."
related: ["20260624_0516_bush_blocks_animals_movement_toggle", "20260710_1635_bush_hiding_metastable_dwell_measure", "20260805_0117_bushrefuge_survives_by_evasion_not_refuge", "20260805_0118_regime_matched_probe_and_run_sweep_probe_dir"]
session_origin: claude_code
session_label: "bush_dwell units correction + hypervigilance mechanism"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: ["20260805_0117_bushrefuge_survives_by_evasion_not_refuge"]
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4efbe660-28c2-4643-b231-d3c6d2635b5a.jsonl
raw_completeness: full
---

# Injury suppresses bush use in bush-refuge agents via a heal-by-rest confound (not hypervigilance)

## Key conclusion
Two things. (1) UNITS: `bush_dwell` in the dwell-sweep CSVs is a FRACTION in [0,1] (= % of episode steps on the bush cell), NOT steps — an earlier read treated 0.42 as "~0 steps" and wrongly concluded the agents don't hide. They DO: predator ~34-42%, rabbit-chase ~26-32%, rabbitwander ~11-14%. (2) FINDING: injury has a robust, replicated, ANTI-hypervigilant effect on cover use in the harmless wandering-rabbit conditions — the agent hides MORE when uninjured (inj00) than injured (inj70). The cause is an environmental confound: injury heals ONLY via the Rest action (stay still), rewarded by the homeostatic drive, so while injured the agent freezes-and-heals in place instead of relocating to cover.

## Evidence, measurements, facts
- Paired (per-checkpoint) inj00-inj70 bush_dwell%, block-bootstrap 95% CI (L=20 blocks, respects checkpoint autocorrelation): **rabbitwander b03 +4.5 [+3.3,+6.4], b04 +5.8 [+3.9,+7.3]; rabbitwander_predsmell b03 +7.0 [+5.1,+9.5], b04 +8.6 [+5.8,+10.9]** — all exclude 0, both agents. Predator/rabbit-chase are NOT robust (block-boot CI spans 0; a naive Wilcoxon FALSELY flagged b03 rabbit-chase p=2.5e-3 -> autocorrelation artifact).
- Per-step confirmation (pooled ~40 ckpts, 121k steps/cell), inj70 rabbitwander split by injury>5: in_bush INJURED window = 1.0% (b03) / 2.9% (b04) vs RECOVERED window 9.9% / 8.3% vs inj00 baseline 13.6% / 10.7%. The gap is made entirely in the injured window.
- Rest-action (action==4) rate while INJURED = 86% (both) vs 55-73% otherwise; when resting while injured, on-bush = 1-3% (rests in the OPEN). So injured -> rest-in-place-to-heal, not travel-to-cover.
- Env mechanics (src/environment/core.py): injury recovers only when `rested` (action 4), accelerating with rest-streak (recovery_accel_rate 0.5); reward = homeostatic drive reduction, drive=(1-sat/100)^2+(injury/100)^2, so healing is directly rewarded; injury heals fully (~-70) over ~35 steps then the agent behaves ~inj00.

## Decisions and actions
- Superseded [[20260805_0117_bushrefuge_survives_by_evasion_not_refuge]] (units error).
- Methodology locked in: (a) bush_dwell is a fraction; (b) use a BLOCK BOOTSTRAP for inj00-vs-inj70 across checkpoints (a training series is autocorrelated; naive t/Wilcoxon over-claim); (c) injury heals to 0, so measure by INJURY WINDOW not whole-episode aggregate.
- The injury manipulation is CONFOUNDED for hypervigilance: raising start-injury also switches on the heal-by-rest drive that dominates and mechanically lowers cover use. Decisive next test (queued): a FROZEN-INJURY probe (recovery_base_rate 0) — prediction: the inj00>inj70 wander gap collapses if it is the rest drive.

## Open questions and follow-ups
- Run the frozen-injury probe to make the mechanism causal (observational injured-window split already strongly implies it).
- To ELICIT genuine hypervigilance, decouple injury from heal-by-rest AND make injury a sustained danger signal; and note hiding only enters the repertoire where evasion fails.

## References
- Supersedes [[20260805_0117_bushrefuge_survives_by_evasion_not_refuge]]. Matched probe: [[20260805_0118_regime_matched_probe_and_run_sweep_probe_dir]]. Bush-refuge feature: [[20260624_0516_bush_blocks_animals_movement_toggle]]. Metastability: [[20260710_1635_bush_hiding_metastable_dwell_measure]].
- Data: results/eval/avoidance/bushrefuge/rppo/ (+ _verify/ independent plots & mechanism figure).
- Raw conversation: synced via `./sync-agent-data.sh claude push`; `claude --resume 4efbe660-28c2-4643-b231-d3c6d2635b5a`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260805_0117_bushrefuge_survives_by_evasion_not_refuge]] (behavior_measures, 2026-08-05) — The 4 bush-refuge rPPO agents (bush blocks predators) survive predator encounter
<!-- END BACKLINKS -->
