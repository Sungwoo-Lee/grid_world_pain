---
id: 20260609_1747_avoidance_is_post_contact_not_preemptive
date: 2026-06-09
time: "17:47"
folder: hypervigilance
tags: [hypervigilance, refutation, learned_lesson]
summary: "Predator-only eval (same chasing-rabbit model, rabbits removed) shows the agent does NOT pre-empt the approaching lone predator — it forages/rests in place until contact and only hides reactively after being hit. The agent's predator avoidance is post-contact / pain-reactive, NOT genuinely anticipatory. The visual-channel rabbit-count signal is real but does not drive reliable pre-emptive avoidance. Supersedes the earlier pre-emptive-elimination reading."
related: ["20260609_1719_predator_discrimination_visual_count_elimination", "20260609_1720_chasing_rabbit_avoidance_damage_driven", "20260609_1721_aggregate_stats_hide_conditional_behavior"]
session_origin: claude_code
session_label: "chasing-rabbit (R4) behaviour deep-dive — predator-only control + trajectory-level read"
importance: high
status: settled
valid_until: null
confidence: medium
supersedes: ["20260609_1719_predator_discrimination_visual_count_elimination"]
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/144f126e-4b23-4e3f-bc6f-cc2491b3f2a4.jsonl
raw_completeness: full
---

# Predator-only control: avoidance is post-contact / pain-reactive, not pre-emptive

## Key conclusion
Evaluating the trained chasing-rabbit agent in a world with the **rabbits removed** (only the patrolling predator) shows the agent does **not** pre-emptively avoid the approaching predator: it forages and rests in place while the lone predator beelines straight into it, and only takes cover *reactively after being hit*. The working conclusion is therefore that the agent's predator avoidance is **post-contact / pain-driven, not genuinely anticipatory**. This supersedes the headline of [[20260609_1719_predator_discrimination_visual_count_elimination]], which read the matched-aggression episode-0 behaviour as genuine pre-emptive avoidance via a rabbit-count "elimination" cue. The **factual** core of that insight survives and is preserved here — the visual channel really does count the neutrals on the agent's own cell (channel 7 = 2.0 = both rabbits) — but that signal does NOT, on the cleanest test, drive reliable pre-emptive avoidance.

## Evidence, measurements, facts
- Predator-only eval: config `configs/experiment/hypervigilance/07-predatorOnly_eval.yaml` (config 06 with the 2 neutral entities removed; `predator_indices=(0,)`, `neutral_indices=()`, obs dim still 27 so the 10M checkpoint loads), 20 deterministic episodes, `results/eval/predatorOnly/models/10000024/`.
- Trajectory read (the substance): in most episodes the agent forages/rests in place as the lone predator approaches and gets hit — EP 13 (survived 501) **eats** at the corner while the predator closes 5→4→3→2→1 then is hit at step 21; EP 7 / EP 3 **rest** in place as the predator beelines in; EP 17 **eats** until contact at step 4. The one flee-and-hide episode (EP 16) is reactive — the predator was already adjacent and damaging it.
- Survival mean 248, 15% reach the 500 cap, mean first predator contact at **step ~11** (much faster than the ~33 with rabbits — nothing slows the beelining predator).
- The aggregate "71% of near-predator moves are away" does NOT imply active fleeing: most near-predator steps are `Rest` (zero displacement → counted as neither toward nor away), so the agent is mostly sitting, not fleeing.
- This reinforces [[20260609_1720_chasing_rabbit_avoidance_damage_driven]] (avoidance is damage/pain-consequence-driven). It also reframes the matched-aggression episode-0 "stop eating + bush-dive at step 42": small injury jumps at steps 39–42 mean that dive may itself have been a *reaction* to a (hiding-predator/rock) hit, not anticipation of the patrolling predator's step-60 contact.

## Decisions and actions
- Adopted conclusion: the chasing-rabbit agent's predator avoidance is **post-contact / pain-reactive, not anticipatory**. The visual rabbit-count signal is present but is not load-bearing for reliable pre-emption.
- Superseded [[20260609_1719_predator_discrimination_visual_count_elimination]] (status flipped to superseded). Its factual sensory observation (ch7 counts on-cell neutrals) is carried forward above.
- This is itself a second instance of the methodology lesson [[20260609_1721_aggregate_stats_hide_conditional_behavior]]: a tidy mechanism ("elimination") that was sensorily plausible and behaviourally suggestive did not survive the clean control test — designing and running the control (predator-only) was what settled it.

## Open questions and follow-ups
- **Confidence is medium, not high**: the predator-only world is OUT-OF-DISTRIBUTION (the agent trained with three animals always present, never zero rabbits), N=20, and pre-contact windows are tiny (~11 steps). The OOD confound is the main weakness.
- The clean, in-distribution confirmation (not yet run): within the 200 matched-aggression trajectories, condition the agent's pre-contact flee/cover response on whether both rabbits are currently "accounted for" (visual ch7 = 2). If pre-emption is absent even when the count is full, that confirms post-contact-only avoidance without the OOD caveat.

## References
- Superseded: [[20260609_1719_predator_discrimination_visual_count_elimination]]; reinforces [[20260609_1720_chasing_rabbit_avoidance_damage_driven]]; methodology: [[20260609_1721_aggregate_stats_hide_conditional_behavior]].
- Config: `configs/experiment/hypervigilance/07-predatorOnly_eval.yaml`; eval + videos: `results/eval/predatorOnly/models/10000024/` (`videos/eval_first5.mp4` = first-5-episode clip).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 144f126e-4b23-4e3f-bc6f-cc2491b3f2a4` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260609_1719_predator_discrimination_visual_count_elimination]] (hypervigilance, 2026-06-09) — The chasing-rabbit rPPO agent can identify an approaching predator BEFORE contac
<!-- END BACKLINKS -->
