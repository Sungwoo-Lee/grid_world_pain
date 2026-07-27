---
id: 20260609_1719_predator_discrimination_visual_count_elimination
date: 2026-06-09
time: "17:19"
folder: hypervigilance
tags: [hypervigilance, learned_lesson, decision]
summary: "The chasing-rabbit rPPO agent can identify an approaching predator BEFORE contact by counting the neutrals on its own cell (visual channel 7 = 2.0 = both rabbits accounted for) and inferring the unaccounted approaching smell must be the predator — discrimination by elimination, not by directly sensing the predator at a distance."
related: ["20260508_1445_sameprop_discriminating_channels", "20260512_1428_sameprop_class_discriminating_defence_event_level", "20260609_1720_chasing_rabbit_avoidance_damage_driven", "20260609_1721_aggregate_stats_hide_conditional_behavior", "20260609_1747_avoidance_is_post_contact_not_preemptive"]
session_origin: claude_code
session_label: "chasing-rabbit (R4) behaviour deep-dive + obs-leak audit + matched-aggression control"
importance: high
status: superseded
valid_until: null
confidence: high
supersedes: []
superseded_by: ["20260609_1747_avoidance_is_post_contact_not_preemptive"]
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/144f126e-4b23-4e3f-bc6f-cc2491b3f2a4.jsonl
raw_completeness: full
---

# Agent discriminates an approaching predator by counting rabbits on its own cell (elimination)

> **SUPERSEDED 2026-06-09** by [[20260609_1747_avoidance_is_post_contact_not_preemptive]]. The factual observation below (visual channel 7 counts the neutrals on the agent's own cell) is still correct, but the behavioural interpretation — that the agent uses it for genuine *pre-emptive* avoidance — did NOT survive the predator-only control test: with rabbits removed the agent does not pre-empt the lone predator, so its avoidance is post-contact / pain-reactive. Read the superseding insight for the corrected conclusion.

## Key conclusion
In the matched-aggression chasing-rabbit world (predator and 2 rabbits share smell `[0,1,0,0,0]`, share `behaviour: hunt`, share all 5 chase params; differ only in damage), the agent reliably takes defensive action against the **approaching predator before the predator contacts it** — and it genuinely *can* do this. The mechanism is **discrimination by elimination via the visual sensor's count**: with `visual_sensor_range: 0` the visual channel only reports animals on the agent's OWN cell, but it reports them as a per-class **count** (sum of one-hots). The two rabbits ride the agent, so the agent literally observes `neutral channel (ch 7) = 2.0` = "both rabbits are on me". It knows there are only two rabbits, so any *third* animal it smells approaching must be the predator. It then stops eating and dives into cover. This is real, sensory-grounded, pre-(predator-)contact discrimination — it does NOT require sensing the predator's class directly at a distance.

## Evidence, measurements, facts
- Direct observation-vector evidence (episode 0 of `results/eval/matchedAggression_traj/.../10000024`): from step 10 onward `obs[visual_ch7] = 2.0` (both rabbits on the agent's cell); `obs[visual_ch5] (predator) = 0.0` throughout the pre-contact window (predator never on the cell). Predator distance falls 12→2→1 over steps 28–42; agent switches from `Eat` (step 33, predator 5 cells) to `Rest` (36) to fleeing into a bush (39–42), ~18 steps before the predator actually lands (step 60).
- The visual channel is a COUNT, verified empirically: `ch7 = 0.0` (no rabbit on cell), `1.0` (one rabbit), `2.0` (both rabbits). Code: `sense_visual` sums `jax.nn.one_hot(animal_visual_channel, 8)` over animals on the agent's cell (`src/environment/sensor.py`); `ANIMAL_CLASS_TO_VIS_CHANNEL = {"predator":5,"neutral":7}`.
- This EXTENDS the already-recorded channel ranking [[20260508_1445_sameprop_discriminating_channels]] ("visual ch.5/ch.7 teaches at contact; extero_nociception is contact-only") and is consistent with [[20260512_1428_sameprop_class_discriminating_defence_event_level]] ("the agent IS class-discriminating at the event level"). The new piece is the **count-based elimination route** and the demonstration that it yields PRE-contact discrimination of an *approaching* predator.
- Earlier in this same session I (wrongly) concluded "matched obs ⟹ the agent cannot discriminate pre-contact" and defended it with confound-explanations and aggregate stats. That conclusion was wrong and even contradicted existing memory; the reasoning post-mortem is captured separately in [[20260609_1721_aggregate_stats_hide_conditional_behavior]].

## Decisions and actions
- Adopt the count-elimination account as the working explanation for the agent's pre-contact predator avoidance in the chasing-rabbit configs.
- Conditional caveat to remember: elimination only works when the rabbits are actually *accounted for* (on/visible to the agent, ch7 count full). When the rabbits are scattered at distance (ch7 = 0), the agent has no elimination signal and reverts to class-blind "flee any approaching animal". This is why aggregate flee-rate looked class-blind (76% vs 76%) — it mixed both regimes.
- Proposed confirmatory test (not yet run): condition the agent's flee/cover response on the current ch7 count — predict that pre-contact predator avoidance is present when ch7 is "full" (both rabbits accounted) and absent when it is not.

## Open questions and follow-ups
- Confirm the policy actually *uses* the count (vs the signal merely being available) via the ch7-conditioned test above.
- Does the same mechanism explain the passive-rabbit (R2.6) and original sameProp results, or do those rely on different cues (movement signature, geography)? R2.6's "avoid predator before contact" was separately explained by spawn geometry + 10-cell detection (predator spawned >10 cells away can't engage).

## References
- Eval data + trajectories: `results/eval/matchedAggression_traj/models/10000024/` (200 recorded episodes); design doc `docs/experiments/active/hypervigilance/sameprop_chasing_rabbit.md`; obs audit `docs/reviews/chasingRabbit_obs_classLeak_audit.md`.
- Related: [[20260508_1445_sameprop_discriminating_channels]], [[20260512_1428_sameprop_class_discriminating_defence_event_level]], [[20260609_1720_chasing_rabbit_avoidance_damage_driven]], [[20260609_1721_aggregate_stats_hide_conditional_behavior]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 144f126e-4b23-4e3f-bc6f-cc2491b3f2a4` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260609_1720_chasing_rabbit_avoidance_damage_driven]] (hypervigilance, 2026-06-09) — Making the rabbits actively hunt the agent (harmless chasers, R4) did NOT create
- [[20260609_1721_aggregate_stats_hide_conditional_behavior]] (hypervigilance, 2026-06-09) — Methodology post-mortem: I reached a confidently-wrong 'the agent cannot discrim
- [[20260609_1747_avoidance_is_post_contact_not_preemptive]] (hypervigilance, 2026-06-09) — Predator-only eval (same chasing-rabbit model, rabbits removed) shows the agent 
<!-- END BACKLINKS -->
