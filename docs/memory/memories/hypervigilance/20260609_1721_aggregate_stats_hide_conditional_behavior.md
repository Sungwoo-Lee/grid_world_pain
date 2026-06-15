---
id: 20260609_1721_aggregate_stats_hide_conditional_behavior
date: 2026-06-09
time: "17:21"
folder: hypervigilance
tags: [learned_lesson, meta, hypervigilance]
summary: "Methodology post-mortem: I reached a confidently-wrong 'the agent cannot discriminate predator from rabbit' conclusion by over-trusting an incomplete theoretical proof and defending it with confound-explanations and aggregate statistics, which averaged away a real CONDITIONAL behaviour. The mechanism only surfaced after I read individual trajectories step-by-step and inspected the agent's raw observation vector — which the user had been pushing me to do for several turns."
related: ["20260508_1445_sameprop_discriminating_channels", "20260512_1428_sameprop_class_discriminating_defence_event_level", "20260609_1719_predator_discrimination_visual_count_elimination", "20260609_1720_chasing_rabbit_avoidance_damage_driven"]
session_origin: claude_code
session_label: "chasing-rabbit (R4) behaviour deep-dive + obs-leak audit + matched-aggression control"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/144f126e-4b23-4e3f-bc6f-cc2491b3f2a4.jsonl
raw_completeness: full
---

# Aggregate stats + a tidy theory hid a real conditional behaviour; raw obs + single trajectories exposed it

## Key conclusion
When analysing whether the RL agent discriminates predator from rabbit, I anchored on an early theoretical claim ("matched smell + visual range 0 ⟹ no class signal at a distance") and treated it as a proof that the agent *could not* discriminate. The proof was incomplete (it ignored that contact reveals class and that the visual channel COUNTS on-cell animals, enabling discrimination-by-elimination — see [[20260609_1719_predator_discrimination_visual_count_elimination]]). I then defended the wrong conclusion across several turns with confound-explanations and aggregate statistics. The correct answer only emerged when I (a) read individual episode trajectories step-by-step and (b) inspected the agent's actual 27-dim observation vector. The user had been telling me to do exactly this the whole time.

## Evidence, measurements, facts
- The misleading aggregates: episode-mean predator-vs-rabbit distance gap; the "fresh-vs-fresh, distance-matched" flee rate (76% vs 76%, looked class-blind); eat-rate-vs-distance (looked like eating *rises* near the predator); M1 interrupted-feeding = 0.000. Each washed out the real effect because the discrimination is **conditional** — it only fires when the rabbits are "accounted for" (visual ch7 count full). Mixing the accounted / not-accounted regimes averaged the conditional effect to zero.
- The decisive evidence was always present in the recorded `obs` array: `visual ch7 = 2.0` (the count of rabbits on the agent's cell). I reasoned ABOUT what the observation should contain (from config/code) instead of reading what it DID contain.
- The wrong conclusion even contradicted existing memory: [[20260512_1428_sameprop_class_discriminating_defence_event_level]] already recorded "the agent IS class-discriminating" and [[20260508_1445_sameprop_discriminating_channels]] already recorded "visual ch5/ch7 teaches at contact". I should have checked memory before concluding.
- Specific failure modes named in the post-mortem: (1) treating an incomplete proof as airtight; (2) confirmation bias — reaching for "it's a confound" whenever data suggested discrimination; (3) over-reliance on aggregates that hide conditional / trajectory-level structure; (4) reasoning about the observation instead of inspecting it; (5) conflating "no direct sensory channel for X" with "X is impossible" (ignored inference/elimination by a recurrent policy).

## Decisions and actions
- Operating rules to apply on future RL-behaviour analysis: (1) Inspect the agent's **actual observation vector** and a handful of **individual trajectories** EARLY, before trusting aggregates. (2) Aggregates can hide conditional effects — when a hypothesis is "the agent does X", check whether X is gated on some state, and bin by that state. (3) When a careful human observer **repeatedly** reports a specific, reproducible observation that contradicts your model, treat it as strong evidence the MODEL is wrong, not as noise to explain away. (4) Check `docs/memory/` before concluding — the project may already have the answer.

## Open questions and follow-ups
- None.

## References
- The finding this lesson is about: [[20260609_1719_predator_discrimination_visual_count_elimination]]; the study verdict it sits under: [[20260609_1720_chasing_rabbit_avoidance_damage_driven]].
- Prior memory that the wrong conclusion contradicted: [[20260512_1428_sameprop_class_discriminating_defence_event_level]], [[20260508_1445_sameprop_discriminating_channels]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 144f126e-4b23-4e3f-bc6f-cc2491b3f2a4` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260609_1719_predator_discrimination_visual_count_elimination]] (hypervigilance, 2026-06-09) — The chasing-rabbit rPPO agent can identify an approaching predator BEFORE contac
- [[20260609_1747_avoidance_is_post_contact_not_preemptive]] (hypervigilance, 2026-06-09) — Predator-only eval (same chasing-rabbit model, rabbits removed) shows the agent 
- [[20260615_1612_testbed_isolation_makes_means_honest]] (hypervigilance, 2026-06-15) — Methodology decision: rather than abandon mean-level measures (we need scalable 
<!-- END BACKLINKS -->
