---
id: 20260609_1720_chasing_rabbit_avoidance_damage_driven
date: 2026-06-09
time: "17:20"
folder: hypervigilance
tags: [hypervigilance, refutation, learned_lesson, decision]
summary: "Making the rabbits actively hunt the agent (harmless chasers, R4) did NOT create predator-level avoidance — the agent tolerates the harmless chaser (lets it ride at distance 0) and forages through it; remove the predator and it ignores the chasers entirely; under matched aggression it dies 91% of episodes. The agent's defensive discrimination is damage / pain-consequence-driven, not chasing-motion-driven."
related: ["20260508_1445_sameprop_discriminating_channels", "20260512_1428_sameprop_class_discriminating_defence_event_level", "20260609_1719_predator_discrimination_visual_count_elimination", "20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride"]
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

# Chasing-rabbit (R4): avoidance is damage-driven, not motion-driven

## Key conclusion
The R4 experiment made the two neutral rabbits actively HUNT the agent (`behaviour: hunt`, detection 10) while keeping them harmless (`class: neutral` ⇒ `is_damaging=False` ⇒ zero damage), to test whether chasing *motion* drives the agent's predator avoidance. It does not. The trained agent **tolerates the harmless chaser** — lets it sit on its own cell (rides ~84% of post-collision steps), forages through it, never interrupts feeding for it. The discrimination the agent shows is driven by **damage / interoceptive pain at contact**, not by the chasing motion or proximity.

## Evidence, measurements, facts
- **No-predator transfer eval** (config `05-noPredator_chasingRabbit_eval.yaml`, all damaging entities removed, only the 2 harmless chasers remain): survival 500/500 every episode (0 deaths), lets rabbits to mean 1.41 cells (within 3 cells 81% of steps), eats 114.8 food/episode, M1 interrupted-feeding 0.000, M5 eat-under-threat 1.08. The agent completely ignores the chasers as threats.
- **Matched-aggression control** (config `06-matchedAggression_chasingRabbit_eval.yaml`, predator's 5 chase params set EQUAL to the rabbits' — detection 10, stamina 60, recovery 1.0, hunt-thresh 0.3, lose-interest 3.0 — so predator and rabbit differ ONLY in damage): survival drops to 273, **91% of episodes end in death (183/200)**, first predator contact at step ~33. When the predator chases as reliably as the rabbits, the agent cannot stay ahead of it — it gets run down and killed. The differentiation it does show is post-contact (after the predator's pain).
- **Distributional vs matched, same final checkpoint (10M), same protocol**: the predator-vs-rabbit pre-contact distance gap (+2.5 cells) collapses to ≈0 once you (a) compare 1 predator to 1 individual rabbit (removing a 2-rabbits-vs-1-predator min artifact) and (b) match the predator's chase aggression to the rabbits' (removing the distributional predator's weaker-chase confound). 1-vs-1 matched gap = −0.07 to −0.24 cells.
- This is consistent with the broader study line: the agent is spatially class-blind but behaviourally class-discriminating ([[20260512_1428_sameprop_class_discriminating_defence_event_level]]), and the discrimination cue is contact-mediated (vision-at-contact / pain), not at-distance motion ([[20260508_1445_sameprop_discriminating_channels]]).

## Decisions and actions
- Built reusable tooling for this study: `scripts/eval_rollout.py --record` (one frozen-checkpoint eval emits both behaviour measures M1/M2/M5 AND `.rec.gz` recordings renderable to MP4 via `scripts/render_recordings.py`); committed `c67cc40`. Three new configs: `04-sameProp_R4_chasingRabbit` (training), `05-noPredator_chasingRabbit_eval`, `06-matchedAggression_chasingRabbit_eval`.
- Verdict adopted: avoidance is damage/pain-consequence-driven. The refined mechanism for HOW the agent identifies an approaching predator pre-contact is captured in [[20260609_1719_predator_discrimination_visual_count_elimination]].

## Open questions and follow-ups
- The chasing rabbit perpetually rides the agent's cell after contact (an artifact of the attack-pause being damage-gated) — to make the harmless chaser behave more like the predator post-contact, see [[20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride]].
- Single seed (42), one architecture (recurrent_ppo); evals on the 10M checkpoint. A second seed would harden the matched-aggression death-rate result.

## References
- Design doc + results + comparison tables: `docs/experiments/active/hypervigilance/sameprop_chasing_rabbit.md`.
- Eval outputs: `results/eval/{noPredator_chasingRabbit, withPredator_final, matchedAggression_final, matchedAggression_traj}/...`; videos under each `videos/eval_*.mp4`.
- Training run: `20260529-212737_recurrent_ppo_04-sameProp_R4_chasingRabbit_s42` (10M, WandB `r3gevy54`).
- Related: [[20260609_1719_predator_discrimination_visual_count_elimination]], [[20260512_1428_sameprop_class_discriminating_defence_event_level]], [[20260508_1445_sameprop_discriminating_channels]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 144f126e-4b23-4e3f-bc6f-cc2491b3f2a4` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260609_1719_predator_discrimination_visual_count_elimination]] (hypervigilance, 2026-06-09) — The chasing-rabbit rPPO agent can identify an approaching predator BEFORE contac
- [[20260609_1721_aggregate_stats_hide_conditional_behavior]] (hypervigilance, 2026-06-09) — Methodology post-mortem: I reached a confidently-wrong 'the agent cannot discrim
- [[20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride]] (env_entities, 2026-06-09) — Two env/rendering findings from the chasing-rabbit work: (1) the renderer has no
<!-- END BACKLINKS -->
