---
id: 20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride
date: 2026-06-09
time: "17:22"
folder: env_entities
tags: [design, learned_lesson, decision]
summary: "Two env/rendering findings from the chasing-rabbit work: (1) the renderer has no agent+neutral composite icon, so a rabbit on the agent's own cell is not drawn — it visually 'disappears' (cosmetic only; the rabbit persists in env state, there is no despawn/respawn). (2) A harmless chasing rabbit perpetually RIDES the agent's cell after contact because the post-contact attack-pause (attack_delay) is set only on damaging contact (at_damaging), so only predators pause/bounce."
related: ["20260529_1823_unified_animal_entity_v2_0_arch", "20260609_1720_chasing_rabbit_avoidance_damage_driven"]
session_origin: claude_code
session_label: "chasing-rabbit (R4) behaviour deep-dive + obs-leak audit + matched-aggression control"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/144f126e-4b23-4e3f-bc6f-cc2491b3f2a4.jsonl
raw_completeness: full
---

# Renderer has no agent+neutral icon (rabbit "disappears"); attack-pause is damage-gated (rabbit rides the agent)

## Key conclusion
Two distinct, verified facts about the v2.0 env that surfaced while debugging the chasing-rabbit videos:
1. **Visual "disappearing rabbit" is a renderer limitation, not a simulation bug.** When a neutral animal (rabbit) occupies the agent's own cell, the renderer does not draw it and falls through to a plain `agent` icon — there is a composite icon for agent+predator / +hiding_predator / +bush / +food, but NONE for agent+neutral. The rabbit is fully alive in env state the whole time (no despawn, no respawn, no "eat animal" mechanic).
2. **The harmless chasing rabbit perpetually rides the agent's cell after contact** because the post-contact "attack pause" is gated on damage. Only damaging animals get their move suppressed after touching the agent, so only the predator bounces away; the rabbit keeps stepping back onto the agent every tick.

## Evidence, measurements, facts
- Renderer: `src/environment/grid_world.py` (~lines 481–488) sets `agent_icon` to `'agent_predator' / 'agent_hiding_predator' / 'agent_bush' / 'agent_food'` on overlap, but the `elif 'neutral' in at_agent:` branch sets plain `'agent'`; and the neutral draw loop skips `draw_icon` when the rabbit is on the agent's cell. No animal despawn/respawn/eat code in `src/environment/core.py` (only resources respawn).
- State-level proof the rabbit persists: stepped the env with a stationary agent — animal count constant at 2 across 70 steps, the colliding rabbit stays at distance 0 for 66 steps, max single-step position jump = 1 cell (no teleport/respawn), `hit_neutral=True` fires throughout.
- Riding mechanism: `should_move = (move_timer<=0) AND (attack_timer<=0)` (`core.py:190`); `new_animal_at = jnp.where(at_damaging, params.animal_attack_delay, new_animal_at)` (`core.py:480`). Predator `attack_delay=3` → pauses 3 steps after each hit → agent escapes → bounce. Rabbit `attack_delay=0` and not `at_damaging` → never pauses → rides. Measured post-collision dist-0 fraction: predator 7.8% (bounces, mean dist 2.55) vs rabbit 49.6% (rides, mean dist 1.09).
- Current matched-config settings: predator and rabbit have identical 5 chase params; they differ in `damaging` (True/False), `attack_delay` (3/0), and the by-class visual channel (5/7, contact-only).

## Decisions and actions
- Renderer fix (proposed, user is creating an `agent_neutral` icon): add an `agent_neutral` composite case in `grid_world.py` so a co-located rabbit stays visible. Cosmetic; does not affect any behaviour measure (measures read true `animal_pos` from state).
- Ride fix (idea, recommended approach = "attack delay", not yet implemented): decouple the post-contact pause from damage so any HUNT animal pauses after contact — set the attack/pause timer on `at_animal & is_hunter` (not just `at_damaging`) AND give the rabbit `attack_delay > 0`. Makes the harmless chaser bounce like the predator without dealing damage, tightening the predator↔rabbit behavioural match. Requires a `core.py` change + re-train/re-eval; route through plan→review.

## Open questions and follow-ups
- Owner hand-off when ready: renderer icon → developer (after user supplies asset); attack-delay decouple → senior-developer plan + developer (touches `core.py` contact logic).
- Even with the attack-pause added, the agent does not flee the harmless rabbit, so it may re-close after the pause — the pause gives a visible bounce rather than full separation.

## References
- Code: `src/environment/grid_world.py` (renderer), `src/environment/core.py` (`_hunt_step`, contact/attack logic), `src/utils/eval_recording.py` (snapshot fields).
- Study context: [[20260609_1720_chasing_rabbit_avoidance_damage_driven]]; env architecture: [[20260529_1823_unified_animal_entity_v2_0_arch]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 144f126e-4b23-4e3f-bc6f-cc2491b3f2a4` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260609_1720_chasing_rabbit_avoidance_damage_driven]] (hypervigilance, 2026-06-09) — Making the rabbits actively hunt the agent (harmless chasers, R4) did NOT create
<!-- END BACKLINKS -->
