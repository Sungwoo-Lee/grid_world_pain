---
id: 20260703_0343_predator_jump_pounce_mechanism
date: 2026-07-03
time: "03:43"
folder: env_entities
tags: [design, decision, learned_lesson]
summary: "Added an opt-in predator JUMP/POUNCE: a hunting predator within a sampled Manhattan attack_range of an UN-hidden agent (cooldown up) teleports onto the agent (hit, prob=attack_success_rate) or a random adjacent cell (miss); any attempt resets the attack_delay cooldown. Bush blocks the jump (agent_hidden reused) so hiding beats running. New config keys attack_range (per-episode rangeable, default [0,0]=off) + attack_success_rate. Disabled = byte-identical no-op (PRNG from the discarded tail key + fold_in). Motivation: a 1-cell/step predator makes roaming a risk-free escape; the v5 noise agent showed early injury-gated avoidance but still roamed instead of hiding."
related: ["20260624_0516_bush_blocks_animals_movement_toggle", "20260630_1630_predator_params_per_episode_ranges", "20260703_0344_attack_range_float_threshold_gotcha"]
session_origin: claude_code
session_label: "basic 05/06/07 promotion + GPU-spec tooling + predator jump mechanism"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/428cfe9b-4b3e-4d99-8065-98b46ee41016.jsonl
raw_completeness: full
---

# Predator jump/pounce — makes roaming risky so the bush becomes the only refuge

## Key conclusion
A normal chasing predator moves only 1 cell/step, so an agent that keeps moving stays one step ahead forever — roaming is a risk-free escape and the bush is never needed. The jump/pounce breaks that: while HUNTING (post-transition `next_state==1`), if the agent is within a per-episode-sampled Manhattan `attack_range`, NOT hidden in a bush, the cooldown is up, and the feature is enabled, the predator's step-move is REPLACED by a pounce. Bernoulli(`attack_success_rate`): success -> teleport onto the agent's cell (the existing damage block delivers the hit, no double-count); miss -> teleport to a random valid Chebyshev-1 neighbour of the agent (stay-put fallback if none). ANY attempt (hit or miss) resets the `attack_delay` cooldown (no spam). Crucially the jump reuses the existing `agent_hidden` gate, so a bushed agent can NEVER be pounced -> hiding beats running (the design goal). Two new predator config keys: `attack_range` (optional, per-episode rangeable like detection_range, default `[0,0]` = disabled) and `attack_success_rate` (scalar, default 0.0). Opt-in: with attack_range absent the feature is a provable no-op.

## Evidence, measurements, facts
- Commit `ea13ea9`. Plan: docs/develop/active/env_entities/PREDATOR_JUMP_MECHANISM.md. Flow: senior-developer plan -> developer impl -> code-reviewer APPROVE (no correctness bugs across PRNG/vmap/double-count/ghost-gating/recompile) -> independent env re-verify.
- Backward-compat proven TWO ways: post-`jax_reset` `state.key` byte-identical to HEAD on the same seed; both parity suites green with NO fixture re-capture. So the ~6 runs training at the time were unaffected. Achieved by taking jump PRNG from the OTHERWISE-DISCARDED tail key in _hunt_step and sampling attack_range via `fold_in(animal_episode_key, 0xA77AC7)` so the size-7 ep_keys split is not widened; whole block behind a static `has_attack_feature` guard.
- Env verification (independent, attack-timer as the jump detector — a position-based classifier was WRONG because a normal chase step also lands adjacent): rate 0.5 -> 52% hit / 48% miss; rate 0.8 -> 84% hit; out-of-range -> 0 jumps; BUSHED agent -> 0/100 jumps (refuge holds); disabled -> 0 jumps.
- Design forks (user-aligned via AskUserQuestion): bush blocks jump YES; miss -> random adjacent; cooldown on ANY attempt; jump only while hunting with attack_range < detection_range. Config caveat: do NOT set attack_delay:0 on a jump-enabled predator (0 cooldown = pounce every step).
- Promoted into canonical basic levels via extends chain: basic/05 = all-combined pressure, basic/06 = 05 + noise, basic/07 = 06 + jump (attack_range [2,2], succ 0.5).

## Decisions and actions
- Shipped opt-in; enabled on basic/07 and a variant. Extends [[20260630_1630_predator_params_per_episode_ranges]] (same distributional-field pattern) and the refuge/[[20260624_0516_bush_blocks_animals_movement_toggle]] bush line.

## Open questions and follow-ups
- Does the jump actually push the trained agent off "roam forever" into the bush? (runs training: basic/07 + the reach-2-or-3 variant.)
- Possible follow-up: make attack_range INTEGER-sampled (randint) so a literal [2,3] means {2,3} — see [[20260703_0344_attack_range_float_threshold_gotcha]].

## References
- Commit `ea13ea9`; plan PREDATOR_JUMP_MECHANISM.md; code core.py _hunt_step jump block + damage block; tests/env/test_predator_jump.py.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 428cfe9b-4b3e-4d99-8065-98b46ee41016` or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260703_0344_attack_range_float_threshold_gotcha]] (env_entities, 2026-07-03) — Distributional animal fields that are FLOAT-sampled (uniform [lo,hi]) but compar
<!-- END BACKLINKS -->
