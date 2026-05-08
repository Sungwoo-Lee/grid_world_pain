---
id: 20260508_1445_sameprop_discriminating_channels
date: 2026-05-08
time: "14:45"
folder: hypervigilance
tags: [hypervigilance, design, learned_lesson]
summary: "Phase 1 code investigation under sameProp olfactory matching: the dominant rabbit-vs-predator discriminator is movement/temporal signature in olfaction (predator HUNT tracks agent_pos, rabbit jitters); visual ch.5/ch.7 at the agent's own cell teaches but does not enable distance avoidance; extero_nociception does NOT broadcast at distance (dist<0.1 only). Practical ablation tool: setting `hunt_stamina_threshold > 1.0` makes HUNT structurally unreachable when stamina is clipped to max_stamina."
related: ["20260508_1444_sameprop_round1_finding_and_confound"]
session_origin: claude_code
session_label: "hypervigilance_sameprop_2026-05-08"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b12b1fea-94dc-491e-998b-735c656560cb.jsonl
raw_completeness: full
---

# Discriminating channels under sameProp olfactory matching (and how to ablate them)

## Key conclusion

When two entity classes share the same olfactory `properties` vector, the RPPO/recurrent-GRU policy can still tell them apart through several non-obvious channels in the current codebase. Ranked by likely impact: (1) **movement/temporal signature in olfaction** — predator HUNT mode tracks `agent_pos` each tick (`src/environment/core.py:156–202`), rabbit jitters randomly (`core.py:253+`); the summed olfactory field's time-derivative correlates with the agent's recent positions only for the predator component, and a recurrent GRU exploits this without any per-step class label; (2) **visual ch.5 vs ch.7 at colocation** — with `visual_sensor_range: 0`, `sense_visual` only fires on the agent's own cell (`src/environment/sensor.py:107–135`), but rabbit→ch.7 (Neutral), predator→ch.5 (Predator) (`sensor.py:190–193`) — gives a definitive class label *at contact*, drives learning but not same-step distance avoidance; (3) **patrol-area asymmetry** — predator roams full grid `[[1,1],[10,10]]`, rabbits are quadrant-locked, so the marginal location distribution of olfactory peaks leaks identity over time. Channels that do **not** discriminate: instantaneous olfactory shape (indistinguishable in expectation when properties match); **extero_nociception is contact-only** (`sensor.py:59–87` checks `dist < 0.1`), and rabbits are not even in its entity list — so the obvious "noxious aura at distance" hypothesis is false. As a practical tool for ablating channel #1: setting `hunt_stamina_threshold > 1.0` while `max_stamina = 30` makes the HUNT entry condition `stamina >= max_stamina * threshold` structurally unreachable (e.g. `1.1 → 33 > 30` clip), so the predator is permanently in PATROL — a clean, no-code-edit way to remove the chase-temporal signature.

## Evidence, measurements, facts

- Phase 1 memo: `docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`. Ranked list confirmed end-to-end by reading source, not inferred from documentation.
- Olfactory aggregation: when two entities at different positions share the same `properties` vector, the olfactory sensor returns a pure additive gradient with `decay_power=2`. Counts and per-position layout do not leak class identity *instantaneously*; identity leakage is purely temporal (channels 1 and 3) or contact-based (channel 2).
- Visual sensor at `visual_sensor_range: 0`: fires only on the agent's own cell. Class identity is encoded as a one-hot at the agent's cell when the agent is colocated with a rabbit (ch.7) or predator (ch.5). Cannot drive distance-based avoidance — only post-contact learning.
- Predator HUNT logic: `core.py:156–202`. State machine with PATROL ↔ HUNT transitions gated by stamina + distance (`detection_range`). HUNT mode tracks `agent_pos` each tick. Rabbit's `move_interval: 1` jitter is class `core.py:253+`, no agent-position dependency.
- extero_nociception: `sensor.py:59–87` checks `dist < 0.1` — contact-only, not distance-broadcast. Rabbit not in the entity list consumed by this sensor. So nociception_intensity (rabbit 0.1 vs predator 0.9) does NOT reach the policy through this channel before contact, which kills the "noxious aura" smoking-gun hypothesis.
- HUNT structural disable trick (Cell A1): `max_stamina: 30`. HUNT entry condition: `stamina >= max_stamina * hunt_stamina_threshold`. With `hunt_stamina_threshold: 1.1` → threshold = 33. Stamina is clipped to `max_stamina = 30` at all times → `30 >= 33` is false → HUNT permanently unreachable. Belt-and-braces: also set `detection_range: 0` so the distance check `dist <= 0` requires colocation (which would be a contact event, not a HUNT transition). Doubly unreachable. Verified by env-config-auditor pre-flight on `02-sameProp_R2_passivePredator.yaml`.
- Implication for ablation design: channel #1 (movement signature) can be ablated with config-only changes (no code edit). Channel #2 (visual at colocation) requires `visual_sensor_enabled: false` (kills two channels at once — visual and any visual-noise contribution). Channel #3 (patrol-area asymmetry) is removed by matching predator patrol_area to rabbit's quadrant, but is observationally entangled with the food/rabbit-quadrant confound (see sibling insight) so cannot be cleanly isolated without also moving food.
- Channel #1 + #3 are the two ablation knobs Round 2 Cell A1 turns simultaneously: predator patrol shrunk to TL `[[1,1],[5,5]]` (matches rabbit TL) AND HUNT disabled.

## Decisions and actions

- Round 2 Cell A1 uses the `hunt_stamina_threshold: 1.1 + detection_range: 0` ablation pattern instead of removing HUNT-related fields entirely. Rationale: keeps the YAML diff surgical, preserves field shape for static-field consistency (no JIT recompile thrash), and is reversible by changing two values back.
- Cell B ("olfaction-only" — `visual_sensor_enabled: false`) is deferred to Round 3 because it would ablate two channels at once (visual contact + visual noise), and Round 2 prefers cleaner one-channel-at-a-time isolation. If Round 2 Cells C + A1 do not fully resolve the question, Cell B becomes the obvious next step.
- The "noxious aura at distance" hypothesis is permanently discarded for this codebase. Any future hypervigilance design that wants nociception-at-distance signaling needs a code change to extero_nociception (not just a config change).
- Documentation: the Phase 1 memo was cross-linked from both Round 1 analysis and Round 2 design docs so the ranking informs every subsequent ablation cell.

## Open questions and follow-ups

- The **exact** strength of channel #1 (movement signature) is what Cell A1 measures. Pre-registered threshold: PredatorHits < RabbitHits − 1.5/ep stably means post-contact-teaching (channel #2) alone is sufficient and movement signature was *not* the dominant cue. PredatorHits ≈ RabbitHits within 1.0/ep means movement signature was dominant. Anywhere in between is borderline and triggers Round 3 with n=2.
- Channel #3 (patrol-area asymmetry) is partially controlled in Cell A1 (predator → TL only) but predator is now confined to one rabbit quadrant — so the BR rabbit quadrant has no co-located predator. This is a deliberate asymmetry that may need its own follow-up cell if it produces unexpected behavior.
- Could `visual_sensor_range > 0` (e.g., 1) restore distance-based class identity even when `properties` match? Likely yes, since the visual one-hot is class-specific. Worth a future ablation if olfaction-only does not produce avoidance.

## References

- Doc: `docs/develop/active/hypervigilance/sameprop_discriminating_channels.md` (Phase 1 ranked-channel memo with file:line cites).
- Doc: `docs/experiments/active/hypervigilance/sameprop_round2_design.md` (Round 2 cells leveraging the ablation tools).
- Source: `src/environment/core.py:156–202` (predator HUNT/PATROL state machine), `core.py:253+` (rabbit jitter), `core.py:300–305` (food regen-without-occupancy-check, irrelevant here but on the radar).
- Source: `src/environment/sensor.py:59–87` (extero_nociception contact-only), `sensor.py:107–135` (visual at agent's own cell), `sensor.py:190–193` (visual class channels: ch.5 Predator, ch.7 Neutral).
- Configs: `configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml`, `02-sameProp_R2_passivePredator.yaml`.
- Sibling insight: `20260508_1444_sameprop_round1_finding_and_confound` (the empirical asymmetry result that this code investigation interprets, plus the food/rabbit confound that limits causal attribution).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume b12b1fea-94dc-491e-998b-735c656560cb` or `python scripts/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b12b1fea-94dc-491e-998b-735c656560cb.jsonl /tmp/20260508_1445.md`.
