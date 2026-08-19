---
id: 20260819_1947_basic04_predator_combinatorics_fully_latent
date: 2026-08-19
time: "19:47"
folder: env_entities
tags: [design, config, hypervigilance, learned_lesson]
summary: "At basic level 04 the mobile predator has 42 distinct DISCRETE per-episode types (detection_range 7 x attack_delay 3 x attack_range 2), giving 1,807 ordered scene loadouts over count 0-2, or 19,877 once the 11 hiding-predator counts are folded in. Crucially NONE of these traits reach the observation: the agent senses only a fixed visual one-hot plus a 2-D smell vector drawn from an INDEPENDENT PRNG stream. From the agent's side there is one predator kind with noisy smell facing a hidden 42-way danger lottery — so the env offers no cue to calibrate graded vigilance against."
related: ["20260529_1823_unified_animal_entity_v2_0_arch", "20260623_0143_per_episode_count_variance_masking", "20260630_1630_predator_params_per_episode_ranges"]
session_origin: claude_code
session_label: "no-hiding-predator verdict + cluster teardown"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4efbe660-28c2-4643-b231-d3c6d2635b5a.jsonl
raw_completeness: full
---

# basic04 predator: 42 latent types per individual, and the agent can observe none of them

## Key conclusion
Level 04 carries **two different predators**: the mobile hunting *entity* `pred` (patrols, detects, chases, pounces) and the `hiding_predator` *resource* (an ambush tile, invisible until stepped on — the thing the no-hide replicate removed). Counting the mobile one's per-episode draws:

- **42 distinct discrete types per individual predator** = detection_range {1..7} x attack_delay {1,2,3} x attack_range {2,3}. Everything from a near-blind slow poker (detect 1, cooldown 3, reach 2) to a map-wide relentless hunter (detect 7, cooldown 1, reach 3).
- **1,807 ordered scene loadouts** over count 0-2 (1 + 42 + 42^2); 946 if predators are treated as interchangeable.
- **19,877** once the 11 hiding-predator counts (2-12) are folded in (10,406 unordered).

**The research-relevant half: none of it is observable.** The observation carries only (a) a fixed visual one-hot, identical for every predator, and (b) a 2-D smell vector (mean 0.7/0.5, sigma 0.3, clipped to [0,1]). `detection_range`, `attack_delay`, `attack_range`, `max_stamina` and `damage` are **fully latent**, and the smell that IS sensed is drawn from an independent PRNG stream, so it is uncorrelated with any of them. The agent cannot distinguish a reach-2/cooldown-3 predator from a reach-3/cooldown-1 one.

That is a well-posed reason to expect a **uniformly cautious** policy rather than a threat-calibrated one: the environment offers nothing to calibrate against. Relevant when interpreting any "the agent isn't showing graded vigilance" result.

## Evidence, measurements, facts
- Verified by BUILDING the env (`load_env_params(load_env_config(...))`), not by reading YAML — per the project's verify-actual-state rule. Both predator slots report `detection_range [1,7]`, `attack_delay [1,3]`, `attack_range [2,3]`, `move_interval [1,1]`.
- Sampling semantics (`src/environment/core.py` jax_reset): integer fields use `randint(lo, hi+1)` = inclusive both ends; the four `DISTRIBUTIONAL_FIELDS` use `uniform(lo,hi)`. Draw shape is `(N,)` so **each predator slot draws independently**. `attack_range` comes from a separate `fold_in(key, 0xA77AC7)` stream, deliberately outside the size-7 `ep_keys` split to preserve PRNG parity.
- Continuous (uncountable, excluded from the combination count): `max_stamina` U(30,150) per episode; smell 2-D Gaussian per episode; `damage` U(15,120) **per hit, not per episode**. Fixed: move_interval 1, stamina_recovery 1.0, hunt_threshold 0.7, lose_interest 1.5, attack_success_rate 0.5 (a per-pounce Bernoulli, never sampled per episode).
- Counts are inclusive-int uniform per entry, and active slots are the first K by rank (`_build_activation_mask`), so which slots are live is deterministic given K.
- Spawn area resolves to the full 10x10 = 100 cells. Positions multiply the space (42 x 100 = 4,200 per individual; ~17.6M ordered two-predator placements) but are scene layout rather than predator identity, so they are kept separate from the type count.
- Latency confirmed in `src/environment/sensor.py`: the only animal-derived observation channels are `animal_visual_property_sampled` (fixed one-hot, std 0) and `animal_property_sampled` via `sense_resource` (the smell). No `*_detect_sampled` / `*_attack_range_sampled` / `*_max_stamina_sampled` reaches the sensor.

## Decisions and actions
- Extends [[20260630_1630_predator_params_per_episode_ranges]] (which established WHICH fields are per-episode rangeable) with the level-04 combinatorics and, new here, the **observability** analysis.
- Design lever recorded for future work: correlating one smell dimension with `detection_range` or `attack_range` would turn the latent lottery into a partially-observable one and give graded vigilance something to learn from. This is a `core.py` sampling change (the streams are currently independent), NOT a config change. Surfaced to the user; not scoped or implemented.

## Open questions and follow-ups
- Unverified whether two predators may spawn on the same cell; the positional counts above assume independence and are therefore an upper bound.

## References
- Config: `configs/environment/experiment/basic/04-jump_attack_10x10.yaml` (entities) inheriting `basic/03-random_init_10x10.yaml` (hiding_predator resource, count 2-12, damage 15-45).
- Code: `src/environment/config_loader.py` (`DISTRIBUTIONAL_FIELDS`, attack_range parsing); `src/environment/core.py` (jax_reset sampling, `_build_activation_mask`, `_sample_property`); `src/environment/sensor.py` (what actually reaches the observation).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 4efbe660-28c2-4643-b231-d3c6d2635b5a` or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
