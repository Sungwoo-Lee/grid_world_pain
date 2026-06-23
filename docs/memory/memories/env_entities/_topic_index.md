# _topic_index.md — `env_entities` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `env_entities` topic.

**Folder definition**: Env entity architecture decisions
**Insights**: 10
**Last updated**: 2026-06-24

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-06-24 | 05:17 | `20260624_0517_bush_spawn_exclusion_free_via_overlap_resolution` | A planned Phase-2 to keep animals from SPAWNING on bush cells needed NO code and zero added reset cost: resolve_overlaps_global already gives every entity a unique cell, so animals never spawn on a bush. Verified empirically 0/2000 resets on the whole-grid default. Cheap test-first probe avoided a needless placement-cost regression. |
| 2026-06-24 | 05:16 | `20260624_0516_bush_blocks_animals_movement_toggle` | Bushes (non-blocking + hides_agent) conceal the agent so hunting predators lose interest, but were not physically impassable to animals. Added a per-obstacle blocks_animals toggle (default off) that blocks ANIMAL movement only (hunt+wander collision), move_agent untouched so the agent still enters. Default-off byte-transparent; commit 523364a. |
| 2026-06-23 | 16:16 | `20260623_1616_rppo_reset_recompile_immune` | rPPO (recurrent_ppo) is recompile-IMMUNE to the new per-episode count-variance: resets ALL envs every step at fixed (num_envs) shape then jnp.where(done,...), and the K-activation is a traced scalar with count_high-static masks. Verified empirically (1 compile, 0 recompiles over 800 steps). Opposite of the Dreamer done-count-sized reset storm. |
| 2026-06-23 | 01:45 | `20260623_0145_proxy_benchmark_contention_misestimate` | A proxy benchmark on a contended GPU mis-estimated the env change's cost (predicted step-SPS +-3%); a clean idle-node re-measure showed a real -13% step-throughput regression driven by the +2 animal hot-path slots (reset +28%, negligible in wall-clock). Lesson: measure perf deltas on a clean idle node. Trimming rabbit 3->2 recovered ~half. |
| 2026-06-23 | 01:44 | `20260623_0144_inactive_resource_slots_revive_respawn` | Blocker: a per-episode activation mask must be threaded through the respawn/regeneration logic too. Inactive resource slots revived on step 1 because update_resources overloaded res_active=False to mean 'eaten, regrow', collapsing every episode to max count. Fix: a separate immutable res_allocated mask gates respawn. |
| 2026-06-23 | 01:43 | `20260623_0143_per_episode_count_variance_masking` | Added per-episode entity-count variance to break layout-overfitting: allocate count_high slots and per-episode activate K~U[low,high] via a boolean mask (res_active + new animal_active/obs_active); default scene moved to whole-grid spawn; scalar count stays backward-compatible; shipped via a two-commit parity discipline. |
| 2026-06-19 | 01:12 | [20260619_0112_configurable_visual_properties_and_std](20260619_0112_configurable_visual_properties_and_std.md) | v3.0 makes the visual sensor config-driven like olfaction: each entity carries a visual_properties vector (len = sensory.visual_vector_size, default 8) instead of a hardcoded one-hot channel, plus optional visual_properties_std for per-episode Gaussian sampling on an INDEPENDENT PRNG stream (fold_in 0x7150A1) so olfaction stays byte-identical. Defaults reproduce today's one-hot (parity-via-defaults). |
| 2026-06-09 | 17:26 | [20260609_1726_doc_audit_surfaces_latent_bugs](20260609_1726_doc_audit_surfaces_latent_bugs.md) | A code-as-truth re-sync of all 14 env docs doubled as a bug-finder, surfacing ~13 latent code findings; top two: overeating_death sets a termination reason but never ends the episode, and info[termination_reason] is unreliable when with_nutrition/with_injury are disabled. |
| 2026-06-09 | 17:22 | [20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride](20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride.md) | Two env findings: (1) renderer has no agent+neutral composite icon, so a rabbit on the agent's cell isn't drawn — visually 'disappears' (cosmetic; rabbit persists in state, no despawn/respawn). (2) Harmless chasing rabbit rides the agent's cell after contact because the post-contact attack-pause (attack_delay) is set only on at_damaging, so only predators bounce; decouple pause from damage to make a harmless chaser bounce. |
| 2026-05-29 | 18:23 | [20260529_1823_unified_animal_entity_v2_0_arch](20260529_1823_unified_animal_entity_v2_0_arch.md) | v2.0 env merges predator + neutral animals into one unified entity class with a static class tag and adds per-episode uniform distributional sampling on 5 core behavioural fields. Shipped via CP1-CP6 atomic refactor across 86 configs; PRNG byte-parity vs v1.4 preserved by a per-subset call pattern. |

---

## Related indexes

| Index | Path | Holds |
|---|---|---|
| Topic registry | [../../ROOT_INDEX.md](../../ROOT_INDEX.md) | All topic-folder metadata |
| Tag dictionary | [../_global_tags.md](../_global_tags.md) | All active tags |
