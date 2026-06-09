# _topic_index.md — `env_entities` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `env_entities` topic.

**Folder definition**: Env entity architecture decisions
**Insights**: 3
**Last updated**: 2026-06-09

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-06-09 | 17:26 | [20260609_1726_doc_audit_surfaces_latent_bugs](20260609_1726_doc_audit_surfaces_latent_bugs.md) | A code-as-truth re-sync of all 14 env docs doubled as a bug-finder, surfacing ~13 latent code findings; top two: overeating_death sets a termination reason but never ends the episode, and info[termination_reason] is unreliable when with_nutrition/with_injury are disabled. |
| 2026-06-09 | 17:22 | [20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride](20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride.md) | Two env findings: (1) renderer has no agent+neutral composite icon, so a rabbit on the agent's cell isn't drawn — visually 'disappears' (cosmetic; rabbit persists in state, no despawn/respawn). (2) Harmless chasing rabbit rides the agent's cell after contact because the post-contact attack-pause (attack_delay) is set only on at_damaging, so only predators bounce; decouple pause from damage to make a harmless chaser bounce. |
| 2026-05-29 | 18:23 | [20260529_1823_unified_animal_entity_v2_0_arch](20260529_1823_unified_animal_entity_v2_0_arch.md) | v2.0 env merges predator + neutral animals into one unified entity class with a static class tag and adds per-episode uniform distributional sampling on 5 core behavioural fields. Shipped via CP1-CP6 atomic refactor across 86 configs; PRNG byte-parity vs v1.4 preserved by a per-subset call pattern. |

---

## Related indexes

| Index | Path | Holds |
|---|---|---|
| Topic registry | [../../ROOT_INDEX.md](../../ROOT_INDEX.md) | All topic-folder metadata |
| Tag dictionary | [../_global_tags.md](../_global_tags.md) | All active tags |
