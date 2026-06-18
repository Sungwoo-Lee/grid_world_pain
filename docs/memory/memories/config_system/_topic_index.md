# _topic_index.md — `config_system` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `config_system` topic.

**Folder definition**: Config loader/layering/schema decisions
**Insights**: 3
**Last updated**: 2026-06-19

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-06-19 | 01:14 | [20260619_0114_config_guide_maintenance_contract](20260619_0114_config_guide_maintenance_contract.md) | docs/environment/CONFIG_GUIDE.md (collaborator-facing) + a Maintenance Contract wired into the 5 config-caring agent profiles (read-before / update-on-change) — the anti-rot mechanism for the v3.0 config system, enforced via profiles not memory. |
| 2026-06-19 | 01:13 | [20260619_0113_configurable_initial_state_ranges](20260619_0113_configurable_initial_state_ranges.md) | body.start_{nutrition,injury}_{low,high} expose the start-randomization bounds (replacing hardcoded max/2); conditional-mandatory (read only when random_start_* is true → zero config ripple); unlocks genuinely hungry/injured starts. |
| 2026-06-19 | 01:11 | [20260619_0111_config_v3_extends_layering_default_base](20260619_0111_config_v3_extends_layering_default_base.md) | v3.0 makes default.yaml the canonical base; experiment configs are sparse extends: environment/default overrides deep-merged at load_env_config(); configs/experiment moved to configs/environment/experiment/archive (91 configs). Opt-in, parity-safe. |

---

## Related indexes

| Index | Path | Holds |
|---|---|---|
| Topic registry | [../../ROOT_INDEX.md](../../ROOT_INDEX.md) | All topic-folder metadata |
| Tag dictionary | [../_global_tags.md](../_global_tags.md) | All active tags |
