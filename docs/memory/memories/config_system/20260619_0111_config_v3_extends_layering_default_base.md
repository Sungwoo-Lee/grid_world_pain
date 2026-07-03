---
id: 20260619_0111_config_v3_extends_layering_default_base
date: 2026-06-19
time: "01:11"
folder: config_system
tags: [config, design, decision, meta]
summary: "v3.0 makes configs/environment/default.yaml the canonical BASE; experiment configs become sparse files that opt into it via a top-level extends: environment/default key, deep-merged at the new load_env_config() chokepoint. The experiment dir moved under configs/environment/experiment/ with all 91 pre-v3.0 configs git-mv'd into archive/."
related: ["20260619_0112_configurable_visual_properties_and_std", "20260619_0113_configurable_initial_state_ranges", "20260619_0114_config_guide_maintenance_contract"]
session_origin: claude_code
session_label: "v3.0 config-system overhaul (default-as-base + extends layering + visual properties + init ranges + CONFIG_GUIDE)"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/96e71c7b-dc03-44c9-a98c-1c2acc86e0d9.jsonl
raw_completeness: full
---

# Config v3.0: default-as-base + opt-in `extends:` layering + experiment reorg

## Key conclusion
The config system was restructured so `configs/environment/default.yaml` is the single canonical BASE and other configs are **sparse overrides** that opt in with a top-level `extends: environment/default` key. The new `load_env_config()` chokepoint in `config_loader.py` resolves `extends:` by deep-merging the base under the child (child wins); a config WITHOUT `extends:` loads standalone, byte-identical to before. The merge reuses the pre-existing `Config.merge()` deep_update — what was added is the resolution chokepoint, not the merge. `configs/experiment/**` was `git mv`'d to `configs/environment/experiment/archive/**` (91 configs, history preserved, byte-unchanged); new live configs go at `configs/environment/experiment/<topic>/`. The whole point is maintainability: schema-wide updates touch only default.yaml, and sparse children inherit.

## Evidence, measurements, facts
- **Deep-merge semantics**: dicts merge recursively; **lists replace wholesale** (the footgun). To suppress a base list block (e.g. `entities:`) a sparse child must write `entities: []` explicitly — omitting it inherits the base's list.
- **`load_env_config` / `_resolve_extends`** (config_loader.py): `extends` is str or list (later overrides earlier; child overrides all bases); cycle detection via a `_seen` frozenset of abspaths; base path resolved from `__file__`-derived `_CONFIGS_ROOT` (CWD-independent); the `extends` key is `pop`'d before the merged Config reaches `load_env_params`, so `get_mandatory` validation runs on the MERGED result (no-fallback rule honoured post-merge).
- **Discovery**: the codebase was already half-layered inconsistently — `dreamer_srl_main.py` did `get_default_config().merge(experiment)`, while test/utility paths loaded standalone. `train.py` loads the full default then merges `--config`, which already replicates extends for the common case (the leftover `extends` key is harmless). The refactor made layering explicit, uniform, and opt-in.
- **default.yaml modernized first** (commit `e233359`): legacy split `predators:`/`neutral_animals:` → unified `entities:` schema; added `behavior_measures:` block; later the 200-line `eval_seeds` literal was compacted to a `{rng, sort}` generator spec (`b509bb4`).
- **Verification**: parity suite green throughout (state parity unchanged); full `tests/env/` suite stayed at 0 failures; code-reviewer APPROVE-WITH-NITS on the `extends` loader; env-config-auditor PASS. The 5-level `basic/` curriculum (`c066d2b`) is the first real `extends:` consumer.

## Decisions and actions
- **Locked**: default-as-base; `extends:` is OPT-IN (no key = standalone, unchanged) — this is what keeps the 91 archived configs and frozen checkpoints safe.
- Archived configs stay FULL standalone (no `extends:`); new configs are authored sparse.
- Parity fixtures renamed to new path-slugs (a move doesn't change behaviour) so the parity gate stays meaningful.
- See [[20260619_0114_config_guide_maintenance_contract]] for how this system is kept documented/maintained, and [[20260619_0113_configurable_initial_state_ranges]] / [[20260619_0112_configurable_visual_properties_and_std]] for v3.0 schema additions that build on this.

## Open questions and follow-ups
- The bulk of live experiment configs are still FULL (archived) — rewriting them as sparse `extends:` files is `experiment-designer` work, not yet done.
- `docs/environment/02_config_schema.md` body is still v2.0 (only a v3.0 banner added) — full sync is a follow-up.

## References
- **Why a new folder**: `config_system` holds config loader / layering / schema / authoring decisions. Closest existing is `env_entities` (env-architecture), but that is entity-specific; the config-loading/layering/archive infrastructure is broader and (per the user) will keep growing. Distinct from `cluster_ops` (infra/ops) and `memory_system_design` (the memory layer).
- Plan: [`CONFIG_LAYERING_AND_EXPERIMENT_REORG`](../../../develop/active/refactors/CONFIG_LAYERING_AND_EXPERIMENT_REORG.md). Guide: [`CONFIG_GUIDE`](../../../environment/CONFIG_GUIDE.md).
- Reviews: `docs/reviews/config_extends_layering.md`, `config_extends_layering_reorg_postimpl.md`.
- Commits: `e233359` (default modernize), `c13a3ac` (extends chokepoint + reorg), `b509bb4` (eval_seeds spec), `c066d2b` (basic curriculum).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 96e71c7b-dc03-44c9-a98c-1c2acc86e0d9` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260619_0112_configurable_visual_properties_and_std]] (env_entities, 2026-06-19) — v3.0 makes the visual sensor config-driven like olfaction: each entity carries a
- [[20260619_0113_configurable_initial_state_ranges]] (config_system, 2026-06-19) — v3.0 exposes the agent's start nutrition/injury randomization bounds as config k
- [[20260619_0114_config_guide_maintenance_contract]] (config_system, 2026-06-19) — To keep the v3.0 config system maintained across sessions, a collaborator-facing
- [[20260703_1507_train_py_ignores_extends_drops_layers]] (config_system, 2026-07-03) — SEVERE: train.py loads --config via Config.load_yaml (plain YAML), which does NO
<!-- END BACKLINKS -->
