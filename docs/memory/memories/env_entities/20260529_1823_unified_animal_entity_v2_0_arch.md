---
id: 20260529_1823_unified_animal_entity_v2_0_arch
date: 2026-05-29
time: "18:23"
folder: env_entities
tags: [design, decision, learned_lesson, meta]
summary: "v2.0 env merges predator + neutral animals into one unified entity class with a static class tag (for sensor channels + metric fan-out) and adds per-episode uniform distributional sampling on 5 core behavioural fields. Shipped via CP1-CP6 atomic refactor across 86 configs; PRNG byte-parity vs v1.4 preserved by a per-subset call pattern."
related: ["20260519_1509_nnx_lax_scan_split_merge_pattern"]
session_origin: claude_code
session_label: "v2.0 env_entities refactor + R3 sameprop-predator-distributional design"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/ba4a22ee-e03f-4d94-a463-bb7d83fd8642.jsonl
raw_completeness: full
---

# v2.0 env: unified animal entity + per-episode distributional sampling

## Key conclusion

The v2.0 branch (created 2026-05-28) replaces the two-class predator-vs-neutral environment with a **single unified animal entity** whose `class` field is a static tag (`predator` / `neutral` / ...) used only for sensor-channel routing and metric fan-out, while behaviour and damage are fully programmable per entity. The schema additionally accepts `[low, high]` ranges on five core behavioural fields — `detection_range`, `max_stamina`, `stamina_recovery_rate`, `hunt_stamina_threshold`, `lose_interest_multiplier` — sampled uniformly per-episode, per-entity. Shipped end-to-end across six checkpoints (CP1-CP6) on 2026-05-28 with byte-parity vs the v1.4 baseline preserved by a per-subset PRNG call pattern; 228 tests pass; 86 configs migrated; local rPPO + dreamer_srl smokes clean; R3 production training launched and progressing at ep 3.9M / 10M without runtime regression.

## Evidence, measurements, facts

- **Architecture lock (Option 2)**: predator and neutral are NOT separate entity types. They share the same `Animal` state arrays; class is a static `pytree_node=False` tag used host-side for (a) which sensor channel each entity contributes to, (b) which metric tag (`predator_TL`, `rabbit_BR`, …) its events fan out to. Movement, hunting, damage, and stamina dynamics are not gated on class — they are gated on per-entity behavioural fields. A "neutral animal" can be configured to chase the agent, and a "predator" can be configured to wander passively, without code changes.
- **5 distributional fields (locked)**: `detection_range`, `max_stamina`, `stamina_recovery_rate`, `hunt_stamina_threshold`, `lose_interest_multiplier`. Scalar form (e.g. `5`) keeps prior behaviour; `[low, high]` form triggers per-episode uniform sampling. Sampling is independent per entity (two predators in the same episode get independent draws); class-coherent sampling is achievable as a config pattern (single entity), not a code feature.
- **PRNG byte-parity technique (per-subset call pattern)**: the unified-array refactor would naively change the PRNG split order vs v1.4 and break determinism on 86 seed-locked configs. Resolved by calling `jax.random.split` and the downstream `update_animals` step **per behaviour subset** (hunt / wander / static — `hunt_idx`, `wander_idx`, `static_idx`) in the same order v1.4 called them per type (predator subset first, neutral subset second). Same call sequence → same threefry-2x32 byte stream → bit-exact parity. Documented as the canonical pattern in plan v0.3 after code-reviewer rejected the initial masked-combine attempt (B1, then N1-N3 in `jax_reset` PRNG threading).
- **`@property` legacy aliases**: `EnvParams.predator_tags` and `EnvParams.neutral_tags` are preserved as `@property` shims that derive from `animal_classes + animal_tags`, so `dreamer_srl_main.py:522-523` and other analysis tooling continue to work without edits. D-CP6-1 (full alias removal) is deferred as a low-priority follow-up.
- **Host-side helper**: `select_by_class(params, class_name: str) -> np.ndarray` in `src/environment/state.py` returns a boolean mask over the unified entity arrays; consumed by metric fan-out and WandB tag breakdowns.
- **86-config migration**: `predator_enabled` removed from every YAML; all moved to the new `animals:` schema. Bulk sweep via `path_subs.sed` (also handled the parallel `configs/models/` reorganisation into algorithm subfolders). CP6 stripped CP1-era xfail markers and rewrote the renderer paths.
- **Validation**: 228 tests pass / 0 fail (CP1 gates G1+G2 fixed at `86e3628`). Local rPPO smoke + dreamer_srl smoke both clean. Production training (R3 predator-distributional, seed 46) launched on n113:0 at 2026-05-28 16:18, ran past ep 3.9M / 10M (39%) by 2026-05-29 16:13 with no runtime regression vs v1.4 baseline.
- **Plan history**: revised v0.1 → v0.4 with two reviewer rejects before sign-off — code-reviewer B1 (masked-combine breaks PRNG parity) → v0.2 per-subset pattern; code-reviewer N1-N3 (`jax_reset` PRNG threading) → v0.3 per-type key-split preservation; errata fold-back to plan body at v0.4. Commits: CP1 `4ef0b6b`, CP3 `a85a951`, CP4 `6685d0e`, CP5 `b783bca`, CP6 `3653247`; plan errata `cddf6f2`; R3 config land `e3e6789`.

## Decisions and actions

- **Architecture**: Option 2 (unified entity with static class tag) locked. Class is metadata only; behaviour is programmable per entity. This is the canonical v2.0 env shape.
- **Sampling**: uniform-only on 5 fields; per-entity independent draws; scalar form preserved as the default.
- **PRNG**: per-subset call pattern is the canonical technique for any future unified-array refactor that needs to preserve seed-locked baselines. Recorded in plan v0.3 and CP1 implementation report. Same shape as [[20260519_1509_nnx_lax_scan_split_merge_pattern]] but applied to `random.split` rather than `lax.scan`.
- **Followups (not yet captured as standalone insights)**:
  - `Episode/sampled_*_<tag>` WandB logging is built in `src/behavior/accumulators.py` (`build_episode_log_dict` + `sampled_wandb_keys`) but not yet wired into `train.py` — CP5 follow-up ticket.
  - `perceptual_noise.enabled` mandatory promotion deferred until after `configs/models/` sweep settles (medium priority).
  - D-CP6-1 legacy info-dict alias removal (low priority).
- **R3 experiment**: this architecture is the substrate for the first sameProp-predator-distributional study — asks whether the agent's avoidance generalises to neutral animals when predator behavioural parameters are no longer static. Design + pre-registered bands in `docs/experiments/active/hypervigilance/sameprop_predator_distributional.md`; mid-training previews at §§9-15. Verdict pending end of training (~2026-05-31).

## Open questions and follow-ups

- Does per-entity sampling (vs per-class) match experimental intent for hypervigilance studies? Current implementation samples independently per entity; "all predators draw from the same per-episode value" requires a config pattern (single entity) rather than a code change.
- `select_by_class` is host-side NumPy. If future analysis needs class-aware metric fan-out inside a JIT'd path, a JAX-side equivalent will be required (none needed for current metric pipeline).
- M5 per-tag rabbit fan-out drift surfaced in R3 first preview (rabbit_TL=1.099 vs rabbit_BR=1.358, gap 0.259) is flagged for the closing analyst; if it's a code-side artefact of the unified-entity refactor (vs a genuine R3 effect), it would surface here. Provisionally a sample-noise artefact pending closing seed-comparison.

## References

- **Why a new folder**: this insight does not fit any existing folder. `nmn_diagnosis` is model-architecture / NMN-specific; `dreamer_diagnosis` is algorithm-specific; `cluster_ops` is infra; `subagent_engineering` is Claude Code tooling; `hypervigilance` is experiments (not env interfaces); `memory_system_design` is this layer's own design. The unified-entity refactor is an **environment-architecture** decision that will shape every future experiment, config, and sensor change — it deserves its own discoverable folder. Closest existing folder is `hypervigilance` (R3 is the experiment that drove it), but the architecture is generic to the env and should not be hidden inside a single experimental track.
- Plan: [`UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING`](../../../develop/active/env_entities/UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING.md) (v0.4, errata fold-back at `cddf6f2`).
- R3 design + bands: [`sameprop_predator_distributional`](../../../experiments/active/hypervigilance/sameprop_predator_distributional.md).
- Related: [[20260519_1509_nnx_lax_scan_split_merge_pattern]] (generalisable NNX + `lax.scan` technique — analogous shape to the per-subset PRNG pattern but applied to `lax.scan` rather than `random.split`).
- Commits: CP1 `4ef0b6b`, CP3 `a85a951`, CP4 `6685d0e`, CP5 `b783bca`, CP6 `3653247`; plan errata `cddf6f2`; R3 config land `e3e6789`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume ba4a22ee-e03f-4d94-a463-bb7d83fd8642` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride]] (env_entities, 2026-06-09) — Two env/rendering findings from the chasing-rabbit work: (1) the renderer has no
- [[20260609_1726_doc_audit_surfaces_latent_bugs]] (env_entities, 2026-06-09) — A code-as-truth re-sync of all 14 env docs doubled as a cheap bug-finder, surfac
- [[20260619_0112_configurable_visual_properties_and_std]] (env_entities, 2026-06-19) — v3.0 makes the visual sensor config-driven like olfaction: each entity carries a
- [[20260630_1630_predator_params_per_episode_ranges]] (env_entities, 2026-06-30) — Predator behavioural params are per-episode randomizable via a [lo,hi] range. 5 
<!-- END BACKLINKS -->
