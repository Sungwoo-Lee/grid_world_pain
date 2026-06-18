---
id: 20260619_0112_configurable_visual_properties_and_std
date: 2026-06-19
time: "01:12"
folder: env_entities
tags: [config, design, decision, learned_lesson]
summary: "v3.0 makes the visual sensor config-driven like olfaction: each entity carries a visual_properties vector (length = sensory.visual_vector_size, default 8) instead of a hardcoded one-hot channel, plus an optional visual_properties_std for per-episode Gaussian sampling drawn on an INDEPENDENT PRNG stream (fold_in(property_key,0x7150A1)) so olfactory sampling stays byte-identical. Defaults reproduce today's one-hot, preserving observation parity."
related: ["20260529_1823_unified_animal_entity_v2_0_arch", "20260619_0111_config_v3_extends_layering_default_base"]
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

# Configurable per-entity visual properties + per-episode `std` sampling

## Key conclusion
The visual sensor now works like the olfactory sensor: every entity (resource/animal/obstacle) carries a config `visual_properties` vector of length `sensory.visual_vector_size` (default 8), consumed by `sense_visual` via the existing property-matrix matmul, instead of a hardcoded `jax.nn.one_hot(channel, 8)`. Each entity also accepts an optional `visual_properties_std`; with `std=0` (default) appearance is deterministic, with `std>0` it is re-sampled per episode (Gaussian, clipped non-negative). The class→channel map is KEPT as the internal default-vector generator, so omitting `visual_properties` reproduces today's one-hot and the observation is byte-identical (parity-via-defaults — same discipline as the v2.0 unified-animal refactor [[20260529_1823_unified_animal_entity_v2_0_arch]]). The decisive correctness technique: visual sampling draws from an INDEPENDENT PRNG stream so olfactory sampling is provably unaffected.

## Evidence, measurements, facts
- **Default channels (V=8)**: predator→5, neutral→7, food→3, hiding_predator→4, all obstacles→6 (rock; tree/bush share it today), background grass/sand/plain→0/1/2. At V≠8 every entity MUST declare `visual_properties` and `sensory.visual_background_properties` (3×V) is required (no silent default).
- **Configurable size**: new `sensory.visual_vector_size` (read-site default 8 — the one sanctioned fallback, to keep the ~86 archived configs zero-edit). Obs width and the `visual` perceptual-noise block both derive from a single `breakdown["Visual"] = num_vis_cells * visual_vector_size`, so changing V auto-resizes the noise block (verified at V=8 and V=5, noise applied at correct width).
- **PRNG independence (the trap, solved)**: `visual_property_key = jax.random.fold_in(property_key, 0x7150A1)` — a fresh constant, NOT the `0xAE1` used by `animal_episode_key`. Verified DIRECTLY: changing visual `std` at a fixed reset key leaves olfactory `res/animal/obs_property_sampled` byte-identical while `animal_visual_property_sampled` changes. `jax_step` re-samples `res_visual_property_sampled` on resource respawn via the same independent stream.
- **Pytree discipline**: the property/std arrays are traced leaves (vmap-safe, no recompile on value change); `visual_vector_size` is static (`pytree_node=False`, shape-determining).
- **NEU/RCK label bug fixed** (labels-only; true encoding rock=6, neutral=7; no observation bytes touched).
- **Parity + behaviour verified end-to-end via ParallelEnv**: std=0 deterministic (sampled==mean), std>0 stochastic (ch5 over 400 eps mean 5.18, std 1.98, min 0), custom background/per-entity vectors flow into the live observation. Live configs got EXPLICIT `visual_properties` (= one-hot) + `visual_properties_std: [0…]` so frozen checkpoints still load while the values are self-documenting.
- Reviews: code-reviewer APPROVE-WITH-NITS, env-config-auditor PASS.

## Decisions and actions
- **Locked v1**: configurable size (default 8), STATIC by default but `std` available; class→channel map kept as hidden default generator; parity-via-defaults is the gate.
- User explicitly relaxed checkpoint-parity as a hard constraint ("project still developing, nothing fixed") but std=0 determinism is preserved anyway as good engineering.
- Render asset-resolver (tint PNGs from the same vector) is OUT of scope — deferred.

## Open questions and follow-ups
- Background appearance has no `std` yet (entity-level only) — possible future symmetry.
- Splitting tree/bush off the shared rock channel (6) would change the observation (needs retrain) — easy explicit edit when wanted.

## References
- Extends [[20260529_1823_unified_animal_entity_v2_0_arch]] (same defaults-preserve-parity discipline; the PRNG-independence concern there was solved by per-subset call ordering, here by an independent `fold_in` stream).
- Built on the v3.0 config layering [[20260619_0111_config_v3_extends_layering_default_base]].
- Plan: [`CONFIGURABLE_VISUAL_PROPERTIES_PLAN`](../../../develop/active/sensors/CONFIGURABLE_VISUAL_PROPERTIES_PLAN.md). Reviews: `docs/reviews/config_visual_properties_postimpl.md`.
- Commits: `cfee42a` (pre-change parity fixtures), `ddff125` (visual_properties), `9cf23bd` (explicit vectors in live configs), `ec61024` (std sampling).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 96e71c7b-dc03-44c9-a98c-1c2acc86e0d9` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260619_0111_config_v3_extends_layering_default_base]] (config_system, 2026-06-19) — v3.0 makes configs/environment/default.yaml the canonical BASE; experiment confi
<!-- END BACKLINKS -->
