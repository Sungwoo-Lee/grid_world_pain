---
id: 20260630_1721_per_episode_variance_dreamer_recompile_safe
date: 2026-06-30
time: "17:21"
folder: env_entities
tags: [dreamer, config, learned_lesson, decision]
summary: "The per-episode environment-variance feature (count ranges via count_high-static allocation + res/animal/obs_active masks, plus per-episode behavioural ranges like predator move_interval[1,3]/detection_range[1,7]) is recompile-SAFE under dreamer_srl, not just rPPO. Basic level 05 (random counts + positions + start nutrition/injury + per-episode predator behaviour) smoke-tested at ~315 XLA compiles == the fixed-count baseline (~309), no storm — because the variance is all traced per-episode VALUES over static-max shapes, never array dimensions."
related: ["20260622_1746_dreamer_srl_recompile_storm_done_count", "20260623_0143_per_episode_count_variance_masking", "20260623_1616_rppo_reset_recompile_immune", "20260630_1720_dreamer_srl_train_step_jit_compile_once"]
session_origin: claude_code
session_label: "dreamer_srl basic training-speed hang: two-bug fix + level 05 launch"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/3c11010b-dbfb-4d17-af58-5b86e54d6815.jsonl
raw_completeness: full
---

# Per-episode env-variance is recompile-safe under Dreamer (basic level 05)

## Key conclusion
The per-episode environment-variance feature — randomized entity counts (allocate `count_high` static slots, per-episode activate K~U[low,high] via `res_active`/`animal_active`/`obs_active` masks) plus per-episode behavioural sampling (predator `move_interval: [1,3]`, `detection_range: [1,7]`, attack_delay, behaviour redrawn each episode) plus random start position/nutrition/injury — is **recompile-safe under dreamer_srl**, confirmed empirically, not just under rPPO. Basic **level 05** (`05-random_init_10x10.yaml`) is the first config to exercise all of this under Dreamer. A `JAX_LOG_COMPILES` smoke (collection-phase, ~3000-3500 steps spanning many resets with different K and different predator params) compiled **~315 distinct functions — essentially identical to the fixed-count baseline (~309)**, with no same-shape recompile storm. The reason: every source of variance is a **traced per-episode VALUE over a static-max shape** (K is a traced scalar bounded by `count_high`; sampled behavioural params are traced floats), never an array dimension or static field — so the env reset+step compile once and cache, exactly as the parallel session showed for rPPO.

## Evidence, measurements, facts
- Level 05 smoke (current fixed dreamer code, node 112): 315 XLA compiles, 137 episodes in 35.6s at 98 SPS, no errors; top compiles `_squeeze`/`scatter`/`broadcast_in_dim` are the same ~40× one-time warmup as fixed-count levels (NOT a repeating storm — compile count does not grow with run length).
- Two L05 variants tested clean: the first (count ranges only, fixed predator) → 315 compiles; the updated (commits `b9712ef`/`2f16809`/`033c255` adding per-episode predator move_interval[1,3]/detection_range[1,7]/attack_delay/behaviour) → also 315 compiles. The new behavioural ranges added zero recompiles.
- Loads + resets clean: grid 10x10, hunt_pred=2 / neutral=2 at `count_high` allocation, masked to K active per episode.
- Depends on dreamer's own reset being fixed-width-masked (the same session's fixes), so the dreamer side has no variable-width reset that the count variance could interact badly with.

## Decisions and actions
- L05 launched into the 6-run basic-curriculum dreamer sweep (seed 42, WandB) and confirmed training. The full sweep runs on even nodes (114×4 + 112×2) per a lab reboot schedule.
- Generalises the rPPO recompile-immunity result to Dreamer: the count-variance masking design ([[20260623_0143_per_episode_count_variance_masking]]) + reset-all-then-`where` pattern ([[20260623_1616_rppo_reset_recompile_immune]]) hold for dreamer_srl once its own reset is fixed-width ([[20260622_1746_dreamer_srl_recompile_storm_done_count]]).
- Standing rule when adding per-episode variance to any config used by Dreamer: keep all variance as traced per-episode values over static-max shapes (count_high allocation + activation masks; sampled `[lo,hi]` fields), never as array dimensions or static (`pytree_node=False`) fields, and it stays compile-safe.

## Open questions and follow-ups
None.

## References
- Plan: `docs/develop/active/refactors/PER_EPISODE_ENV_VARIANCE.md`. Config: `configs/environment/experiment/basic/05-random_init_10x10.yaml`.
- Same-session train_step fix that made any dreamer training viable: [[20260630_1720_dreamer_srl_train_step_jit_compile_once]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 3c11010b-dbfb-4d17-af58-5b86e54d6815` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
