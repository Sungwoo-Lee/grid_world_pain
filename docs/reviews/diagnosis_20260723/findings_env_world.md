# Diagnosis findings — env world/sensing unit (2026-07-23)

**Unit**: `src/environment/grid_world.py` + `src/environment/sensor.py`
**Reviewer**: JAX/Flax RL correctness pass (report-only; no code modified)

**Headline**: The unit's biggest surprise is structural — `src/environment/grid_world.py` is **not** world construction at all. It is a stale, orphaned copy of the renderer (`src/environment/renderer.py`) with **zero importers** anywhere in `src/`, `scripts/`, `tests/`, or `train.py` (only `SOURCES.txt` packaging metadata and the generated `graph.html` mention it). It has fallen behind the live renderer and carries a guaranteed-crash `COLORS` KeyError that the live renderer already fixed. The actual sensing pipeline (`sensor.py`) is in good shape: observation layout, breakdown parity, noise alignment, masking, coordinates, bounds, and PRNG use all check out; findings there are latent/quality-grade only.

---

## Findings

### F1 — [P2] src/environment/grid_world.py (entire file, esp. line 602)
**Claim**: The file is a dead, stale duplicate of `renderer.py`; if ever imported and used it crashes with `KeyError` on the first frame where the agent takes damage.
**Failure scenario**: Any caller doing `from src.environment.grid_world import render_jax_state` (plausible — the filename suggests it is the world model; this very diagnosis task assumed so) and passing `info={'damage': >0}` hits line 602: `COLORS['dmg_hiding_predator']` / `COLORS['dmg_predator']` / `COLORS['dmg_obstacle']` — none of these keys exist in this file's `COLORS` dict (grid_world.py:104-126). The tuple list at line 602 is built eagerly, so the KeyError fires regardless of segment values.
**Evidence**: `md5sum` differs from `renderer.py`; `diff` shows renderer.py has (a) the three `dmg_*` color keys (renderer.py:127-131), (b) `min(1.0, pct)` clamping in `draw_dual_capsule_bar` (renderer.py:145,154) missing here (grid_world.py:139,148 — bars overflow their trough when state/obs > 1, e.g. olfaction >= 2.0 at contact), (c) the `obs_only` pod feature, and (d) current icon basenames (`hiding_predator` vs this file's stale `danger`/`agent_danger`, which no longer resolve against `assets/` -> icons silently `None`, falling back to markers). Grep confirms zero Python importers: all real callers (`scripts/eval/render_recordings.py:93`, `scripts/eval/benchmark_render.py:39`, `scripts/dreamer/visualize_dream.py:159`, `scripts/media/record_env_demo.py:14`, `src/environment/renderer_v2.py`) import `src.environment.renderer`.
**Fix direction**: Delete or `git mv` to an archive location (per project archive convention); at minimum add a module docstring "DEAD — superseded by renderer.py".

### F2 — [P2] src/environment/renderer.py:306-309 (same bug in the dead copy at grid_world.py:295-298 + wrong default labels grid_world.py:291)
**Claim**: The categorical Visual pod's color array is ordered `[..., predator, NEUTRAL, ROCK]` while the sensor's visual channel order is `5=Predator, 6=Rock, 7=Neutral` — so in eval videos the bar carrying rock data (correctly labeled `RCK` via labels passed from `build_sensory_viz`) is tinted the neutral-animal cyan, and the `NEU` bar is tinted rock-gray.
**Failure scenario**: Any eval video with `visual_sensor_enabled` and V=8: a researcher color-matching the Visual pod bars against arena entities concludes a neutral animal is in view when it is actually a rock (and vice versa). Data and labels are correct; only colors mislead. The default *label* fallback (`renderer.py:302`, `['...','NEU','RCK']`) is also swapped relative to the sensor's channel map, but is unreachable in practice because `build_sensory_viz` (sensor.py:449-452) always passes the correct `['...','RCK','NEU']`.
**Evidence**: sensor.py:146-171 docstring fixes channel map (6: Rock, 7: Neutral); `draw_boresight_diamond`'s parallel `feature_keys`/`feature_colors` (renderer.py:225-228) have the CORRECT rock-before-neutral order — only `draw_categorical_visual` is swapped.
**Fix direction**: Swap the last two entries of `feature_colors` in `draw_categorical_visual` (and fix the default label fallback to match sensor channel order).

### F3 — [P2] src/environment/renderer.py:462-491 (arena) and :534-536 (minimap); mirrored at grid_world.py:446-475, :518-520
**Claim**: Arena icon drawing and minimap dots ignore `state.animal_active` and `state.obs_active`, so per-episode-deactivated animals and obstacles (PER_EPISODE_ENV_VARIANCE) are rendered as if physically present, while every sensor correctly treats them as absent.
**Failure scenario**: In a config using per-episode entity subsets, an eval video shows a "predator" adjacent to the agent that emits zero nociception/olfaction/visual signal and cannot attack; a trajectory-story qualitative read ("agent ignores the predator!") is confounded by a ghost. Resources are handled correctly (`res_active` checked at renderer.py:455 and in minimap masks :532-533); only animals and obstacles pass all-ones masks (`np.ones(p_pos.shape[0])` etc., renderer.py:534-536).
**Evidence**: Compare with sensor.py, where every path masks by `animal_active`/`obs_active` (sensor.py:42, 78, 87, 208-213, 322-324).
**Fix direction**: Thread `state.animal_active` / `state.obs_active` into the arena loops and `plot_entity_dots` masks (viz-only change).

### F4 — [P2] src/environment/sensor.py:262 vs :284-285
**Claim**: Guard asymmetry in `apply_perceptual_noise`: the sigma/alpha/mode loop indexes `modality_map[sensor_name]` unguarded, while the clip_min/clip_max concats filter with `if name in modality_map` — the filter is currently dead, and if the first loop were ever "fixed" to match it, the clip arrays would silently mis-align against `obs.shape` for the skipped modality.
**Failure scenario**: A YAML typo in a modality key (e.g. `olfactory:` for `olfaction:`) is *silently dropped* by `config_loader.py:1512-1516` (`if k in _YAML_KEY_TO_SENSOR_NAME` — no error). If the corresponding sensor is enabled, the sigma loop then raises a bare `KeyError: 'Olfaction'` at JIT trace time with no pointer to the YAML; if the sensor is disabled, the typo passes completely silently (noise config never applied). Same family as the OPEN "config-boundary silent failures" row, but this specific trap (noise-modality key typo + in-sensor guard asymmetry) is not among its four listed traps.
**Evidence**: sensor.py:261-264 (unguarded `modality_map[sensor_name]`), :284-285 (guarded comprehensions); config_loader.py:1493-1516 (silent drop of unknown keys).
**Fix direction**: Validate `set(breakdown) subset-of set(noise_modality_order)` (and reject unknown YAML modality keys) at config-load time with a clear error; make the two code paths in `apply_perceptual_noise` symmetric.

### F5 — [P2] src/environment/sensor.py:288 (latent)
**Claim**: When `perceptual_noise_enabled` is true, the per-modality clip is applied to the *clean* signal even for modalities whose noise mode is `none` (mode 0, `sigma_eff = 0`), so a mode-0 modality's observation can differ from the `apply_noise=False` "true" observation whenever its natural range exceeds `[clip_min, clip_max]`.
**Failure scenario**: A config sets a modality to mode `none` with clip `[0, 1]` while the signal legitimately exceeds 1 (e.g. visual cell with background + stacked entities, or olfaction 2.0 at contact) — the agent's obs is silently saturated and eval's REAL-vs-OBS panels show a discrepancy in a "noise-free" modality. In the shipped `configs/environment/default.yaml` the clips are chosen to cover natural ranges (extero-noc/olfaction/visual clip_max=100, location [-1,1]), so this is latent, not active.
**Evidence**: sensor.py:277-288 — `sigma_eff` respects mode but `jnp.clip(obs + noise, clip_min, clip_max)` applies unconditionally to all modalities.
**Fix direction**: Only clip modalities with mode != 0 (e.g. `jnp.where(mode == 0, obs, clipped)`), or document clipping as intentional sensor saturation.

### F6 — [P2] src/environment/sensor.py:116-144 (latent)
**Claim**: `get_visual_offsets` contains dead sort code (`dist` computed at :138, never used) and the range-1 special case (:141-142, order C,U,R,D,L) disagrees with the generic path's ring-1 ordering (C,U,R,**L,D** — generation order at :127-134 emits `[0,-1]` before `[1,0]`), so growing a sensor range from 1 to 2 *permutes* the first-ring channel order in addition to appending cells.
**Failure scenario**: Any cross-config analysis, probe, or transfer that assumes the range-1 observation layout is a stable prefix of the range-2 layout silently reads Down as Left and Left as Down. Within any single run this is harmless — sensors, breakdown, and both renderers all call the same function, so each config is self-consistent.
**Evidence**: enumerated the generic loop for d=1: dr=-1 -> `[-1,0]`; dr=0 -> `[0,1]`,`[0,-1]`; dr=1 -> `[1,0]` — vs the hardcoded `[[0,0],[-1,0],[0,1],[1,0],[0,-1]]`.
**Fix direction**: Make the generic path emit the same per-ring U,R,D,L order (or delete the special case and document generation order); remove the dead `dist` sort remnant.

---

## Fixed-bug regression check (all PASS — no regressions found)

- **Noise painted on wrong sensory channel** (`2e4ac34`): fix intact — channel order is YAML-driven end to end: `config_loader.py:1512` builds `noise_modality_order` from YAML key order via `_YAML_KEY_TO_SENSOR_NAME` (names exactly match `get_observation_breakdown` keys), `apply_perceptual_noise` maps by *name* (not position), and `build_sensory_viz` (sensor.py:392-459) walks `obs`/`true_obs` with lock-stepped pointers over the same breakdown. Note F2 is a *color-tint* cosmetic issue in the renderer, not a recurrence of this data-channel bug.
- **Noise invisible in eval video** (`80d3b70`): fix intact — `build_sensory_viz` accepts and threads `true_obs`; `scripts/eval/render_recordings.py:117` passes `ep['true_obs']` through to the side-by-side panels.
- **Observability gate fires at step 0** (`84014e4`): fix was config-side (fixed start position eliminates step-0 contact); nothing in this unit re-introduces a step-0 contact path — `sense_extero_nociception` and `last_collision_noc` (initialized 0.0 at core.py:1357) are purely state-driven. Cannot regress from within this unit.
- **Fractional attack/detection range inclusive-integer fix** (`7ff8d1f`): lives in config_loader/core (attack_range sampling, core.py:1300), outside this unit; the olfactory `dist <= radius` comparison in `sense_resource` (sensor.py:17) is intentionally continuous Euclidean and is a different mechanism — unaffected.
- **JAX read-only arrays crash downstream** (`d729e2c`): renderer converts via `np.array(...)` copies (writable) throughout — consistent with the fix.

## Reviewed but clean

- **Observation layout parity**: `get_observation` (sensor.py:291-348) and `get_observation_breakdown` (:350-390) enable/order branches match one-for-one (Injury, Nutrition, Satiation, InteroNoc, ExteroNoc, Olfaction, Collision, Proprioception, Visual, Location); Collision/Visual cell counts `2r^2+2r+1` match `get_visual_offsets` cardinality (verified by ring enumeration: ring d has 4d cells).
- **Noise vector alignment**: sigma/alpha/mode/clip arrays are concatenated in breakdown order with breakdown dims -> element-aligned with the assembled obs; any dim mismatch would fail loudly at trace.
- **PRNG**: `obs_key = fold_in(state.key, 999)` (sensor.py:294) is a distinct stream — core.py splits `state.key` 6-ways per step and stores the carried key (core.py:505, 838), and no other call site folds `state.key` with 999; deterministic per (state, step), which is correct for reproducible obs-noise.
- **JIT safety**: all Python-level branches (`perceptual_noise_enabled`, `injury_observable`, `*_enabled`, `visual_vector_size`, `sensor_range`, `noise_modality_order`) are `struct.field(pytree_node=False)` statics in `EnvParams` (state.py:258-293); `@jax.jit(static_argnames=['apply_noise'])` is valid; noise arrays are traced but indexed only with static ints.
- **Masking**: visual/olfactory/nociception/collision all correctly AND with `res_active`/`animal_active`/`obs_active`; `sense_visual` doubly protects OOB cells (`safe_coords` for background gather + final `* is_in_bounds`); inactive-entity props zeroed *before* the matmul so parked/dead entities cannot contribute.
- **Coordinates/bounds**: egocentric offsets are `[dr, dc]` row-major, bounds checked against `[height, width]` in the same order; `sense_location` normalization to [-1,1] correct; no wrap-around anywhere (by design — OOB is collision).
- **Interoceptive nociception**: buffer stores absolute injury (core.py:115, slot 0 = newest), kernel is sum-normalized alpha with k[0]=0 (config_loader.py:1294-1304) -> convolved/max_injury in [0,1]; passthrough mode consistent.
- **`sense_resource` singularity**: `jnp.where(dist<0.001, 2.0, 1/(d^p + 1e-10))` — both branches finite, no NaN (env is not differentiated, so no gradient-through-where hazard).
- **Empty-array reductions**: all `jnp.max(..., initial=0.0)` guarded; `num_animal == 0` concat paths handled.
