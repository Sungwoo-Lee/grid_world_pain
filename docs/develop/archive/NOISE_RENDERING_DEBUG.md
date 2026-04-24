# Perceptual Noise Rendering Debug Harness

> **Status**: PLANNED
> **Opened**: 2026-04-22
> **Related**: [NOISE_RENDERING_RENDERER_FIX.md](NOISE_RENDERING_RENDERER_FIX.md) — follow-up renderer improvements
> **Artifacts**: [docs/environment/10_perceptual_noise.md](../environment/10_perceptual_noise.md), [docs/environment/12_renderer.md](../environment/12_renderer.md)

---

## Context

The user reported two rendering issues in the eval video produced by `evaluation_core._run_single_env_eval` when running [configs/experiment/labmeeting/basic-03-noise.yaml](../../configs/experiment/labmeeting/basic-03-noise.yaml):

1. **Olfactory / Collision / Visual pods** — only one layer is visible, not the expected "real (ghosted) + noised (solid)" dual view.
2. **Extero Nociception pod** — `REAL:` and `OBS:` labels always print the same value.

Before proposing any renderer change we need to confirm whether the problem is (a) the data passed to the renderer, (b) the renderer itself, or (c) the noise implementation silently not firing. Today we have no way to inspect the `sensory_data` dict the renderer actually receives on a given step, and no way to inspect an individual frame without seeking inside an MP4.

## Analysis

### Data-path trace

`_run_single_env_eval` at [src/utils/evaluation_core.py:306-478](../../src/utils/evaluation_core.py#L306-L478):

```
record_true_obs = config.get_mandatory('testing.record_true_observations') and record_stats    # line 166
  |
true_obs = get_observation(state, params_ref, apply_noise=False) if record_true_obs else None  # lines 341, 467
  |
get_sensory_viz(obs, true_obs)                                                                  # lines 420, 475
  |
  └── for each sensor: *_true = slice(true_obs_vec) if true_obs_vec is not None else *_obs      # lines 374, 380, 386, 402
  |
render_jax_state(..., sensory_data=viz, ...)                                                    # lines 417, 472
```

**Failure mode**: whenever `record_true_obs` is `False` (either `testing.record_true_observations: false` OR `testing.record_stats: false`), `true_obs` is `None`, and every `*_true` in the viz dict silently falls back to the noisy `*_obs`. The renderer then overlays two identical layers — symptom #1 for spectrum/diamond/visual, symptom #2 for intensity/nociception.

### Existing stats CSV already captures the raw numerics

`_write_episode_stats` at [src/utils/evaluation_core.py:44-132](../../src/utils/evaluation_core.py#L44-L132) writes one row per step to `{results_dir}/stats/{checkpoint_pct}/{episode:06d}ep_stats.csv`. When `record_stats + record_true_observations` are both true, that row already contains for every step:

| Column group | Columns | Source |
|---|---|---|
| Step context | `step, pos_r, pos_c, action, reward` | always |
| Body state | `satiation, nutrition, injury, rest_streak` | always |
| Events | `event_ate, event_collided, event_rested, damage_total, damage_hiding_predator, damage_predator, damage_obstacle` | always |
| **Noised observation** | `obs_intero_satiation, obs_intero_nutrition, obs_intero_injury, obs_noc, obs_olf_{0..V-1}, obs_coll_r{dr}c{dc}, obs_prop_{0..A-1}, obs_vis_{0..W-1}, obs_loc_r, obs_loc_c` | `record_stats` |
| **Ground-truth observation** | `true_intero_*, true_noc, true_olf_*, true_coll_*, true_prop_*, true_vis_*, true_loc_*` | `record_true_obs` |
| World entities | `res_{i}_r/c/active, pred_{i}_r/c, neutral_{i}_r/c, obs_entity_{i}_r/c` | `record_stats` |
| Closing | `termination_reason, max_satiation, max_injury` | `record_stats` |

**Derivable post-hoc from the existing CSV** (no new columns needed):

- `delta_*` per modality: `obs_* − true_*`
- `injury_norm = injury / max_injury`
- `σ_eff` per modality: needs the static per-modality mode/σ_base/α/clip, which are constant across a run

The only missing piece for numeric noise verification is the **static per-run noise config** (σ_base, α, mode, clip_min, clip_max, modality_order). That belongs in a one-time sidecar next to the CSV, not in every row.

### Renderer trace (reference)

- `intensity` pod ([renderer.py:603-609](../../src/environment/renderer.py#L603-L609)) reads `true_intensity` with fallback to `s_data['intensity']` — if equal, both stacked bars overlap and the `REAL:`/`OBS:` labels show the same number.
- `spectrum` pod ([renderer.py:610-626](../../src/environment/renderer.py#L610-L626)) reads `true_vector` with fallback to `obs_vec`.
- `diamond` / `visual_grid` pod ([renderer.py:627-650](../../src/environment/renderer.py#L627-L650), `draw_categorical_visual` at [renderer.py:284-345](../../src/environment/renderer.py#L284-L345)) reads `true_vec` with fallback to `obs_v`.

The renderer branches look consistent. What the debug harness needs to prove is whether the `sensory_data` dict arriving at the renderer actually contains distinct `vector`/`true_vector` arrays on a given step. Reading that dict back out of the flat CSV is lossy (type/shape info is gone). A per-step JSON snapshot is the direct way.

### Noise-implementation audit (static code review)

Comparing [src/environment/sensor.py:197-241](../../src/environment/sensor.py#L197-L241) against [docs/environment/10_perceptual_noise.md](../environment/10_perceptual_noise.md):

| Doc claim | Implementation location | Matches? |
|---|---|---|
| `modality_map = {name: index}` from `noise_modality_order` | sensor.py:206 | ✅ |
| Iterate `get_observation_breakdown(params)` and broadcast σ_base, α, mode to each sensor's dim | sensor.py:213-217 | ✅ |
| `σ_eff = mode==2 ? σ_base*(1+α*inj) : mode==1 ? σ_base : 0.0` | sensor.py:230-234 | ✅ |
| Clip per-modality via `noise_clip_min/max` | sensor.py:237-241 | ✅ |
| PRNG key: `jax.random.fold_in(state.key, 999)` | sensor.py:247 | ✅ |
| YAML key order is single source of truth | config_loader.py (per memory note 2026-04-12) | ✅ |

No static defects. Runtime verification still needed on the exact config the user runs — handled by the CP checklist below using the existing CSV.

## Implementation Plan

### Design

Reuse the existing stats CSV for numeric verification; add only the missing pieces that the CSV cannot capture.

**Reuses unchanged** (no code changes in these):
- `_write_episode_stats` CSV — already has `obs_*` and `true_*` per modality.
- `scripts/analyze_noise_diagnostics.py` — already post-processes those columns.

**New artifacts** — three additions, all gated by a single new mandatory flag `testing.noise_debug_output`:

1. **One-time `noise_config.json` sidecar** (per run, not per step). Written once at stats_dir creation. Captures `noise_modality_order`, `noise_modes`, `noise_sigmas`, `noise_injury_scales`, `noise_clip_min`, `noise_clip_max`, `perceptual_noise_enabled`, `max_injury`. This lets any analysis script compute σ_eff per step from the CSV's `injury` column without hardcoding config values.
2. **Per-step `sensory_data/step_{NNNN}.json`** (per episode). The exact `sensory_data` list passed to `render_jax_state` on that step, with `vector`/`true_vector` (or `intensity`/`true_intensity`) preserved as nested arrays. Diagnoses "is the renderer receiving dual data or degenerate data?" directly.
3. **Per-step `frames/step_{NNNN}.png`** (per episode). The RGB frame that would otherwise only exist inside the MP4, saved individually with `matplotlib.image.imsave` before `save_jax_video` concatenates them. Lets the user inspect arbitrary frames by filename.

All three go under `{results_dir}/noise_debug/{checkpoint_pct}/ep_{episode:03d}/`. When `noise_debug_output: false`, zero extra I/O; existing behavior is byte-identical.

`noise_debug_output: true` **forces `record_stats=true` AND `record_true_observations=true`** via a warning log (both already required for meaningful CSV rows; without them the exercise is pointless).

Scope is **single-env path only** — the user is running 1 env per the current eval config; parallel path is out of scope and will carry a comment pointing to this plan if/when needed.

### File Changes

#### `configs/evaluation/default.yaml` (append)

```yaml
# BEFORE:
testing:
  seed: 8217
  evaluation_episodes: 1
  num_envs: 1
  render_video: true
  record_stats: true
  record_true_observations: true

# AFTER:
testing:
  seed: 8217
  evaluation_episodes: 1
  num_envs: 1
  render_video: true
  record_stats: true
  record_true_observations: true
  # Noise visualization debug. When true, additionally writes:
  #   - {results_dir}/noise_debug/{checkpoint_pct}/noise_config.json  (one per run)
  #   - .../ep_{NNN}/sensory_data/step_{NNNN}.json                    (one per step)
  #   - .../ep_{NNN}/frames/step_{NNNN}.png                           (one per step)
  # Forces record_stats + record_true_observations on.
  # Mandatory key — no fallback default.
  noise_debug_output: false
```

#### `src/utils/evaluation_core.py` — setup block (around lines 160-170)

```python
# BEFORE:
record_true_obs = config.get_mandatory('testing.record_true_observations') and record_stats

# AFTER:
record_true_obs = config.get_mandatory('testing.record_true_observations') and record_stats
noise_debug_output = config.get_mandatory('testing.noise_debug_output')
if noise_debug_output:
    if not record_stats:
        print("[WARN] testing.noise_debug_output=true requires record_stats; forcing on for this run.", flush=True)
        record_stats = True
    if not record_true_obs:
        print("[WARN] testing.noise_debug_output=true requires record_true_observations; forcing on for this run.", flush=True)
        record_true_obs = True
```

Pass `noise_debug_output` through to `_run_single_env_eval()` at the call site (around line 261).

#### `src/utils/evaluation_core.py` — new helper `_write_noise_config_sidecar` (top of file, next to `_write_episode_stats`)

```python
def _write_noise_config_sidecar(noise_debug_root, params):
    """Write one-time static noise config sidecar. Called once per run before any episode starts."""
    import json
    os.makedirs(noise_debug_root, exist_ok=True)
    payload = {
        'perceptual_noise_enabled': bool(params.perceptual_noise_enabled),
        'max_injury': float(params.max_injury),
        'noise_modality_order': list(params.noise_modality_order),
        'noise_modes': np.asarray(params.noise_modes).tolist(),
        'noise_sigmas': np.asarray(params.noise_sigmas).tolist(),
        'noise_injury_scales': np.asarray(params.noise_injury_scales).tolist(),
        'noise_clip_min': np.asarray(params.noise_clip_min).tolist(),
        'noise_clip_max': np.asarray(params.noise_clip_max).tolist(),
    }
    with open(os.path.join(noise_debug_root, 'noise_config.json'), 'w') as f:
        json.dump(payload, f, indent=2)
```

#### `src/utils/evaluation_core.py` — new helper `_dump_sensory_data_and_frame`

```python
def _dump_sensory_data_and_frame(ep_dir, step, sensory_data, frame_rgb):
    """Write per-step sensory_data JSON + frame PNG. Called when noise_debug_output is enabled."""
    import json
    from matplotlib.image import imsave

    def _to_py(v):
        if isinstance(v, (jnp.ndarray, np.ndarray)): return np.asarray(v).tolist()
        if isinstance(v, (list, tuple)):             return [_to_py(x) for x in v]
        if isinstance(v, dict):                      return {k: _to_py(x) for k, x in v.items()}
        if hasattr(v, 'item'):                       return v.item()
        return v

    sd_dir = os.path.join(ep_dir, 'sensory_data')
    fr_dir = os.path.join(ep_dir, 'frames')
    os.makedirs(sd_dir, exist_ok=True)
    os.makedirs(fr_dir, exist_ok=True)
    with open(os.path.join(sd_dir, f'step_{step:04d}.json'), 'w') as f:
        json.dump(_to_py(sensory_data), f, indent=2)
    imsave(os.path.join(fr_dir, f'step_{step:04d}.png'), frame_rgb)
```

#### `src/utils/evaluation_core.py` — `_run_single_env_eval` integration

**Once per run**, before the episode loop (after `record_true_obs` line, around line 170 — pass `noise_debug_root` into the single-env path):

```python
if noise_debug_output:
    noise_debug_root = os.path.join(results_dir, 'noise_debug', str(checkpoint_pct))
    _write_noise_config_sidecar(noise_debug_root, params_ref)
else:
    noise_debug_root = None
```

**Inside `_run_single_env_eval`**, set up per-episode dir at the start of each episode:

```python
# After `ep_true_obs = [] if record_true_obs else None` (~line 338):
if noise_debug_output:
    ep_noise_dir = os.path.join(noise_debug_root, f'ep_{ep+1:03d}')
    os.makedirs(ep_noise_dir, exist_ok=True)
else:
    ep_noise_dir = None
```

**Augment the two render sites** — the step-0 block (~line 413-423) and the per-step block (~line 469-478). For each: capture the returned frame into a local variable, then dump:

```python
# BEFORE (step-0 block):
if render_video:
    if debug: print(f"    [Render] Initial frame...", end="", flush=True)
    state_for_render = jax.device_get(state)
    all_frames.append(render_jax_state(
        state_for_render, params_ref, episode=ep+1, step=0,
        train_episode=checkpoint_pct,
        sensory_data=get_sensory_viz(obs, true_obs),
        info=None,
        icon_config=icon_config
    ))

# AFTER:
if render_video:
    if debug: print(f"    [Render] Initial frame...", end="", flush=True)
    state_for_render = jax.device_get(state)
    viz0 = get_sensory_viz(obs, true_obs)
    frame0 = render_jax_state(
        state_for_render, params_ref, episode=ep+1, step=0,
        train_episode=checkpoint_pct, sensory_data=viz0,
        info=None, icon_config=icon_config,
    )
    all_frames.append(frame0)
    if noise_debug_output:
        _dump_sensory_data_and_frame(ep_noise_dir, 0, viz0, frame0)
```

Apply the same capture+dump pattern to the per-step block. No new JAX evaluations introduced — `viz` and `frame` are already computed; the dump only uses host-side arrays.

## Checkpoints

- [ ] **CP1** — With `noise_debug_output: false`: run eval. Verify the `noise_debug/` directory is **not** created; stats CSV content unchanged byte-for-byte vs. a previous baseline run. Confirms zero regression in the default path.
- [ ] **CP2** — With `noise_debug_output: true` and `record_true_observations: false`: verify both `[WARN]` lines appear and the artifacts are still produced (force-on path works; CSV has `true_*` columns).
- [ ] **CP3** — Open `noise_debug/{chk}/noise_config.json`. Verify `noise_modality_order == ('Injury','Nutrition','Satiation','Extero Nociception','Olfaction','Collision','Proprioception','Visual','Location')` for the default 9-modality config, and `noise_sigmas[i]` matches the YAML values.
- [ ] **CP4** — Open one `{epdir}/stats/.../000001ep_stats.csv` row where `injury > 0`: compute `obs_olf_0 − true_olf_0` and `obs_vis_0 − true_vis_0`. Expect non-zero magnitudes consistent with σ_base*(1+α*injury_norm) from the sidecar. Confirms olfactory + visual noise fires and scales with injury. Repeat for `obs_coll_*` (expect ~±0.01 constant jitter) and `obs_prop_*` (~±0.05 constant).
- [ ] **CP5** — Open `{epdir}/sensory_data/step_NNNN.json`. Verify `Olfactory`, `Extero Nociception`, `Collision`, `Visual` entries each contain **both** `vector`/`intensity` AND `true_vector`/`true_intensity` as **distinct** arrays. If they are element-wise equal, the bug is upstream of the renderer — look at `record_true_obs` gating at [evaluation_core.py:166](../../src/utils/evaluation_core.py#L166).
- [ ] **CP6** — Open `{epdir}/frames/step_NNNN.png` for the same step as CP5. If the JSON confirms distinct vectors but the PNG still shows only one layer (no ghosted real + solid obs), the bug is in the renderer itself — escalate to [NOISE_RENDERING_RENDERER_FIX.md](NOISE_RENDERING_RENDERER_FIX.md).

## Implementation Report

> **Implemented by**: _TBD_
> **Date**: _TBD_

<!-- Filled by the implementing agent after code changes are made. -->

## Verification Report

> **Verified by**: _TBD_
> **Date**: _TBD_

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `configs/evaluation/default.yaml` | Append `noise_debug_output: false` | | |
| `src/utils/evaluation_core.py` | Read new mandatory key, force-on guards, thread `noise_debug_output` + `noise_debug_root` into single-env path | | |
| `src/utils/evaluation_core.py` | New helpers `_write_noise_config_sidecar`, `_dump_sensory_data_and_frame` | | |
| `src/utils/evaluation_core.py` (single-env loop) | Per-episode `ep_noise_dir`; capture+dump at step-0 and per-step render sites | | |

**Conclusion**: _TBD_

---

## Design Note — Why Not Add Columns to the Existing CSV?

The first draft of this plan proposed a parallel `noise_debug.csv` with `delta_*` and `σ_eff_*` columns. On review, both are derivable post-hoc from the existing stats CSV plus the one-time `noise_config.json` sidecar:

- `delta = obs − true` — one pandas expression.
- `σ_eff = sigma[idx] * (1 + alpha[idx] * injury / max_injury)` when `mode[idx]==2`, else `sigma[idx]` when `mode[idx]==1`, else 0 — again computed from the sidecar + `injury` column.

Keeping the computed columns out of the row avoids (a) doubling the per-step I/O, (b) drift between the formula written in the CSV vs. the formula in `apply_perceptual_noise`, and (c) a wider `stat_headers` that would need to stay in sync with the breakdown. The sidecar + existing CSV covers the numeric verification; the JSON + PNG covers what the CSV cannot (renderer-input structure and individual frames).
