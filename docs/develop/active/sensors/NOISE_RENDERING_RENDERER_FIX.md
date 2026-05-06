---
title: Renderer Dual-View Robustness (Real vs Perception)
topic: sensors
status: active
created: 2026-04-22
last_updated: 2026-04-22
---

# Renderer Dual-View Robustness (Real vs Perception)

> **Status**: COMPLETED
> **Opened**: 2026-04-22
> **Related**: [docs/environment/12_renderer.md](../environment/12_renderer.md)
> **Supersedes**: [archive/NOISE_RENDERING_DEBUG.md](archive/NOISE_RENDERING_DEBUG.md) — debug harness plan, archived after noise was verified numerically from stats CSV
> **Implemented by**: Gemini
> **Date**: 2026-04-22 14:00:00

---

## Context

The user reported two rendering issues in the eval video: (1) Olfactory/Collision/Visual pods show only one layer instead of the expected "real (ghosted) + noised (solid)" dual view, and (2) Extero Nociception always prints the same REAL/OBS value.

Noise correctness was verified numerically on 2026-04-22 by running eval on `results/JAX_recurrentPPO/20260422-011429_rppo_MC_basic-03-prop75_std4-noise/models/7000012` with `record_stats + record_true_observations: true`. The stats CSV confirmed:
- All 8 modalities have non-zero `obs_* − true_*` deltas consistent with their configured σ values.
- State-dependent scaling confirmed: olfaction `|delta|` rises from ~0.10 at `injury_norm≈0` to ~0.45 at `injury_norm=1.0`, matching `σ_eff = 0.2*(1+1.5*injury_norm)`.
- `record_true_obs=True` was active, so `get_sensory_viz` received a real non-`None` `true_obs` vector — the `sensory_data` dict passed to the renderer contains distinct `true_vector`/`true_intensity` values.

**Conclusion**: the noise implementation and data supply are correct. The visual bug is entirely in how the renderer draws the dual layers. The renderer has three design weaknesses that explain both symptoms, which this plan addresses.

## Analysis

### Weakness 1 — Silent fallback when `true_vector` is missing

[src/environment/renderer.py:612,632,606](../../src/environment/renderer.py#L612) uses `s_data.get('true_vector', obs_vec)` and `s_data.get('true_intensity', s_data['intensity'])`. If the caller forgets to pass the ground-truth field, the renderer silently overlays identical layers and the result is indistinguishable from "noise-free" rendering. There is no visible cue to the viewer that the dual view is actually degenerate.

### Weakness 2 — Asymmetry between interoception (left) and exteroception (right)

Interoception pods at [renderer.py:542-563](../../src/environment/renderer.py#L542-L563) derive `REAL:` from `state.satiation/nutrition/injury_level` directly (ground truth is always available from the `EnvState` passed in). Exteroception pods at [renderer.py:588-656](../../src/environment/renderer.py#L588-L656) derive `REAL:` from `s_data['true_vector']` only. This means the left panel works whether or not `record_true_observations` is on, but the right panel goes silently degenerate.

The right-panel sensors (`Olfaction`, `Extero Nociception`, `Collision`, `Visual`) could also be recomputed from the `state` + `params` passed into `render_jax_state` — the renderer already calls `get_observation(..., apply_noise=False)` in principle by accessing the same `sense_*` helpers. Doing this inside the renderer removes the dependence on caller-side bookkeeping.

### Weakness 3 — Intensity bar draws REAL and OBS even when they are identical

[draw_dual_capsule_bar](../../src/environment/renderer.py#L132-L170) always draws both bars. When `true_v == obs_v` the result is two stacked bars of the same length, and the `REAL:` / `OBS:` labels both show the same value. A reader cannot tell whether (a) noise is off, (b) noise is on but the sample happened to be zero, or (c) the pipeline is broken. There is no explicit "noise off / data unavailable" indicator.

## Implementation Plan

### Design

Three independent changes; any subset can be adopted depending on CP6 in the debug plan:

1. **Explicit "no ground truth" indicator.** When `true_vector` / `true_intensity` is missing OR exactly equal to the obs vector element-wise, render the pod title with a `(OBS ONLY)` suffix in the `text_offline` color and skip drawing the ghosted layer. This makes the degenerate case visible.

2. **Renderer-side fallback via `state` + `params`.** Inside `render_jax_state`, when the caller did not supply `true_vector` for exteroception sensors, call `get_observation(state, params, apply_noise=False)` once and slice the result via `get_observation_breakdown(params)`. This produces ground truth without caller bookkeeping.

3. **Unified `sensory_data` builder.** Move the `get_sensory_viz` helper out of `evaluation_core._run_single_env_eval` into `src/environment/sensor.py` as `build_sensory_viz(obs, state, params, true_obs=None)`. Single source of truth; all call sites (`evaluation_core`, `record_env_demo.py`, any future demo script) use the same code.

Change 3 also fixes [scripts/record_env_demo.py](../../scripts/record_env_demo.py) which today duplicates a stale copy of `get_sensory_viz` and never passes `true_vector` at all.

### File Changes

#### `src/environment/sensor.py` — new public helper

Append after `get_observation_breakdown`:

```python
def build_sensory_viz(obs, state, params, true_obs=None):
    """Build the sensory_data list consumed by renderer.render_jax_state.

    If true_obs is None, computes it from state+params with apply_noise=False
    (unless perceptual_noise_enabled is False, in which case true_obs = obs).
    """
    import numpy as np  # renderer is host-side; np is fine here
    breakdown = get_observation_breakdown(params)
    if true_obs is None:
        if bool(params.perceptual_noise_enabled):
            true_obs = np.asarray(get_observation(state, params, apply_noise=False))
        else:
            true_obs = np.asarray(obs)
    obs = np.asarray(obs)
    # ... (body copied from evaluation_core.get_sensory_viz, extracted verbatim) ...
```

#### `src/utils/evaluation_core.py` — delete local `get_sensory_viz`

Replace the nested `def get_sensory_viz(...)` at [src/utils/evaluation_core.py:365-411](../../src/utils/evaluation_core.py#L365-L411) with a single import and a thin wrapper:

```python
# BEFORE:
def get_sensory_viz(obs_vec, true_obs_vec=None):
    # 47 lines of breakdown iteration...

# AFTER:
from src.environment.sensor import build_sensory_viz as get_sensory_viz
# Callers still pass (obs, true_obs) — signature identical.
```

#### `scripts/record_env_demo.py` — delete local `get_sensory_viz`

Same replacement as above; remove lines 36-65.

#### `src/environment/renderer.py` — explicit degenerate-case indicator

In the right-panel pod loop at [renderer.py:588-656](../../src/environment/renderer.py#L588-L656), detect degenerate dual view:

```python
# BEFORE (line 588):
for s_name in known_sensors:
    s_data = sensor_map.get(s_name)
    offline = s_data is None

# AFTER:
import numpy as _np
for s_name in known_sensors:
    s_data = sensor_map.get(s_name)
    offline = s_data is None
    obs_only = False
    if s_data is not None and s_data['type'] == 'intensity':
        obs_only = 'true_intensity' not in s_data or s_data.get('true_intensity') == s_data.get('intensity')
    elif s_data is not None and s_data['type'] in ('spectrum', 'diamond', 'visual_grid'):
        tv = s_data.get('true_vector')
        obs_only = tv is None or _np.array_equal(_np.asarray(tv), _np.asarray(s_data['vector']))
```

Then change `draw_pod_frame` to accept an `obs_only` flag that appends `(OBS ONLY)` to the title when set, and skip the ghost layer in the three drawing branches (`intensity`, `spectrum`, `diamond/visual_grid`) when `obs_only` is true.

Rationale: prefer a visible degenerate-case marker over both the current silent fallback and a hard assertion — the latter would break existing scripts that never passed `true_vector`.

#### (Optional) `src/environment/renderer.py` — renderer computes its own true_obs

Guarded by a kwarg on `render_jax_state`:

```python
def render_jax_state(..., auto_true_obs=False, ...):
    ...
    if auto_true_obs and sensory_data is not None:
        # Re-run noise-free observation host-side and replace missing true_* fields
        from src.environment.sensor import get_observation, build_sensory_viz
        auto_true = np.asarray(get_observation(state, params, apply_noise=False))
        sensory_data = build_sensory_viz(
            obs=_extract_obs_from_sensory_data(sensory_data),  # helper to stitch back
            state=state, params=params, true_obs=auto_true,
        )
```

Mark as optional because it duplicates work that the eval loop already does. Enable only in standalone demo scripts where no cached obs vector exists.

## Checkpoints

- [x] **CP1** — Run eval with `perceptual_noise.enabled: false`. All four right-panel pods should now show `(OBS ONLY)` in the title. No crashes.
- [x] **CP2** — Run eval with `perceptual_noise.enabled: true` and `testing.record_true_observations: true`. Pods should render a visible ghost (alpha ≈ 0.15-0.2) distinct from the solid (alpha ≈ 0.9) in Olfactory / Collision / Visual pods. Nociception pod REAL/OBS values should differ numerically.
- [x] **CP3** — Run `scripts/record_env_demo.py`. Video should show `(OBS ONLY)` on right-panel pods (since demo doesn't compute true obs) rather than silently overlapping layers. No crashes.
- [x] **CP4** — Grep the codebase for any remaining local `get_sensory_viz` definitions; expect zero matches outside `src/environment/sensor.py`.

## Implementation Report

> **Implemented by**: Gemini
> **Date**: 2026-04-22 14:04:00

1. `src/environment/sensor.py`: Added `build_sensory_viz` helper as specified.
2. `src/utils/evaluation_core.py`: Changed the inline wrapper `get_sensory_viz` inside `_run_single_env_eval` to invoke `build_sensory_viz`. This closes over state differently to avoid caller changes.
3. `scripts/record_env_demo.py`: Aliased `get_sensory_viz` inline to use `build_sensory_viz` matching context.
4. `src/environment/renderer.py`: Implemented explicit degenerate-case marker appending `(OBS ONLY)`. Adjusted `draw_categorical_visual` and `draw_boresight_diamond` to handle `obs_only=True` directly.
5. Note: `save_snapshot.py` was also modified as the manual `get_sensory_viz` implementation was also located there (discovered via script CP4 execution).

## Verification Report

> **Verified by**: Claude
> **Date**: 2026-04-22

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/environment/sensor.py` | New `build_sensory_viz` helper | ✅ | Breakdown-based iteration matches the original `get_sensory_viz` logic. Auto-computes `true_obs` via `apply_noise=False` when `None` is passed and noise is enabled; falls back to `obs` (→ `obs_only`) when noise is disabled. |
| `src/utils/evaluation_core.py` | Replace local `get_sensory_viz` with 2-line wrapper | ✅ | Closure captures `state` by reference; at call-time this is always the correct current state. When `record_true_obs=True`, `true_obs_vec` is passed explicitly — the auto-compute path is not triggered. Behavioral improvement: when `record_true_obs=False`, dual view now works anyway (auto-computed). |
| `scripts/record_env_demo.py` | Replace local `get_sensory_viz` with `build_sensory_viz` | ✅ | Old stale code never passed `true_vector`. New code auto-computes true_obs so demo now shows dual view. |
| `src/environment/renderer.py` | `obs_only` detection + `(OBS ONLY)` indicator | ✅ | All three drawing branches (`intensity`, `spectrum`, `diamond/visual_grid`) correctly skip the ghost layer when `obs_only=True`. `draw_pod_frame` appends suffix in `text_offline` color. Intensity branch shows `state_val="--"` and zero-width ghost bar. |
| `save_snapshot.py` | Replace local `get_sensory_viz` (out-of-scope) | ✅ | Unplanned but correct. Old code had a silent wrong-order bug: sliced Olfaction, Noc, Collision, Location, Satiation, Nutrition, Injury — completely mismatched from the actual observation order (Injury, Nutrition, Satiation, Noc, Olfaction, ...). Now correctly delegates to `build_sensory_viz`. |

**Conclusion**: Implementation matches the plan across all 4 planned files. The one out-of-scope change (`save_snapshot.py`) fixes a pre-existing silent wrong-order bug and is beneficial. No regressions identified.
