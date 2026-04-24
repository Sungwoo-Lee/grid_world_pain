# 12 — Renderer & Visualization

> **Source**: `src/environment/renderer.py` | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## Overview

`render_jax_state(state, params, ...)` (`renderer.py:343`) converts a single `EnvState` + `EnvParams` snapshot into an RGB numpy array (H × W × 3). Frames can be saved as an MP4 video with `save_jax_video`. The renderer is pure Python (NumPy + Matplotlib) and runs on CPU — it is called after data is collected from the JAX environment, never inside JIT.

The renderer produces a **three-panel telemetry dashboard** designed for research presentations. It shows both the agent's objective state (ground truth) and its noisy perception side-by-side where applicable.

---

## Dashboard Layout (3 Panels)

Figure size: `14 × 10 inches` at configurable DPI. Uses `GridSpec` with `width_ratios=[0.8, 2.2, 0.8]`.

```
┌────────────────┬───────────────────────────┬────────────────┐
│  LEFT PANEL    │     CENTRE PANEL          │  RIGHT PANEL   │
│  INTEROCEPTION │     ULTIMATE ARENA        │  EXTEROCEPTION │
│                │     (local grid view)     │                │
│  - Satiation   │                           │  - Nociception │
│  - Nutrition   │     (minimap inset        │  - Olfaction   │
│  - Injury      │      at bottom)           │  - Collision   │
│  - Run context │                           │  - Visual      │
│                │                           │  - Location    │
│                │                           │  - Action Pod  │
└────────────────┴───────────────────────────┴────────────────┘
```

---

## Left Panel — Interoception Vitals

Draws three **dual capsule bars** (`draw_dual_capsule_bar`) — one each for Satiation, Nutrition, and Injury. Each bar shows:
- **Reality** (ghosted, translucent): the actual `EnvState` value.
- **Perception** (solid, opaque): the agent's observed value (from `sensory_data` if provided).

Below the vitals: a **Run Context pod** showing episode number, grid scale, and current step.

At the bottom: a **minimap** (`ax_minimap` inset) showing the full grid with entity positions as small colored dots and a viewport rectangle indicating the local view window.

---

## Centre Panel — Arena

Shows a **local view window** (`local_view_size × local_view_size`) centred on the agent. The window clamps to grid boundaries when the agent is near an edge.

Rendering order:
1. Background grid lines.
2. Terrain tile colors: white (plain), light green (grass), amber (sand).
3. Entity icons: food, danger, predator, rock/bush/tree, neutral animal.
4. Agent icon — composite icon used when agent co-occupies a cell with another entity:
   - Agent + predator → `agent_predator`
   - Agent + danger → `agent_danger`
   - Agent + bush → `agent_bush`
   - Agent + food → `agent_food`

---

## Right Panel — Exteroception Pods

Each sensor modality is drawn as a **pod** (`draw_pod_frame`). Pods are rendered top-down; the loop stops when `y_cursor < 0.20` to leave room for the Action Pod.

Pod types by sensor:
| Sensor | Pod type | Visualization |
|--------|----------|---------------|
| Extero Nociception | `intensity` | Dual capsule bar (reality + perception) |
| Olfaction | `spectrum` | Per-channel bar chart |
| Collision | `diamond` or `visual_grid` | Categorical spatial grid |
| Visual | `diamond` or `visual_grid` | Categorical spatial grid |
| Location | `text` | Coordinate string |

Pods are marked **OFFLINE** (greyed out) when the sensor is not enabled or `sensory_data` doesn't include that sensor.

**Damage pod** (bottom of exteroception, when damage > 0): segmented bar showing proportional split of `damage_danger`, `damage_predator`, `damage_obstacle`.

**Action pod** (fixed bottom): shows the current action name and arrow symbol.

---

## Icon Loading

`_load_icons(icon_config)` (`renderer.py:30`) loads icons from `assets/` relative to the project root. Supported formats: `.png`, `.jpg`, `.jpeg`, `.svg`, `.gif`, `.webp`. SVG requires `cairosvg` (optional dependency).

Icon keys and their filenames (default config):
| Key | Filename |
|-----|---------|
| `agent` | `agent` |
| `food` | `food` |
| `danger` | `danger` |
| `predator` | `predator` |
| `agent_food` | `agent_food` |
| `agent_danger` | `agent_danger` |
| `agent_predator` | `agent_predator` |
| `rock` | `rock` |
| `bush` | `bush` |
| `agent_bush` | `bush_agent` |
| `neutral` | `neutral` |

Icons are globally cached in `_ICON_CACHE` after the first load. If a file is not found, the key maps to `None` and a fallback marker is used: matplotlib scatter plot with shape and color depending on entity type.

Custom icon mappings can be passed via `icon_config` kwarg to `render_jax_state`.

---

## Iterative Renderer Development Pipeline

This section is the reference for any renderer development cycle — adding a new pod, changing the layout engine, redesigning cards, or fixing a visual bug. Follow this workflow regardless of which renderer module is being changed.

---

### Minimal Harness Script Pattern

Every renderer iteration needs a standalone script that loads a saved model, steps it for one episode, calls the renderer, and writes PNGs to `tmp/`. The script must be runnable from the project root in under 90 seconds.

**Canonical harness**: `tmp/test_v2_render.py` — use this as the template for any future harness.

Minimum structure a harness must follow:

```python
import os, sys
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. Load config + env params
config = Config(yaml.safe_load(open(config_path)))
params = load_env_params(config)

# 2. Build model — include ALL required kwargs (see pitfalls below)
model = ActorCriticRNN(
    input_dim=..., action_dim=...,
    hidden_size=config.get_mandatory('agent.hidden_size'),
    rngs=nnx.Rngs(key),
    rnn_type=config.get_mandatory('agent.rnn_type'),
    activation=config.get_mandatory('agent.activation'),
    modulation_config=...,
    observation_breakdown=get_observation_breakdown(params),   # required
    encoding_config=config.to_dict().get('agent', {}),         # required
)

# 3. Restore weights — use merge_restored, not bare nnx.update (see pitfalls)
restored = checkpointer.restore(CHECKPOINT)
peeled = peel(restored['model'])
current_struct = to_pure_dict(nnx.state(model, nnx.Param))
nnx.update(model, merge_restored(current_struct, peeled))

# 4. Init hidden state — never pass None directly to generic_inference
h_state = model.initial_state(batch_size=None) if hasattr(model, 'initial_state') else None

# 5. Episode loop: step → build sensory_data → render → collect frame
for step_n in range(MAX_STEPS):
    action, _, _, h_state, _ = generic_inference(model, obs[None, :], h_state, eval_mode=True)
    state, reward, done, _ = jax_step(state, int(jnp.squeeze(action)), params)
    obs = get_observation(state, params)
    true_obs = get_observation(state, params, apply_noise=False)
    viz = build_sensory_viz(obs, state, params, true_obs)
    frames.append(render_fn(jax.device_get(state), params,
                             episode=1, step=step_n, action=int(...),
                             sensory_data=viz, icon_config=...))
    if done: break

# 6. Save to tmp/ (never to results/)
save_jax_video(frames, 'tmp/render_test.mp4', fps=5)
subprocess.run(['ffmpeg', '-y', '-i', 'tmp/render_test.mp4',
                'tmp/render_frames/step_%04d.png'], check=True)
```

**Reference checkpoint** (use for all renderer debugging — noise is enabled, multiple entity types visible):
```
results/JAX_recurrentPPO/20260422-011429_rppo_MC_basic-03-prop75_std4-noise/
  models/7500046
```

Run from project root:
```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python tmp/test_v2_render.py
```
Typical wall time: 30–60 s. Output overwrites `tmp/render_frames/` on every run.

---

### Iteration Loop

```
edit src/environment/renderer_*.py
  └─→ python tmp/test_v2_render.py          (~45s)
        └─→ inspect tmp/render_frames/ in IDE
              └─→ identify issue → edit → repeat
```

Frames are overwritten each run — no cleanup needed between iterations.

---

### Frame Inspection Guide

The reference episode (checkpoint `7500046`, seed 42) consistently produces 11–12 frames. Each frame tests a different scenario:

| Frame | Scenario | What to verify |
|-------|----------|----------------|
| `step_0001.png` | Initial state, no action | Arena fully populated; all pods render; action pod shows `—` |
| `step_0004.png` – `step_0007.png` | Mid-episode, healthy agent | Nutrition drift (Δ < 0); olfactory bars vary per position; minimap agent dot moves |
| `step_0009.png` – `step_0011.png` | High-injury, predator contact | Nociception fires (`obs > 0`, `real > 0`, `Δ` non-zero); Injury bar shows OBS vs REAL split; Visual ghost bars appear; Collision directional bars fire |
| `step_0012.png` | Episode end (`done=True`) | Final state rendered cleanly; no crashes on zero-health edge case |

**The injury frames (9–11) are the most diagnostic** — they exercise all dual-view paths simultaneously: intensity pod with non-trivial Δ, spectrum ghost bars, categorical ghost bars, and the vital card tick marker all at once.

**Checklist per frame:**
- [ ] No overlapping text or bars
- [ ] Cards have visible borders and backgrounds
- [ ] Header banner shows correct step number
- [ ] Vitals: big OBS number, bar proportional to OBS, REAL tick at correct position, Δ sign correct
- [ ] Olfactory: ghost bars (lighter) visible behind solid bars when noise active
- [ ] Nociception: shows Δ value and REAL tick when noise active; `(OBS ONLY)` absent
- [ ] Collision: directional labels `C U R D L` readable; fires on correct direction
- [ ] Visual: ghost bars visible with distinct color/alpha from solid
- [ ] Arena: entity icons at correct grid positions; agent icon correct for co-occupancy
- [ ] Minimap: agent dot at correct position; viewport rectangle aligned with arena view

---

### Known Pitfalls

These errors are guaranteed to appear when writing a new harness or renderer function. Solutions below.

#### 1. `ValueError: encoding_config is required for ObservationEncoder`

`ActorCriticRNN` silently accepts but later crashes if `encoding_config` is omitted — there is no fallback. Always pass both:

```python
model = ActorCriticRNN(
    ...,
    observation_breakdown=get_observation_breakdown(params),
    encoding_config=config.to_dict().get('agent', {}),
)
```

#### 2. `KeyError: 'Invalid key: 0'` on `nnx.update`

Orbax restores `Sequential` layer indices as strings (`'0'`, `'1'`), but Flax NNX expects integers. Bare `nnx.update(model, peel(restored))` crashes. Use the key-reconciliation pattern:

```python
from flax.nnx.statelib import to_pure_dict

def peel(st):
    if isinstance(st, dict):
        return st['value'] if 'value' in st else {k: peel(v) for k, v in st.items()}
    return st

def merge_restored(module_state, restored_state):
    if isinstance(module_state, dict):
        out = {}
        for k in module_state:
            rkey = k if k in restored_state else (str(k) if str(k) in restored_state else None)
            out[k] = merge_restored(module_state[k], restored_state[rkey]) if rkey else module_state[k]
        return out
    return restored_state

peeled = peel(restored['model'])
current_struct = to_pure_dict(nnx.state(model, nnx.Param))
nnx.update(model, merge_restored(current_struct, peeled))
```

Both helpers are in `tmp/test_v2_render.py` and the canonical versions live in `evaluation.py` (`peel_nnx_state`, `_merge_restored_into_module_state`).

#### 3. `AttributeError: 'NoneType' object has no attribute 'ndim'` in GRU/LSTM cell

Passing `h_state=None` to `generic_inference` crashes inside the RNN cell. Always initialise:

```python
h_state = model.initial_state(batch_size=None) if hasattr(model, 'initial_state') else None
```

#### 4. Ghost layers (dual view) not visible — `obs_only` always `True`

Using `np.array_equal` or `==` for the `obs_only` guard treats noise-perturbed floats as identical to ground truth when differences are sub-epsilon. Use a tolerance:

```python
# Bug: noise deltas < machine epsilon treated as "no noise"
obs_only = np.array_equal(obs_vec, true_vec)

# Fix: intentional threshold
obs_only = tv is None or np.allclose(obs_vec, true_vec, atol=1e-6)
```

This was confirmed as the root cause of olfactory / visual / collision ghost bars disappearing even when noise was enabled.

#### 5. Arena icons wrong size after changing `local_view_size`

Icon zoom in the arena is in data coordinates (grid cells), not axes fraction. The scale factor auto-adjusts with view size:

```python
scale_factor = (4.0 / view_size) * icon_scale
```

If icons appear too large or too small, pass `icon_scale` (float multiplier) to the renderer entry point — do not hardcode zoom values inside drawing functions.

#### 6. `constrained_layout` warning: axes too small to fit decorators

Happens when a `subplot_mosaic` row is given a very small `height_ratio` (< 0.4) and the text in the pod is larger than the axes. Fix: increase the height ratio for that row, or reduce font size in the drawing function for that pod.

---

### Swapping the Active Renderer in Production

The production eval pipeline imports the renderer in two places in `src/utils/evaluation_core.py`:

```python
# Line ~150 (render setup):
from src.environment.renderer import render_jax_state, save_jax_video

# Line ~310 (single-env path):
from src.environment.renderer import render_jax_state
```

To point the entire eval pipeline at a new renderer module, change both imports. The alias keeps all call sites unchanged:

```python
from src.environment.renderer_NEW import render_jax_state_NEW as render_jax_state, save_jax_video
```

**Gate**: only swap after the frame inspection checklist above passes for all reference frames. Run a full eval episode with `--render` after swapping and confirm the generated MP4 looks correct before committing.

---

## Video Export

`save_jax_video(frames, output_path, fps=5, quiet=False)` (`renderer.py:698`)

- `frames`: list of RGB numpy arrays `(H, W, 3)`.
- Uses `imageio.mimsave` to write MP4 at the given FPS.
- Creates output directory if needed.
- `quiet=True`: suppresses imageio stdout/stderr by redirecting to `/dev/null`.

Typical usage in evaluation scripts:
```python
frames = []
for step in range(max_steps):
    frame = render_jax_state(state, params, step=step, ...)
    frames.append(frame)
save_jax_video(frames, "output/episode.mp4", fps=10)
```

---

## Color Palette

The renderer uses a fixed professional palette (`COLORS` dict, `renderer.py:103`) designed for white-background presentations:

| Key | Hex | Used for |
|-----|-----|---------|
| `satiation` | `#0D9488` (teal) | Satiation bar |
| `nutrition` | `#D97706` (amber) | Nutrition bar |
| `injury` | `#BE123C` (crimson) | Injury bar |
| `food` | `#059669` (green) | Food entities |
| `danger` | `#DC2626` (red) | Danger entities |
| `predator` | `#111827` (near-black) | Predator entities |
| `action` | `#2563EB` (cobalt) | Agent, sensor pods |
| `rock` | `#4B5563` (steel grey) | Rock obstacles |
| `neutral` | `#0891B2` (cyan) | Neutral animals |

---

## `sensory_data` Interface

The renderer accepts `sensory_data` — a list of dicts, one per sensor — to show perceived (noisy) values alongside ground truth. Each dict has structure:

```python
{
    'name': 'Satiation',          # sensor name
    'type': 'intensity',          # 'intensity' | 'spectrum' | 'diamond' | 'visual_grid' | 'text'
    'intensity': 0.85,            # observed value (for intensity type)
    'true_intensity': 0.90,       # ground truth (optional)
    'vector': [...],              # for spectrum/diamond types
    'true_vector': [...],         # ground truth vector (optional)
    'range': 1,                   # for diamond types
    'num_features': 8,            # for diamond/visual_grid types
}
```

If `sensory_data=None`, vitals bars display ground truth only (no perception bar).

---

## Clarifications / FAQ

**Q: Is the renderer JIT-compatible?**
A: **No.** It is pure Python (NumPy + Matplotlib), runs on CPU, and is designed to be called *after* pulling data off the JAX device with `jax.device_get(state)`. Never call `render_jax_state` inside a `@jax.jit` block or a `jax.lax.scan` body.

**Q: Can I render a batched `EnvState` (multiple envs)?**
A: No — `render_jax_state` expects a single env's state. To render multiple envs, index into the batched state per-env (`jax.tree.map(lambda x: x[i], batched_state)`) and call the renderer for each.

**Q: What's the difference between `sensory_data=None` and omitting a sensor from `sensory_data`?**
A: `sensory_data=None` disables all perception overlays globally — vitals show ground truth only. Listing `sensory_data` but omitting a specific sensor grey-outs that pod with an **OFFLINE** label (see right panel).

**Q: Does the renderer read `apply_noise` or does it need separate noisy-and-clean observations?**
A: It doesn't know about noise itself. The caller must produce both `obs = get_observation(state, params)` and `true_obs = get_observation(state, params, apply_noise=False)` and pass them into the `build_sensory_viz` helper. The renderer just plots what's given.

**Q: Why is the "dual view" ghost layer sometimes invisible?**
A: Almost always a tolerance bug: equality checks between float arrays fail silently for sub-epsilon noise. Use `np.allclose(..., atol=1e-6)` per Pitfall 4. If ghost bars remain missing even with tolerance applied, confirm `sensory_data['true_vector']` (or `true_intensity`) is actually populated.

**Q: Can I customise the icon set without changing code?**
A: Yes — pass `icon_config` kwarg to `render_jax_state`. The dict maps icon keys (`agent`, `food`, `danger`, etc.) to file paths. Missing files fall back to matplotlib scatter markers. Cache is global in `_ICON_CACHE` — clear it if you swap files in-process.

**Q: What does the `local_view_size` param do?**
A: Sets the arena side length (in cells) centred on the agent. Smaller = more zoomed in. On grids smaller than `local_view_size`, the view is clamped to grid extents. Default 7 is chosen for readable 10×10 arenas.

**Q: What if the agent stands on top of an entity the renderer doesn't have a composite icon for?**
A: Only four composites exist: `agent_predator`, `agent_danger`, `agent_bush`, `agent_food`. Other co-occupancies (e.g. agent on rock) fall back to drawing just the agent icon on top — the underlying entity may be partially occluded. If you need a new composite, add an icon file and register its key.

**Q: Does the minimap always show the full grid?**
A: Yes. The minimap is drawn at a fixed small size regardless of grid dimensions. On very large grids (>50×50), dot density may make it unreadable — consider subsampling entities for the minimap if this matters.

**Q: How does the renderer handle entities with `count: 0`?**
A: They aren't in the arrays (config_loader skips zero-count expansions). The renderer naturally omits them. No special case needed.

**Q: What FPS should I use for video export?**
A: `fps=5` for slow-motion debugging (default), `fps=10-15` for natural-speed training videos. Higher fps produces smaller perceived step durations.

**Q: Does the renderer need a display / X server?**
A: No. Matplotlib's `Agg` backend is used (no display needed). Renders correctly on headless servers and inside containers.

**Q: What's the memory cost per frame?**
A: A typical 14×10 inch figure at DPI 100 is ~1400×1000×3 ≈ 4 MB as uint8. For a 1000-step episode that's 4 GB. Consider rendering every Nth step for long episodes, or stream frames directly to `imageio.get_writer()` instead of accumulating a list.

**Q: Why use `matplotlib` instead of a faster library (e.g. Pillow, pygame)?**
A: Matplotlib provides the dashboard layout (GridSpec), coordinate axes, and consistent font rendering required for presentation-quality figures. For training-speed visualisation, there is no attempt to optimise — render infrequently.

**Q: How do I add a new pod to the right panel?**
A: (1) Decide the pod type (`intensity`, `spectrum`, `diamond`, `visual_grid`, or `text`); (2) add an entry to the sensor's `sensory_data` dict emitted by `build_sensory_viz`; (3) add a drawing case in the pod loop in `renderer.py` if the type is new. Follow the iterative harness workflow above.

**Q: The action pod shows `—` — why?**
A: Renderer couldn't infer action name. Pass `action=<int>` explicitly to `render_jax_state`, or pass a string mapping via `action_name` kwarg.

**Q: Does the renderer handle multi-predator configs?**
A: Yes — it iterates `state.pred_pos` and renders each. Same for resources and neutrals. Each entity draws its own icon.

**Q: What if I want headless video export without the dashboard (just the arena)?**
A: Not directly supported. You'd need a variant `render_arena_only` that returns just the centre panel. Consider extracting `_draw_arena` from `renderer.py` and calling it standalone.
