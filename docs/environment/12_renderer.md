# 12 — Renderer & Visualization

> **Source**: `src/environment/renderer.py` | **Back to hub**: [README](README.md)

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
