# 12 — Renderer & Visualization

> **Sources**: `src/environment/renderer.py` (V1, current production default) · `src/environment/renderer_v2.py` (V2, newer declarative layout) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## What this document covers

The GridWorld Pain environment has two renderers. Both convert a single `EnvState` + `EnvParams` snapshot into an RGB image that can be assembled into an MP4 video. The image is a multi-panel telemetry dashboard — a research figure that shows the agent's ground-truth internal state (satiation, nutrition, injury) on the left, a zoomed-in map of the arena in the centre, and the agent's noisy sensory readings on the right. When noise is active, most panels show both the clean "true" reading and the agent's noised "observed" reading side by side, so you can directly see the gap introduced by perceptual noise.

**Renderer V1** (`src/environment/renderer.py`, entry point `render_jax_state`) is the current production default. It is the version called by every production eval and demo script.

**Renderer V2** (`src/environment/renderer_v2.py`, entry point `render_jax_state_v2`) is a newer refactor. It produces a very similar dashboard but uses a declarative layout engine (`subfigures` + `subplot_mosaic`) that makes panel sizing more predictable and eliminates the manual y-cursor bookkeeping of V1. V2 is a drop-in replacement (identical public signature) but is not yet the default in production eval — it is available for experiments and will become the default after passing the full frame inspection checklist.

Both renderers are pure Python (NumPy + Matplotlib, Agg backend) and run on CPU. They are called *after* pulling state data off the JAX device with `jax.device_get`, never inside a JIT block.

**Key v2.0 changes reflected in both renderers:**
- The entity formerly called `danger` is now `hiding_predator` throughout (icon keys, color keys, boresight feature labels).
- A new **interoceptive nociception** panel (a time-averaged pain signal derived from injury history) appears in the left panel of V1 when enabled via `params.interoceptive_nociception_enabled`.
- Animals use a unified `animal_pos` array; per-class rendering is done via `select_by_class(params, 'predator')` / `select_by_class(params, 'neutral')` — the old separate `pred_*` / `neutral_*` arrays are gone.

---

## Which renderer is the current default?

`render_jax_state` from `src/environment/renderer.py` (V1) is the current production default. It is imported by:

- `scripts/render_recordings.py` (offline post-hoc video export, called automatically by eval) — `renderer.py:45`
- `scripts/record_env_demo.py` — `renderer.py:15`
- `scripts/benchmark_render.py` — `renderer.py:39`
- `save_snapshot.py` — `renderer.py:14`

`render_jax_state_v2` from `src/environment/renderer_v2.py` is available and used in `test_c4.py` but not yet wired into the eval pipeline.

To swap the entire pipeline to V2, change the imports in `scripts/render_recordings.py` lines 31 and 45 (see "Swapping the Active Renderer" section below).

---

## Dashboard Layout

### V1 (`render_jax_state`) — flat GridSpec

Figure size: `14 × 10 inches` at configurable DPI. Uses a flat `GridSpec(1, 3)` with `width_ratios=[0.8, 2.2, 0.8]`. All drawing within the left and right panels is done with y-cursor arithmetic in axes (transAxes) coordinates, so pod positions are computed at render time. The minimap is an `inset_axes` anchored inside `ax_left`.

```
┌──────────────────┬─────────────────────────────┬──────────────────┐
│  LEFT PANEL      │     CENTRE PANEL             │  RIGHT PANEL     │
│  (ax_left)       │     ARENA — LOCAL VIEW       │  (ax_right)      │
│                  │     (ax_grid)                │                  │
│  INTEROCEPTION   │     grid lines, terrain,     │  EXTEROCEPTION   │
│  - Satiation bar │     entity icons             │  sensor pods:    │
│  - Nutrition bar │                              │  - Olfactory     │
│  - Injury bar    │                              │  - Extero Noc    │
│  - Intero Noc*   │                              │  - Collision     │
│    (* if enabled)│                              │  - Visual        │
│                  │                              │  - LOC           │
│  Run Context pod │                              │  Action pod      │
│  MINIMAP (inset) │                              │                  │
└──────────────────┴─────────────────────────────┴──────────────────┘
```

### V2 (`render_jax_state_v2`) — declarative subfigures

Figure size: `14 × 8 inches` (shorter than V1) at configurable DPI. Uses `fig.subfigures(2, 1, height_ratios=[0.055, 0.945])` to create a slim header banner across the full width, then splits the body into three subfigures with `width_ratios=[1.0, 2.4, 1.0]`. Each column uses `subplot_mosaic` to define named axes — panel overlap is structurally impossible with this layout engine.

```
┌─────────────────────────────────────────────────────────────────────┐
│ HEADER BANNER: "GridWorld · Homeostasis Dashboard  Ep N · WxH · St N"│
├──────────────────┬─────────────────────────────┬────────────────────┤
│  LEFT COLUMN     │     CENTRE COLUMN           │  RIGHT COLUMN      │
│  sf_left         │     sf_center               │  sf_right          │
│                  │                             │                    │
│  [satiation]     │  [arena]  ← 3 equal rows   │  [olfactory]       │
│  [nutrition]     │  [arena]                    │  [nociception]     │
│  [injury]        │  [arena]                    │  [collision]       │
│                  │  [minimap] ← 0.65 ratio     │  [visual]          │
│                  │                             │  [action]          │
└──────────────────┴─────────────────────────────┴────────────────────┘
```

Height ratios for right column: `[olfactory:1, nociception:1, collision:1.4, visual:1.4, action:0.55]`.

---

## Entry Points & Signatures

Both renderers share an identical public signature:

```python
render_jax_state(
    state,           # EnvState (single env, host numpy — call jax.device_get first)
    params,          # EnvParams
    episode=None,    # episode number for header display
    step=None,       # step number for header display
    train_episode=None,  # fallback for episode if episode=None
    dpi=100,
    icon_scale=1.0,  # float multiplier for arena icon zoom
    action=None,     # int action index (0–5), or None for "—"
    sensory_data=None,  # list of sensor dicts; None = ground truth only
    info=None,       # unused in V1/V2 right-panel loop (was for damage pod in legacy)
    icon_config=None,   # dict mapping icon keys to asset filenames
)
# Returns: (H, W, 3) uint8 numpy array
```

V1 entry point: `src/environment/renderer.py:355` (`render_jax_state`)
V2 entry point: `src/environment/renderer_v2.py:216` (`render_jax_state_v2`)

`save_jax_video` signature (defined in `renderer.py:734`, re-exported by `renderer_v2.py:29`):

```python
save_jax_video(frames, output_path, fps=5, quiet=False)
# frames: list of (H, W, 3) uint8 arrays
# quiet=True: suppresses imageio stdout/stderr via fd-level redirect
```

---

## V1 Draw Helpers

These functions are defined in `renderer.py` and are also imported and reused by `renderer_v2.py`.

### `draw_dual_capsule_bar` (`renderer.py:134`)

Draws a horizontal capsule-shaped progress bar on any axes using `transAxes` coordinates. The bar has two layers:
- **Reality layer** (translucent, alpha 0.3): the actual `EnvState` value (ground truth), drawn at full width of the capsule.
- **Perception layer** (opaque, alpha 1.0, thinner): the agent's observed (possibly noisy) value, drawn as a narrower inner bar.

Text labels show `REAL: X.XX` to the right at top and `OBS: X.XX` at bottom. Used for Satiation, Nutrition, Injury, and Interoceptive Nociception in the left panel.

### `draw_pod_frame` (`renderer.py:174`)

Draws the outer border and title banner for a sensor pod in the right panel. Accepts `offline=True` (greyed out, "OFFLINE" watermark) and `obs_only=True` (appends "(OBS ONLY)" to the title, lightens text color).

### `draw_boresight_diamond` (`renderer.py:200`)

Renders a Manhattan-diamond spatial grid for directional sensors (Collision at `r=0`, Visual at `r>1`). Uses the same offset logic as `sensor.py:get_visual_offsets` for perfect spatial alignment. Each grid cell shows:
- A ghosted (alpha 0.15) icon or color patch for the ground-truth feature (skipped when `obs_only=True`).
- A solid (alpha 0.8–0.9) icon or color patch for the observed feature.

Feature slots (8-channel): grass, sand, plain, food, **hiding_predator** (slot 4 — was `danger` in V1 legacy), predator, rock, neutral.

### `draw_categorical_visual` (`renderer.py:291`)

Renders a flat stacked bar chart for Collision and Visual sensors when range ≤ 1. Cells are laid out left-to-right: for `r=1` the 5-cell layout is `C U R D L` (Centre, Up, Right, Down, Left). Each cell shows ghost bars (true) behind solid bars (observed) for each feature channel. Labels (GRS, SND, PLN, FOD, DNG, PRD, NEU, RCK) rotate 90° under each bar group.

> Note: `grid_world.py` contains an older copy of this function that uses `'HPR'` instead of `'DNG'` in the fallback label list — a legacy discrepancy. The authoritative version in `renderer.py:302` uses `'DNG'`.

---

## V2 Card/Pod Functions

These are defined in `renderer_v2.py` and are used exclusively by `render_jax_state_v2`. They each own their own axes object (passed in as `ax`) and use `transAxes` coordinates internally — no manual y-cursor needed.

### `draw_vital_card` (`renderer_v2.py:86`)

For each interoception vital (Satiation, Nutrition, Injury) in the left column. Layout:
- Top-left: label in small caps.
- Top-right: large OBS value (14pt bold, colored with the vital's color).
- Middle: horizontal bar (`_hbar`) with OBS fill + a thin vertical REAL tick mark.
- Bottom: `real X.XX  ·  Δ ±X.XX` (the gap between observed and true, signed).

The tick mark design makes it easy to see how far the agent's perception is from reality at a glance: when there is no noise, OBS bar and REAL tick coincide.

### `draw_offline_card` (`renderer_v2.py:107`)

Placeholder for any vital or sensor that has no data. Shows a greyed-out title and "OFFLINE" watermark.

### `draw_intensity_pod` (`renderer_v2.py:115`)

For the Extero Nociception pod in the right column (external pain signal). Same layout as `draw_vital_card` but uses the action color (cobalt blue) instead of a vital-specific color. When `obs_only=True` (no noise-free reference available), substitutes a note "obs only — no noise-free reference" for the Δ line.

### `draw_spectrum_pod` (`renderer_v2.py:140`)

For the Olfactory pod — a 5-channel scent signal. Draws stacked vertical bars normalized to the maximum observed value. Ghost bars (alpha 0.15) show the true signal; solid bars (alpha 0.9) show the observed signal. When `obs_only=True`, ghost bars are suppressed.

### `draw_categorical_pod` (`renderer_v2.py:168`)

Wrapper for Collision and Visual pods. Calls `draw_categorical_visual` (shared from `renderer.py`) with margins tuned for V2 axes geometry (title occupies top ~16%, feature labels use bottom ~8%). Shows "NO SIGNALS" text when all observed values are zero.

### `draw_action_pod` (`renderer_v2.py:188`)

Small badge at the bottom of the right column. Shows the action name (UP / RIGHT / DOWN / LEFT / REST / EAT) and a Unicode arrow symbol. Shows `—` when `action=None`.

### `_hbar` (`renderer_v2.py:58`)

Internal helper used by `draw_vital_card` and `draw_intensity_pod`. Draws: grey trough rectangle, colored OBS fill using `FancyBboxPatch`, and a thin vertical REAL tick line that extends slightly above and below the bar for visibility.

### `_style_card` / `_card_title` (`renderer_v2.py:40`, `renderer_v2.py:49`)

Internal helpers. `_style_card` sets the axes background color and spine style for a card. `_card_title` writes the card's label text, adding `(OBS ONLY)` or `OFFLINE` suffixes as needed.

---

## Left Panel — Interoception (V1 detailed)

V1 left panel draws directly on `ax_left` using `transAxes` coordinates. A `y_ptr` cursor starts at 0.95 and decrements after each bar.

**Bar spacing:** When `params.interoceptive_nociception_enabled=True`, the renderer detects 4 bars are needed and compresses spacing to `bar_step=0.10`; otherwise `bar_step=0.15` for 3 bars (`renderer.py:553-554`).

**Vitals rendered:**
1. **Satiation** — teal (`#0D9488`), `state.satiation / params.max_satiation`.
2. **Nutrition** — amber (`#D97706`), `state.nutrition / params.max_nutrition`.
3. **Injury** — crimson (`#BE123C`), `state.injury_level / params.max_injury`.
4. **Interoceptive Nociception** (violet `#7C3AED`, *only when `params.interoceptive_nociception_enabled`*) — a time-averaged pain signal computed from `state.nociception_history_buffer` convolved with `params.interoceptive_kernel`, normalized by `max_injury`. This is the agent's internal "memory of pain" — it persists beyond the acute injury that triggered it. When `sensory_data` includes an `'Intero Nociception'` entry, that value is used directly; otherwise the renderer recomputes it locally (`renderer.py:593-603`).

Below the vitals: a **Run Context pod** showing Episode, grid Scale (WxH), and Step.

At the bottom: a **MINIMAP** inset (`inset_axes` at 80% width, 25% height, `loc='lower center'`). The minimap shows the full grid as a terrain-colored image with colored dots for all entity types (food: green, hiding_predator: red, predator: near-black, obstacles: steel grey, neutrals: cyan). A cobalt rectangle marks the current arena view window; a cobalt dot marks the agent's position.

---

## Right Panel — Exteroception Pods (V1 detailed)

V1 iterates through `known_sensors = ['Olfactory', 'Extero Nociception', 'Collision', 'Visual', 'LOC']`. For each, it looks up `sensor_map` (built from `sensory_data`) and draws the pod. The loop breaks when `y_cursor < 0.20` to leave room for the Action Pod at `y=0.03`.

Pod heights:
- Default: `pod_h = 0.11`
- `diamond` / `visual_grid` types: `pod_h = 0.20` (extra room for the bar labels)

**`obs_only` flag** — set `True` when no noise-free reference is available:
- `intensity` type: `'true_intensity'` key absent, or equals `'intensity'` by `==` (note: uses exact equality, see bug note in Known Pitfalls).
- `spectrum` / `diamond` / `visual_grid`: `'true_vector'` absent, or `np.array_equal(...)` (same exact-equality caveat).

V2 fixes the `obs_only` check to use `np.allclose(..., atol=1e-6)` for vector types and an absolute difference `< 1e-6` for scalars.

| Sensor name | Pod type | Visualization |
|-------------|----------|---------------|
| `Olfactory` | `spectrum` | 5-channel vertical bar chart; ghost bars for true signal |
| `Extero Nociception` | `intensity` | Dual capsule bar; REAL tick / OBS fill; `--` when obs_only |
| `Collision` | `diamond` or `visual_grid` | Categorical grid (`r` cells × 1 feature channel) |
| `Visual` | `diamond` or `visual_grid` | Categorical grid (`r²` cells × 8 feature channels) |
| `LOC` | `text` | Coordinate string |

**`r ≤ 1` vs `r > 1`:** For Collision and Visual, when range `r ≤ 1` the renderer calls `draw_categorical_visual` (flat bar chart); when `r > 1` it calls `draw_boresight_diamond` (spatial diamond layout). The boresight diamond is the spatial "radar" view — each grid cell in the sensor's range is drawn at its correct relative position around the agent.

**Action pod** (fixed at bottom `y=0.03`): shows action name and Unicode arrow. Action mapping: 0=UP↑, 1=RIGHT→, 2=DOWN↓, 3=LEFT←, 4=REST⊝, 5=EAT✙.

---

## Centre Panel — Arena

Both V1 and V2 draw the same arena content. The arena shows a **local view window** of `view_size × view_size` cells centred on the agent. Window bounds are clamped to grid edges, and for grids smaller than `view_size`, the view covers the whole grid.

**Rendering order within the arena:**
1. Background grid lines (light grey `#F3F4F6`).
2. Terrain tile fill: plain (white/bg), grass (light green `#ECFDF5`), sand (light amber `#FFFBEB`).
3. Resource entities (`state.res_pos / res_active`): food (type 0) → `food` icon; hiding-predator trap (type 1) → `hiding_predator` icon.
4. Predator animals (sliced from `state.animal_pos` via `select_by_class(params, 'predator')`): `predator` icon.
5. Obstacles (`state.obs_pos`): icon from `params.obstacle_names[obs_type[i]]` (e.g. `rock`, `bush`).
6. Neutral animals (sliced via `select_by_class(params, 'neutral')`): `neutral` icon.
7. Agent — composite icon selected by what shares the agent's cell:
   - Agent + predator → `agent_predator`
   - Agent + hiding_predator → `agent_hiding_predator`
   - Agent + bush → `agent_bush`
   - Agent + food → `agent_food`
   - Agent alone (or with neutral) → `agent`

Icon zoom: `zoom = 0.038 * scale_factor` where `scale_factor = (4.0 / view_size) * icon_scale`. This keeps icons the same visual size regardless of the local view window size.

---

## `danger` → `hiding_predator` Rename

In v2.0, the entity class formerly called `danger` was renamed to `hiding_predator` throughout the codebase. In the renderer this affects:

- **Icon keys** (`renderer.py:46-49`): `'hiding_predator': 'hiding_predator'` and `'agent_hiding_predator': 'agent_hiding_predator'` (the asset file names also changed from `danger.png` / `agent_danger.png`).
- **Color keys** (`renderer.py:123`): `'hiding_predator': '#DC2626'` (was `'danger': '#DC2626'`).
- **Damage color keys**: `'dmg_hiding_predator': '#EF4444'`.
- **Boresight feature labels** (`renderer.py:219`): slot 4 is now `'hiding_predator'`.
- **Fallback scatter marker** (`renderer.py:443`): `hiding_predator` → `('X', red)`.

`grid_world.py` (the legacy copy of V1) still maps `'hiding_predator': 'danger'` in its icon config (`grid_world.py:46`), meaning it will try to load `danger.png` instead of `hiding_predator.png`. If only the new asset name exists on disk, the icon falls back to the scatter marker.

---

## Interoceptive Nociception Panel

The interoceptive nociception panel is a signal that represents the agent's accumulated "pain history" — not just the current injury level, but a convolution of recent injury events over time using a kernel stored in `params.interoceptive_kernel`. This models the biological phenomenon of central sensitization: pain can persist or be amplified after the initial injury.

**V1 render logic** (`renderer.py:581-606`):
- Only rendered when `params.interoceptive_nociception_enabled=True`.
- Preferred data source: `sensory_data['Intero Nociception']` dict with `'intensity'` (observed, possibly noisy) and `'true_intensity'` (noise-free convolved signal).
- Fallback when no sensory_data: recomputes the convolved signal directly in the renderer by applying `params.interoceptive_kernel` to `state.nociception_history_buffer`, normalized by `max_injury`.
- When `params.interoceptive_convolution_enabled=False`, falls back to `state.injury_level / max_inj` directly.
- Drawn with violet color `#7C3AED` (COLORS key `'intero_noc'`).

**V2**: the left column of V2 only has three fixed vital card axes (`satiation`, `nutrition`, `injury`). The interoceptive nociception panel is **not yet present in V2** — it would require adding a fourth entry to the `subplot_mosaic` list (`renderer_v2.py:260-262`) and a `draw_vital_card` call for it.

---

## Dual-View (True vs Observed) Rendering

When `sensory_data` is provided with both observed and true values, the renderer shows both simultaneously:

- **Vitals / intensity pods**: translucent wide bar = ground truth; solid narrow bar = observation. The `_hbar` function in V2 uses a vertical tick mark for ground truth rather than a separate bar.
- **Spectrum (olfactory)**: light ghost bars (alpha 0.15–0.2) behind solid bars (alpha 0.9).
- **Categorical (collision / visual)**: ghost patches or icons (alpha 0.1–0.15) behind solid patches or icons.

When `sensory_data=None`, all panels display ground truth only (no perception overlay).

When a sensor is listed in `known_sensors` but absent from `sensory_data`, its pod is drawn as **OFFLINE** (greyed out, "OFFLINE" watermark).

**`obs_only` flag caveat (Pitfall 4):** V1 uses `np.array_equal` to detect "no noise" for vector sensors, which fails for sub-epsilon float perturbations — ghost layers disappear even when noise is active. V2 fixes this with `np.allclose(..., atol=1e-6)`. For scalar intensity sensors, V1 uses `==` (exact equality); V2 uses `abs(ti - obs_v) < 1e-6`. When using V1, pass `true_vector` / `true_intensity` from `get_observation(state, params, apply_noise=False)` and verify ghost layers appear on injury frames.

---

## Icon Loading

`_load_icons(icon_config)` (`renderer.py:31`) loads PNG/JPG/SVG/GIF/WebP icons from `assets/` at the project root. Icons are loaded once and cached globally in `_ICON_CACHE`. SVG loading requires `cairosvg` (optional); missing `cairosvg` silently skips SVG files. All formats are converted to RGBA numpy arrays via PIL. Missing files map to `None`; fallback rendering uses matplotlib scatter markers.

**Current default icon config** (defined in `renderer.py:43-55`):

| Key | Asset filename | Fallback marker |
|-----|---------------|-----------------|
| `agent` | `agent` | circle, cobalt |
| `food` | `food` | diamond, green |
| `hiding_predator` | `hiding_predator` | X, red |
| `predator` | `predator` | triangle, near-black |
| `agent_food` | `agent_food` | — |
| `agent_hiding_predator` | `agent_hiding_predator` | — |
| `agent_predator` | `agent_predator` | — |
| `rock` | `rock` | square, steel grey |
| `bush` | `bush` | — |
| `agent_bush` | `bush_agent` | — |
| `neutral` | `neutral` | circle, cyan |

Custom icon mappings can be passed via `icon_config` kwarg to `render_jax_state` / `render_jax_state_v2`. Clear `_ICON_CACHE` (set to `None`) if swapping asset files in-process.

> **Legacy discrepancy**: `grid_world.py:46` still maps `'hiding_predator': 'danger'` and `'agent_hiding_predator': 'agent_danger'`, pointing at old asset names. `renderer.py` uses the correct `'hiding_predator'` / `'agent_hiding_predator'` names.

---

## Video Export

### `save_jax_video` in `renderer.py` (`renderer.py:734`)

Uses `imageio.get_writer` (streaming, frame-by-frame):

```python
save_jax_video(frames, output_path, fps=5, quiet=False)
```

- `frames`: list of `(H, W, 3)` uint8 numpy arrays.
- Creates the output directory if needed (`os.makedirs`).
- `quiet=True`: suppresses imageio stdout/stderr by redirecting stdout/stderr fds to `/dev/null` and restoring them in a `finally` block.
- `renderer_v2.py` re-exports `save_jax_video` directly from `renderer.py` (`renderer_v2.py:29`) — there is a single implementation used by both renderers.

### `save_jax_video` in `grid_world.py` (`grid_world.py:701`)

Uses `imageio.mimsave` (loads all frames at once). This is the older implementation in the legacy copy. **Production code uses the `renderer.py` version** (streaming `get_writer`), not this one.

### Typical usage pattern

```python
frames = []
for step in range(max_steps):
    frame = render_jax_state(state_host, params, episode=ep, step=step,
                              action=action_idx, sensory_data=viz)
    frames.append(frame)
save_jax_video(frames, "output/episode.mp4", fps=5)
```

The offline recording pipeline (`scripts/render_recordings.py`) uses multiprocessing workers, each calling `render_jax_state` + `save_jax_video` to produce one MP4 per episode, then optionally concatenating them.

---

## `sensory_data` Interface

The renderer accepts `sensory_data` — a list of dicts, one per sensor — to show perceived (noisy) values alongside ground truth. Each dict has the structure:

```python
{
    'name': 'Extero Nociception',   # must match an entry in known_sensors
    'type': 'intensity',            # 'intensity' | 'spectrum' | 'diamond' | 'visual_grid' | 'text'
    'intensity': 0.85,              # observed value (for intensity type)
    'true_intensity': 0.90,         # ground truth (optional; omit if no noise-free ref)
    'vector': [...],                # for spectrum/diamond/visual_grid types
    'true_vector': [...],           # ground truth vector (optional)
    'range': 1,                     # for diamond/visual_grid types
    'num_features': 8,              # for diamond/visual_grid types
    'labels': ['GRS', ...],         # optional custom feature labels
    'value_text': '(5, 7)',         # for text type
}
```

`build_sensory_viz` in `src/environment/sensor.py:371` is the canonical way to construct this list from an observation vector.

If `sensory_data=None`, all panels display ground truth only. If `sensory_data` is provided but a specific sensor is missing, that pod is shown as OFFLINE.

---

## Color Palette

The renderer uses a shared `COLORS` dict (`renderer.py:103`) designed for white-background presentations:

| Key | Hex | Used for |
|-----|-----|---------|
| `satiation` | `#0D9488` (teal) | Satiation bar |
| `nutrition` | `#D97706` (amber) | Nutrition bar |
| `injury` | `#BE123C` (crimson) | Injury bar |
| `intero_noc` | `#7C3AED` (violet) | Interoceptive nociception bar |
| `action` | `#2563EB` (cobalt) | Agent, sensor pods, minimap viewport |
| `food` | `#059669` (green) | Food entities |
| `hiding_predator` | `#DC2626` (red) | Hiding-predator entities (was `danger`) |
| `predator` | `#111827` (near-black) | Predator entities |
| `rock` | `#4B5563` (steel grey) | Rock obstacles |
| `neutral` | `#0891B2` (cyan) | Neutral animals |
| `dmg_hiding_predator` | `#EF4444` | Damage-pod hiding_predator segment |
| `dmg_predator` | `#111827` | Damage-pod predator segment |
| `dmg_obstacle` | `#6B7280` | Damage-pod obstacle segment |

V2 adds two tokens via `COLORS.setdefault`: `card_bg: '#F9FAFB'` and `arena_border: '#D1D5DB'`.

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
edit src/environment/renderer.py (or renderer_v2.py)
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

V1 (`renderer.py`) uses `np.array_equal` or `==` for the `obs_only` guard, which treats noise-perturbed floats as identical to ground truth when differences are sub-epsilon. V2 fixes this. For V1:

```python
# Bug in V1 renderer.py:639 — noise deltas < machine epsilon treated as "no noise"
obs_only = tv is None or np.array_equal(np.asarray(tv), np.asarray(s_data['vector']))

# V2 fix (renderer_v2.py:506):
obs_only = tv is None or np.allclose(obs_vec, true_vec, atol=1e-6)
```

This was confirmed as the root cause of olfactory / visual / collision ghost bars disappearing even when noise was enabled. This bug is still present in V1 production code.

#### 5. Arena icons wrong size after changing `local_view_size`

Icon zoom in the arena is in data coordinates (grid cells), not axes fraction. The scale factor auto-adjusts with view size:

```python
scale_factor = (4.0 / view_size) * icon_scale
```

If icons appear too large or too small, pass `icon_scale` (float multiplier) to the renderer entry point — do not hardcode zoom values inside drawing functions.

#### 6. `constrained_layout` warning: axes too small to fit decorators

Happens in V2 when a `subplot_mosaic` row is given a very small `height_ratio` (< 0.4) and the text in the pod is larger than the axes. Fix: increase the height ratio for that row, or reduce font size in the drawing function for that pod.

---

### Swapping the Active Renderer in Production

The production offline render pipeline imports the renderer in two places in `scripts/render_recordings.py`:

```python
# Line 31 (worker init):
from src.environment.renderer import _load_icons

# Line 45 (render worker body):
from src.environment.renderer import render_jax_state, save_jax_video
```

To point the entire pipeline at V2:

```python
from src.environment.renderer_v2 import render_jax_state_v2 as render_jax_state
from src.environment.renderer import _load_icons, save_jax_video  # save_jax_video is re-exported by v2 too
```

**Gate**: only swap after the frame inspection checklist above passes for all reference frames. Run a full eval episode with `--render` after swapping and confirm the generated MP4 looks correct before committing.

---

## Clarifications / FAQ

**Q: Is the renderer JIT-compatible?**
A: **No.** It is pure Python (NumPy + Matplotlib), runs on CPU, and is designed to be called *after* pulling data off the JAX device with `jax.device_get(state)`. Never call `render_jax_state` inside a `@jax.jit` block or a `jax.lax.scan` body.

**Q: Can I render a batched `EnvState` (multiple envs)?**
A: No — `render_jax_state` expects a single env's state. To render multiple envs, index into the batched state per-env (`jax.tree.map(lambda x: x[i], batched_state)`) and call the renderer for each.

**Q: What's the difference between `sensory_data=None` and omitting a sensor from `sensory_data`?**
A: `sensory_data=None` disables all perception overlays globally — vitals show ground truth only. Providing `sensory_data` but omitting a specific sensor grey-outs that pod with an **OFFLINE** label.

**Q: Does the renderer read `apply_noise` or does it need separate noisy-and-clean observations?**
A: It doesn't know about noise itself. The caller must produce both `obs = get_observation(state, params)` and `true_obs = get_observation(state, params, apply_noise=False)` and pass them into the `build_sensory_viz` helper. The renderer just plots what's given.

**Q: Why is the "dual view" ghost layer sometimes invisible?**
A: Almost always a tolerance bug — see Pitfall 4. In V1, equality checks between float arrays fail silently for sub-epsilon noise. V2 uses `np.allclose(..., atol=1e-6)`. If ghost bars remain missing even with tolerance applied, confirm `sensory_data['true_vector']` (or `true_intensity`) is actually populated.

**Q: Does V2 include the interoceptive nociception panel?**
A: Not yet. V2's left column has three fixed vital card axes. Adding interoceptive nociception to V2 requires adding a fourth entry (e.g. `['intero_noc']`) to the `subplot_mosaic` in `render_jax_state_v2` (`renderer_v2.py:260-262`) and adding a `draw_vital_card` call for it.

**Q: Can I customise the icon set without changing code?**
A: Yes — pass `icon_config` kwarg to `render_jax_state` or `render_jax_state_v2`. The dict maps icon keys to base filenames (without extension) in `assets/`. Missing files fall back to matplotlib scatter markers. Cache is global in `_ICON_CACHE` — set it to `None` to force a reload.

**Q: What does the `local_view_size` param do?**
A: Sets the arena side length (in cells) centred on the agent. Smaller = more zoomed in. On grids smaller than `local_view_size`, the view is clamped to grid extents. Default 7 is chosen for readable 10×10 arenas.

**Q: What if the agent stands on top of an entity the renderer doesn't have a composite icon for?**
A: Five composites exist: `agent_predator`, `agent_hiding_predator`, `agent_bush`, `agent_food`, and the fallback plain `agent` (used for neutral co-occupancy). Other co-occupancies fall back to the plain agent icon drawn on top. If you need a new composite, add an icon file and register its key.

**Q: Does the minimap always show the full grid?**
A: Yes. The minimap is drawn at a fixed small size regardless of grid dimensions. On very large grids (>50×50), dot density may make it unreadable — consider subsampling entities for the minimap if this matters.

**Q: What FPS should I use for video export?**
A: `fps=5` for slow-motion debugging (default), `fps=10-15` for natural-speed training videos.

**Q: Does the renderer need a display / X server?**
A: No. Matplotlib's `Agg` backend is used (`renderer.py:10`). Renders correctly on headless servers and inside containers.

**Q: What's the memory cost per frame?**
A: V1 at 14×10 inches, DPI 100: ~1400×1000×3 ≈ 4 MB as uint8. V2 at 14×8 inches: ~1400×800×3 ≈ 3.4 MB. For a 1000-step episode that's ~3–4 GB. Consider rendering every Nth step for long episodes, or stream frames directly to `imageio.get_writer()` instead of accumulating a list.

**Q: Does the renderer handle multi-predator configs?**
A: Yes — it iterates `state.animal_pos[_pred_mask]` and renders each. Same for resources and neutrals. The `select_by_class` helper provides the per-class boolean mask.

**Q: How do I add a new pod to the right panel in V1?**
A: (1) Decide the pod type (`intensity`, `spectrum`, `diamond`, `visual_grid`, or `text`); (2) add an entry to `known_sensors` in `render_jax_state`; (3) add a drawing case in the pod loop if the type is new; (4) add an entry to the `sensory_data` dict emitted by `build_sensory_viz`. Follow the iterative harness workflow.

**Q: How do I add a new card to V2?**
A: Add a new key to the `subplot_mosaic` list in the appropriate column (`sf_left` or `sf_right`), add its `height_ratios` entry, and add a `draw_*_card/pod` call after the existing ones.

**Q: The action pod shows `—` — why?**
A: `action=None` was passed (or not passed). Pass `action=<int>` explicitly to the renderer entry point.

**Q: What if I want headless video export without the dashboard (just the arena)?**
A: Not directly supported by either renderer. You'd need to extract the arena drawing block into a standalone function — consider a `render_arena_only` variant.

**Q: How does the renderer handle entities with `count: 0`?**
A: They aren't in the arrays (config_loader skips zero-count expansions). The renderer naturally omits them. No special case needed.
