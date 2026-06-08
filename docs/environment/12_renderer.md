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

---

## Implementation deep-dive — host-side code, not JAX

> **Key mindset shift.** Everything in this section runs eagerly on the CPU, never under `jax.jit` or `jax.vmap`. The renderer is Python + NumPy + Matplotlib. Its "advanced API" story is about the **host/device boundary** (pulling JAX arrays to CPU), **`select_by_class`** (recovering per-class animals from the unified array), and **Matplotlib/imageio idioms**. Primer anchors `jax-pytrees` and `scatter-index` are the only ones that genuinely apply.

### Why this matters for a JAX learner

In the rest of the codebase, `EnvState` lives on the GPU as a JAX pytree of `jnp.ndarray` values — see [jax-pytrees](00_jax_primer.md#jax-pytrees). The renderer cannot touch those arrays directly. The caller must convert the state to NumPy first (`jax.device_get(state)` or equivalently wrapping each field with `np.array(...)`). Once converted, every field is a plain NumPy array and normal Python/NumPy/Matplotlib code applies.

`select_by_class` is the host-side mirror of the [scatter-index](00_jax_primer.md#scatter-index) pattern: instead of writing results back into a fixed-size array, it reads a per-class *boolean mask* out of the config tuple `params.animal_classes`. The result is a NumPy boolean array, not a JAX array — so you can use it for standard Python indexing (`array[mask]`), which would be illegal inside a JIT-traced kernel.

---

### `select_by_class` — per-class mask from the unified animal array

```python
def select_by_class(params, class_name: str) -> np.ndarray:
    """Return a boolean NumPy mask of length N selecting animals of the given class.

    Args:
        params: EnvParams holds `animal_classes` tuple (pytree_node=False, len N).
        class_name: one of 'predator', 'neutral', or any future class string.

    Returns:
        np.ndarray[bool, shape (N,)] -- True at index i iff animal_classes[i] == class_name.

    Usage:
        pred_mask = select_by_class(params, 'predator')
        pred_pos = np.array(state.animal_pos)[pred_mask]   # shape [N_pred, 2]

    Notes:
        - This is a host-side (NumPy) helper, not a JAX traced function. Call it
          from analysis scripts, renderers, and eval code, not inside jit'd kernels.
        - The mask is derived from `params.animal_classes`, a static
          `pytree_node=False` tuple, so it is config-constant and allocation-free
          when called repeatedly with the same params.
    """
    return np.array([c == class_name for c in params.animal_classes], dtype=bool)
```

*Source: `src/environment/state.py:7–28`*

> **API notes.** `params.animal_classes` is a `pytree_node=False` tuple — a [static field](00_jax_primer.md#static-dynamic) on `EnvParams`. That is why it is safe to iterate over in Python: it is never a traced value. The result is a plain NumPy bool array, usable for standard fancy-indexing (`state.animal_pos[mask]`) after the state has been pulled to host. Compare to the [scatter-index](00_jax_primer.md#scatter-index) pattern in the JAX kernel, which uses precomputed index tuples rather than boolean masks because JAX cannot trace dynamic shapes.

---

### `_load_icons` — icon cache

Short description: loads PNG/JPG/SVG/WebP assets from the `assets/` directory the first time it is called; subsequent calls return the cached dict immediately. Missing files produce `None` entries; the arena draw function falls back to matplotlib scatter markers.

```python
def _load_icons(icon_config=None):
    """Loads icons from assets directory based on config."""
    global _ICON_CACHE
    if _ICON_CACHE is not None:
        return _ICON_CACHE

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assets_path = os.path.join(base_dir, 'assets')

    if icon_config is None:
        icon_config = {
            'agent': 'agent',
            'food': 'food',
            'hiding_predator': 'hiding_predator',
            'predator': 'predator',
            'agent_food': 'agent_food',
            'agent_hiding_predator': 'agent_hiding_predator',
            'agent_predator': 'agent_predator',
            'rock': 'rock',
            'bush': 'bush',
            'agent_bush': 'bush_agent',
            'neutral': 'neutral'
        }

    supported_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.gif', '.webp']

    icons = {}
    for key, base_filename in icon_config.items():
        found = False
        for ext in supported_extensions:
            filename = f"{base_filename}{ext}" if not base_filename.endswith(ext) else base_filename
            path = os.path.join(assets_path, filename)
            if os.path.exists(path):
                try:
                    if ext.lower() == '.svg':
                        if CAIROSVG_AVAILABLE:
                            png_data = cairosvg.svg2png(url=path)
                            icons[key] = np.array(Image.open(BytesIO(png_data)))
                            found = True
                        # else: silently skipped -- no cairosvg
                    else:
                        img = Image.open(path)
                        if img.mode != 'RGBA':
                            img = img.convert('RGBA')
                        icons[key] = np.array(img)
                        found = True
                    if found:
                        break
                except Exception:
                    pass
        if not found:
            icons[key] = None   # draw_icon falls back to scatter marker

    _ICON_CACHE = icons
    return icons
```

*Source: `src/environment/renderer.py:31–96`*

> **API notes.** This is pure host-side code — PIL/NumPy, no JAX. The cache is a module-level global; set `_ICON_CACHE = None` to force a reload when swapping asset files in-process. SVG support requires the optional `cairosvg` package; missing it silently skips SVGs (no warning printed by default). The `icon_config` kwarg lets you pass a custom `{key: filename}` mapping to `render_jax_state` / `render_jax_state_v2` without editing source.

---

### Draw helpers (V1 — shared with V2)

#### `draw_dual_capsule_bar` — two-layer vitals bar

Short description: draws a horizontal capsule progress bar on any axes using `transAxes` (0–1) coordinates. A wide translucent layer shows the ground-truth value; a narrower opaque layer shows the observed (possibly noisy) value. Used for every vital in V1's left panel.

```python
def draw_dual_capsule_bar(ax, x, y, w, h, state_pct, obs_pct, color,
                          label=None, state_val=None, obs_val=None,
                          transform=None):
    """Draws a professional capsule-style progress bar showing reality vs perception."""
    # Background (Trough)
    bg_rect = matplotlib.patches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h/2}",
        facecolor='#F3F4F6', edgecolor='none', transform=transform, zorder=0
    )
    ax.add_patch(bg_rect)

    # State (Reality) -- thicker, alpha 0.3
    if state_pct > 0:
        val_w = max(h, w * min(1.0, state_pct))
        st_rect = matplotlib.patches.FancyBboxPatch(
            (x, y), val_w, h, boxstyle=f"round,pad=0,rounding_size={h/2}",
            facecolor=color, edgecolor='none', alpha=0.3, transform=transform, zorder=1
        )
        ax.add_patch(st_rect)

    # Observation (Perception) -- thinner inner bar
    if obs_pct > 0:
        val_w = max(h*0.6, w * min(1.0, obs_pct))
        obs_rect = matplotlib.patches.FancyBboxPatch(
            (x, y + h*0.2), val_w, h*0.6,
            boxstyle=f"round,pad=0,rounding_size={h*0.3}",
            facecolor=color, edgecolor='none', transform=transform, zorder=2
        )
        ax.add_patch(obs_rect)

    # ... (omitted: 10 lines of text label drawing -- label, REAL: value, OBS: value) ...
```

*Source: `src/environment/renderer.py:134–172`*

> **API notes.** Runs entirely on host; all coordinates are `transAxes` fractions (0 = left/bottom, 1 = right/top of the axes). No JAX involved. The `transform=ax.transAxes` argument is passed through to every `add_patch` / `ax.text` call so the same function works on any axes object without knowing pixel dimensions.

---

#### `draw_pod_frame` — sensor pod border and title

Short description: draws the rectangular border, accent top-line, and title text for a sensor pod. The `offline=True` flag greys everything out and adds an "OFFLINE" watermark; `obs_only=True` appends "(OBS ONLY)" to the title.

```python
def draw_pod_frame(ax, x, y, w, h, title, offline=False, obs_only=False, transform=None):
    """Draws a modular Telemetry Pod frame."""
    rect = plt.Rectangle((x, y), w, h, facecolor=COLORS['bg'],
                         edgecolor=COLORS['border'],
                         linewidth=0.5, transform=transform, zorder=0)
    ax.add_patch(rect)

    border_color = COLORS['border'] if not offline else COLORS['text_offline']
    ax.plot([x, x + w], [y + h, y + h], color=border_color,
            linewidth=1.5, transform=transform, zorder=1)

    t_color = COLORS['text_label'] if not offline else COLORS['text_offline']
    if obs_only:
        title = title + " (OBS ONLY)"
        t_color = COLORS['text_offline']

    ax.text(x + 0.02, y + h + 0.015, title.upper(), color=t_color,
            fontsize=8, fontweight='bold', transform=transform)

    if offline:
        ax.text(x + w/2, y + h/2, "OFFLINE", color=COLORS['text_offline'],
                ha='center', va='center', fontsize=9, fontweight='bold',
                alpha=0.5, transform=transform)
```

*Source: `src/environment/renderer.py:174–196`*

> **API notes.** Host-side only. The title is drawn *above* the pod rectangle (at `y + h + 0.015`) — the pod frame is the content area; the title sits in the gap above it. This is why V1's y-cursor must account for title height when stacking pods.

---

#### `draw_boresight_diamond` — spatial sensor grid (range > 1)

Short description: draws a Manhattan-diamond spatial grid for directional sensors when range > 1. Each grid cell is drawn at its correct relative position around the agent's centre. Ghost icons/patches (alpha 0.10–0.15) show the ground-truth feature; solid patches/icons show what the agent observed. Uses the same offset table as `sensor.py:get_visual_offsets` for spatial alignment.

```python
def draw_boresight_diamond(ax, x, y, size, vec, r, num_features,
                           true_vec=None, icons=None, obs_only=False,
                           transform=None):
    """
    Draws a schematic Manhattan diamond grid for directional sensors.
    Uses the same offset logic as sensor.py for perfect spatial alignment.
    """
    offsets = np.array(get_visual_offsets(r))  # host-side: np.array(), not jnp

    if len(vec) != len(offsets) * num_features:
        return

    obs_grid = np.array(vec).reshape(len(offsets), num_features)
    true_grid = (np.array(true_vec).reshape(len(offsets), num_features)
                 if true_vec is not None else None)

    feature_keys = [
        'grass', 'sand', 'plain', 'food',
        'hiding_predator',   # slot 4 -- was 'danger' in legacy
        'predator', 'rock', 'neutral',
    ]
    feature_colors = [
        '#A1DFA1', '#F2D7D5', '#FFFFFF', COLORS['food'],
        COLORS['hiding_predator'], COLORS['predator'],
        COLORS['rock'], COLORS['neutral']
    ]

    for i, (dr, dc) in enumerate(offsets):
        cell_x, cell_y = x + dc * size, y - dr * size

        # Cell background border
        # ... (omitted: 3 lines of plt.Rectangle grid cell border) ...

        obs_vals = obs_grid[i]
        true_vals = true_grid[i] if true_grid is not None else obs_vals

        if num_features == 1:
            # Collision: single-channel indicator
            if not obs_only and true_vals[0] > 0.5:   # ghosted ground truth
                ax.add_patch(plt.Rectangle(
                    (cell_x - size*0.48, cell_y - size*0.48), size*0.96, size*0.96,
                    facecolor=COLORS['hiding_predator'], alpha=0.15,
                    transform=transform, zorder=2))
            if obs_vals[0] > 0.5:                       # solid observed
                ax.add_patch(plt.Rectangle(
                    (cell_x - size*0.35, cell_y - size*0.35), size*0.7, size*0.7,
                    facecolor=COLORS['hiding_predator'], alpha=0.8,
                    transform=transform, zorder=3))
        else:
            # Visual (8-channel): ghosted reality, then solid observation
            if not obs_only:
                true_active = np.where(true_vals > 0.1)[0]
                true_entities = [idx for idx in true_active if idx >= 3]
                for feat_idx in true_active:
                    icon_key = feature_keys[feat_idx % len(feature_keys)]
                    icon_img = icons.get(icon_key) if icons else None
                    if icon_img is not None and (feat_idx >= 3 or not true_entities):
                        imagebox = OffsetImage(icon_img, zoom=size * 0.35)
                        ab = AnnotationBbox(imagebox, (cell_x, cell_y),
                                            frameon=False, pad=0,
                                            xycoords=transform if transform else 'data')
                        ab.set_alpha(0.15)
                        ax.add_artist(ab)
                    else:
                        color = feature_colors[feat_idx % len(feature_colors)]
                        ax.add_patch(plt.Rectangle(
                            (cell_x - size*0.48, cell_y - size*0.48),
                            size*0.96, size*0.96,
                            facecolor=color, alpha=0.1,
                            transform=transform, zorder=2))

            obs_active = np.where(obs_vals > 0.1)[0]
            obs_entities = [idx for idx in obs_active if idx >= 3]
            for feat_idx in obs_active:
                icon_key = feature_keys[feat_idx % len(feature_keys)]
                icon_img = icons.get(icon_key) if icons else None
                if icon_img is not None and (feat_idx >= 3 and len(obs_entities) == 1):
                    imagebox = OffsetImage(icon_img, zoom=size * 0.35)
                    ab = AnnotationBbox(imagebox, (cell_x, cell_y),
                                        frameon=False, pad=0,
                                        xycoords=transform if transform else 'data')
                    ax.add_artist(ab)
                elif feat_idx >= 3:
                    color = feature_colors[feat_idx % len(feature_colors)]
                    ax.add_patch(plt.Rectangle(
                        (cell_x - size*0.35, cell_y - size*0.35),
                        size*0.7, size*0.7,
                        facecolor=color, alpha=0.9,
                        transform=transform, zorder=4))
```

*Source: `src/environment/renderer.py:200–289`*

> **API notes.** Host-side only; `np.array(get_visual_offsets(r))` converts the JAX-origin offset list to NumPy immediately so the rest of the loop is plain Python. The `xycoords=transform if transform else 'data'` idiom passes the axes transform through to `AnnotationBbox` so icons land correctly regardless of whether the caller is using `transAxes` or `transData` coordinates.

---

#### `draw_categorical_visual` — flat bar chart for sensors (range ≤ 1)

Short description: for Collision and Visual sensors with range ≤ 1, draws a flat stacked bar chart — one group of bars per spatial cell (`C U R D L` layout for 5-cell), one bar per feature channel within each group. Ghost bars (alpha 0.15) behind solid bars (alpha 0.9) show truth vs. observed.

```python
def draw_categorical_visual(ax, x, y, w, h, obs_vec, r, num_features,
                            true_vec=None, labels=None, obs_only=False,
                            transform=None):
    """Draws a categorical bar chart for visual observations (V7).
    y: bottom of the pod frame
    h: total height of the pod frame
    """
    obs_grid = np.array(obs_vec).reshape(-1, num_features)
    true_grid = (np.array(true_vec).reshape(-1, num_features)
                 if true_vec is not None else obs_grid)
    num_cells = obs_grid.shape[0]

    if labels is None:
        feature_labels = ['GRS', 'SND', 'PLN', 'FOD', 'DNG', 'PRD', 'NEU', 'RCK']
    else:
        feature_labels = labels

    # ... (omitted: 8 lines of feature_colors list + cell_labels setup) ...

    # Internal margins (fraction of h)
    margin_bottom = 0.04
    margin_top    = 0.02
    bar_y = y + margin_bottom
    bar_h = h - (margin_bottom + margin_top)

    cell_gap_ratio = 1.15
    unit_w = w / (num_cells * num_features + max(num_cells - 1, 0) * cell_gap_ratio)

    for c_idx in range(num_cells):
        start_x = x + c_idx * (num_features + cell_gap_ratio) * unit_w

        if num_cells > 1:
            ax.text(start_x + (num_features * unit_w)/2, bar_y + bar_h + 0.002,
                    cell_labels[c_idx],
                    color=COLORS['text_label'], fontsize=5.5, fontweight='bold',
                    ha='center', transform=transform)

        for f_idx in range(num_features):
            bx = start_x + f_idx * unit_w
            bw = unit_w * 0.8
            color = feature_colors[f_idx % len(feature_colors)]

            # Ground truth (ghosted)
            if not obs_only:
                true_val = float(true_grid[c_idx, f_idx])
                ax.add_patch(plt.Rectangle(
                    (bx, bar_y), bw, bar_h * true_val,
                    facecolor=color, alpha=0.15,
                    transform=transform, zorder=1))

            # Perception (solid)
            obs_val = float(obs_grid[c_idx, f_idx])
            ax.add_patch(plt.Rectangle(
                (bx + bw*0.1, bar_y), bw*0.8, bar_h * obs_val,
                facecolor=color, alpha=0.9,
                transform=transform, zorder=2))

            # Feature labels (rotated 90 deg, inside frame bottom)
            if (num_cells == 1 or c_idx == 0) and f_idx < len(feature_labels):
                ax.text(bx + bw*1.1/2, y + 0.002, feature_labels[f_idx],
                        color=COLORS['text_label'],
                        fontsize=4.0, ha='center', va='bottom',
                        rotation=90, transform=transform)
```

*Source: `src/environment/renderer.py:291–353`*

> **API notes.** Host-side only. The `feature_labels` fallback list uses `'DNG'` (not `'HPR'`) — the authoritative v2.0 label. The legacy copy in `grid_world.py` still uses `'HPR'`. Both are cosmetic labels only; they do not affect which feature channel maps to which entity class.

---

### `render_jax_state` — V1 entry point (full)

Short description: the main V1 render function. Takes a single `EnvState` (already converted to host NumPy by the caller), creates a `14x10` figure, draws three panels (left: interoception vitals + minimap, centre: arena local view, right: exteroception sensor pods), and returns a `(H, W, 3)` uint8 NumPy array.

```python
def render_jax_state(state, params, episode=None, step=None, train_episode=None,
                     dpi=100, icon_scale=1.0, action=None,
                     sensory_data=None, info=None, icon_config=None):
    """
    Render a JAX EnvState to an RGB numpy array (Industrial White V2).
    """
    from src.environment.core import calculate_drive
    start_time = time.time()
    icons = _load_icons(icon_config)
    global _FIG_CACHE

    # Grid dimensions -- all .array() / int() calls pull JAX scalars to Python
    height, width = int(params.height), int(params.width)
    view_size = int(params.local_view_size)
    agent_pos = np.array(state.agent_pos)   # host/device boundary: JAX -> NumPy
    ar, ac = int(agent_pos[0]), int(agent_pos[1])

    # Local view window bounds
    half_view = view_size // 2
    r_start = max(0, min(height - view_size, ar - half_view))
    c_start = max(0, min(width  - view_size, ac - half_view))
    r_start = max(0, r_start)
    c_start = max(0, c_start)
    r_end = min(height, r_start + view_size)
    c_end = min(width,  c_start + view_size)

    plt.close('all')

    # Figure: flat GridSpec, 3 columns
    fig = plt.figure(figsize=(14, 10), dpi=dpi)
    fig.patch.set_facecolor(COLORS['bg'])
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 2.2, 0.8])
    ax_left  = fig.add_subplot(gs[0, 0])   # INTEROCEPTION + Minimap
    ax_grid  = fig.add_subplot(gs[0, 1])   # ARENA
    ax_right = fig.add_subplot(gs[0, 2])   # EXTEROCEPTION + Action

    for ax in [ax_left, ax_grid, ax_right]:
        ax.set_facecolor(COLORS['bg'])
        ax.axis('off')

    canvas = FigureCanvas(fig)

    # ---- Centre: arena local view -----------------------------------------------
    ax_grid.set_xlim(c_start - 0.5, c_end - 0.5)
    ax_grid.set_ylim(r_start - 0.5, r_end - 0.5)
    ax_grid.invert_yaxis()
    ax_grid.set_aspect('equal')
    ax_grid.axis('off')

    # Background grid lines
    for x in range(c_start, c_end + 1):
        ax_grid.vlines(x - 0.5, r_start - 0.5, r_end - 0.5,
                       colors=COLORS['grid'], linewidth=0.5)
    for y in range(r_start, r_end + 1):
        ax_grid.hlines(y - 0.5, c_start - 0.5, c_end - 0.5,
                       colors=COLORS['grid'], linewidth=0.5)

    # Terrain tiles
    location_colors = {0: COLORS['bg'], 1: '#ECFDF5', 2: '#FFFBEB'}
    loc_grid = np.array(params.grid_location_type)  # host/device boundary
    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            l_type = int(loc_grid[r, c])
            if l_type != 0:
                ax_grid.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    color=location_colors.get(l_type, COLORS['bg']), zorder=0))

    # Icon drawing helper (inner closure)
    scale_factor = (4.0 / view_size) * icon_scale
    def draw_icon(ax, r, c, icon_key, zoom=0.038, s_fac=1.0, is_axes_coords=False):
        img = icons.get(icon_key)
        if img is not None:
            imagebox = OffsetImage(img, zoom=zoom * s_fac)
            ab = AnnotationBbox(imagebox, (c, r), frameon=False, pad=0,
                                xycoords='data' if not is_axes_coords else ax.transAxes)
            ax.add_artist(ab)
        else:
            m_map = {'agent':('o',COLORS['action']), 'food':('D',COLORS['food']),
                     'hiding_predator':('X',COLORS['hiding_predator']),
                     'predator':('v',COLORS['predator']), 'rock':('s',COLORS['rock']),
                     'neutral':('o',COLORS['neutral'])}
            m, clr = m_map.get(icon_key, ('s', 'grey'))
            ax.plot(c, r, marker=m, markersize=12*s_fac, color=clr,
                    markeredgecolor='white', markeredgewidth=1,
                    transform=ax.transData if not is_axes_coords else ax.transAxes)

    # ---- Entity drawing -- host/device boundary at np.array() calls ----------
    res_pos, res_type, res_active = (np.array(state.res_pos),   # JAX -> NumPy
                                     np.array(params.res_type),
                                     np.array(state.res_active))
    at_agent = []

    for i in range(len(res_active)):
        if not res_active[i]: continue
        rr, rc = int(res_pos[i, 0]), int(res_pos[i, 1])
        if rr == ar and rc == ac:
            at_agent.append('food' if res_type[i] == 0 else 'hiding_predator')
        elif r_start <= rr < r_end and c_start <= rc < c_end:
            draw_icon(ax_grid, rr, rc,
                      'food' if res_type[i] == 0 else 'hiding_predator',
                      zoom=0.035, s_fac=scale_factor)

    # select_by_class: recover per-class slice from unified animal_pos array
    _pred_mask = select_by_class(params, 'predator')    # NumPy bool mask
    p_pos = np.array(state.animal_pos)[_pred_mask]      # host/device boundary
    for i in range(p_pos.shape[0]):
        pr, pc = int(p_pos[i, 0]), int(p_pos[i, 1])
        if pr == ar and pc == ac:
            at_agent.append('predator')
        elif r_start <= pr < r_end and c_start <= pc < c_end:
            draw_icon(ax_grid, pr, pc, 'predator', zoom=0.045, s_fac=scale_factor)

    o_pos = np.array(state.obs_pos)     # host/device boundary
    obs_types = np.array(params.obs_type)
    obs_hides = (np.array(params.obs_hides_agent)
                 if hasattr(params, 'obs_hides_agent')
                 else np.zeros(o_pos.shape[0], dtype=bool))
    for i in range(o_pos.shape[0]):
        or_, oc = int(o_pos[i, 0]), int(o_pos[i, 1])
        if or_ == ar and oc == ac:
            if obs_hides[i]:
                at_agent.append('bush')
        elif r_start <= or_ < r_end and c_start <= oc < c_end:
            draw_icon(ax_grid, or_, oc, params.obstacle_names[obs_types[i]],
                      zoom=0.035, s_fac=scale_factor)

    _neut_mask = select_by_class(params, 'neutral')     # NumPy bool mask
    n_pos = np.array(state.animal_pos)[_neut_mask]      # host/device boundary
    for i in range(n_pos.shape[0]):
        nr, nc = int(n_pos[i, 0]), int(n_pos[i, 1])
        if nr == ar and nc == ac:
            at_agent.append('neutral')
        elif r_start <= nr < r_end and c_start <= nc < c_end:
            draw_icon(ax_grid, nr, nc, 'neutral', zoom=0.035, s_fac=scale_factor)

    # Composite agent icon: priority order -- predator > hiding_predator > bush > food
    agent_icon = 'agent'
    if 'predator'          in at_agent: agent_icon = 'agent_predator'
    elif 'hiding_predator' in at_agent: agent_icon = 'agent_hiding_predator'
    elif 'bush'            in at_agent: agent_icon = 'agent_bush'
    elif 'food'            in at_agent: agent_icon = 'agent_food'
    draw_icon(ax_grid, ar, ac, agent_icon, zoom=0.035, s_fac=scale_factor)

    # ---- Minimap (inset inside ax_left) -----------------------------------------
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_minimap = inset_axes(ax_left, width="80%", height="25%",
                            loc='lower center', borderpad=2.2)
    ax_minimap.axis('off')
    ax_minimap.set_aspect('equal')
    ax_minimap.set_xlim(-0.5, width - 0.5)
    ax_minimap.set_ylim(-0.5, height - 0.5)
    ax_minimap.invert_yaxis()

    minimap_img = np.ones((height, width, 3))
    for l_id, color_hex in location_colors.items():
        if l_id == 0: continue
        from matplotlib.colors import to_rgb
        minimap_img[loc_grid == l_id] = to_rgb(color_hex)
    ax_minimap.imshow(minimap_img,
                      extent=(-0.5, width-0.5, height-0.5, -0.5),
                      zorder=0, alpha=0.5)

    def plot_entity_dots(positions, active_mask, color):
        valid = positions[active_mask]
        if len(valid) > 0:
            ax_minimap.scatter(valid[:, 1], valid[:, 0],
                               s=2.5, color=color, edgecolors='none',
                               alpha=0.8, zorder=2)

    plot_entity_dots(res_pos, res_active & (res_type == 0), COLORS['food'])
    plot_entity_dots(res_pos, res_active & (res_type == 1), COLORS['hiding_predator'])
    plot_entity_dots(p_pos, np.ones(p_pos.shape[0], dtype=bool), COLORS['predator'])
    plot_entity_dots(o_pos, np.ones(o_pos.shape[0], dtype=bool), COLORS['rock'])
    plot_entity_dots(n_pos, np.ones(n_pos.shape[0], dtype=bool), COLORS['neutral'])

    ax_minimap.add_patch(plt.Rectangle(
        (c_start-0.5, r_start-0.5), view_size, view_size,
        fill=False, edgecolor=COLORS['action'],
        linewidth=0.8, alpha=0.6, zorder=3))
    ax_minimap.plot(ac, ar, 'o', color=COLORS['action'],
                    markersize=3, markeredgecolor='white',
                    markeredgewidth=0.4, zorder=4)
    ax_left.text(0.5, 0.32, "MINIMAP", color=COLORS['text_label'],
                 fontsize=7, fontweight='bold', ha='center',
                 transform=ax_left.transAxes)

    # ---- Left panel: interoception vitals ---------------------------------------
    y_ptr = 0.95
    ax_left.text(0.05, y_ptr, "INTEROCEPTION", color=COLORS['text_main'],
                 fontsize=10, fontweight='black', transform=ax_left.transAxes)
    y_ptr -= 0.12

    sensor_map = {s['name']: s for s in sensory_data} if sensory_data else {}

    # 4 bars need tighter spacing than 3
    intero_noc_enabled = bool(getattr(params, 'interoceptive_nociception_enabled', False))
    bar_step = 0.10 if intero_noc_enabled else 0.15

    # Satiation
    sat_real, max_sat = float(state.satiation), float(params.max_satiation)
    sat_obs_data = sensor_map.get('Satiation', {'intensity': sat_real/max_sat})
    sat_obs = float(sat_obs_data.get('intensity', 0))
    draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04,
                          sat_real/max_sat, sat_obs, COLORS['satiation'],
                          "Satiation", f"{sat_real/max_sat:.2f}", f"{sat_obs:.2f}",
                          transform=ax_left.transAxes)
    y_ptr -= bar_step

    # Nutrition
    nut_real, max_nut = float(state.nutrition), float(params.max_nutrition)
    nut_obs_data = sensor_map.get('Nutrition', {'intensity': nut_real/max_nut})
    nut_obs = float(nut_obs_data.get('intensity', 0))
    draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04,
                          nut_real/max_nut, nut_obs, COLORS['nutrition'],
                          "Nutrition", f"{nut_real/max_nut:.2f}", f"{nut_obs:.2f}",
                          transform=ax_left.transAxes)
    y_ptr -= bar_step

    # Injury
    inj_real, max_inj = float(state.injury_level), float(params.max_injury)
    inj_obs_data = sensor_map.get('Injury', {'intensity': inj_real/max_inj})
    inj_obs = float(inj_obs_data.get('intensity', 0))
    draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04,
                          inj_real/max_inj, inj_obs, COLORS['injury'],
                          "Injury", f"{inj_real/max_inj:.2f}", f"{inj_obs:.2f}",
                          transform=ax_left.transAxes)
    y_ptr -= bar_step

    # Interoceptive Nociception (only when enabled)
    if intero_noc_enabled:
        intero_obs_data = sensor_map.get('Intero Nociception')
        if intero_obs_data is not None:
            intero_real = float(intero_obs_data.get('true_intensity',
                                                    intero_obs_data.get('intensity', 0.0)))
            intero_obs  = float(intero_obs_data.get('intensity', intero_real))
        else:
            # Fallback: recompute the convolved signal host-side
            import numpy as _np
            if bool(getattr(params, 'interoceptive_convolution_enabled', False)):
                buf = _np.asarray(state.nociception_history_buffer)  # host/device boundary
                ker = _np.asarray(params.interoceptive_kernel)
                intero_real = float(_np.sum(buf * ker) / max(float(params.max_injury), 1e-6))
            else:
                intero_real = float(state.injury_level) / max_inj
            intero_obs = intero_real

        draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04,
                              intero_real, intero_obs, COLORS['intero_noc'],
                              "Intero Noc", f"{intero_real:.2f}", f"{intero_obs:.2f}",
                              transform=ax_left.transAxes)
        y_ptr -= bar_step

    # ... (omitted: ~12 lines of Run Context pod -- episode, scale, step text labels) ...

    # ---- Right panel: exteroception pods ----------------------------------------
    y_cursor = 0.95
    ax_right.text(0.05, y_cursor, "EXTEROCEPTION", color=COLORS['text_main'],
                  fontsize=11, fontweight='black', transform=ax_right.transAxes)
    y_cursor -= 0.05
    pod_h_default = 0.11

    known_sensors = ['Olfactory', 'Extero Nociception', 'Collision', 'Visual', 'LOC']

    for s_name in known_sensors:
        s_data = sensor_map.get(s_name)
        offline = s_data is None

        # obs_only detection -- NOTE: V1 uses exact equality (== / np.array_equal),
        # which silently fails for sub-epsilon float noise. See Pitfall 4.
        obs_only = False
        if s_data is not None and s_data['type'] == 'intensity':
            obs_only = ('true_intensity' not in s_data
                        or s_data.get('true_intensity') == s_data.get('intensity'))
        elif s_data is not None and s_data['type'] in ('spectrum', 'diamond', 'visual_grid'):
            tv = s_data.get('true_vector')
            obs_only = tv is None or np.array_equal(
                np.asarray(tv), np.asarray(s_data['vector']))  # BUG: exact equality

        pod_h = 0.20 if (not offline and s_data.get('type') in ('diamond', 'visual_grid')) \
                else pod_h_default
        y_frame_bottom = y_cursor - pod_h
        draw_pod_frame(ax_right, 0.05, y_frame_bottom, 0.9, pod_h, s_name,
                       offline=offline, obs_only=obs_only,
                       transform=ax_right.transAxes)

        if not offline:
            if s_data['type'] == 'intensity':
                true_v = float(s_data.get('true_intensity', s_data['intensity']))
                obs_v  = float(s_data['intensity'])
                if obs_only:
                    draw_dual_capsule_bar(ax_right, 0.15, y_frame_bottom+0.02,
                                          0.7, 0.07, 0, obs_v, COLORS['action'],
                                          state_val="--", obs_val=f"{obs_v:.2f}",
                                          transform=ax_right.transAxes)
                else:
                    draw_dual_capsule_bar(ax_right, 0.15, y_frame_bottom+0.02,
                                          0.7, 0.07, true_v, obs_v, COLORS['action'],
                                          state_val=f"{true_v:.2f}", obs_val=f"{obs_v:.2f}",
                                          transform=ax_right.transAxes)
            elif s_data['type'] == 'spectrum':
                obs_vec = np.array(s_data['vector'])
                true_vec = np.array(s_data.get('true_vector', obs_vec))
                max_val = max(np.max(obs_vec), np.max(true_vec))
                scale = 1.0 if max_val <= 1.0 else (1.0 / max_val)
                n = len(obs_vec)
                sw = 0.7 / n
                for i in range(n):
                    vx = 0.15 + i * sw
                    if not obs_only:
                        ax_right.add_patch(plt.Rectangle(
                            (vx, y_frame_bottom+0.02), sw*0.8, 0.07*true_vec[i]*scale,
                            color=COLORS['action'], alpha=0.2,
                            transform=ax_right.transAxes))
                    ax_right.add_patch(plt.Rectangle(
                        (vx, y_frame_bottom+0.02), sw*0.8, 0.07*obs_vec[i]*scale,
                        color=COLORS['action'], alpha=0.9,
                        transform=ax_right.transAxes))
            elif s_data['type'] in ('diamond', 'visual_grid'):
                r = int(s_data.get('range', 1))
                num_features = int(s_data.get('num_features', 1))
                obs_v  = np.array(s_data['vector'])
                true_v = np.array(s_data.get('true_vector', obs_v))
                if r <= 1:
                    draw_categorical_visual(ax_right, 0.08, y_frame_bottom, 0.84, pod_h,
                                            obs_v, r, num_features, true_vec=true_v,
                                            labels=s_data.get('labels'), obs_only=obs_only,
                                            transform=ax_right.transAxes)
                else:
                    origin_x, origin_y = 0.5, y_cursor - pod_h/2
                    cell_size = 0.12 / (2*r + 1)
                    draw_boresight_diamond(ax_right, origin_x, origin_y, cell_size,
                                           obs_v, r, num_features, true_vec=true_v,
                                           icons=icons, obs_only=obs_only,
                                           transform=ax_right.transAxes)
                if not np.any(obs_v > 0.1) and not np.any(true_v > 0.1):
                    ax_right.text(0.5, y_cursor - pod_h/2, "NO SIGNALS",
                                  color=COLORS['text_offline'], fontsize=6,
                                  ha='center', transform=ax_right.transAxes)
            elif s_data['type'] == 'text':
                ax_right.text(0.5, y_cursor - pod_h/2,
                              s_data.get('value_text', '--'),
                              color=COLORS['text_main'], fontsize=10,
                              fontweight='bold', ha='center', va='center',
                              transform=ax_right.transAxes, fontfamily='monospace')

        y_cursor -= (pod_h + 0.03)
        if y_cursor < 0.20: break   # save room for action pod

    # ---- Action pod (fixed at bottom) -------------------------------------------
    action_y = 0.03
    if action is not None:
        draw_pod_frame(ax_right, 0.05, action_y, 0.9, 0.13, "Current Action",
                       transform=ax_right.transAxes)
        action_names = {0:"UP", 1:"RIGHT", 2:"DOWN", 3:"LEFT", 4:"REST", 5:"EAT"}
        arrows        = {0:"↑",  1:"→",    2:"↓",    3:"←",    4:"⊝",    5:"✙"}
        act_name = action_names.get(int(action), f"{action}")
        arrow    = arrows.get(int(action), "•")
        ax_right.text(0.5, action_y+0.085, act_name, color=COLORS['text_main'],
                      fontsize=10, fontweight='black', ha='center',
                      transform=ax_right.transAxes)
        ax_right.text(0.5, action_y+0.025, arrow, color=COLORS['action'],
                      fontsize=18, fontweight='black', ha='center',
                      transform=ax_right.transAxes)

    # ---- Render to numpy array ---------------------------------------------------
    canvas.draw()
    s, (w, h) = canvas.print_to_buffer()
    image = np.frombuffer(s, dtype='uint8').reshape((int(h), int(w), 4))
    return image[:, :, :3]   # drop alpha channel
```

*Source: `src/environment/renderer.py:355–731`*

> **API notes.** This function runs eagerly on host — never under `jax.jit` or `jax.vmap`. The host/device boundary crossings are every `np.array(state.<field>)` call: `state.agent_pos`, `state.res_pos`, `state.res_active`, `state.animal_pos`, `state.obs_pos`, `state.satiation`, `state.nutrition`, `state.injury_level`, `state.nociception_history_buffer`, `params.grid_location_type`. After those conversions, nothing JAX-specific is used; the rest is plain NumPy + Matplotlib. `select_by_class` — see [jax-pytrees](00_jax_primer.md#jax-pytrees) and [scatter-index](00_jax_primer.md#scatter-index) — is called twice (lines 462 and 484) to slice predator and neutral positions from the unified `animal_pos` array. The `plt.close('all')` at the top of the function prevents figure accumulation across frames.

---

### `render_jax_state_v2` — V2 entry point (full)

Short description: the V2 render function. Drop-in replacement for `render_jax_state` with the same signature. Uses `subfigures` + `subplot_mosaic` to create a named-axes layout — panels cannot overlap because each has its own `Axes` object. Shorter figure (`14x8` vs `14x10`), slim header banner across the full width, constrained layout engine.

```python
def render_jax_state_v2(state, params,
                        episode=None, step=None, train_episode=None,
                        dpi=100, icon_scale=1.0, action=None,
                        sensory_data=None, info=None, icon_config=None):
    """
    Render a JAX EnvState to an RGB numpy array -- Dashboard V2.

    Drop-in replacement for render_jax_state(); identical signature.
    """
    icons = _load_icons(icon_config)

    height, width = int(params.height), int(params.width)
    view_size = int(params.local_view_size)
    agent_pos = np.array(state.agent_pos)   # host/device boundary: JAX -> NumPy
    ar, ac = int(agent_pos[0]), int(agent_pos[1])

    half_view = view_size // 2
    r_start = max(0, min(height - view_size, ar - half_view))
    c_start = max(0, min(width  - view_size, ac - half_view))
    r_start, c_start = max(0, r_start), max(0, c_start)
    r_end = min(height, r_start + view_size)
    c_end = min(width,  c_start + view_size)

    plt.close('all')

    # ---- Declarative figure layout (V2 key difference) ---------------------------
    fig = plt.figure(figsize=(14, 8), dpi=dpi, layout='constrained')
    fig.patch.set_facecolor(COLORS['bg'])

    sf_header, sf_body = fig.subfigures(2, 1, height_ratios=[0.055, 0.945])
    sf_header.patch.set_facecolor(COLORS['bg'])
    sf_body.patch.set_facecolor(COLORS['bg'])

    sf_left, sf_center, sf_right = sf_body.subfigures(
        1, 3, width_ratios=[1.0, 2.4, 1.0], wspace=0.02)
    for sf in (sf_left, sf_center, sf_right):
        sf.patch.set_facecolor(COLORS['bg'])

    # Header: single axes
    ax_hdr = sf_header.subplots(1, 1)
    ax_hdr.set_facecolor(COLORS['bg'])
    ax_hdr.axis('off')

    # Left column: 3 vital cards (named axes)
    left_axes = sf_left.subplot_mosaic(
        [['satiation'], ['nutrition'], ['injury']],
        gridspec_kw={'hspace': 0.35})

    # Centre column: large arena + compact minimap
    center_axes = sf_center.subplot_mosaic(
        [['arena'], ['arena'], ['arena'], ['minimap']],
        gridspec_kw={'hspace': 0.06, 'height_ratios': [1, 1, 1, 0.65]})

    # Right column: sensor pods + action badge
    right_axes = sf_right.subplot_mosaic(
        [['olfactory'],
         ['nociception'],
         ['collision'],
         ['visual'],
         ['action']],
        gridspec_kw={'hspace': 0.4, 'height_ratios': [1, 1, 1.4, 1.4, 0.55]})

    sensor_map = {s['name']: s for s in sensory_data} if sensory_data else {}

    # ---- Header banner -----------------------------------------------------------
    disp_ep = episode if episode is not None else (train_episode or '--')
    ax_hdr.text(0.02, 0.5, 'GridWorld · Homeostasis Dashboard',
                transform=ax_hdr.transAxes,
                color=COLORS['text_main'], fontsize=11, fontweight='bold', va='center')
    ax_hdr.text(0.98, 0.5,
                f'Ep {disp_ep}  ·  {width}x{height}  ·  Step {step or "--"}',
                transform=ax_hdr.transAxes,
                color=COLORS['text_label'], fontsize=9,
                ha='right', va='center', fontfamily='monospace')
    ax_hdr.axhline(0.0, color=COLORS['border'], linewidth=1.0)

    # ---- Left panel: vital cards -------------------------------------------------
    # All state reads are plain float() -- scalar JAX arrays auto-convert via float()
    sat_real = float(state.satiation) / float(params.max_satiation)
    sat_data = sensor_map.get('Satiation', {})
    sat_obs  = float(sat_data.get('intensity', sat_real))
    draw_vital_card(left_axes['satiation'], 'Satiation',
                    sat_real, sat_obs,
                    f'{sat_real:.2f}', f'{sat_obs:.2f}', COLORS['satiation'])

    nut_real = float(state.nutrition) / float(params.max_nutrition)
    nut_data = sensor_map.get('Nutrition', {})
    nut_obs  = float(nut_data.get('intensity', nut_real))
    draw_vital_card(left_axes['nutrition'], 'Nutrition',
                    nut_real, nut_obs,
                    f'{nut_real:.2f}', f'{nut_obs:.2f}', COLORS['nutrition'])

    inj_real = float(state.injury_level) / float(params.max_injury)
    inj_data = sensor_map.get('Injury', {})
    inj_obs  = float(inj_data.get('intensity', inj_real))
    draw_vital_card(left_axes['injury'], 'Injury',
                    inj_real, inj_obs,
                    f'{inj_real:.2f}', f'{inj_obs:.2f}', COLORS['injury'])

    # ---- Centre panel: arena -----------------------------------------------------
    ax_arena = center_axes['arena']
    ax_arena.set_facecolor('#ECFDF5')
    for spine in ax_arena.spines.values():
        spine.set_edgecolor(COLORS['arena_border'])
        spine.set_linewidth(2.0)
    ax_arena.set_xticks([])
    ax_arena.set_yticks([])
    ax_arena.set_xlim(c_start - 0.5, c_end - 0.5)
    ax_arena.set_ylim(r_start - 0.5, r_end - 0.5)
    ax_arena.invert_yaxis()
    ax_arena.set_aspect('equal')
    ax_arena.set_title('ARENA -- LOCAL VIEW',
                        fontsize=7, color=COLORS['text_label'],
                        fontweight='bold', pad=4)

    # Grid lines + terrain tiles
    # ... (omitted: ~15 lines of grid vlines/hlines + terrain patches -- identical logic to V1) ...

    # Entity drawing -- host/device boundary at np.array() calls
    scale_factor = (4.0 / view_size) * icon_scale

    def draw_icon(r, c, icon_key, zoom=0.038, s_fac=1.0):
        # ... (omitted: 12 lines -- identical fallback logic to V1, draws on ax_arena) ...
        pass

    res_pos    = np.array(state.res_pos)        # host/device boundary
    res_type   = np.array(params.res_type)
    res_active = np.array(state.res_active)     # host/device boundary
    _pred_mask = select_by_class(params, 'predator')          # NumPy bool mask
    p_pos      = np.array(state.animal_pos)[_pred_mask]       # host/device boundary
    o_pos      = np.array(state.obs_pos)        # host/device boundary
    obs_types  = np.array(params.obs_type)
    obs_hides  = (np.array(params.obs_hides_agent)
                  if hasattr(params, 'obs_hides_agent')
                  else np.zeros(o_pos.shape[0], dtype=bool))
    _neut_mask = select_by_class(params, 'neutral')           # NumPy bool mask
    n_pos      = np.array(state.animal_pos)[_neut_mask]       # host/device boundary

    at_agent = []
    # ... (omitted: ~45 lines of entity iteration -- identical co-occupancy logic to V1) ...

    agent_icon = 'agent'
    if 'predator'          in at_agent: agent_icon = 'agent_predator'
    elif 'hiding_predator' in at_agent: agent_icon = 'agent_hiding_predator'
    elif 'bush'            in at_agent: agent_icon = 'agent_bush'
    elif 'food'            in at_agent: agent_icon = 'agent_food'
    draw_icon(ar, ac, agent_icon, zoom=0.035, s_fac=scale_factor)

    # ---- Centre panel: minimap ---------------------------------------------------
    ax_mini = center_axes['minimap']   # named axes -- no inset_axes needed
    _style_card(ax_mini, facecolor='card_bg')
    ax_mini.set_xlim(-0.5, width - 0.5)
    ax_mini.set_ylim(-0.5, height - 0.5)
    ax_mini.invert_yaxis()
    ax_mini.set_aspect('equal')
    ax_mini.set_title('MINIMAP', fontsize=6, color=COLORS['text_label'],
                      fontweight='bold', pad=3)

    minimap_img = np.ones((height, width, 3))
    for l_id, color_hex in location_colors.items():
        if l_id != 0:
            minimap_img[loc_grid == l_id] = to_rgb(color_hex)
    ax_mini.imshow(minimap_img,
                   extent=(-0.5, width-0.5, height-0.5, -0.5),
                   zorder=0, alpha=0.5)

    # ... (omitted: ~15 lines of minimap scatter dots + viewport rectangle -- identical to V1) ...

    # ---- Right panel: sensor pods (V2 uses named axes per pod) ------------------
    pod_map = [
        ('olfactory',   'Olfactory'),
        ('nociception', 'Extero Nociception'),
        ('collision',   'Collision'),
        ('visual',      'Visual'),
    ]

    for ax_key, s_name in pod_map:
        ax = right_axes[ax_key]
        s_data = sensor_map.get(s_name)

        if s_data is None:
            draw_offline_card(ax, s_name)
            continue

        s_type = s_data.get('type', 'spectrum')

        if s_type == 'intensity':
            ti = s_data.get('true_intensity')
            # V2 fix: use abs() < 1e-6 instead of exact == (fixes Pitfall 4)
            obs_only = (ti is None
                        or abs(float(ti) - float(s_data['intensity'])) < 1e-6)
            obs_v  = float(s_data['intensity'])
            true_v = float(s_data.get('true_intensity', obs_v))
            draw_intensity_pod(ax, s_name,
                               min(1.0, obs_v), min(1.0, true_v),
                               f'{obs_v:.2f}', f'{true_v:.2f}',
                               obs_only=obs_only)

        elif s_type == 'spectrum':
            obs_vec  = np.asarray(s_data['vector'])
            tv       = s_data.get('true_vector')
            true_vec = np.asarray(tv) if tv is not None else obs_vec
            # V2 fix: np.allclose instead of np.array_equal (fixes Pitfall 4)
            obs_only = tv is None or np.allclose(obs_vec, true_vec, atol=1e-6)
            draw_spectrum_pod(ax, s_name, obs_vec, true_vec, obs_only=obs_only)

        elif s_type in ('diamond', 'visual_grid'):
            obs_vec      = np.asarray(s_data['vector'])
            tv           = s_data.get('true_vector')
            true_vec     = np.asarray(tv) if tv is not None else obs_vec
            obs_only     = tv is None or np.allclose(obs_vec, true_vec, atol=1e-6)
            r_range      = int(s_data.get('range', 1))
            num_features = int(s_data.get('num_features', 1))
            draw_categorical_pod(ax, s_name, obs_vec, true_vec,
                                 r_range, num_features,
                                 labels=s_data.get('labels'),
                                 obs_only=obs_only)

        elif s_type == 'text':
            _style_card(ax)
            _card_title(ax, s_name)
            ax.text(0.5, 0.48, s_data.get('value_text', '--'),
                    transform=ax.transAxes, color=COLORS['text_main'],
                    fontsize=10, fontweight='bold',
                    ha='center', va='center', fontfamily='monospace')
        else:
            draw_offline_card(ax, s_name)

    draw_action_pod(right_axes['action'], action)

    # ---- Render to numpy array ---------------------------------------------------
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf, (w, h) = canvas.print_to_buffer()
    image = np.frombuffer(buf, dtype='uint8').reshape((int(h), int(w), 4))
    plt.close(fig)
    return image[:, :, :3]   # drop alpha channel
```

*Source: `src/environment/renderer_v2.py:216–540`*

> **API notes.** Host-side only. `subfigures` + `subplot_mosaic` is a Matplotlib 3.4+ API that assigns each panel its own `Axes` object at creation time — no y-cursor arithmetic, no `inset_axes`, no overlap risk. The `layout='constrained'` flag (Matplotlib 3.5+) handles padding automatically; it can warn when axes are too small for their decorators (Pitfall 6). Unlike V1, V2 closes the figure (`plt.close(fig)`) at the end rather than relying on the global `plt.close('all')` at the start of the next call. The `obs_only` detection uses `abs() < 1e-6` (scalar) and `np.allclose(..., atol=1e-6)` (vector) — the fix for V1's exact-equality bug (Pitfall 4). No JAX is used after the `np.array(state.agent_pos)` boundary crossing at the top.

---

### `save_jax_video` — streaming video export

Short description: writes a list of `(H, W, 3)` uint8 NumPy frames to an MP4 file using `imageio.get_writer` (frame-by-frame streaming). The `quiet=True` flag silences imageio's stdout/stderr by redirecting file descriptors to `/dev/null` at the OS level — not just Python-level redirection.

```python
def save_jax_video(frames, output_path, fps=5, quiet=False):
    """
    Save a list of RGB frames as an MP4 video.

    Args:
        frames: List of numpy arrays (H, W, 3)
        output_path: Path to save the video
        fps: Frames per second
        quiet: Suppress output messages
    """
    import os
    import imageio

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if quiet:
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        old_stdout_fd = os.dup(sys.stdout.fileno())
        old_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, sys.stdout.fileno())
            os.dup2(devnull, sys.stderr.fileno())
            os.close(devnull)
            with imageio.get_writer(output_path, fps=fps) as writer:
                for frame in frames:
                    writer.append_data(frame)
        finally:
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
    else:
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"Saved video to {output_path}")
```

*Source: `src/environment/renderer.py:734–773`*

> **API notes.** Host-side only — no JAX. The production implementation uses `imageio.get_writer` (streaming, one frame at a time, constant memory), unlike the legacy `grid_world.py` copy which uses `imageio.mimsave` (loads all frames into memory at once). `renderer_v2.py` re-exports `save_jax_video` directly from `renderer.py` (`renderer_v2.py:29`) — there is a single implementation. The fd-level redirect in `quiet` mode (`os.dup`/`os.dup2`) silences C-library output that Python-level `sys.stdout` capture cannot catch — relevant because imageio's ffmpeg subprocess writes directly to fd 1/2.

---

### V2 card/pod helpers (signatures + instructive core)

#### `_hbar` — OBS fill bar + REAL tick

```python
def _hbar(ax, bar_left, bar_bottom, bar_width, bar_height,
          obs_pct, real_pct, color, obs_only=False):
    """Horizontal OBS bar + thin vertical REAL tick marker."""
    # Trough
    ax.add_patch(matplotlib.patches.FancyBboxPatch(
        (bar_left, bar_bottom), bar_width, bar_height,
        boxstyle='round,pad=0,rounding_size=0.02',
        facecolor='#F3F4F6', edgecolor='none',
        transform=ax.transAxes, zorder=0))
    # OBS fill
    obs_w = max(0.02, bar_width * min(1.0, obs_pct))
    ax.add_patch(matplotlib.patches.FancyBboxPatch(
        (bar_left, bar_bottom), obs_w, bar_height,
        boxstyle='round,pad=0,rounding_size=0.02',
        facecolor=color, edgecolor='none', alpha=0.85,
        transform=ax.transAxes, zorder=2))
    # REAL tick (vertical line at real_pct position)
    if not obs_only:
        tick_x = bar_left + bar_width * min(1.0, real_pct)
        ax.plot([tick_x, tick_x],
                [bar_bottom - 0.06, bar_bottom + bar_height + 0.06],
                color=COLORS['text_main'], linewidth=1.8,
                transform=ax.transAxes, zorder=3,
                solid_capstyle='round')
```

*Source: `src/environment/renderer_v2.py:58–81`*

#### `draw_vital_card` — interoception vital card

```python
def draw_vital_card(ax, label, real_pct, obs_pct,
                    real_val_str, obs_val_str, color):
    """Interoception vital: label + big OBS value + bar + REAL tick + delta."""
    _style_card(ax)
    ax.text(0.05, 0.93, label.upper(),
            transform=ax.transAxes, color=COLORS['text_label'],
            fontsize=7, fontweight='bold', va='top')
    ax.text(0.95, 0.93, obs_val_str,
            transform=ax.transAxes, color=color,
            fontsize=14, fontweight='bold', ha='right', va='top')

    _hbar(ax, 0.05, 0.40, 0.90, 0.26, obs_pct, real_pct, color, obs_only=False)

    delta = obs_pct - real_pct
    sign = '+' if delta >= 0 else ''
    ax.text(0.05, 0.08,
            f"real {real_val_str}  ·  delta {sign}{delta:.2f}",
            transform=ax.transAxes,
            color=COLORS['text_label'], fontsize=6, va='bottom')
```

*Source: `src/environment/renderer_v2.py:86–104`*

#### `draw_intensity_pod` — extero nociception pod

```python
def draw_intensity_pod(ax, label, obs_pct, real_pct,
                       obs_val_str, real_val_str, obs_only=False):
    """Extero nociception: same layout as vital card but uses action color."""
    _style_card(ax)
    _card_title(ax, label, obs_only=obs_only)
    ax.text(0.95, 0.93, obs_val_str,
            transform=ax.transAxes, color=COLORS['action'],
            fontsize=12, fontweight='bold', ha='right', va='top')

    _hbar(ax, 0.05, 0.40, 0.90, 0.26, obs_pct, real_pct,
          COLORS['action'], obs_only=obs_only)

    if obs_only:
        ax.text(0.05, 0.08, 'obs only -- no noise-free reference',
                transform=ax.transAxes,
                color=COLORS['text_offline'], fontsize=5.5, va='bottom')
    else:
        delta = obs_pct - real_pct
        sign = '+' if delta >= 0 else ''
        ax.text(0.05, 0.08,
                f"real {real_val_str}  ·  delta {sign}{delta:.2f}",
                transform=ax.transAxes,
                color=COLORS['text_label'], fontsize=6, va='bottom')
```

*Source: `src/environment/renderer_v2.py:115–137`*

#### `draw_spectrum_pod` — olfactory 5-channel bars

```python
def draw_spectrum_pod(ax, label, obs_vec, true_vec, obs_only=False):
    """Olfactory 5-channel spectrum bars."""
    _style_card(ax)
    _card_title(ax, label, obs_only=obs_only)

    obs_vec  = np.asarray(obs_vec,  dtype=float)
    true_vec = np.asarray(true_vec, dtype=float)
    n = len(obs_vec)
    all_vals = np.concatenate([obs_vec, true_vec]) if not obs_only else obs_vec
    max_val  = float(np.max(all_vals)) if np.max(all_vals) > 0.01 else 1.0
    scale    = 1.0 / max_val

    bar_x, bar_y = 0.06, 0.14
    bar_w, bar_h = 0.88, 0.65
    sw = bar_w / n
    for i in range(n):
        vx = bar_x + i * sw
        if not obs_only:
            ax.add_patch(plt.Rectangle(
                (vx, bar_y), sw * 0.80, bar_h * float(true_vec[i]) * scale,
                facecolor=COLORS['action'], alpha=0.15,
                transform=ax.transAxes, zorder=1))
        ax.add_patch(plt.Rectangle(
            (vx, bar_y), sw * 0.80, bar_h * float(obs_vec[i]) * scale,
            facecolor=COLORS['action'], alpha=0.9,
            transform=ax.transAxes, zorder=2))
```

*Source: `src/environment/renderer_v2.py:140–165`*

#### `draw_categorical_pod` — Collision/Visual wrapper

```python
def draw_categorical_pod(ax, label, obs_vec, true_vec,
                          r, num_features, labels=None, obs_only=False):
    """Collision / Visual categorical grid, using draw_categorical_visual."""
    _style_card(ax)
    _card_title(ax, label, obs_only=obs_only)
    draw_categorical_visual(
        ax, 0.05, 0.08, 0.90, 0.78,
        obs_vec, r, num_features,
        true_vec=true_vec, labels=labels,
        obs_only=obs_only,
        transform=ax.transAxes)

    if not np.any(np.asarray(obs_vec) > 0.1):
        ax.text(0.5, 0.50, 'NO SIGNALS',
                transform=ax.transAxes, color=COLORS['text_offline'],
                fontsize=6, ha='center', va='center')
```

*Source: `src/environment/renderer_v2.py:168–185`*

#### `draw_action_pod` — action badge

```python
def draw_action_pod(ax, action):
    """Small action badge at bottom of right column."""
    _style_card(ax)
    ax.text(0.05, 0.93, 'CURRENT ACTION',
            transform=ax.transAxes, color=COLORS['text_label'],
            fontsize=6.5, fontweight='bold', va='top')

    action_names = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT',
                    4: 'REST', 5: 'EAT'}
    arrows        = {0: '↑', 1: '→', 2: '↓', 3: '←', 4: '⊝', 5: '✙'}

    if action is not None:
        act_name = action_names.get(int(action), str(action))
        arrow    = arrows.get(int(action), '•')
        ax.text(0.5, 0.66, act_name,
                transform=ax.transAxes, color=COLORS['text_main'],
                fontsize=11, fontweight='black', ha='center', va='center')
        ax.text(0.5, 0.20, arrow,
                transform=ax.transAxes, color=COLORS['action'],
                fontsize=17, fontweight='black', ha='center', va='center')
    else:
        ax.text(0.5, 0.48, '--',
                transform=ax.transAxes, color=COLORS['text_offline'],
                fontsize=12, ha='center', va='center')
```

*Source: `src/environment/renderer_v2.py:188–211`*

> **API notes (V2 card helpers).** All V2 card/pod functions take `ax` as their first argument and use `transform=ax.transAxes` internally — no coordinate arithmetic outside the function. `_style_card` sets background and spine style; `_card_title` writes the title with `(OBS ONLY)` or `OFFLINE` suffix. `draw_categorical_pod` is a thin wrapper that delegates to the shared `draw_categorical_visual` from `renderer.py`, passing fixed margins (`0.05, 0.08, 0.90, 0.78`) tuned for the V2 axes geometry.
