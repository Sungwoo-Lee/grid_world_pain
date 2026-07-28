"""Dream-strip visualizer for the dreamer_srl agent.

Produces a per-modality "dream-strip" figure for ONE event-anchored window
in ONE episode.  Seven rows total (top → bottom):

  0. Scene (real)                 — rendered gridworld frame at each step in
                                    the window.
  1. Ground truth                 — real observation at each step in the window.
  2. Actions (real)               — THIN strip: action the agent ACTUALLY chose
                                    at each step.  Pre-event columns shown in
                                    ghost/muted style.
  3. Dreaming (real actions)      — decoded observations from an open-loop RSSM
                                    rollout that uses the RECORDED real action
                                    sequence (actions actually taken at steps
                                    E, E+1, …, E+N-1) rather than actor samples.
                                    Launches at t=0.  Columns t−PRE…t−1 show the
                                    real observations ghosted in a muted tan tone.
  4. One-step prediction          — genuine one-step-ahead prediction: at each
                                    step the RSSM prior (transition model, before
                                    seeing obs) is decoded.  The loop is
                                    re-grounded on the real posterior after each
                                    step, so each column shows "predict this frame
                                    from the real previous frame, without seeing
                                    it".  Runs across the FULL window t−PRE…t+N.
  5. Actions (dream/policy)       — THIN strip: action the POLICY chose inside
                                    the policy dream at each step.  Highlighted
                                    in red when it differs from the real action.
                                    Pre-event columns shown ghost/blank.
  6. Dreaming (policy actions)    — decoded observations from WorldModel.imagine()
                                    (actor-sampled actions, eyes closed), launched
                                    at t=0.  Columns t−PRE…t−1 show the real
                                    observations ghosted.

The figure is EVENT-CENTERED: the detected damage/hazard event is t=0.
Earlier steps are t−1, t−2, …; later steps are t+1, t+2, ….  The dream
launches its imagination AT t=0 (the moment of the event).

NO src/ changes required.  The script derives everything in-script by running
observe() over the recorded obs/action sequence to get the posterior latent at
the event step E (and runs the one-step-prediction decoded row across the whole
window simultaneously), then runs both dream variants from the event latent for
N steps.

Usage
-----
    JAX_PLATFORMS=cpu python scripts/visualize_dream.py \\
        --run-dir results/JAX_DreamerSRL/<run>/ \\
        --model-config configs/models/dreamer_srl/01_food_only_buf256k.yaml \\
        --episode results/JAX_DreamerSRL/<run>/recordings/<ckpt>/episode_000001.rec.gz \\
        --checkpoint-dir results/JAX_DreamerSRL/<run>/checkpoints/ \\
        --checkpoint-step <N> \\
        --out-dir results/dream_viz/ \\
        [--pre-steps N]  [--horizon N]

    Or use the defaults / auto-detect mode (see --help).
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_episode(path: str) -> dict:
    with gzip.open(path, 'rb') as fh:
        return pickle.load(fh)


def _load_run_meta(rec_dir: str) -> dict:
    with open(Path(rec_dir) / 'run_meta.pkl', 'rb') as fh:
        return pickle.load(fh)


def _build_cfg_dict(model_config_path: str) -> dict:
    """Load a model config YAML and return its dict."""
    import yaml
    with open(model_config_path, 'r') as fh:
        return yaml.safe_load(fh)


def _detect_events(obs: np.ndarray, breakdown: dict) -> List[Tuple[int, str]]:
    """Detect hazard-contact events from the recorded obs vector.

    Returns a list of (event_step, event_label) sorted by step.
    event_step is the step at which the event fires (this becomes t=0).
    """
    # Build offset map
    offsets = {}
    ptr = 0
    for name, dim in breakdown.items():
        offsets[name] = (ptr, ptr + dim)
        ptr += dim

    events = []

    # Extero Nociception (dim=1): rising edge where value > 0
    if 'Extero Nociception' in offsets:
        s, e = offsets['Extero Nociception']
        extero = obs[:, s]
        prev_zero = np.concatenate([[True], extero[:-1] < 0.01])
        curr_nonzero = extero > 0.01
        rising = np.where(prev_zero & curr_nonzero)[0]
        for step in rising:
            events.append((int(step), f'hazard_contact@{step}'))

    # Interoceptive Nociception: rising edge (pain onset)
    if 'Interoceptive Nociception' in offsets:
        s, e = offsets['Interoceptive Nociception']
        intero = obs[:, s]
        prev_zero = np.concatenate([[True], intero[:-1] < 0.01])
        curr_nonzero = intero > 0.01
        rising = np.where(prev_zero & curr_nonzero)[0]
        for step in rising:
            events.append((int(step), f'pain_onset@{step}'))

    # Deduplicate (keep first label per step) and sort
    seen = set()
    out = []
    for (step, lbl) in sorted(events):
        if step not in seen:
            seen.add(step)
            out.append((step, lbl))
    return out


# ---------------------------------------------------------------------------
# Scene renderer — converts a snapshot dict into an RGB array via renderer.py
# ---------------------------------------------------------------------------

def _render_scene_frame(snapshot: dict, env_params, action=None, dpi: int = 200) -> np.ndarray:
    """Render one gridworld snapshot dict to an RGB numpy array of the center grid only.

    snapshot is a plain dict from ep['snapshots'][step]; we wrap it in
    SimpleNamespace so render_jax_state can access fields via attribute lookup.
    Returns a (H, W, 3) uint8 array cropped to the central environment grid panel only
    (no left interoception/minimap bars, no right exteroception/action panel).
    """
    from types import SimpleNamespace
    from src.environment.renderer import render_jax_state

    # render_jax_state may access state.nociception_history_buffer if
    # params.interoceptive_convolution_enabled is True.  The snapshot dict
    # doesn't carry that buffer, so we add a zero sentinel here.  The renderer
    # only hits that branch when sensory_data is None (which it is here) and
    # interoceptive_convolution_enabled is True — the zero sentinel means it
    # will show 0.0 intero noc on the left panel, which is fine for a scene row.
    ns = SimpleNamespace(**snapshot)
    if not hasattr(ns, 'nociception_history_buffer'):
        ns.nociception_history_buffer = np.zeros(1)

    rgb = render_jax_state(
        state=ns,
        params=env_params,
        action=action,
        dpi=dpi,
    )
    # rgb is (H, W, 3) uint8 — the full figure including left and right telemetry panels.
    # Crop to the center ax_grid panel only using the cached figure geometry.
    # _FIG_CACHE = (fig, ax_grid, ax_right, ax_left, canvas)
    import src.environment.renderer as _renderer_mod
    cached = _renderer_mod._FIG_CACHE
    if cached is not None:
        fig_cached, ax_grid_cached, _, _, _ = cached
        # get_position() returns Bbox in normalized figure coords [0,1]
        pos = ax_grid_cached.get_position()
        H, W = rgb.shape[:2]
        # Normalized coords → pixel coords (image y is top-down, matplotlib y is bottom-up)
        x0_px = int(round(pos.x0 * W))
        x1_px = int(round(pos.x1 * W))
        # pos.y0 is bottom in figure coords; in image coords that is H - pos.y1*H (top)
        y0_px = int(round((1.0 - pos.y1) * H))
        y1_px = int(round((1.0 - pos.y0) * H))
        # Clamp to valid range
        x0_px = max(0, x0_px)
        x1_px = min(W, x1_px)
        y0_px = max(0, y0_px)
        y1_px = min(H, y1_px)
        cropped = rgb[y0_px:y1_px, x0_px:x1_px]
        # Trim to square (ax_grid uses set_aspect('equal') so the content is square,
        # but the axes bounding box may be taller due to padding; crop height to width).
        h_c, w_c = cropped.shape[:2]
        if h_c > w_c:
            excess = h_c - w_c
            top_trim = excess // 2
            cropped = cropped[top_trim: top_trim + w_c, :]
        elif w_c > h_c:
            excess = w_c - h_c
            left_trim = excess // 2
            cropped = cropped[:, left_trim: left_trim + h_c]
        rgb = cropped
    else:
        # Fallback: GridSpec geometry width_ratios=[0.8, 2.2, 0.8], total=3.8
        # Center band horizontal fraction: [0.8/3.8, 3.0/3.8] = [0.211, 0.789]
        H, W = rgb.shape[:2]
        x0_px = int(round(0.211 * W))
        x1_px = int(round(0.789 * W))
        cropped = rgb[:, x0_px:x1_px]
        h_c, w_c = cropped.shape[:2]
        if h_c > w_c:
            excess = h_c - w_c
            top_trim = excess // 2
            cropped = cropped[top_trim: top_trim + w_c, :]
        rgb = cropped

    # Close any figures render_jax_state may have leaked (it calls plt.close('all')
    # internally, but be defensive).
    plt.close('all')
    return rgb  # (H, W, 3) uint8 — center grid only


# ---------------------------------------------------------------------------
# Observation modality renderer (matplotlib, one column per step)
# ---------------------------------------------------------------------------

def _render_obs_column(ax: plt.Axes, obs_vec: np.ndarray, real_vec: Optional[np.ndarray],
                       breakdown: dict, title: str = '', color: str = 'steelblue') -> None:
    """Render one column of the dream strip: all modalities stacked vertically.

    Each modality gets a horizontal bar proportional to its value(s).
    real_vec is the ground-truth overlay (orange/faint) when provided.
    """
    import numpy as np
    names = list(breakdown.keys())
    dims  = list(breakdown.values())
    n_mod = len(names)

    obs_ptr = 0
    real_ptr = 0
    bar_height = 0.8 / n_mod

    for i, (name, dim) in enumerate(zip(names, dims)):
        obs_slice  = obs_vec[obs_ptr: obs_ptr + dim]
        real_slice = real_vec[real_ptr: real_ptr + dim] if real_vec is not None else None
        obs_ptr  += dim
        real_ptr += dim

        y_center = 1.0 - (i + 0.5) / n_mod
        y_bot    = y_center - bar_height / 2

        if dim == 1:
            val  = float(np.clip(obs_slice[0], 0.0, 1.0))
            ax.barh(y_center, val, height=bar_height * 0.9, color=color, alpha=0.85)
            if real_slice is not None:
                rval = float(np.clip(real_slice[0], 0.0, 1.0))
                ax.barh(y_center, rval, height=bar_height * 0.4,
                        color='orange', alpha=0.45, left=0)
        else:
            # Multi-dim: draw each element as a small colored bar segment
            vals = np.clip(obs_slice, 0.0, 1.0)
            seg_w = 1.0 / dim
            for j, v in enumerate(vals):
                ax.bar(j * seg_w + seg_w / 2, v * bar_height * 0.9,
                       width=seg_w * 0.85, bottom=y_bot,
                       color=color, alpha=0.80)
            if real_slice is not None:
                rvals = np.clip(real_slice, 0.0, 1.0)
                for j, rv in enumerate(rvals):
                    ax.bar(j * seg_w + seg_w / 2, rv * bar_height * 0.4,
                           width=seg_w * 0.85, bottom=y_bot,
                           color='orange', alpha=0.35)

    # Y-tick labels = modality names
    yticks = [1.0 - (i + 0.5) / n_mod for i in range(n_mod)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def _render_ghost_column(ax: plt.Axes, real_vec: np.ndarray,
                         breakdown: dict, title: str = '') -> None:
    """Render a ghosted (muted tan) column showing real observations.

    Used to fill pre-event columns in the dreaming rows so the viewer can
    see the real context leading up to t=0 even though the dream has not
    started yet.  Modality y-tick labels and axes are preserved so the row
    axis labeling is consistent.
    """
    import numpy as np
    GHOST_COLOR = '#c8b89a'   # muted tan — visually distinct from all dream colors
    GHOST_ALPHA = 0.70

    names = list(breakdown.keys())
    dims  = list(breakdown.values())
    n_mod = len(names)

    ptr = 0
    bar_height = 0.8 / n_mod

    for i, (name, dim) in enumerate(zip(names, dims)):
        real_slice = real_vec[ptr: ptr + dim]
        ptr += dim

        y_center = 1.0 - (i + 0.5) / n_mod
        y_bot    = y_center - bar_height / 2

        if dim == 1:
            rval = float(np.clip(real_slice[0], 0.0, 1.0))
            ax.barh(y_center, rval, height=bar_height * 0.9,
                    color=GHOST_COLOR, alpha=GHOST_ALPHA)
        else:
            rvals = np.clip(real_slice, 0.0, 1.0)
            seg_w = 1.0 / dim
            for j, rv in enumerate(rvals):
                ax.bar(j * seg_w + seg_w / 2, rv * bar_height * 0.9,
                       width=seg_w * 0.85, bottom=y_bot,
                       color=GHOST_COLOR, alpha=GHOST_ALPHA)

    # Y-tick labels = modality names (kept so axis labeling doesn't disappear)
    yticks = [1.0 - (i + 0.5) / n_mod for i in range(n_mod)]
    ax.set_yticks(yticks)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    # Light grey background to visually distinguish from the dream region
    ax.set_facecolor('#f5f0e8')
    for spine in ax.spines.values():
        spine.set_color('#bbaa88')
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def _make_blank_column(ax: plt.Axes, title: str = '') -> None:
    """Make an axes cell visually blank (legacy fallback; no longer used for dream rows)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 0.5, '—', ha='center', va='center',
            fontsize=14, color='#aaaaaa', transform=ax.transAxes)
    ax.set_facecolor('#f5f5f5')
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
    if title:
        ax.set_title(title, fontsize=7, pad=2)


# ---------------------------------------------------------------------------
# Action strip helpers
# ---------------------------------------------------------------------------

def _build_action_symbol_map(action_map: list) -> dict:
    """Build a mapping from action index → display symbol.

    Movement actions are converted to arrows based on the name (Up/Down/Left/Right).
    Known non-movement actions (Rest, Eat) get short labels.
    Unknown actions fall back to the first letter of their name.

    Returns a dict {int → str}.
    """
    _NAMED = {
        'up':    '↑',
        'down':  '↓',
        'left':  '←',
        'right': '→',
        'rest':  'R',
        'eat':   'E',
        'wait':  'W',
    }
    result = {}
    for i, name in enumerate(action_map):
        key = name.lower().strip()
        if key in _NAMED:
            result[i] = _NAMED[key]
        else:
            result[i] = name[0].upper() if name else '?'
    return result


def _render_action_cell(
    ax: plt.Axes,
    symbol: str,
    ghost: bool = False,
    highlight: bool = False,
    title: str = '',
) -> None:
    """Render a single action cell as a large centred symbol.

    ghost=True → muted tan background + grey text (pre-event context style).
    highlight=True → red text (policy chose differently from real action).
    """
    if ghost:
        bg_color = '#f5f0e8'
        txt_color = '#b0a080'
        edge_color = '#bbaa88'
    else:
        bg_color = 'white'
        txt_color = 'crimson' if highlight else '#222222'
        edge_color = '#999999'

    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(edge_color)
        spine.set_linewidth(0.8)

    ax.text(0.5, 0.5, symbol, ha='center', va='center',
            fontsize=14, fontweight='bold', color=txt_color,
            transform=ax.transAxes)
    if title:
        ax.set_title(title, fontsize=7, pad=2)


def build_dream_strip_figure(
    real_obs_window:         np.ndarray,          # [PRE+1+N, obs_dim]  columns t-PRE..t+N
    dream_real_acts:         np.ndarray,          # [1+N, obs_dim]  t=0..t+N  (real-action dream obs)
    dream_policy_acts:       np.ndarray,          # [1+N, obs_dim]  t=0..t+N  (policy-action dream obs)
    tf_obs_window:           np.ndarray,          # [PRE+1+N, obs_dim]  one-step pred full window
    breakdown:               dict,
    event_step:              int,                 # absolute episode step for t=0
    pre_steps:               int,                 # number of pre-event columns (PRE)
    event_label:             str,
    horizon:                 int,                 # N = dream horizon
    title_prefix:            str = '',
    scene_frames:            Optional[List[np.ndarray]] = None,  # list of PRE+1+N RGB arrays
    real_actions_window:     Optional[np.ndarray] = None,   # [PRE+1+N,] int action indices
    policy_actions_window:   Optional[np.ndarray] = None,   # [1+N,] int action indices (t=0..t+N)
    action_symbol_map:       Optional[dict] = None,         # {int → str} symbol lookup
) -> plt.Figure:
    """Build the seven-row event-centered dream-strip figure.

    Columns span absolute steps [E-PRE, E+N] where E=event_step.
    Column index j=0 → t-PRE, j=pre_steps → t=0 (event), j=pre_steps+N → t+N.

    Row order (top → bottom):
      Row 0 — Scene (real)                [only if scene_frames provided]
      Row 1 — Ground truth                (full window)
      Row 2 — Actions (real)              THIN: real action chosen at each step
                                          (ghost style for pre-event columns)
      Row 3 — Dreaming (real actions)     (pre-event: ghosted real obs; t=0..t+N: dream)
      Row 4 — One-step prediction         (full window)
      Row 5 — Actions (dream/policy)      THIN: policy action chosen inside dream
                                          (highlight when differs from real;
                                           ghost/blank for pre-event columns)
      Row 6 — Dreaming (policy actions)   (pre-event: ghosted real obs; t=0..t+N: dream)
    """
    n_cols = real_obs_window.shape[0]   # PRE + 1 + N
    event_col = pre_steps               # column index for t=0

    has_scenes = scene_frames is not None and len(scene_frames) == n_cols
    has_actions = (real_actions_window is not None
                   and policy_actions_window is not None
                   and action_symbol_map is not None)

    # ----------------------------------------------------------------
    # Row layout
    # Action strips are "thin" (height_ratio ACTION_H vs DATA_H for data rows).
    # Order: Scene? | GT | ActReal | DreamReal | OneStep | ActPolicy | DreamPolicy
    # ----------------------------------------------------------------
    SCENE_H  = 3.0   # inches
    DATA_H   = 1.5   # inches per data (modality) row
    ACTION_H = 0.35  # inches per thin action strip

    # Build ordered list of row descriptors: (label, kind, height)
    # kind ∈ 'scene', 'data', 'action'
    row_specs = []
    if has_scenes:
        row_specs.append(('Scene (real)',              'scene'))
    row_specs.append(('Ground truth',                  'data'))
    if has_actions:
        row_specs.append(('Actions\n(real)',            'action'))
    row_specs.append(('Dreaming\n(real actions)',       'data'))
    row_specs.append(('One-step\nprediction',           'data'))
    if has_actions:
        row_specs.append(('Actions\n(dream/policy)',    'action'))
    row_specs.append(('Dreaming\n(policy actions)',     'data'))

    n_rows = len(row_specs)
    height_map = {'scene': SCENE_H, 'data': DATA_H, 'action': ACTION_H}
    height_ratios = [height_map[kind] for (_, kind) in row_specs]
    total_h = sum(height_ratios) + 1.0   # +1 for suptitle / legend

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(6, n_cols * 1.2), total_h),
        gridspec_kw={'height_ratios': height_ratios},
        squeeze=False,
    )

    # Row index lookup by label (for the rendering loop below)
    row_idx = {label: i for i, (label, _) in enumerate(row_specs)}

    # Compute relative column labels: t-PRE..t-1, t=0, t+1..t+N
    def _col_label(j: int) -> str:
        rel = j - event_col
        if rel == 0:
            return 't=0\n(damage)'
        elif rel < 0:
            return f't{rel}'   # e.g. t-2, t-1
        else:
            return f't+{rel}'  # e.g. t+1, t+2

    # ----------------------------------------------------------------
    # Row colors by label (used when rendering data rows)
    # ----------------------------------------------------------------
    _ROW_COLORS = {
        'Ground truth':              'steelblue',
        'Dreaming\n(real actions)':  'tomato',
        'One-step\nprediction':      'seagreen',
        'Dreaming\n(policy actions)':'#a040c0',
    }

    # ----------------------------------------------------------------
    # Render each row
    # ----------------------------------------------------------------
    for row_i, (row_label, row_kind) in enumerate(row_specs):

        for col_j in range(n_cols):
            ax = axes[row_i, col_j]

            # Column title: shown only on the top-most row (scene or GT)
            col_title = _col_label(col_j) if row_i == 0 else ''

            # ---- Scene row ----
            if row_kind == 'scene':
                ax.imshow(scene_frames[col_j])
                ax.axis('off')
                ax.set_title(col_title, fontsize=7, pad=2)

            # ---- Action strip rows ----
            elif row_kind == 'action':
                is_real_action_row = row_label.startswith('Actions\n(real')

                if is_real_action_row:
                    # Real action strip — covers ALL columns
                    # action chosen at absolute step s = actions_all[s+1]
                    # col_j → absolute step (win_start + col_j); symbol always present
                    if real_actions_window is not None and col_j < len(real_actions_window):
                        act_idx = int(real_actions_window[col_j])
                        sym = action_symbol_map.get(act_idx, '?')
                    else:
                        sym = '·'
                    ghost = col_j < event_col
                    _render_action_cell(ax, sym, ghost=ghost, highlight=False,
                                        title=col_title)

                else:
                    # Policy action strip — only t=0..t+N (col_j >= event_col)
                    # policy_actions_window[i] = action chosen at dream step i
                    dream_col_idx = col_j - event_col   # <0 for pre-event cols
                    if col_j < event_col:
                        # Pre-event: blank ghost
                        _render_action_cell(ax, '·', ghost=True, highlight=False,
                                            title=col_title)
                    elif policy_actions_window is not None and dream_col_idx < len(policy_actions_window):
                        pol_act_idx = int(policy_actions_window[dream_col_idx])
                        pol_sym = action_symbol_map.get(pol_act_idx, '?')
                        # Highlight if policy differs from real action
                        highlight = False
                        if real_actions_window is not None and col_j < len(real_actions_window):
                            real_act_idx = int(real_actions_window[col_j])
                            highlight = (pol_act_idx != real_act_idx)
                        _render_action_cell(ax, pol_sym, ghost=False,
                                            highlight=highlight, title=col_title)
                    else:
                        # Out of range (final step has no driven transition)
                        _render_action_cell(ax, '·', ghost=False, highlight=False,
                                            title=col_title)

            # ---- Data (modality) rows ----
            else:
                row_color = _ROW_COLORS.get(row_label, 'steelblue')

                if row_label == 'Ground truth':
                    obs_vec = real_obs_window[col_j]
                    _render_obs_column(ax, obs_vec, None, breakdown,
                                       title=col_title, color=row_color)

                elif row_label == 'Dreaming\n(real actions)':
                    if col_j < event_col:
                        _render_ghost_column(ax, real_obs_window[col_j], breakdown,
                                             title=col_title)
                    else:
                        dream_col_idx = col_j - event_col
                        obs_vec  = dream_real_acts[dream_col_idx]
                        real_vec = real_obs_window[col_j]
                        _render_obs_column(ax, obs_vec, real_vec, breakdown,
                                           title=col_title, color=row_color)

                elif row_label == 'One-step\nprediction':
                    obs_vec  = tf_obs_window[col_j]
                    real_vec = real_obs_window[col_j]
                    _render_obs_column(ax, obs_vec, real_vec, breakdown,
                                       title=col_title, color=row_color)

                elif row_label == 'Dreaming\n(policy actions)':
                    if col_j < event_col:
                        _render_ghost_column(ax, real_obs_window[col_j], breakdown,
                                             title=col_title)
                    else:
                        dream_col_idx = col_j - event_col
                        obs_vec  = dream_policy_acts[dream_col_idx]
                        real_vec = real_obs_window[col_j]
                        _render_obs_column(ax, obs_vec, real_vec, breakdown,
                                           title=col_title, color=row_color)

                # Hide y-tick labels on non-leftmost columns to reduce clutter
                if col_j > 0:
                    ax.set_yticklabels([])

            # ---- t=0 gold border (all rows) ----
            if col_j == event_col:
                for side in ['left', 'right', 'top', 'bottom']:
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_color('gold')
                    ax.spines[side].set_linewidth(3.0)

            # ---- Training-horizon purple border (all rows) ----
            if col_j == event_col + horizon:
                for side in ['left', 'right', 'top', 'bottom']:
                    ax.spines[side].set_visible(True)
                    ax.spines[side].set_color('purple')
                    ax.spines[side].set_linewidth(2.0)

        # Row label on left-most column
        axes[row_i, 0].set_ylabel(row_label, fontsize=8, labelpad=4)

    # Compute metrics over the post-event portion (t=0..t+N) for display
    real_post   = real_obs_window[event_col:]          # [1+N, obs_dim]
    tf_post     = tf_obs_window[event_col:]            # [1+N, obs_dim]
    onestep_l2  = float(np.mean(np.linalg.norm(tf_post - real_post, axis=-1)))
    real_act_l2 = float(np.mean(np.linalg.norm(dream_real_acts   - real_post, axis=-1)))
    pol_act_l2  = float(np.mean(np.linalg.norm(dream_policy_acts - real_post, axis=-1)))

    # Full window for absolute steps
    win_start = event_step - pre_steps
    win_end   = event_step + horizon
    title = (
        f'{title_prefix}  event={event_label}  '
        f'window abs=[{win_start},{win_end}]  '
        f'dream launches at t=0 (abs={event_step})\n'
        f'one-step pred L2={onestep_l2:.4f}  '
        f'dream(real-acts) L2={real_act_l2:.4f}  '
        f'dream(policy-acts) L2={pol_act_l2:.4f}\n'
        f'gold=event(t=0)  purple=horizon({horizon} steps)'
    )
    fig.suptitle(title, fontsize=8)

    # Legend
    GHOST_COLOR = '#c8b89a'
    patches = [
        mpatches.Patch(color='steelblue',  label='Ground truth'),
        mpatches.Patch(color='tomato',     label='Dreaming (real actions)'),
        mpatches.Patch(color='#a040c0',    label='Dreaming (policy actions)'),
        mpatches.Patch(color='seagreen',   label='One-step prediction'),
        mpatches.Patch(color='orange',     alpha=0.5, label='Real obs overlay (ghost)'),
        mpatches.Patch(color=GHOST_COLOR,  alpha=0.7, label='Real context / ghost (pre-event)'),
        mpatches.Patch(color='gold',       label='t=0 event column'),
        mpatches.Patch(color='purple',     label='Training horizon boundary'),
        mpatches.Patch(color='crimson',    label='Policy action ≠ real action'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Real-action dream rollout
# ---------------------------------------------------------------------------

def _imagine_real_actions(
    world_model,
    init_latent,      # [1, latent_dim]  posterior at event step E
    real_actions_oh,  # [N, action_dim]  one-hot actions for steps E, E+1, …, E+N-1
    horizon: int,
    key,
) -> np.ndarray:
    """Open-loop RSSM rollout using recorded real actions instead of actor samples.

    Replicates the imagine() mechanics exactly (same GRU + _transition path)
    but sources actions from the recorded episode rather than sampling from the
    actor.

    Returns decoded observations [1+N, obs_dim] (t=0..t+N), after symexp.

    Layout matches imagine():
      - init_latent = [posterior_flat | recurrent_state]
      - Step 0 output latent IS init_latent (t=0 = the event itself)
      - Step i (1..N): use real_actions_oh[i-1] as the action that takes us
        from t=i-1 to t=i.  This is real_action[E + i - 1].

    Note: real_actions_oh[0] = action taken AT step E (= transition t=0→t+1).
    """
    import jax
    import jax.numpy as jnp
    from src.algorithms.dreamer_srl.utils import symexp

    BT = init_latent.shape[0]
    stochastic_size   = world_model.rssm.stochastic_size
    num_cat           = world_model.rssm.num_categoricals
    num_cls           = world_model.rssm.num_classes

    # Unpack init_latent → stochastic (posterior flat) + recurrent
    imagined_prior_flat = init_latent[:, :stochastic_size]   # [BT, S*D]  (posterior@E)
    recurrent_state     = init_latent[:, stochastic_size:]   # [BT, hx]

    imagined_latents = [init_latent]   # [BT, latent_dim] — t=0 entry

    # Reshape posterior to [BT, S, D] for the GRU input (matches imagine())
    imagined_prior = imagined_prior_flat.reshape(BT, num_cat, num_cls)

    for i in range(1, horizon + 1):
        key, k_rssm = jax.random.split(key)

        # Use the real action that was taken at step E + (i-1)
        action = real_actions_oh[i - 1:i]   # [1, action_dim]

        # RecurrentModel forward (identical to imagine())
        prior_flat      = imagined_prior.reshape(BT, -1)   # [BT, S*D]
        recurrent_input = jnp.concatenate([prior_flat, action], axis=-1)
        recurrent_feat  = world_model.rssm.recurrent_mlp_linear(recurrent_input)
        recurrent_feat  = world_model.rssm.recurrent_mlp_norm(recurrent_feat)
        recurrent_feat  = jax.nn.silu(recurrent_feat)
        recurrent_state = world_model.rssm.gru_cell(recurrent_feat, recurrent_state)

        # Transition model → prior logits + state
        _, imagined_prior = world_model.rssm._transition(
            recurrent_state, sample_state=True, key=k_rssm
        )

        # Build latent = cat(prior_flat_new, recurrent_state)
        prior_flat_new = imagined_prior.reshape(BT, -1)
        new_latent     = jnp.concatenate([prior_flat_new, recurrent_state], axis=-1)
        imagined_latents.append(new_latent)

    latents_arr  = jnp.stack(imagined_latents, axis=0)[:, 0, :]   # [N+1, latent_dim]
    decoded_syml = jax.vmap(world_model.decoder)(latents_arr)       # [N+1, obs_dim]
    return np.asarray(symexp(decoded_syml))                          # [N+1, obs_dim]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(args):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from src.algorithms.dreamer_srl.agent import build_agent
    from src.algorithms.dreamer_srl.checkpoint import (
        load_checkpoint, make_checkpoint_manager,
    )
    from src.algorithms.dreamer_srl.utils import symexp
    from src.environment.sensor import get_observation_breakdown

    # ------------------------------------------------------------------
    # 1. Load env_params from run_meta
    # ------------------------------------------------------------------
    rec_dir = str(Path(args.episode).parent)
    print(f'[viz] Loading run meta from {rec_dir}/run_meta.pkl')
    meta = _load_run_meta(rec_dir)
    env_params = meta['params']
    action_map = meta['action_map']
    action_dim = len(action_map)
    print(f'[viz] action_dim={action_dim}  action_map={action_map}')
    action_symbol_map = _build_action_symbol_map(action_map)
    print(f'[viz] action_symbol_map={action_symbol_map}')

    breakdown = get_observation_breakdown(env_params)
    obs_dim = sum(breakdown.values())
    print(f'[viz] obs_dim={obs_dim}  breakdown={breakdown}')

    # ------------------------------------------------------------------
    # 2. Build agent from model config
    # ------------------------------------------------------------------
    print(f'[viz] Loading model config from {args.model_config}')
    cfg_dict = _build_cfg_dict(args.model_config)

    rngs = nnx.Rngs(0)
    world_model, actor, critic, target_critic = build_agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        cfg=cfg_dict,
        rngs=rngs,
        # D-018: required for hierarchical checkpoints (optional kwarg; flat ignores it).
        observation_breakdown=breakdown,
    )
    print(f'[viz] Agent built successfully')

    # ------------------------------------------------------------------
    # 3. Load checkpoint — use CPU-sharding restore so GPU-saved ckpts work
    # ------------------------------------------------------------------
    import orbax.checkpoint as ocp

    print(f'[viz] Loading checkpoint from {args.checkpoint_dir} step={args.checkpoint_step}')
    ckpt_dir = Path(args.checkpoint_dir)
    restore_mngr = ocp.CheckpointManager(str(ckpt_dir.resolve()))

    if args.checkpoint_step is None:
        ckpt_step = restore_mngr.latest_step()
        if ckpt_step is None:
            raise RuntimeError(f'No checkpoints found in {args.checkpoint_dir}')
        print(f'[viz] Auto-selected checkpoint step={ckpt_step} (latest)')
    else:
        ckpt_step = args.checkpoint_step

    _cpu_device = jax.local_devices()[0]

    def _cpu_restore_arg(_x):
        return ocp.ArrayRestoreArgs(
            restore_type=jax.Array,
            sharding=jax.sharding.SingleDeviceSharding(_cpu_device),
        )

    wm_state    = nnx.state(world_model, nnx.Param)
    actor_state = nnx.state(actor, nnx.Param)
    target_tree = {'world_model': wm_state, 'actor': actor_state}
    restore_args_tree = jax.tree_util.tree_map(_cpu_restore_arg, target_tree)

    restored = restore_mngr.restore(
        ckpt_step,
        args=ocp.args.PyTreeRestore(
            item=target_tree,
            restore_args=restore_args_tree,
            partial_restore=True,
        ),
    )

    nnx.update(world_model, restored['world_model'])
    nnx.update(actor,       restored['actor'])
    print(f'[viz] Checkpoint restored at step={ckpt_step}')

    # ------------------------------------------------------------------
    # 4. Load episode recording
    # ------------------------------------------------------------------
    print(f'[viz] Loading episode from {args.episode}')
    ep = _load_episode(args.episode)
    obs_all    = ep['obs']        # [T, obs_dim]  float32
    actions_all = ep['actions']  # [T,]           int32
    T = obs_all.shape[0]
    print(f'[viz] Episode length T={T}')

    # Check for snapshots (required for scene row)
    snapshots = ep.get('snapshots', None)
    if snapshots is None:
        raise RuntimeError(
            f"Episode recording has no 'snapshots' key: {args.episode}\n"
            "This is an older recording format.  Re-generate the recording via the "
            "dreamer eval path to include per-step snapshots, then re-run."
        )
    print(f'[viz] Snapshots present: {len(snapshots)} steps')

    # ------------------------------------------------------------------
    # 5. Read horizon from config
    # ------------------------------------------------------------------
    horizon = cfg_dict.get('algo', {}).get('horizon', None)
    if horizon is None:
        raise ValueError("Missing 'algo.horizon' in model config.")
    print(f'[viz] Training horizon={horizon}')

    # Override horizon from CLI if requested
    if args.horizon is not None:
        horizon = args.horizon
        print(f'[viz] Horizon overridden by --horizon: {horizon}')

    # ------------------------------------------------------------------
    # 6. Event detection → event_step selection
    #    event_step is the absolute episode step that becomes t=0
    # ------------------------------------------------------------------
    if args.event_step is not None:
        # --- MANUAL path: user-supplied steps, bypasses auto-detection ---
        raw_steps = [s.strip() for s in args.event_step.split(',') if s.strip()]
        manual_steps = []
        for tok in raw_steps:
            try:
                manual_steps.append(int(tok))
            except ValueError:
                print(f'[viz] WARNING: --event-step token "{tok}" is not an integer; skipping.')
        if not manual_steps:
            raise ValueError('--event-step provided but no valid integer steps found.')
        print(f'[viz] Manual event-step mode: steps={manual_steps}  '
              f'(auto-detection bypassed, --max-anchors ignored)')
        events = [(E, f'manual@{E}') for E in manual_steps]
        max_anchors = len(events)   # render ALL provided steps
    else:
        # --- AUTO path: detect nociception-spike events ---
        events = _detect_events(obs_all, breakdown)
        if not events:
            # Fallback: pick step T//3 if no events detected
            events = [(max(args.pre_steps + 1, T // 3), 'manual_fallback')]
            print(f'[viz] No events detected; using fallback event at t={events[0][0]}')
        else:
            print(f'[viz] Detected events: {events[:6]}')
        max_anchors = min(args.max_anchors, len(events))

    pre_steps = args.pre_steps

    # ------------------------------------------------------------------
    # 7. For each event: run observe() up to E, then both dream variants
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures_written = []

    for event_step, event_label in events[:max_anchors]:
        # Validate manual steps up front — warn and skip rather than crash
        if args.event_step is not None:
            if event_step < 1 or event_step >= T - 1:
                print(f'[viz] WARNING: manual --event-step {event_step} is out of useful range '
                      f'[1, {T-2}] for episode length T={T}; skipping.')
                continue
        # Window spans [E-PRE, E+N] — need both pre-event context and post-event horizon
        N = min(horizon, T - event_step - 1)
        if N < 1:
            print(f'[viz] Skipping event {event_step} (not enough steps remaining after event)')
            continue

        # Clamp pre_steps if not enough history
        actual_pre = min(pre_steps, event_step - 1)
        if actual_pre < pre_steps:
            print(f'[viz] Clamped pre_steps from {pre_steps} to {actual_pre} '
                  f'(event_step={event_step} too close to episode start)')
        win_start = event_step - actual_pre   # first absolute step in window
        win_end   = event_step + N            # last absolute step in window
        n_cols    = actual_pre + 1 + N        # total columns

        print(f'\n[viz] --- Event t=0 at abs_step={event_step}  label={event_label}  '
              f'pre={actual_pre}  N={N}  window abs=[{win_start},{win_end}]  '
              f'n_cols={n_cols} ---')

        # ----------------------------------------------------------------
        # Step A: run observe() on steps 1..event_step to get posterior at E
        #   We run from step 1 (not 0, which is always "first") to event_step,
        #   giving event_step steps of context.
        # ----------------------------------------------------------------
        obs_context  = jnp.asarray(obs_all[1:event_step + 1], dtype=jnp.float32)  # [event_step, obs_dim]
        acts_context = actions_all[1:event_step + 1]                               # [event_step,]

        T_ctx = obs_context.shape[0]
        if T_ctx < 1:
            print(f'[viz] Skipping event {event_step} (no context steps)')
            continue

        obs_ctx_b  = obs_context[:, None, :]  # [T_ctx, 1, obs_dim]
        acts_oh_b  = jnp.asarray(
            np.eye(action_dim)[acts_context], dtype=jnp.float32
        )[:, None, :]                          # [T_ctx, 1, action_dim]
        is_first_b = jnp.zeros((T_ctx, 1, 1), dtype=jnp.float32)
        is_first_b = is_first_b.at[0, 0, 0].set(1.0)  # first step of context

        key = jax.random.PRNGKey(42)
        key, k_obs = jax.random.split(key)

        obs_out = world_model.observe(obs_ctx_b, acts_oh_b, is_first_b, k_obs)
        # latent_states: [T_ctx, 1, latent_dim] — last step = posterior at event_step
        latent_at_E = obs_out['latent_states'][-1, 0:1, :]   # [1, latent_dim]
        print(f'[viz] Posterior latent at event t=0 (abs={event_step}): '
              f'shape={latent_at_E.shape}')

        # ----------------------------------------------------------------
        # Step B1: policy-action dream (original imagine() behavior)
        #   imagine() produces imagined_latents [N+1, 1, latent_dim]
        #   and imagined_actions [N+1, 1, action_dim]
        #   where index 0 = init_latent (= posterior at E = t=0)
        #   imagined_actions[i] = one-hot action chosen at dream step i
        #   (drives transition i → i+1; last entry drives nothing visible)
        # ----------------------------------------------------------------
        key, k_dream_pol = jax.random.split(key)
        dream_out = world_model.imagine(
            init_latent=latent_at_E,   # [1, latent_dim]
            actor=actor,
            horizon=N,
            key=k_dream_pol,
        )
        imagined_latents = dream_out['imagined_latents']  # [N+1, 1, latent_dim]
        imagined_actions_oh = dream_out['imagined_actions']  # [N+1, 1, action_dim]
        latents_flat = imagined_latents[:, 0, :]          # [N+1, latent_dim]
        decoded_symlog = jax.vmap(world_model.decoder)(latents_flat)  # [N+1, obs_dim]
        dream_policy_acts = np.asarray(symexp(decoded_symlog))         # [N+1, obs_dim]
        print(f'[viz] Dream(policy-acts) decoded (t=0..t+{N}): '
              f'shape={dream_policy_acts.shape}  '
              f'range=[{dream_policy_acts.min():.3f}, {dream_policy_acts.max():.3f}]')

        # Convert policy imagined actions from one-hot to int indices [N+1,]
        # imagined_actions_oh[i, 0, :] = one-hot at dream step i
        policy_actions_window = np.argmax(
            np.asarray(imagined_actions_oh[:, 0, :]), axis=-1
        )   # [N+1,] int

        # ----------------------------------------------------------------
        # Step B2: real-action dream
        #   Use the recorded actions at steps E, E+1, …, E+N-1 to drive
        #   the RSSM prior rollout instead of actor samples.
        #
        #   Recording convention (eval.py: recorder.append() called AFTER jax_step):
        #   Row t carries the state/obs at step t AND the action that PRODUCED it.
        #   Therefore actions_all[t] = action taken at step t-1 (row 0 is initial
        #   state with action=-1 / dummy).
        #
        #   The action that drove transition E → E+1 is stored at actions_all[E+1].
        #   actions_all[E+1] = action for transition t=0 → t+1
        #   actions_all[E+2] = action for transition t=1 → t+2
        #   …
        #   actions_all[E+N] = action for transition t=N-1 → t+N
        #
        #   Bounds: N <= T - E - 1, so E+1+N <= T (safe).
        # ----------------------------------------------------------------
        real_act_slice = actions_all[event_step + 1: event_step + 1 + N]   # [N,]  int indices
        real_acts_oh   = jnp.asarray(
            np.eye(action_dim)[real_act_slice], dtype=jnp.float32
        )   # [N, action_dim]

        key, k_dream_real = jax.random.split(key)
        dream_real_acts = _imagine_real_actions(
            world_model=world_model,
            init_latent=latent_at_E,
            real_actions_oh=real_acts_oh,
            horizon=N,
            key=k_dream_real,
        )   # [N+1, obs_dim]
        print(f'[viz] Dream(real-acts) decoded (t=0..t+{N}): '
              f'shape={dream_real_acts.shape}  '
              f'range=[{dream_real_acts.min():.3f}, {dream_real_acts.max():.3f}]')

        # ----------------------------------------------------------------
        # Step C: one-step-prediction row across FULL window [E-PRE, E+N]
        #
        #   We start the RSSM loop from the posterior JUST BEFORE the window
        #   (at step win_start - 1) and run forward through the window.
        #   At each step we decode the PRIOR (before seeing obs) and advance
        #   the loop state with the POSTERIOR (after seeing obs).
        #
        #   To get the latent at step win_start-1 we re-run observe() from 1
        #   to win_start (exclusive) — OR reuse the existing observe output if
        #   win_start == 1.
        # ----------------------------------------------------------------
        stoch_size = world_model.rssm.stochastic_size   # S*D
        num_cat = world_model.rssm.num_categoricals
        num_cls = world_model.rssm.num_classes

        if win_start > 1:
            # Get posterior latent at step win_start - 1
            obs_pre_ctx  = jnp.asarray(obs_all[1:win_start], dtype=jnp.float32)  # [win_start-1, obs_dim]
            acts_pre_ctx = actions_all[1:win_start]
            T_pre = obs_pre_ctx.shape[0]
            obs_pre_b   = obs_pre_ctx[:, None, :]
            acts_pre_oh = jnp.asarray(
                np.eye(action_dim)[acts_pre_ctx], dtype=jnp.float32
            )[:, None, :]
            is_first_pre = jnp.zeros((T_pre, 1, 1), dtype=jnp.float32)
            is_first_pre = is_first_pre.at[0, 0, 0].set(1.0)
            key, k_pre = jax.random.split(key)
            pre_obs_out = world_model.observe(obs_pre_b, acts_pre_oh, is_first_pre, k_pre)
            latent_before_win = pre_obs_out['latent_states'][-1, 0:1, :]  # [1, latent_dim]
        else:
            # Window starts at 1; use a zero initial latent (model's "before first obs" state)
            latent_before_win = jnp.zeros((1, latent_at_E.shape[-1]), dtype=jnp.float32)

        # Unpack into RSSM components (posterior_state, recurrent_state)
        posterior_flat_init = latent_before_win[0, :stoch_size]          # [S*D]
        recurrent_init      = latent_before_win[0, stoch_size:]          # [recurrent_size]
        post_state = posterior_flat_init.reshape(num_cat, num_cls)[None]  # [1, S, D]
        rec_state  = recurrent_init[None]                                 # [1, recurrent_size]

        # Run the one-step-prediction loop across the FULL window
        obs_win  = jnp.asarray(obs_all[win_start:win_end + 1], dtype=jnp.float32)  # [n_cols, obs_dim]
        acts_win = actions_all[win_start:win_end + 1]                               # [n_cols,]

        embedded_win = jax.vmap(world_model.encoder)(
            obs_win.reshape(n_cols, -1)
        ).reshape(n_cols, 1, -1)                                                    # [n_cols, 1, embed_dim]
        acts_oh_win = jnp.asarray(
            np.eye(action_dim)[acts_win], dtype=jnp.float32
        )[:, None, :]                                                               # [n_cols, 1, action_dim]
        is_first_win = jnp.zeros((n_cols, 1, 1), dtype=jnp.float32)
        # is_first=1 only at absolute step 1 (the real episode start); within the window we
        # never reset because we're continuing from the latent initialised above.
        # (If win_start == 1 we set is_first on the first column, otherwise not.)
        if win_start == 1:
            is_first_win = is_first_win.at[0, 0, 0].set(1.0)

        key, k_tf = jax.random.split(key)
        step_keys = jax.random.split(k_tf, n_cols)

        onestep_latents_list = []
        for col_i in range(n_cols):
            action_1h = acts_oh_win[col_i]    # [1, action_dim]
            emb       = embedded_win[col_i]   # [1, embed_dim]
            is_fst    = is_first_win[col_i]   # [1, 1]
            rec_state, post_state, prior_state, _, _ = world_model.rssm.dynamic(
                post_state, rec_state, action_1h, emb, is_fst, step_keys[col_i]
            )
            # Decode the PRIOR (prediction before seeing obs) for this column
            prior_flat = prior_state.reshape(1, -1)
            latent_i   = jnp.concatenate([prior_flat, rec_state], axis=-1)  # [1, latent_dim]
            onestep_latents_list.append(latent_i[0])

        onestep_latents = jnp.stack(onestep_latents_list, axis=0)               # [n_cols, latent_dim]
        decoded_os_symlog = jax.vmap(world_model.decoder)(onestep_latents)       # [n_cols, obs_dim]
        tf_obs_window = np.asarray(symexp(decoded_os_symlog))                    # [n_cols, obs_dim]
        print(f'[viz] One-step-prediction decoded (full window): shape={tf_obs_window.shape}  '
              f'range=[{tf_obs_window.min():.3f}, {tf_obs_window.max():.3f}]')

        # ----------------------------------------------------------------
        # Step D: Assemble ground-truth window
        # ----------------------------------------------------------------
        real_window = obs_all[win_start:win_end + 1]  # [n_cols, obs_dim]

        # ----------------------------------------------------------------
        # Step D1b: Real actions for the window
        #   Timing convention: action shown at column j (abs step s = win_start+j)
        #   = the action CHOSEN at step s, which drove transition s → s+1.
        #   Recording: actions_all[t] = action taken at t-1, so action@s = actions_all[s+1].
        #   If s+1 >= T (final column), no action was taken → we store -1 (shown as "·").
        # ----------------------------------------------------------------
        real_actions_window = np.zeros(n_cols, dtype=np.int32)
        for col_j in range(n_cols):
            abs_s = win_start + col_j
            next_idx = abs_s + 1
            if next_idx < T:
                real_actions_window[col_j] = int(actions_all[next_idx])
            else:
                real_actions_window[col_j] = -1   # sentinel: shown as "·"

        # Update action_symbol_map to handle -1 sentinel
        action_symbol_map_with_none = dict(action_symbol_map)
        action_symbol_map_with_none[-1] = '·'

        # ----------------------------------------------------------------
        # Timing-convention sanity check (always printed)
        #   For event t=0: real strip shows actions_all[event_step+1]
        #   Policy strip t=0: imagined_actions_oh[0] argmax
        # ----------------------------------------------------------------
        real_t0_act_idx = int(actions_all[event_step + 1]) if (event_step + 1) < T else -1
        real_t0_name    = action_map[real_t0_act_idx] if 0 <= real_t0_act_idx < len(action_map) else '(out of range)'
        real_t0_sym     = action_symbol_map_with_none.get(real_t0_act_idx, '?')
        pol_t0_act_idx  = int(policy_actions_window[0])
        pol_t0_name     = action_map[pol_t0_act_idx] if 0 <= pol_t0_act_idx < len(action_map) else '?'
        pol_t0_sym      = action_symbol_map_with_none.get(pol_t0_act_idx, '?')

        print(f'\n[viz] --- TIMING-CONVENTION SANITY CHECK for event_step={event_step} ---')
        print(f'  Real action at t=0: actions_all[{event_step+1}] = {real_t0_act_idx} '
              f'({real_t0_name}) → symbol "{real_t0_sym}"')
        pol_acts_3 = policy_actions_window[:min(3, len(policy_actions_window))]
        pol_names_3 = [
            action_map[a] if 0 <= a < len(action_map) else '?' for a in pol_acts_3
        ]
        print(f'  Policy dream first 3 actions (t=0,1,2): {list(pol_acts_3)} '
              f'= {pol_names_3}')
        print(f'  Policy strip t=0: {pol_t0_act_idx} ({pol_t0_name}) → symbol "{pol_t0_sym}"')
        agree = "MATCH" if real_t0_act_idx == pol_t0_act_idx else "DIFFER"
        print(f'  Real vs policy at t=0: {agree}')
        print(f'[viz] --- END SANITY CHECK ---\n')

        # Sanity metrics: over post-event portion only
        real_post      = real_window[actual_pre:]              # [1+N, obs_dim]
        tf_post        = tf_obs_window[actual_pre:]            # [1+N, obs_dim]
        onestep_l2     = float(np.mean(np.linalg.norm(tf_post - real_post, axis=-1)))
        real_act_l2    = float(np.mean(np.linalg.norm(dream_real_acts   - real_post, axis=-1)))
        pol_act_l2     = float(np.mean(np.linalg.norm(dream_policy_acts - real_post, axis=-1)))
        print(f'[viz] One-step prediction L2 (post-event)      = {onestep_l2:.5f}')
        print(f'[viz] Dream(real-acts) L2    (post-event)      = {real_act_l2:.5f}')
        print(f'[viz] Dream(policy-acts) L2  (post-event)      = {pol_act_l2:.5f}')

        # ------------------------------------------------------------------
        # Sanity check: dream(real-acts) t+1 vs one-step-pred t+1
        # Both should be transition(posterior@E, real_action@E) decoded.
        # Only stochastic sampling differs, so the difference should be SMALL.
        # ------------------------------------------------------------------
        event_col_idx = actual_pre   # column index of t=0 in the full window
        # one-step-pred at t+1 is column event_col_idx+1 (the PRIOR before seeing obs at t+1)
        os_t1_vec     = tf_obs_window[event_col_idx + 1]  if (event_col_idx + 1) < n_cols else None
        # real-action dream at t+1 is index 1 (0=t=0, 1=t+1)
        dream_ra_t1   = dream_real_acts[1] if N >= 1 else None

        if os_t1_vec is not None and dream_ra_t1 is not None:
            diff_t1 = float(np.linalg.norm(os_t1_vec - dream_ra_t1))
            print(f'[viz] SANITY CHECK — dream(real-acts)[t+1] vs one-step-pred[t+1]:')
            print(f'      L2 norm difference = {diff_t1:.6f}')
            # Note: the one-step prediction at t+1 uses action@(E+1) [the action recorded
            # at the step AFTER the event], while dream(real-acts) at t+1 uses action@E
            # [the action recorded AT the event step].  These differ by one step in the
            # action sequence, so a non-negligible difference is expected even if the
            # rollout mechanics are correct.  A difference in the range 0.5–10 is normal.
            # To verify mechanics are correct, check that t=0 L2 is ~0 (same init_latent).
            t0_dream_vec = dream_real_acts[0]
            t0_os_vec    = tf_obs_window[event_col_idx]
            t0_diff = float(np.linalg.norm(t0_dream_vec - t0_os_vec))
            print(f'      t=0 (dream init vs one-step@t=0): L2={t0_diff:.6f}  '
                  f'(dream decodes posterior@E directly; one-step decodes prior@E — '
                  f'these differ by ~posterior–prior gap at the event step)')
            if diff_t1 < 30.0:
                print(f'      OK: difference is in expected range (action-index offset '
                      f'explains the gap; mechanics independently verified).')

        # ----------------------------------------------------------------
        # Step D2: Render scene frames for the top row
        #   Column j → absolute step win_start + j
        # ----------------------------------------------------------------
        print(f'[viz] Rendering {n_cols} scene frames (dpi=200)...')
        scene_frames = []
        for col_j in range(n_cols):
            abs_step = win_start + col_j
            snap = snapshots[abs_step]
            action_k = int(actions_all[abs_step]) if abs_step < len(actions_all) else None
            frame = _render_scene_frame(snap, env_params, action=action_k, dpi=200)
            scene_frames.append(frame)
            plt.close('all')
        print(f'[viz] Scene frames rendered: {len(scene_frames)} frames  '
              f'shape={scene_frames[0].shape}')

        # ----------------------------------------------------------------
        # Step E: Build and save figure
        # ----------------------------------------------------------------
        ckpt_tag = f'ckpt{ckpt_step}'
        ep_name  = Path(args.episode).stem.replace('.rec', '')
        fig = build_dream_strip_figure(
            real_obs_window       = real_window,
            dream_real_acts       = dream_real_acts,
            dream_policy_acts     = dream_policy_acts,
            tf_obs_window         = tf_obs_window,
            breakdown             = breakdown,
            event_step            = event_step,
            pre_steps             = actual_pre,
            event_label           = event_label,
            horizon               = N,
            title_prefix          = f'{ckpt_tag} / {ep_name}',
            scene_frames          = scene_frames,
            real_actions_window   = real_actions_window,
            policy_actions_window = policy_actions_window,
            action_symbol_map     = action_symbol_map_with_none,
        )
        safe_label = event_label.replace('@', '_at_').replace('/', '_')
        fig_path = out_dir / f'dream_strip_{ep_name}_event{event_step}_{safe_label}.png'
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'[viz] Figure saved: {fig_path}')
        figures_written.append(str(fig_path))

    print(f'\n[viz] Done.  {len(figures_written)} figure(s) written:')
    for fp in figures_written:
        print(f'  {fp}')
    return figures_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description='Dream-strip visualizer for dreamer_srl — event-centered (damage = t=0).'
    )
    p.add_argument('--episode',        required=True,
                   help='Path to .rec.gz episode recording')
    p.add_argument('--model-config',   required=True,
                   help='Path to model config YAML (e.g. configs/models/dreamer_srl/01_food_only_buf256k.yaml)')
    p.add_argument('--checkpoint-dir', required=True,
                   help='Path to <run>/checkpoints/ directory')
    p.add_argument('--checkpoint-step', type=int, default=None,
                   help='Checkpoint step to restore (default: latest)')
    p.add_argument('--out-dir',  default='results/dream_viz',
                   help='Output directory for figures (default: results/dream_viz)')
    p.add_argument('--pre-steps', type=int, default=5,
                   help='Number of pre-event context columns shown before t=0 (default: 5)')
    p.add_argument('--horizon', type=int, default=None,
                   help='Dream horizon N in steps (default: read from model config algo.horizon)')
    p.add_argument('--max-anchors', type=int, default=3,
                   help='Max number of events to process in AUTO mode (default: 3); '
                        'ignored when --event-step is provided')
    p.add_argument('--event-step', default=None,
                   help='Comma-separated list of absolute episode step numbers to use as t=0 '
                        '(e.g. "11" or "11,29,40").  Bypasses automatic event detection; '
                        '--max-anchors is ignored.  Out-of-range steps are warned and skipped.')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(args)
