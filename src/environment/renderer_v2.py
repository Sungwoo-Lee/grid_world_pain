"""
JAX Environment Renderer V2 — Declarative Dashboard Layout.

Key changes vs renderer.py:
- Uses Figure.subfigures() + subplot_mosaic() + layout='constrained'.
  Every panel gets its own axes; overlap is structurally impossible.
- No y_cursor bookkeeping, no fig.text() with hard-coded coords.
- Each vital and sensor pod is drawn by a dedicated function that owns
  its axes, using transAxes (0-1) coords internally.
- Arena + Minimap live in the centre column as separate named axes.
- Slim header banner replaces the Run Context pod.

Public API: render_jax_state_v2(...) — same signature as render_jax_state().
            save_jax_video is re-exported unchanged from renderer.py.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.colors import to_rgb

from src.environment.renderer import (
    _load_icons, COLORS,
    draw_categorical_visual, draw_boresight_diamond,
    save_jax_video,
)
from src.environment.sensor import get_visual_offsets  # noqa: F401 (used by draw_boresight_diamond)

# Additional tokens for v2 (extend the inherited palette)
COLORS.setdefault('card_bg', '#F9FAFB')
COLORS.setdefault('arena_border', '#D1D5DB')

# ── Card helpers ──────────────────────────────────────────────────────────────

def _style_card(ax, facecolor='card_bg', border_color='border'):
    ax.set_facecolor(COLORS[facecolor])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS[border_color])
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])


def _card_title(ax, title, obs_only=False, offline=False):
    color = (COLORS['text_offline'] if (offline or obs_only)
             else COLORS['text_label'])
    suffix = '  (OBS ONLY)' if obs_only else ('  OFFLINE' if offline else '')
    ax.text(0.05, 0.93, (title + suffix).upper(),
            transform=ax.transAxes, color=color,
            fontsize=6.5, fontweight='bold', va='top')


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
    # REAL tick
    if not obs_only:
        tick_x = bar_left + bar_width * min(1.0, real_pct)
        ax.plot([tick_x, tick_x],
                [bar_bottom - 0.06, bar_bottom + bar_height + 0.06],
                color=COLORS['text_main'], linewidth=1.8,
                transform=ax.transAxes, zorder=3,
                solid_capstyle='round')


# ── Drawing functions — one per card type ─────────────────────────────────────

def draw_vital_card(ax, label, real_pct, obs_pct,
                    real_val_str, obs_val_str, color):
    """Interoception vital: label + big OBS value + bar + REAL tick + Δ."""
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
            f"real {real_val_str}  ·  Δ {sign}{delta:.2f}",
            transform=ax.transAxes,
            color=COLORS['text_label'], fontsize=6, va='bottom')


def draw_offline_card(ax, label):
    _style_card(ax)
    _card_title(ax, label, offline=True)
    ax.text(0.5, 0.48, 'OFFLINE',
            transform=ax.transAxes, color=COLORS['text_offline'],
            fontsize=9, fontweight='bold', ha='center', va='center', alpha=0.5)


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
        ax.text(0.05, 0.08, 'obs only — no noise-free reference',
                transform=ax.transAxes,
                color=COLORS['text_offline'], fontsize=5.5, va='bottom')
    else:
        delta = obs_pct - real_pct
        sign = '+' if delta >= 0 else ''
        ax.text(0.05, 0.08,
                f"real {real_val_str}  ·  Δ {sign}{delta:.2f}",
                transform=ax.transAxes,
                color=COLORS['text_label'], fontsize=6, va='bottom')


def draw_spectrum_pod(ax, label, obs_vec, true_vec, obs_only=False):
    """Olfactory 5-channel spectrum bars."""
    _style_card(ax)
    _card_title(ax, label, obs_only=obs_only)

    obs_vec = np.asarray(obs_vec, dtype=float)
    true_vec = np.asarray(true_vec, dtype=float)
    n = len(obs_vec)
    all_vals = np.concatenate([obs_vec, true_vec]) if not obs_only else obs_vec
    max_val = float(np.max(all_vals)) if np.max(all_vals) > 0.01 else 1.0
    scale = 1.0 / max_val

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


def draw_categorical_pod(ax, label, obs_vec, true_vec,
                          r, num_features, labels=None, obs_only=False):
    """Collision / Visual categorical grid, using draw_categorical_visual."""
    _style_card(ax)
    _card_title(ax, label, obs_only=obs_only)
    # Leave top ~16% for title, bottom ~8% for feature labels.
    # draw_categorical_visual handles its own internal margins.
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


def draw_action_pod(ax, action):
    """Small action badge at bottom of right column."""
    _style_card(ax)
    ax.text(0.05, 0.93, 'CURRENT ACTION',
            transform=ax.transAxes, color=COLORS['text_label'],
            fontsize=6.5, fontweight='bold', va='top')

    action_names = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT',
                    4: 'REST', 5: 'EAT'}
    arrows = {0: '↑', 1: '→', 2: '↓', 3: '←', 4: '⊝', 5: '✙'}

    if action is not None:
        act_name = action_names.get(int(action), str(action))
        arrow = arrows.get(int(action), '•')
        ax.text(0.5, 0.66, act_name,
                transform=ax.transAxes, color=COLORS['text_main'],
                fontsize=11, fontweight='black', ha='center', va='center')
        ax.text(0.5, 0.20, arrow,
                transform=ax.transAxes, color=COLORS['action'],
                fontsize=17, fontweight='black', ha='center', va='center')
    else:
        ax.text(0.5, 0.48, '—',
                transform=ax.transAxes, color=COLORS['text_offline'],
                fontsize=12, ha='center', va='center')


# ── Main render entry point ───────────────────────────────────────────────────

def render_jax_state_v2(state, params,
                        episode=None, step=None, train_episode=None,
                        dpi=100, icon_scale=1.0, action=None,
                        sensory_data=None, info=None, icon_config=None):
    """
    Render a JAX EnvState to an RGB numpy array — Dashboard V2.

    Drop-in replacement for render_jax_state(); identical signature.
    """
    icons = _load_icons(icon_config)

    height, width = int(params.height), int(params.width)
    view_size = int(params.local_view_size)
    agent_pos = np.array(state.agent_pos)
    ar, ac = int(agent_pos[0]), int(agent_pos[1])

    half_view = view_size // 2
    r_start = max(0, min(height - view_size, ar - half_view))
    c_start = max(0, min(width - view_size, ac - half_view))
    r_start, c_start = max(0, r_start), max(0, c_start)
    r_end = min(height, r_start + view_size)
    c_end = min(width, c_start + view_size)

    plt.close('all')

    # ── Figure: declarative layout ─────────────────────────────────────
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

    # Left column: 3 vital cards
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

    # ── Header banner ──────────────────────────────────────────────────
    disp_ep = episode if episode is not None else (train_episode or '--')
    ax_hdr.text(0.02, 0.5, 'GridWorld · Homeostasis Dashboard',
                transform=ax_hdr.transAxes,
                color=COLORS['text_main'], fontsize=11, fontweight='bold', va='center')
    ax_hdr.text(0.98, 0.5,
                f'Ep {disp_ep}  ·  {width}×{height}  ·  Step {step or "--"}',
                transform=ax_hdr.transAxes,
                color=COLORS['text_label'], fontsize=9,
                ha='right', va='center', fontfamily='monospace')
    ax_hdr.axhline(0.0, color=COLORS['border'], linewidth=1.0)

    # ── Left panel: vital cards ─────────────────────────────────────────
    sat_real = float(state.satiation) / float(params.max_satiation)
    sat_data = sensor_map.get('Satiation', {})
    sat_obs = float(sat_data.get('intensity', sat_real))
    draw_vital_card(left_axes['satiation'], 'Satiation',
                    sat_real, sat_obs,
                    f'{sat_real:.2f}', f'{sat_obs:.2f}', COLORS['satiation'])

    nut_real = float(state.nutrition) / float(params.max_nutrition)
    nut_data = sensor_map.get('Nutrition', {})
    nut_obs = float(nut_data.get('intensity', nut_real))
    draw_vital_card(left_axes['nutrition'], 'Nutrition',
                    nut_real, nut_obs,
                    f'{nut_real:.2f}', f'{nut_obs:.2f}', COLORS['nutrition'])

    inj_real = float(state.injury_level) / float(params.max_injury)
    inj_data = sensor_map.get('Injury', {})
    inj_obs = float(inj_data.get('intensity', inj_real))
    draw_vital_card(left_axes['injury'], 'Injury',
                    inj_real, inj_obs,
                    f'{inj_real:.2f}', f'{inj_obs:.2f}', COLORS['injury'])

    # ── Centre panel: arena ────────────────────────────────────────────
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
    ax_arena.set_title('ARENA — LOCAL VIEW',
                        fontsize=7, color=COLORS['text_label'],
                        fontweight='bold', pad=4)

    # Grid lines
    for x in range(c_start, c_end + 1):
        ax_arena.vlines(x - 0.5, r_start - 0.5, r_end - 0.5,
                        colors=COLORS['grid'], linewidth=0.5)
    for y in range(r_start, r_end + 1):
        ax_arena.hlines(y - 0.5, c_start - 0.5, c_end - 0.5,
                        colors=COLORS['grid'], linewidth=0.5)

    # Terrain tiles
    location_colors = {0: COLORS['bg'], 1: '#ECFDF5', 2: '#FFFBEB'}
    loc_grid = np.array(params.grid_location_type)
    for row in range(r_start, r_end):
        for col in range(c_start, c_end):
            l_type = int(loc_grid[row, col])
            if l_type != 0:
                ax_arena.add_patch(plt.Rectangle(
                    (col - 0.5, row - 0.5), 1, 1,
                    color=location_colors.get(l_type, COLORS['bg']), zorder=0))

    # Icon helpers
    scale_factor = (4.0 / view_size) * icon_scale

    def draw_icon(r, c, icon_key, zoom=0.038, s_fac=1.0):
        img = icons.get(icon_key)
        if img is not None:
            imagebox = OffsetImage(img, zoom=zoom * s_fac)
            ab = AnnotationBbox(imagebox, (c, r), frameon=False, pad=0)
            ax_arena.add_artist(ab)
        else:
            m_map = {
                'agent': ('o', COLORS['action']),
                'food': ('D', COLORS['food']),
                'hiding_predator': ('X', COLORS['hiding_predator']),
                'predator': ('v', COLORS['predator']),
                'rock': ('s', COLORS['rock']),
                'neutral': ('o', COLORS['neutral']),
            }
            m, clr = m_map.get(icon_key, ('s', 'grey'))
            ax_arena.plot(c, r, marker=m, markersize=12 * s_fac,
                          color=clr, markeredgecolor='white', markeredgewidth=1)

    # Entities
    res_pos = np.array(state.res_pos)
    res_type = np.array(params.res_type)
    res_active = np.array(state.res_active)
    p_pos = np.array(state.pred_pos)
    o_pos = np.array(state.obs_pos)
    obs_types = np.array(params.obs_type)
    obs_hides = (np.array(params.obs_hides_agent)
                 if hasattr(params, 'obs_hides_agent')
                 else np.zeros(o_pos.shape[0], dtype=bool))
    n_pos = np.array(state.neutral_pos)

    at_agent = []

    for i in range(len(res_active)):
        if not res_active[i]:
            continue
        rr, rc = int(res_pos[i, 0]), int(res_pos[i, 1])
        icon = 'food' if res_type[i] == 0 else 'hiding_predator'
        if rr == ar and rc == ac:
            at_agent.append(icon)
        elif r_start <= rr < r_end and c_start <= rc < c_end:
            draw_icon(rr, rc, icon, zoom=0.035, s_fac=scale_factor)

    for i in range(p_pos.shape[0]):
        pr, pc = int(p_pos[i, 0]), int(p_pos[i, 1])
        if pr == ar and pc == ac:
            at_agent.append('predator')
        elif r_start <= pr < r_end and c_start <= pc < c_end:
            draw_icon(pr, pc, 'predator', zoom=0.045, s_fac=scale_factor)

    for i in range(o_pos.shape[0]):
        or_, oc = int(o_pos[i, 0]), int(o_pos[i, 1])
        if or_ == ar and oc == ac:
            if obs_hides[i]:
                at_agent.append('bush')
        elif r_start <= or_ < r_end and c_start <= oc < c_end:
            draw_icon(or_, oc, params.obstacle_names[obs_types[i]],
                      zoom=0.035, s_fac=scale_factor)

    for i in range(n_pos.shape[0]):
        nr, nc = int(n_pos[i, 0]), int(n_pos[i, 1])
        if nr == ar and nc == ac:
            at_agent.append('neutral')
        elif r_start <= nr < r_end and c_start <= nc < c_end:
            draw_icon(nr, nc, 'neutral', zoom=0.035, s_fac=scale_factor)

    agent_icon = 'agent'
    if 'predator' in at_agent:
        agent_icon = 'agent_predator'
    elif 'hiding_predator' in at_agent:
        agent_icon = 'agent_hiding_predator'
    elif 'bush' in at_agent:
        agent_icon = 'agent_bush'
    elif 'food' in at_agent:
        agent_icon = 'agent_food'
    draw_icon(ar, ac, agent_icon, zoom=0.035, s_fac=scale_factor)

    # ── Centre panel: minimap ──────────────────────────────────────────
    ax_mini = center_axes['minimap']
    _style_card(ax_mini, facecolor='card_bg')
    ax_mini.set_xlim(-0.5, width - 0.5)
    ax_mini.set_ylim(-0.5, height - 0.5)
    ax_mini.invert_yaxis()
    ax_mini.set_aspect('equal')
    ax_mini.set_title('MINIMAP',
                       fontsize=6, color=COLORS['text_label'],
                       fontweight='bold', pad=3)

    minimap_img = np.ones((height, width, 3))
    for l_id, color_hex in location_colors.items():
        if l_id != 0:
            minimap_img[loc_grid == l_id] = to_rgb(color_hex)
    ax_mini.imshow(minimap_img,
                   extent=(-0.5, width - 0.5, height - 0.5, -0.5),
                   zorder=0, alpha=0.5)

    def plot_dots(positions, mask, color, s=3):
        valid = positions[mask]
        if len(valid) > 0:
            ax_mini.scatter(valid[:, 1], valid[:, 0],
                            s=s, color=color, edgecolors='none',
                            alpha=0.9, zorder=2)

    plot_dots(res_pos, res_active & (res_type == 0), COLORS['food'])
    plot_dots(res_pos, res_active & (res_type == 1), COLORS['hiding_predator'])
    ax_mini.scatter(p_pos[:, 1], p_pos[:, 0], s=3,
                    color=COLORS['predator'], edgecolors='none', alpha=0.9, zorder=2)
    ax_mini.scatter(o_pos[:, 1], o_pos[:, 0], s=2,
                    color=COLORS['rock'], edgecolors='none', alpha=0.6, zorder=2)
    ax_mini.scatter(n_pos[:, 1], n_pos[:, 0], s=3,
                    color=COLORS['neutral'], edgecolors='none', alpha=0.9, zorder=2)

    ax_mini.add_patch(plt.Rectangle(
        (c_start - 0.5, r_start - 0.5), view_size, view_size,
        fill=False, edgecolor=COLORS['action'],
        linewidth=1.0, alpha=0.8, zorder=3))
    ax_mini.plot(ac, ar, 'o', color=COLORS['action'], markersize=4,
                 markeredgecolor='white', markeredgewidth=0.5, zorder=4)

    # ── Right panel: sensor pods ───────────────────────────────────────
    pod_map = [
        ('olfactory',    'Olfactory'),
        ('nociception',  'Extero Nociception'),
        ('collision',    'Collision'),
        ('visual',       'Visual'),
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
            obs_only = (ti is None
                        or abs(float(ti) - float(s_data['intensity'])) < 1e-6)
            obs_v = float(s_data['intensity'])
            true_v = float(s_data.get('true_intensity', obs_v))
            draw_intensity_pod(ax, s_name,
                               min(1.0, obs_v), min(1.0, true_v),
                               f'{obs_v:.2f}', f'{true_v:.2f}',
                               obs_only=obs_only)

        elif s_type == 'spectrum':
            obs_vec = np.asarray(s_data['vector'])
            tv = s_data.get('true_vector')
            true_vec = np.asarray(tv) if tv is not None else obs_vec
            obs_only = tv is None or np.allclose(obs_vec, true_vec, atol=1e-6)
            draw_spectrum_pod(ax, s_name, obs_vec, true_vec, obs_only=obs_only)

        elif s_type in ('diamond', 'visual_grid'):
            obs_vec = np.asarray(s_data['vector'])
            tv = s_data.get('true_vector')
            true_vec = np.asarray(tv) if tv is not None else obs_vec
            obs_only = tv is None or np.allclose(obs_vec, true_vec, atol=1e-6)
            r_range = int(s_data.get('range', 1))
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

    # ── Render to numpy ────────────────────────────────────────────────
    canvas = FigureCanvas(fig)
    canvas.draw()
    buf, (w, h) = canvas.print_to_buffer()
    image = np.frombuffer(buf, dtype='uint8').reshape((int(h), int(w), 4))
    plt.close(fig)
    return image[:, :, :3]
