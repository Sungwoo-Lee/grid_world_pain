"""
JAX Environment Renderer.

Provides rendering utilities that convert JAX EnvState to RGB frames for visualization.
Works with the JAX-native environment without requiring the PyTorch GridWorld class.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
from io import BytesIO
from PIL import Image

try:
    import cairosvg
    CAIROSVG_AVAILABLE = True
except ImportError:
    CAIROSVG_AVAILABLE = False

# Imports for JAX EnvState
import jax.numpy as jnp

# Global cache for icons to avoid reloading every frame
_ICON_CACHE = None

def _load_icons(icon_config=None):
    """Loads icons from assets directory based on config."""
    global _ICON_CACHE
    if _ICON_CACHE is not None:
        return _ICON_CACHE
        
    # Calculate assets path relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assets_path = os.path.join(base_dir, 'assets')
    
    # Default icons mapping if not provided
    if icon_config is None:
        icon_config = {
            'agent': 'agent',
            'food': 'food',
            'danger': 'danger',
            'predator': 'predator',
            'agent_food': 'agent_food',
            'agent_danger': 'agent_danger',
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
                            # Convert SVG to PNG in memory then to numpy array
                            png_data = cairosvg.svg2png(url=path)
                            icons[key] = np.array(Image.open(BytesIO(png_data)))
                            found = True
                        else:
                            # print(f"Warning: cairosvg not available for {path}")
                            pass
                    else:
                        # Use PIL to ensure alpha channels (transparency) are correctly handled for all formats
                        img = Image.open(path)
                        if img.mode != 'RGBA':
                             img = img.convert('RGBA')
                        icons[key] = np.array(img)
                        found = True
                    
                    if found:
                        break
                except Exception as e:
                    # print(f"Error loading icon {path}: {e}")
                    pass
        
        if not found:
            # print(f"Warning: Icon not found for key '{key}' with extensions {supported_extensions}")
            icons[key] = None
            
    _ICON_CACHE = icons
    return icons


# Global cache for figure, axes, and canvas to avoid recreating them every frame
_FIG_CACHE = None

# Professional Color Palette (Industrial White / Technical)
# Designed for high-contrast white backgrounds (Presentations)
COLORS = {
    'bg': '#FFFFFF',
    'pod_bg': '#F9FAFB',
    'border': '#E5E7EB',
    'text_main': '#1F2937',
    'text_label': '#6B7280',
    'text_offline': '#9CA3AF',
    'grid': '#F3F4F6',
    
    # Technical Tones
    'satiation': '#0D9488',   # Teal
    'nutrition': '#D97706',   # Amber
    'injury': '#BE123C',      # Ruby/Crimson
    'mod': '#7C3AED',         # Violet
    'action': '#2563EB',      # Cobalt
    
    # Entity Tones (Muted but distinct)
    'food': '#059669',
    'danger': '#DC2626',
    'predator': '#111827',
    'rock': '#4B5563',
    'neutral': '#0891B2',
    
    # Damage Segments
    'dmg_danger': '#EF4444',
    'dmg_predator': '#111827',
    'dmg_obstacle': '#6B7280'
}

def draw_dual_capsule_bar(ax, x, y, w, h, state_pct, obs_pct, color, label=None, state_val=None, obs_val=None, transform=None):
    """Draws a professional capsule-style progress bar showing reality vs perception."""
    # Background (Trough)
    bg_rect = matplotlib.patches.FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={h/2}", 
        facecolor='#F3F4F6', edgecolor='none', transform=transform, zorder=0
    )
    ax.add_patch(bg_rect)
    
    # State (Reality) - Thicker, slightly translucent base
    if state_pct > 0:
        val_w = max(h, w * state_pct)
        st_rect = matplotlib.patches.FancyBboxPatch(
            (x, y), val_w, h, boxstyle=f"round,pad=0,rounding_size={h/2}", 
            facecolor=color, edgecolor='none', alpha=0.3, transform=transform, zorder=1
        )
        ax.add_patch(st_rect)
        
    # Observation (Perception) - Thinner interior bar or Marker
    if obs_pct > 0:
        val_w = max(h*0.6, w * obs_pct)
        obs_rect = matplotlib.patches.FancyBboxPatch(
            (x, y + h*0.2), val_w, h*0.6, boxstyle=f"round,pad=0,rounding_size={h*0.3}", 
            facecolor=color, edgecolor='none', transform=transform, zorder=2
        )
        ax.add_patch(obs_rect)

    # Labels
    if label:
        ax.text(x, y + h + 0.02, label.upper(), color=COLORS['text_label'], 
                fontsize=7, fontweight='bold', transform=transform)
    
    # Value labels (Reality: Bold black, Obs: Smaller Gray)
    if state_val is not None:
        ax.text(x + w, y + h + 0.02, f"REAL: {state_val}", color=COLORS['text_main'], 
                fontsize=7, fontweight='bold', ha='right', transform=transform, fontfamily='monospace')
    if obs_val is not None:
        ax.text(x + w, y - 0.03, f"OBS:  {obs_val}", color=COLORS['text_label'], 
                fontsize=6, fontweight='bold', ha='right', transform=transform, fontfamily='monospace')

def draw_pod_frame(ax, x, y, w, h, title, offline=False, transform=None):
    """Draws a modular Telemetry Pod frame."""
    rect = plt.Rectangle((x, y), w, h, facecolor=COLORS['bg'], edgecolor=COLORS['border'], 
                         linewidth=0.5, transform=transform, zorder=0)
    ax.add_patch(rect)
    
    # Accent top border
    border_color = COLORS['border'] if not offline else COLORS['text_offline']
    ax.plot([x, x + w], [y + h, y + h], color=border_color, linewidth=1.5, transform=transform, zorder=1)
    
    # Title
    t_color = COLORS['text_label'] if not offline else COLORS['text_offline']
    ax.text(x + 0.02, y + h + 0.015, title.upper(), color=t_color, 
            fontsize=8, fontweight='bold', transform=transform)
    
    if offline:
        ax.text(x + w/2, y + h/2, "OFFLINE", color=COLORS['text_offline'], 
                ha='center', va='center', fontsize=9, fontweight='bold', 
                alpha=0.5, transform=transform)

from src.environment.sensor import get_visual_offsets

def draw_boresight_diamond(ax, x, y, size, vec, r, num_features, true_vec=None, icons=None, transform=None):
    """
    Draws a schematic Manhattan diamond grid for directional sensors.
    Uses the same offset logic as sensor.py for perfect spatial alignment.
    """
    # Use the unified offset logic from sensor.py
    offsets = np.array(get_visual_offsets(r))
    
    if len(vec) != len(offsets) * num_features:
        return 
        
    obs_grid = np.array(vec).reshape(len(offsets), num_features)
    true_grid = np.array(true_vec).reshape(len(offsets), num_features) if true_vec is not None else None
    
    feature_keys = [
        'grass',    # 0: Grass
        'sand',     # 1: Sand
        'plain',    # 2: Plain
        'food',     # 3: Food
        'danger',   # 4: Danger
        'predator', # 5: Predator
        'rock',     # 6: Rock
        'neutral',  # 7: Neutral Animal
    ]
    
    feature_colors = [
        '#A1DFA1', '#F2D7D5', '#FFFFFF', COLORS['food'], COLORS['danger'], 
        COLORS['predator'], COLORS['rock'], COLORS['neutral']
    ]
    
    for i, (dr, dc) in enumerate(offsets):
        cell_x, cell_y = x + dc * size, y - dr * size
        
        # Cell background
        rect = plt.Rectangle((cell_x - size/2, cell_y - size/2), size, size, 
                             facecolor='none', edgecolor=COLORS['grid'], linewidth=0.3, transform=transform, zorder=1)
        ax.add_patch(rect)
        
        obs_vals = obs_grid[i]
        true_vals = true_grid[i] if true_grid is not None else obs_vals
        
        if num_features == 1:
            # Collision Style: Grid-based indicator
            if true_vals[0] > 0.5: # Reality: Ghosted fill
                ax.add_patch(plt.Rectangle((cell_x - size*0.48, cell_y - size*0.48), size*0.96, size*0.96, 
                             facecolor=COLORS['danger'], alpha=0.15, transform=transform, zorder=2))
            if obs_vals[0] > 0.5: # Perception: Solid block
                ax.add_patch(plt.Rectangle((cell_x - size*0.35, cell_y - size*0.35), size*0.7, size*0.7, 
                             facecolor=COLORS['danger'], alpha=0.8, transform=transform, zorder=3))
        else:
            # Visual Style: Grid-based indicators
            # 1. Draw Reality (Ghosted)
            true_active = np.where(true_vals > 0.1)[0]
            # Prioritize entities (Index 3+) over terrain (0,1,2)
            true_entities = [idx for idx in true_active if idx >= 3]
            
            for feat_idx in true_active:
                icon_key = feature_keys[feat_idx % len(feature_keys)]
                icon_img = icons.get(icon_key) if icons else None
                if icon_img is not None and (feat_idx >= 3 or not true_entities):
                    # Fixed scaling: ensure icon fits within the cell (size * 0.35 zoom)
                    imagebox = OffsetImage(icon_img, zoom=size * 0.35)
                    ab = AnnotationBbox(imagebox, (cell_x, cell_y), frameon=False, pad=0, xycoords=transform if transform else 'data')
                    ab.set_alpha(0.15) 
                    ax.add_artist(ab)
                else:
                    color = feature_colors[feat_idx % len(feature_colors)]
                    ax.add_patch(plt.Rectangle((cell_x - size*0.48, cell_y - size*0.48), size*0.96, size*0.96, 
                                 facecolor=color, alpha=0.1, transform=transform, zorder=2))
            
            # 2. Draw Perception (Solid)
            obs_active = np.where(obs_vals > 0.1)[0]
            obs_entities = [idx for idx in obs_active if idx >= 3]
            
            for feat_idx in obs_active:
                icon_key = feature_keys[feat_idx % len(feature_keys)]
                icon_img = icons.get(icon_key) if icons else None
                
                if icon_img is not None and (feat_idx >= 3 and len(obs_entities) == 1):
                    imagebox = OffsetImage(icon_img, zoom=size * 0.35)
                    ab = AnnotationBbox(imagebox, (cell_x, cell_y), frameon=False, pad=0, xycoords=transform if transform else 'data')
                    ax.add_artist(ab)
                elif feat_idx >= 3:
                    color = feature_colors[feat_idx % len(feature_colors)]
                    ax.add_patch(plt.Rectangle((cell_x - size*0.35, cell_y - size*0.35), size*0.7, size*0.7, 
                                 facecolor=color, alpha=0.9, transform=transform, zorder=4))
    
    # Center markers removed to reduce visual clutter as requested
    pass

def draw_categorical_visual(ax, x, y, w, h, obs_vec, r, num_features, true_vec=None, labels=None, transform=None):
    """Draws a categorical bar chart for visual observations (V7).
    y: bottom of the pod frame
    h: total height of the pod frame
    """
    obs_grid = np.array(obs_vec).reshape(-1, num_features)
    true_grid = np.array(true_vec).reshape(-1, num_features) if true_vec is not None else obs_grid
    num_cells = obs_grid.shape[0]
    
    # Defaults for 8-channel (fallback)
    if labels is None:
        feature_labels = ['GRS', 'SND', 'PLN', 'FOD', 'DNG', 'PRD', 'NEU', 'RCK']
    else:
        feature_labels = labels
        
    feature_colors = [
        '#A1DFA1', '#F2D7D5', '#FFFFFF', COLORS['food'], COLORS['danger'], 
        COLORS['predator'], COLORS['neutral'], COLORS['rock']
    ]
    # Rotate colors if we have more features
    while len(feature_colors) < num_features:
        feature_colors += feature_colors
    # Cell labels for spatial context (Center, Up, Right, Down, Left)
    cell_labels = ['C', 'U', 'R', 'D', 'L'] if num_cells == 5 else [f'C{i}' for i in range(num_cells)]
    if num_cells <= 1: cell_labels = ['']

    # Internal Margins (Percent of pod height h)
    margin_bottom = 0.04 
    margin_top = 0.02
    
    bar_y = y + margin_bottom
    bar_h = h - (margin_bottom + margin_top)

    # Calculate horizontal distribution
    cell_gap_ratio = 1.15
    unit_w = w / (num_cells * num_features + max(num_cells - 1, 0) * cell_gap_ratio)
    
    for c_idx in range(num_cells):
        start_x = x + c_idx * (num_features + cell_gap_ratio) * unit_w
        
        # Group Label for Spatial Orientation
        if num_cells > 1:
            ax.text(start_x + (num_features * unit_w)/2, bar_y + bar_h + 0.002, cell_labels[c_idx], 
                    color=COLORS['text_label'], fontsize=5.5, fontweight='bold', ha='center', transform=transform)
            
        for f_idx in range(num_features):
            bx = start_x + f_idx * unit_w
            bw = unit_w * 0.8
            color = feature_colors[f_idx % len(feature_colors)]
            
            # Ground Truth (Ghosted)
            true_val = float(true_grid[c_idx, f_idx])
            ax.add_patch(plt.Rectangle((bx, bar_y), bw, bar_h * true_val, facecolor=color, alpha=0.15, transform=transform, zorder=1))
            
            # Perception (Solid)
            obs_val = float(obs_grid[c_idx, f_idx])
            ax.add_patch(plt.Rectangle((bx + bw*0.1, bar_y), bw*0.8, bar_h * obs_val, facecolor=color, alpha=0.9, transform=transform, zorder=2))
            
            # Categorical Labels inside frame
            if (num_cells == 1 or c_idx == 0) and f_idx < len(feature_labels):
                ax.text(bx + bw*1.1/2, y + 0.002, feature_labels[f_idx], color=COLORS['text_label'], 
                        fontsize=4.0, ha='center', va='bottom', rotation=90, transform=transform)

def render_jax_state(state, params, episode=None, step=None, train_episode=None, dpi=100, icon_scale=1.0, action=None, sensory_data=None, info=None, icon_config=None):
    """
    Render a JAX EnvState to an RGB numpy array (Industrial White V2).
    """
    from src.environment.core import calculate_drive
    start_time = time.time()
    icons = _load_icons(icon_config)
    global _FIG_CACHE
    
    # Grid dimensions
    height, width = int(params.height), int(params.width)
    view_size = int(params.local_view_size)
    agent_pos = np.array(state.agent_pos)
    ar, ac = int(agent_pos[0]), int(agent_pos[1])
    
    # Calculate local window bounds
    half_view = view_size // 2
    r_start = max(0, min(height - view_size, ar - half_view))
    c_start = max(0, min(width - view_size, ac - half_view))
    r_end, c_end = r_start + view_size, c_start + view_size
    
    # Force creation to ensure V2 aesthetics
    plt.close('all')
    
    # Figure setup: Wide layout for presentation (Now 10in tall for V7)
    fig = plt.figure(figsize=(14, 10), dpi=dpi)
    fig.patch.set_facecolor(COLORS['bg'])
    
    # GridSpec: [Left Telemetry] [Ultimate Arena] [Right Telemetry]
    # Center Arena ratio pushed to 2.2 for true fullscreen feel
    gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 2.2, 0.8])
    
    ax_left = fig.add_subplot(gs[0, 0])      # INTEROCEPTION + Minimap
    ax_grid = fig.add_subplot(gs[0, 1])      # ULTIMATE ARENA (Maximized)
    ax_right = fig.add_subplot(gs[0, 2])     # EXTEROCEPTION + Action
    
    for ax in [ax_left, ax_grid, ax_right]:
        ax.set_facecolor(COLORS['bg'])
        ax.axis('off')
        
    canvas = FigureCanvas(fig)
    _FIG_CACHE = (fig, ax_grid, ax_right, ax_left, canvas)

    # --- 1. Center Arena: Maximized Local View ---
    ax_grid.set_xlim(c_start - 0.5, c_end - 0.5)
    ax_grid.set_ylim(r_start - 0.5, r_end - 0.5)
    ax_grid.invert_yaxis()
    ax_grid.set_aspect('equal')
    ax_grid.axis('off')
    
    # ... (rest of local grid and minimap logic remains same) ...
    
    # -- Local Grid --
    ax_grid.set_xlim(c_start - 0.5, c_end - 0.5)
    ax_grid.set_ylim(r_start - 0.5, r_end - 0.5)
    ax_grid.invert_yaxis()
    ax_grid.set_aspect('equal')
    ax_grid.axis('off')
    
    # Background grid
    for x in range(c_start, c_end + 1):
        ax_grid.vlines(x - 0.5, r_start - 0.5, r_end - 0.5, colors=COLORS['grid'], linewidth=0.5)
    for y in range(r_start, r_end + 1):
        ax_grid.hlines(y - 0.5, c_start - 0.5, c_end - 0.5, colors=COLORS['grid'], linewidth=0.5)
        
    # Location types
    location_colors = {0: COLORS['bg'], 1: '#ECFDF5', 2: '#FFFBEB'}
    loc_grid = np.array(params.grid_location_type)
    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            l_type = int(loc_grid[r, c])
            if l_type != 0:
                ax_grid.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=location_colors.get(l_type, COLORS['bg']), zorder=0))

    # Icon Drawing
    scale_factor = (4.0 / view_size) * icon_scale
    def draw_icon(ax, r, c, icon_key, zoom=0.038, s_fac=1.0, is_axes_coords=False):
        img = icons.get(icon_key)
        if img is not None:
            imagebox = OffsetImage(img, zoom=zoom * s_fac)
            ab = AnnotationBbox(imagebox, (c, r), frameon=False, pad=0, xycoords='data' if not is_axes_coords else ax.transAxes)
            ax.add_artist(ab)
        else:
            m_map = {'agent':('o',COLORS['action']), 'food':('D',COLORS['food']), 'danger':('X',COLORS['danger']), 'predator':('v',COLORS['predator']), 'rock':('s',COLORS['rock']), 'neutral':('o',COLORS['neutral'])}
            m, clr = m_map.get(icon_key, ('s', 'grey'))
            ax.plot(c, r, marker=m, markersize=12*s_fac, color=clr, markeredgecolor='white', markeredgewidth=1, transform=ax.transData if not is_axes_coords else ax.transAxes)

    # Entities
    res_pos, res_type, res_active = np.array(state.res_pos), np.array(params.res_type), np.array(state.res_active)
    # Entities at agent position (for composite visualization)
    at_agent = []

    # Entities
    res_pos, res_type, res_active = np.array(state.res_pos), np.array(params.res_type), np.array(state.res_active)
    for i in range(len(res_active)):
        if not res_active[i]: continue
        rr, rc = int(res_pos[i, 0]), int(res_pos[i, 1])
        if rr == ar and rc == ac:
            at_agent.append('food' if res_type[i] == 0 else 'danger')
        elif r_start <= rr < r_end and c_start <= rc < c_end:
            draw_icon(ax_grid, rr, rc, 'food' if res_type[i] == 0 else 'danger', zoom=0.035, s_fac=scale_factor)
            
    p_pos = np.array(state.pred_pos)
    for i in range(p_pos.shape[0]):
        pr, pc = int(p_pos[i, 0]), int(p_pos[i, 1])
        if pr == ar and pc == ac:
            at_agent.append('predator')
        elif r_start <= pr < r_end and c_start <= pc < c_end:
            draw_icon(ax_grid, pr, pc, 'predator', zoom=0.045, s_fac=scale_factor)
            
    o_pos = np.array(state.obs_pos)
    obs_types = np.array(params.obs_type)
    obs_hides = np.array(params.obs_hides_agent) if hasattr(params, 'obs_hides_agent') else np.zeros(o_pos.shape[0], dtype=bool)
    for i in range(o_pos.shape[0]):
        or_, oc = int(o_pos[i, 0]), int(o_pos[i, 1])
        if or_ == ar and oc == ac:
            obs_icon_name = params.obstacle_names[obs_types[i]]
            if obs_hides[i]:
                at_agent.append('bush')
        elif r_start <= or_ < r_end and c_start <= oc < c_end:
            obs_icon = params.obstacle_names[obs_types[i]]
            draw_icon(ax_grid, or_, oc, obs_icon, zoom=0.035, s_fac=scale_factor)
            
    n_pos = np.array(state.neutral_pos)
    for i in range(n_pos.shape[0]):
        nr, nc = int(n_pos[i, 0]), int(n_pos[i, 1])
        if nr == ar and nc == ac:
            at_agent.append('neutral')
        elif r_start <= nr < r_end and c_start <= nc < c_end:
            draw_icon(ax_grid, nr, nc, 'neutral', zoom=0.035, s_fac=scale_factor)

    # Determine Agent Icon (Normal vs Composite Overlap)
    agent_icon = 'agent'
    if 'predator' in at_agent:
        agent_icon = 'agent_predator'
    elif 'danger' in at_agent:
        agent_icon = 'agent_danger'
    elif 'bush' in at_agent:
        agent_icon = 'agent_bush'
    elif 'food' in at_agent:
        agent_icon = 'agent_food'
    elif 'neutral' in at_agent:
        agent_icon = 'agent'
        
    draw_icon(ax_grid, ar, ac, agent_icon, zoom=0.035, s_fac=scale_factor)
    
    # -- Sidebar Minimap (Integrated into ax_left) --
    # In V4, the minimap moves to the bottom of the left panel
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_minimap = inset_axes(ax_left, width="80%", height="25%", loc='lower center', borderpad=2.2)
    
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
    ax_minimap.imshow(minimap_img, extent=(-0.5, width-0.5, height-0.5, -0.5), zorder=0, alpha=0.5)
    
    # Restoration of Global Entities on Minimap (Smaller dots for sidebar)
    def plot_entity_dots(positions, active_mask, color):
        valid = positions[active_mask]
        if len(valid) > 0:
            ax_minimap.scatter(valid[:, 1], valid[:, 0], s=2.5, color=color, edgecolors='none', alpha=0.8, zorder=2)

    plot_entity_dots(res_pos, res_active & (res_type == 0), COLORS['food'])
    plot_entity_dots(res_pos, res_active & (res_type == 1), COLORS['danger'])
    plot_entity_dots(p_pos, np.ones(p_pos.shape[0], dtype=bool), COLORS['predator'])
    plot_entity_dots(o_pos, np.ones(o_pos.shape[0], dtype=bool), COLORS['rock'])
    plot_entity_dots(n_pos, np.ones(n_pos.shape[0], dtype=bool), COLORS['neutral'])

    ax_minimap.add_patch(plt.Rectangle((c_start-0.5, r_start-0.5), view_size, view_size, fill=False, edgecolor=COLORS['action'], linewidth=0.8, alpha=0.6, zorder=3))
    ax_minimap.plot(ac, ar, 'o', color=COLORS['action'], markersize=3, markeredgecolor='white', markeredgewidth=0.4, zorder=4)
    
    # Title for Minimap in Sidebar (V5.2 Polish)
    ax_left.text(0.5, 0.32, "MINIMAP", color=COLORS['text_label'], fontsize=7, fontweight='bold', ha='center', transform=ax_left.transAxes)

    # --- 2. Left Panel: Vitals (Dual View) ---
    y_ptr = 0.95
    ax_left.text(0.05, y_ptr, "INTEROCEPTION", color=COLORS['text_main'], fontsize=10, fontweight='black', transform=ax_left.transAxes)
    y_ptr -= 0.12
    
    # Known sensor observations for mapping
    sensor_map = {s['name']: s for s in sensory_data} if sensory_data else {}
    
    # Satiation
    sat_real, max_sat = float(state.satiation), float(params.max_satiation)
    sat_obs_data = sensor_map.get('Satiation', {'intensity': sat_real/max_sat})
    sat_obs = float(sat_obs_data.get('intensity', 0))
    draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04, sat_real/max_sat, sat_obs, COLORS['satiation'], 
                          "Satiation", f"{sat_real/max_sat:.2f}", f"{sat_obs:.2f}", transform=ax_left.transAxes)
    y_ptr -= 0.15
    
    # Nutrition
    nut_real, max_nut = float(state.nutrition), float(params.max_nutrition)
    nut_obs_data = sensor_map.get('Nutrition', {'intensity': nut_real/max_nut})
    nut_obs = float(nut_obs_data.get('intensity', 0))
    draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04, nut_real/max_nut, nut_obs, COLORS['nutrition'], 
                          "Nutrition", f"{nut_real/max_nut:.2f}", f"{nut_obs:.2f}", transform=ax_left.transAxes)
    y_ptr -= 0.15
    
    # Injury
    inj_real, max_inj = float(state.injury_level), float(params.max_injury)
    inj_obs_data = sensor_map.get('Injury', {'intensity': inj_real/max_inj})
    inj_obs = float(inj_obs_data.get('intensity', 0))
    draw_dual_capsule_bar(ax_left, 0.05, y_ptr, 0.9, 0.04, inj_real/max_inj, inj_obs, COLORS['injury'], 
                          "Injury", f"{inj_real/max_inj:.2f}", f"{inj_obs:.2f}", transform=ax_left.transAxes)
    y_ptr -= 0.16
    
    draw_pod_frame(ax_left, 0.05, 0.32, 0.9, 0.16, "Run Context", transform=ax_left.transAxes)
    ctxt_y = 0.43
    ax_left.text(0.1, ctxt_y, f"EPISODE:", color=COLORS['text_label'], fontsize=7, transform=ax_left.transAxes)
    # Use sequential episode (defaulting to train_episode if episode is None)
    disp_ep = episode if episode is not None else (train_episode or '--')
    ax_left.text(0.9, ctxt_y, f"{disp_ep}", color=COLORS['text_main'], fontsize=7, fontweight='bold', ha='right', transform=ax_left.transAxes)
    ctxt_y -= 0.04
    ax_left.text(0.1, ctxt_y, f"SCALE:", color=COLORS['text_label'], fontsize=7, transform=ax_left.transAxes)
    ax_left.text(0.9, ctxt_y, f"{width}x{height}", color=COLORS['text_main'], fontsize=7, fontweight='bold', ha='right', transform=ax_left.transAxes)
    ctxt_y -= 0.04
    ax_left.text(0.1, ctxt_y, f"STEP:", color=COLORS['text_label'], fontsize=7, transform=ax_left.transAxes)
    ax_left.text(0.9, ctxt_y, f"{step or '--'}", color=COLORS['text_main'], fontsize=7, fontweight='bold', ha='right', transform=ax_left.transAxes)

    # --- 3. Right Panel: Sensory Telemetry (Observations) ---
    y_cursor = 0.95
    ax_right.text(0.05, y_cursor, "EXTEROCEPTION", color=COLORS['text_main'], fontsize=11, fontweight='black', transform=ax_right.transAxes)
    y_cursor -= 0.05
    pod_h_default = 0.11
    

    known_sensors = ['Olfactory', 'Extero Nociception', 'Collision', 'Visual', 'LOC']
    
    for s_name in known_sensors:
        s_data = sensor_map.get(s_name)
        offline = s_data is None
        
        # Dynamic pod height: Visual/Diamond pods get more room for bars + labels
        if not offline and s_data.get('type') in ('diamond', 'visual_grid'):
            pod_h = 0.20
        else:
            pod_h = pod_h_default
        
        y_frame_bottom = y_cursor - pod_h
        draw_pod_frame(ax_right, 0.05, y_frame_bottom, 0.9, pod_h, s_name, offline=offline, transform=ax_right.transAxes)
        
        if not offline:
            px, py, pw, ph = 0.15, y_frame_bottom + 0.02, 0.7, 0.07
            if s_data['type'] == 'intensity':
                # Dual Capsule Bar for Intensity Sensors (e.g. Nociception)
                true_v = float(s_data.get('true_intensity', s_data['intensity']))
                obs_v = float(s_data['intensity'])
                # Adding numerical labels for research clarity
                draw_dual_capsule_bar(ax_right, px, py, pw, ph, true_v, obs_v, COLORS['action'], 
                                      state_val=f"{true_v:.2f}", obs_val=f"{obs_v:.2f}", transform=ax_right.transAxes)
            elif s_data['type'] == 'spectrum':
                obs_vec = np.array(s_data['vector'])
                true_vec = np.array(s_data.get('true_vector', obs_vec))
                n = len(obs_vec)
                sw = pw / n
                for i in range(n):
                    vx = px + i * sw
                    # Show True (Ghosted)
                    ax_right.add_patch(plt.Rectangle((vx, py), sw*0.8, ph*true_vec[i], color=COLORS['action'], alpha=0.2, transform=ax_right.transAxes))
                    # Show Observed (Solid)
                    ax_right.add_patch(plt.Rectangle((vx, py), sw*0.8, ph*obs_vec[i], color=COLORS['action'], alpha=0.9, transform=ax_right.transAxes))
            elif s_data['type'] == 'diamond' or s_data['type'] == 'visual_grid':
                # V7 Categorical Spectrum Upgrade (Dual View Distribution)
                r = int(s_data.get('range', 1))
                num_features = int(s_data.get('num_features', 1))
                obs_v = np.array(s_data['vector'])
                true_v = np.array(s_data.get('true_vector', obs_v))
                
                # Distribution Plot Geometry
                px, py, pw, ph = 0.1, y_cursor - pod_h + 0.05, 0.8, 0.08
                
                # If range > 1, we fallback to schematic diamond (too many bars)
                # But for research standard r=0,1 we use the categorical spectrum
                if r <= 1:
                    draw_categorical_visual(ax_right, 0.08, y_frame_bottom, 0.84, pod_h, obs_v, r, num_features, 
                                           true_vec=true_v, labels=s_data.get('labels'), transform=ax_right.transAxes)
                else:
                    origin_x, origin_y = 0.5, y_cursor - pod_h/2
                    cell_size = 0.12 / (2*r + 1)
                    draw_boresight_diamond(ax_right, origin_x, origin_y, cell_size, obs_v, r, num_features, 
                                           true_vec=true_v, icons=icons, transform=ax_right.transAxes)
                
                if not np.any(obs_v > 0.1) and not np.any(true_v > 0.1):
                    ax_right.text(0.5, y_cursor - pod_h/2, "NO SIGNALS", color=COLORS['text_offline'], 
                                  fontsize=6, ha='center', transform=ax_right.transAxes)
            elif s_data['type'] == 'text':
                # Text-based display (e.g. coordinates or state labels)
                ax_right.text(0.5, y_cursor - pod_h/2, s_data.get('value_text', '--'), 
                              color=COLORS['text_main'], fontsize=10, fontweight='bold', 
                              ha='center', va='center', transform=ax_right.transAxes, fontfamily='monospace')

        y_cursor -= (pod_h + 0.03)  # 0.03 gap for next pod title
        if y_cursor < 0.20: break # Save space for Action Pod

    # --- 4. Action Pod (Fixed Position bottom floor for V7.3) ---
    action_y = 0.03
    if action is not None:
        draw_pod_frame(ax_right, 0.05, action_y, 0.9, 0.13, "Current Action", transform=ax_right.transAxes)
        action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT", 4: "REST", 5: "EAT"}
        arrows = {0: "↑", 1: "→", 2: "↓", 3: "←", 4: "⊝", 5: "✙"}
        act_name, arrow = action_names.get(int(action), f"{action}"), arrows.get(int(action), "•")
        
        ax_right.text(0.5, action_y + 0.085, act_name, color=COLORS['text_main'], 
                      fontsize=10, fontweight='black', ha='center', transform=ax_right.transAxes)
        ax_right.text(0.5, action_y + 0.025, arrow, color=COLORS['action'], 
                      fontsize=18, fontweight='black', ha='center', transform=ax_right.transAxes)

    canvas.draw()
    s, (w, h) = canvas.print_to_buffer()
    image = np.frombuffer(s, dtype='uint8').reshape((int(h), int(w), 4))
    return image[:, :, :3]
    # print(f"      [Profile] Render took {time.time() - start_time:.4f}s")


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
            imageio.mimsave(output_path, frames, fps=fps)
        finally:
            os.dup2(old_stdout_fd, sys.stdout.fileno())
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)
    else:
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Saved video to {output_path}")
