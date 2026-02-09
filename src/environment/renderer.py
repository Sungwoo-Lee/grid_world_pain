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
                        icons[key] = plt.imread(path)
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

def render_jax_state(state, params, episode=None, step=None, dpi=100, icon_scale=1.0, action=None, sensory_data=None, icon_config=None):
    """
    Render a JAX EnvState to an RGB numpy array.
    
    Args:
        state: JAX EnvState dataclass
        params: JAX EnvParams dataclass
        episode: Optional episode number for display
        step: Optional step number for display
        dpi: Resolution (default 100)
        icon_scale: Scale factor for icons
        action: Optional action index taken
        sensory_data: Optional list of sensory component dicts:
            [{'name': str, 'vector': ndarray, 'color': str, 'type': 'radial'|'spectrum'}, ...]
        icon_config: Optional dict mapping entity keys to filenames
    """
    start_time = time.time()
    icons = _load_icons(icon_config)
    global _FIG_CACHE
    
    # Colors
    bg_color = '#FFFFFF'
    text_color = '#343A40'
    grid_lines = '#DEE2E6'
    agent_color = '#339AF0'
    food_color = '#40C057'
    danger_color = '#FA5252'
    
    # Grid dimensions
    height = int(params.height)
    width = int(params.width)
    view_size = int(params.local_view_size)
    
    # Calculate local window bounds centered at agent
    agent_pos = np.array(state.agent_pos)
    ar, ac = int(agent_pos[0]), int(agent_pos[1])
    
    half_view = view_size // 2
    r_start = max(0, ar - half_view)
    r_end = min(height, r_start + view_size)
    # Adjust r_start if window hit the bottom edge
    r_start = max(0, r_end - view_size)
    
    c_start = max(0, ac - half_view)
    c_end = min(width, c_start + view_size)
    # Adjust c_start if window hit the right edge
    c_start = max(0, c_end - view_size)
    
    # Figure setup
    has_sensory = sensory_data is not None and len(sensory_data) > 0
    
    need_new_fig = False
    if _FIG_CACHE is not None:
        cached_fig, cached_grid, cached_stats, cached_sensory, cached_canvas, cached_minimap = _FIG_CACHE
        if (has_sensory and cached_sensory is None) or (not has_sensory and cached_sensory is not None):
            need_new_fig = True
            plt.close(cached_fig)
    else:
        need_new_fig = True

    if need_new_fig:
        # Layout: [Grid (L)] [Stats (TR)] [Sensory (BR)] [Minimap (BL)]
        # We'll use a 3x2 grid spec
        fig_width = 11 if has_sensory else 9
        fig = plt.figure(figsize=(fig_width, 7), dpi=dpi)
        fig.patch.set_facecolor(bg_color)
        
        gs = fig.add_gridspec(3, 3, width_ratios=[1.8, 0.6, 0.6], height_ratios=[0.4, 0.3, 0.3])
        
        ax_grid = fig.add_subplot(gs[:, 0]) # Large local view
        ax_stats = fig.add_subplot(gs[0, 1:]) # Top stats
        ax_sensory = fig.add_subplot(gs[1:, 1]) # Sensory modules
        ax_minimap = fig.add_subplot(gs[1:, 2]) # Global minimap
             
        canvas = FigureCanvas(fig)
        _FIG_CACHE = (fig, ax_grid, ax_stats, ax_sensory, canvas, ax_minimap)
    else:
        fig, ax_grid, ax_stats, ax_sensory, canvas, ax_minimap = _FIG_CACHE
        ax_grid.clear()
        ax_stats.clear()
        ax_minimap.clear()
        if ax_sensory: ax_sensory.clear()
    
    # --- 1. Draw Local Grid ---
    ax_grid.set_facecolor(bg_color)
    ax_grid.set_xlim(c_start - 0.5, c_end - 0.5)
    ax_grid.set_ylim(r_start - 0.5, r_end - 0.5)
    ax_grid.invert_yaxis()
    ax_grid.set_aspect('equal')
    ax_grid.axis('off')
    ax_grid.set_title("LOCAL VIEW", color='#868E96', fontsize=10, fontweight='bold', pad=10)
    
    # Scaling factor for icons (based on visible window)
    scale_factor = (4.0 / view_size) * icon_scale
    
    # Grid lines and Background Tiles (Slicing)
    location_colors = {0: bg_color, 1: '#D3F9D8', 2: '#FFF4E6'}
    loc_grid = np.array(params.grid_location_type)
    
    for r in range(r_start, r_end):
        for c in range(c_start, c_end):
            l_type = int(loc_grid[r, c])
            if l_type != 0:
                ax_grid.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=location_colors.get(l_type, bg_color), ec='none', zorder=0))

    # Grid lines for the visible area
    for x in range(c_start, c_end + 1):
        ax_grid.vlines(x - 0.5, r_start - 0.5, r_end - 0.5, colors=grid_lines, linestyles='-', linewidth=0.8, alpha=0.5)
    for y in range(r_start, r_end + 1):
        ax_grid.hlines(y - 0.5, c_start - 0.5, c_end - 0.5, colors=grid_lines, linestyles='-', linewidth=0.8, alpha=0.5)
    
    def draw_icon(ax, r, c, icon_key, zoom=0.038, s_fac=1.0):
        img = icons.get(icon_key)
        if img is not None:
            imagebox = OffsetImage(img, zoom=zoom * s_fac)
            ab = AnnotationBbox(imagebox, (c, r), frameon=False, pad=0)
            ax.add_artist(ab)
        else:
            markers = {'agent': ('o', agent_color), 'food': ('D', food_color), 
                       'danger': ('X', danger_color), 'predator': ('v', '#212529')}
            m, color = markers.get(icon_key, ('s', 'grey'))
            ax.plot(c, r, marker=m, markersize=14 * s_fac, color=color, markeredgecolor='white', markeredgewidth=1.5)

    # Filter and draw entities
    def is_visible(r, c):
        return r_start <= r < r_end and c_start <= c < c_end

    # Resources
    res_positions = np.array(state.res_pos)
    res_types = np.array(params.res_type)
    res_active = np.array(state.res_active)
    
    any_danger_here, any_food_here = False, False
    for i in range(len(res_active)):
        if not res_active[i]: continue
        rr, rc = int(res_positions[i, 0]), int(res_positions[i, 1])
        is_here = (rr == ar and rc == ac)
        if is_here:
            if res_types[i] == 0: any_food_here = True
            else: any_danger_here = True
        elif is_visible(rr, rc):
            draw_icon(ax_grid, rr, rc, 'food' if res_types[i]==0 else 'danger', zoom=0.035, s_fac=scale_factor)
    
    # Predators
    pred_positions = np.array(state.pred_pos)
    num_pred = pred_positions.shape[0]
    predator_here = False
    for i in range(num_pred):
        pr, pc = int(pred_positions[i, 0]), int(pred_positions[i, 1])
        if pr == ar and pc == ac: predator_here = True
        elif is_visible(pr, pc):
            draw_icon(ax_grid, pr, pc, 'predator', zoom=0.045, s_fac=scale_factor)
            # Stamina bar logic simplified for speed
            if float(params.pred_max_stamina[i]) > 0:
                st_pct = float(state.pred_stamina[i]) / float(params.pred_max_stamina[i])
                ax_grid.add_patch(plt.Rectangle((pc-0.3*scale_factor, pr-0.4*scale_factor), 0.6*scale_factor, 0.08*scale_factor, color='#F1F3F5', alpha=0.5))
                ax_grid.add_patch(plt.Rectangle((pc-0.3*scale_factor, pr-0.4*scale_factor), 0.6*scale_factor*st_pct, 0.08*scale_factor, color='#FD7E14', alpha=0.8))

    # Rocks
    obs_positions = np.array(state.obs_pos)
    for i in range(obs_positions.shape[0]):
        or_, oc = int(obs_positions[i, 0]), int(obs_positions[i, 1])
        if is_visible(or_, oc) and not (or_ == ar and oc == ac):
            draw_icon(ax_grid, or_, oc, 'rock', zoom=0.035, s_fac=scale_factor)
            
    # Neutral Animals
    neutral_positions = np.array(state.neutral_pos)
    for i in range(neutral_positions.shape[0]):
        nr, nc = int(neutral_positions[i, 0]), int(neutral_positions[i, 1])
        if is_visible(nr, nc) and not (nr == ar and nc == ac):
            draw_icon(ax_grid, nr, nc, 'neutral', zoom=0.035, s_fac=scale_factor)

    # Agent
    if predator_here: draw_icon(ax_grid, ar, ac, 'agent_predator', zoom=0.055, s_fac=scale_factor)
    elif any_danger_here: draw_icon(ax_grid, ar, ac, 'agent_danger', zoom=0.048, s_fac=scale_factor)
    elif any_food_here: draw_icon(ax_grid, ar, ac, 'agent_food', zoom=0.048, s_fac=scale_factor)
    else: draw_icon(ax_grid, ar, ac, 'agent', zoom=0.035, s_fac=scale_factor)
    
    # --- 1.5. Draw Minimap ---
    ax_minimap.set_facecolor('#F8F9FA')
    ax_minimap.set_xlim(-0.5, width - 0.5)
    ax_minimap.set_ylim(-0.5, height - 0.5)
    ax_minimap.invert_yaxis()
    ax_minimap.set_aspect('equal')
    ax_minimap.axis('off')
    ax_minimap.set_title("MINIMAP", color='#868E96', fontsize=8, fontweight='bold', pad=5)
    
    # Show full grid background
    minimap_img = np.zeros((height, width, 3))
    # Fill with base color
    minimap_img[:, :] = [1, 1, 1] # White
    for l_id, color_hex in location_colors.items():
        if l_id == 0: continue
        # Convert hex to RGB 0-1
        from matplotlib.colors import to_rgb
        minimap_img[loc_grid == l_id] = to_rgb(color_hex)
    
    ax_minimap.imshow(minimap_img, extent=(-0.5, width-0.5, height-0.5, -0.5), zorder=1)
    
    # Draw local view rectangle
    rect = plt.Rectangle((c_start - 0.5, r_start - 0.5), view_size, view_size, linewidth=1, edgecolor='#228BE6', facecolor='none', alpha=0.8, zorder=5)
    ax_minimap.add_patch(rect)
    
    # Draw agent on minimap
    ax_minimap.plot(ac, ar, marker='o', markersize=3, color=agent_color, markeredgecolor='white', markeredgewidth=0.5, zorder=10)
    
    # Draw Rocks on minimap
    obs_x = obs_positions[:, 1]
    obs_y = obs_positions[:, 0]
    ax_minimap.scatter(obs_x, obs_y, s=1, c='#868E96', marker='s', zorder=5)
    
    # Draw Resources on minimap
    food_mask = jnp.logical_and(res_active, res_types == 0)
    danger_mask = jnp.logical_and(res_active, res_types == 1)
    
    if jnp.any(food_mask):
        ax_minimap.scatter(res_positions[food_mask, 1], res_positions[food_mask, 0], s=1, c=food_color, marker='o', zorder=6)
    if jnp.any(danger_mask):
        ax_minimap.scatter(res_positions[danger_mask, 1], res_positions[danger_mask, 0], s=1, c=danger_color, marker='x', zorder=6)
        
    # Draw Predators on minimap
    if num_pred > 0:
        ax_minimap.scatter(pred_positions[:, 1], pred_positions[:, 0], s=2, c='#212529', marker='v', zorder=7)
        
    # Draw Neutral Animals on minimap
    if neutral_positions.shape[0] > 0:
        ax_minimap.scatter(neutral_positions[:, 1], neutral_positions[:, 0], s=1.5, c='#15aabf', marker='o', zorder=7)
    

    
    # --- 2. Draw Stats Panel ---
    ax_stats.set_facecolor(bg_color)
    ax_stats.axis('off')
    y_cursor = 1.0 
    
    y_cursor -= 0.15 
    ax_stats.text(0.5, y_cursor, "INTEROCEPTIVE AI", color=text_color, ha='center', fontsize=12, fontweight='bold', transform=ax_stats.transAxes)
    ax_stats.plot([0.2, 0.8], [y_cursor - 0.05, y_cursor - 0.05], color='#ADB5BD', transform=ax_stats.transAxes, linewidth=1)
    
    y_cursor -= 0.15
    ep_str = f"EPISODE: {episode}" if episode is not None else "EP: --"
    step_str = f"STEP:    {step}" if step is not None else "STEP: --"
    ax_stats.text(0.1, y_cursor, ep_str, color='#495057', fontsize=9, transform=ax_stats.transAxes, fontfamily='monospace', weight='bold')
    ax_stats.text(0.1, y_cursor - 0.08, step_str, color='#495057', fontsize=9, transform=ax_stats.transAxes, fontfamily='monospace', weight='bold')
    
    # Satiation bar
    y_cursor -= 0.12 # Reduced gap
    if params.max_satiation > 0:
        y_cursor -= 0.08
        satiation = float(state.satiation)
        max_sat = float(params.max_satiation)
        sat_pct = max(0, min(1, satiation / max_sat))
        
        ax_stats.text(0.1, y_cursor + 0.08, f"SATIATION: {satiation:.1f}/{max_sat:.0f}", color=text_color, fontsize=8, fontweight='bold', transform=ax_stats.transAxes)
        ax_stats.add_patch(plt.Rectangle((0.1, y_cursor), 0.8, 0.06, facecolor='#F1F3F5', transform=ax_stats.transAxes))
        ax_stats.add_patch(plt.Rectangle((0.1, y_cursor), 0.8 * sat_pct, 0.06, facecolor=food_color, transform=ax_stats.transAxes))
    
    # Injury bar
    if params.max_injury > 0:
        y_cursor -= 0.15
        injury = float(state.injury_level)
        max_inj = float(params.max_injury)
        inj_pct = max(0, min(1, injury / max_inj))
        
        ax_stats.text(0.1, y_cursor + 0.08, f"INJURY: {injury:.1f}/{max_inj:.0f}", color=text_color, fontsize=8, fontweight='bold', transform=ax_stats.transAxes)
        ax_stats.add_patch(plt.Rectangle((0.1, y_cursor), 0.8, 0.06, facecolor='#F1F3F5', transform=ax_stats.transAxes))
        ax_stats.add_patch(plt.Rectangle((0.1, y_cursor), 0.8 * inj_pct, 0.06, facecolor=danger_color, transform=ax_stats.transAxes))
    
    # Status Badge and Action
    y_cursor -= 0.20
    
    if predator_here or any_danger_here:
        status_text, status_bg = "DANGER", danger_color
    elif any_food_here:
        status_text, status_bg = "FOOD", food_color
    else:
        status_text, status_bg = "CLEAR", '#ADB5BD'
        
    ax_stats.text(0.8, y_cursor + 0.05, status_text, color='white', ha='center', va='center', fontsize=8, fontweight='bold', 
                  transform=ax_stats.transAxes,
                  bbox=dict(boxstyle='round,pad=0.3', facecolor=status_bg, edgecolor='none'))

    if action is not None:
         # Action names indexing (JAX env standard: 0-3 directions, 4: Rest, 5: Eat)
         action_names = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
         if params.rest_action_enabled: action_names[4] = "REST"
         if params.eat_action_enabled: action_names[4 + int(params.rest_action_enabled)] = "EAT"
         
         act_str = action_names.get(int(action), f"ACT {action}")
         ax_stats.text(0.4, y_cursor + 0.05, f"ACTION: {act_str}", color='#495057', ha='center', fontsize=11, weight='bold', transform=ax_stats.transAxes)

    # --- 3. Draw Sensory Modules ---
    if ax_sensory and has_sensory:
        import math
        ax_sensory.set_facecolor(bg_color)
        ax_sensory.axis('off')
        ax_sensory.text(0.5, 0.95, "SENSORY MODULES", color=text_color, ha='center', fontsize=10, fontweight='bold', transform=ax_sensory.transAxes)
        
        # Use a dynamic layout engine for slots
        sensor_weights = []
        for s in sensory_data:
            sensor_weights.append(0.5 if s.get('side_by_side', False) else 1.0)
            
        # Calculate row counts (a row is full when weight sums to 1.0)
        rows = []
        current_row = []
        acc_w = 0.0
        for i, s in enumerate(sensory_data):
            w = sensor_weights[i]
            if acc_w + w > 1.05: # Float safety
                rows.append(current_row)
                current_row = [s]
                acc_w = w
            else:
                current_row.append(s)
                acc_w += w
        if current_row:
            rows.append(current_row)
            
        num_rows = len(rows)
        top_y, bottom_y = 0.85, 0.10
        available_h = top_y - bottom_y
        row_h = available_h / max(1, num_rows)
        
        for r_idx, row_sensors in enumerate(rows):
            y_center = top_y - (r_idx * row_h) - (row_h / 2)
            num_in_row = len(row_sensors)
            row_slot_w = 0.8 / num_in_row
            
            for c_idx, sensor in enumerate(row_sensors):
                slot_x = 0.1 + c_idx * (0.8 / num_in_row)
                slot_w = row_slot_w * 0.9 # Small gap
                bar_h = min(0.06, row_h * 0.4)
                
                name = sensor['name']
                vector = np.array(sensor.get('vector', []))
                
                # Label
                ax_sensory.text(slot_x, y_center + 0.08, name.upper(), color='#868E96', fontsize=7, fontweight='bold', transform=ax_sensory.transAxes)
            
            if sensor['type'] == 'intensity':
                val = float(sensor.get('intensity', 0))
                color = sensor.get('color', '#339AF0')
                ax_sensory.add_patch(plt.Rectangle((slot_x, y_center - bar_h/2), slot_w, bar_h, color='#F1F3F5', transform=ax_sensory.transAxes))
                ax_sensory.add_patch(plt.Rectangle((slot_x, y_center - bar_h/2), slot_w * min(1.0, val), bar_h, color=color, transform=ax_sensory.transAxes))
            
            elif sensor['type'] == 'radial':
                 # Draw small radial bars
                 num_ch = len(vector)
                 bar_w = slot_w / num_ch
                 gap = bar_w * 0.2
                 act_w = bar_w - gap
                 color = sensor.get('color', '#339AF0')
                 for ci in range(num_ch):
                      val = max(0, min(1.0, float(vector[ci])))
                      bx = slot_x + ci * bar_w
                      ax_sensory.add_patch(plt.Rectangle((bx, y_center - bar_h/2), act_w, bar_h, color='#F1F3F5', transform=ax_sensory.transAxes))
                      if val > 0:
                           sh = bar_h * val
                           ax_sensory.add_patch(plt.Rectangle((bx, y_center - bar_h/2), act_w, sh, color=color, transform=ax_sensory.transAxes, alpha=0.9))
            
            elif sensor['type'] == 'spectrum':
                 num_ch = len(vector)
                 bar_w = slot_w / num_ch
                 gap = bar_w * 0.2
                 act_w = bar_w - gap
                 
                 palette = ['#40C057', '#FA5252', '#339AF0', '#fab005', '#be4bdb', '#15aabf']
                 for ci in range(num_ch):
                      val = max(0, min(1.0, float(vector[ci]) / 2.0))
                      bx = slot_x + ci * bar_w
                      c = palette[ci % len(palette)]
                      
                      ax_sensory.add_patch(plt.Rectangle((bx, y_center - bar_h/2), act_w, bar_h, color='#F1F3F5', transform=ax_sensory.transAxes))
                      if val > 0:
                           sh = bar_h * val
                           ax_sensory.add_patch(plt.Rectangle((bx, y_center - bar_h/2), act_w, sh, color=c, transform=ax_sensory.transAxes, alpha=0.9))
            
            elif sensor['type'] == 'text':
                val_text = sensor.get('value_text', '')
                color = sensor.get('color', '#343A40')
                ax_sensory.text(slot_x, y_center - 0.02, val_text, color=color, fontsize=8, weight='bold', transform=ax_sensory.transAxes, fontfamily='monospace')
            
            elif sensor['type'] == 'diamond':
                # Manhattan Spatial Layout with Icons
                r = sensor.get('range', 0)
                num_features = sensor.get('num_features', 1)
                num_cells = len(vector) // num_features
                grid_data = vector.reshape(num_cells, num_features)
                
                def get_local_offsets(sr):
                    off = []
                    for d in range(sr + 1):
                        if d == 0: off.append([0, 0])
                        else:
                            for dr in range(-d, d + 1):
                                dc_abs = d - abs(dr)
                                if dc_abs == 0: off.append([dr, 0])
                                else:
                                    off.append([dr, dc_abs])
                                    off.append([dr, -dc_abs])
                    return np.array(off)
                
                cell_offsets = get_local_offsets(r)
                
                # Scale within slot_w and row_h
                max_span = 2 * r + 1
                # Adjust cell size to fit the slot
                cell_size = min(slot_w / max_span, row_h * 0.6 / max_span)
                
                origin_x = slot_x + slot_w / 2
                origin_y = y_center
                
                for ci in range(min(num_cells, len(cell_offsets))):
                    dr, dc = cell_offsets[ci]
                    cx = origin_x + dc * cell_size
                    cy = origin_y - dr * cell_size # dr positive is Up
                    
                    # Draw cell background
                    ax_sensory.add_patch(plt.Rectangle((cx - cell_size*0.45, cy - cell_size*0.45), cell_size*0.9, cell_size*0.9, color='#F1F3F5', transform=ax_sensory.transAxes, alpha=0.5))
                    
                    if num_features == 1:
                        # Collision style (binary)
                        val = float(grid_data[ci, 0])
                        if val > 0.5:
                            ax_sensory.add_patch(plt.Rectangle((cx - cell_size*0.4, cy - cell_size*0.4), cell_size*0.8, cell_size*0.8, color='#FA5252', transform=ax_sensory.transAxes, alpha=0.8))
                    else:
                        # Visual style - Icons or Colors
                        active_indices = np.where(grid_data[ci] > 0)[0]
                        if len(active_indices) > 0:
                            num_active = len(active_indices)
                            # Separate location features (0,1,2) from objects (3,4,5,6)
                            loc_indices = [i for i in active_indices if i < 3]
                            obj_indices = [i for i in active_indices if i >= 3]
                            
                            # Draw background color for location
                            bg_map = {0:'#D3F9D8', 1:'#FFF4E6', 2:'#FFFFFF'} # Grass, Sand, Plain
                            if loc_indices:
                                # Prioritize most dominant or just last one for simplicity
                                ax_sensory.add_patch(plt.Rectangle((cx - cell_size*0.45, cy - cell_size*0.45), cell_size*0.9, cell_size*0.9, color=bg_map.get(loc_indices[-1], '#FFFFFF'), transform=ax_sensory.transAxes, zorder=0))

                            if obj_indices:
                                num_obj = len(obj_indices)
                                m_rows = int(np.ceil(np.sqrt(num_obj)))
                                m_cols = int(np.ceil(num_obj / m_rows))
                                m_size = cell_size * 0.8 / max(m_rows, m_cols)
                                
                                icon_map = {3:'food', 4:'danger', 5:'predator', 6:'rock', 7:'neutral'}
                                
                                for mi, idx in enumerate(obj_indices):
                                    m_r = mi // m_cols
                                    m_c = mi % m_cols
                                    mx = cx - (m_cols*m_size)/2 + (m_c + 0.5)*m_size
                                    my = cy + (m_rows*m_size)/2 - (m_r + 0.5)*m_size
                                    
                                    icon_key = icon_map.get(idx)
                                    if icon_key and icons.get(icon_key) is not None:
                                        img = icons[icon_key]
                                        # Zoom factor needs to be small
                                        zoom_val = 0.0125 * (cell_size / 0.1) * (1.0 / max(m_rows, m_cols))
                                        imagebox = OffsetImage(img, zoom=zoom_val)
                                        ab = AnnotationBbox(imagebox, (mx, my), frameon=False, pad=0, xycoords=ax_sensory.transAxes)
                                        ax_sensory.add_artist(ab)
                                    else:
                                        # Fallback
                                        fallback_colors = ['#40C057', '#FA5252', '#212529', '#868E96', '#15aabf'] # Food, Danger, Predator, Rock, Neutral
                                        c = fallback_colors[idx-3] if (idx-3) < len(fallback_colors) else '#ADB5BD'
                                        ax_sensory.add_patch(plt.Circle((mx, my), m_size*0.4, color=c, transform=ax_sensory.transAxes))

    # Render directly from canvas (removing tight_layout)
    canvas.draw()
    
    # print_to_buffer is the Agg-native way to get the array
    s, (w, h) = canvas.print_to_buffer()
    image = np.frombuffer(s, dtype='uint8').reshape((int(h), int(w), 4))
    
    # Convert RGBA to RGB
    rgb = image[:, :, :3]
    
    # print(f"      [Profile] Render took {time.time() - start_time:.4f}s")
    return rgb


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
