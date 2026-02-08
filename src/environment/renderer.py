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

# Imports for JAX EnvState
import jax.numpy as jnp

# Global cache for icons to avoid reloading every frame
_ICON_CACHE = None

def _load_icons():
    """Lengths icons from assets directory."""
    global _ICON_CACHE
    if _ICON_CACHE is not None:
        return _ICON_CACHE
        
    # Calculate assets path relative to this file
    # src/environment/jax_env/renderer.py -> ... -> assets/
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    assets_path = os.path.join(base_dir, 'assets')
    
    icon_files = {
        'agent': 'agent.jpg',
        'food': 'food.jpg',
        'danger': 'danger.jpg',
        'predator': 'predator.jpg',
        'agent_food': 'agent_food.jpg',
        'agent_danger': 'agent_danger.jpg',
        'agent_predator': 'agent_predator.jpg'
    }
    
    icons = {}
    for key, filename in icon_files.items():
        path = os.path.join(assets_path, filename)
        if os.path.exists(path):
            icons[key] = plt.imread(path)
        else:
            # print(f"Warning: Icon not found at {path}")
            icons[key] = None
            
    _ICON_CACHE = icons
    return icons


# Global cache for figure, axes, and canvas to avoid recreating them every frame
_FIG_CACHE = None

def render_jax_state(state, params, episode=None, step=None, dpi=100, icon_scale=1.0, action=None, sensory_data=None):
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
        
    Returns:
        numpy array of shape (H, W, 3) representing the RGB frame
    """
    start_time = time.time()
    icons = _load_icons()
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
    
    # Figure setup (Reuse if possible, but handle sensory_data presence changes)
    has_sensory = sensory_data is not None and len(sensory_data) > 0
    
    # We check if the cached figure matches the current layout requirement
    need_new_fig = False
    if _FIG_CACHE is not None:
        cached_fig, cached_grid, cached_stats, cached_sensory, cached_canvas = _FIG_CACHE
        # If the number of subplots changed, we need to rebuild
        if (has_sensory and cached_sensory is None) or (not has_sensory and cached_sensory is not None):
            need_new_fig = True
            plt.close(cached_fig)
    else:
        need_new_fig = True

    if need_new_fig:
        fig_width = 10 if has_sensory else 8
        fig = plt.figure(figsize=(fig_width, 6), dpi=dpi)
        fig.patch.set_facecolor(bg_color)
        
        if has_sensory:
             gs = fig.add_gridspec(2, 2, width_ratios=[1.5, 1], height_ratios=[0.4, 0.6])
             ax_grid = fig.add_subplot(gs[:, 0])
             ax_stats = fig.add_subplot(gs[0, 1])
             ax_sensory = fig.add_subplot(gs[1, 1])
        else:
             gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
             ax_grid = fig.add_subplot(gs[0])
             ax_stats = fig.add_subplot(gs[1])
             ax_sensory = None
             
        canvas = FigureCanvas(fig)
        _FIG_CACHE = (fig, ax_grid, ax_stats, ax_sensory, canvas)
    else:
        fig, ax_grid, ax_stats, ax_sensory, canvas = _FIG_CACHE
        ax_grid.clear()
        ax_stats.clear()
        if ax_sensory: ax_sensory.clear()
    
    # --- 1. Draw Grid ---
    ax_grid.set_facecolor(bg_color)
    ax_grid.set_xlim(-0.5, width - 0.5)
    ax_grid.set_ylim(-0.5, height - 0.5)
    ax_grid.invert_yaxis()
    ax_grid.set_aspect('equal')
    ax_grid.axis('off')
    
    # Scaling factor based on baseline 4x4 grid (matching grid_world.py)
    scale_factor = (4.0 / max(width, height)) * icon_scale
    
    # Grid lines
    for x in range(width + 1):
        ax_grid.vlines(x - 0.5, -0.5, height - 0.5, colors=grid_lines, linestyles='-', linewidth=0.8, alpha=0.5)
    for y in range(height + 1):
        ax_grid.hlines(y - 0.5, -0.5, width - 0.5, colors=grid_lines, linestyles='-', linewidth=0.8, alpha=0.5)
    
    def draw_icon(ax, r, c, icon_key, zoom=0.038):
        img = icons.get(icon_key)
        if img is not None:
            imagebox = OffsetImage(img, zoom=zoom * scale_factor)
            ab = AnnotationBbox(imagebox, (c, r), frameon=False, pad=0)
            ax.add_artist(ab)
        else:
            # Fallback markers
            markers = {'agent': ('o', agent_color), 'food': ('D', food_color), 
                       'danger': ('X', danger_color), 'predator': ('v', '#212529')}
            m, color = markers.get(icon_key, ('s', 'grey'))
            ax.plot(c, r, marker=m, markersize=14 * scale_factor, color=color, markeredgecolor='white', markeredgewidth=1.5)

    # Agent position
    agent_pos = np.array(state.agent_pos)
    ar, ac = int(agent_pos[0]), int(agent_pos[1])
    
    # Resources
    res_positions = np.array(state.res_pos)   # [num_res, 2]
    res_types = np.array(params.res_type)     # [num_res] - from params
    res_active = np.array(state.res_active)   # [num_res]
    
    any_danger_here = False
    any_food_here = False
    
    num_res = len(res_active)
    for i in range(num_res):
        if not res_active[i]:
            continue
        rr, rc = int(res_positions[i, 0]), int(res_positions[i, 1])
        
        is_here = (rr == ar and rc == ac)
        
        if res_types[i] == 0:  # Food
            if is_here: any_food_here = True
            else: draw_icon(ax_grid, rr, rc, 'food', zoom=0.035)
        else:  # Danger
            if is_here: any_danger_here = True
            else: draw_icon(ax_grid, rr, rc, 'danger', zoom=0.035)
    
    # Predators
    pred_positions = np.array(state.pred_pos)  # [num_pred, 2]
    pred_stamina = np.array(state.pred_stamina) # [num_pred]
    pred_max_stamina = np.array(params.pred_max_stamina) # [num_pred]
    num_pred = pred_positions.shape[0]
    
    predator_here = False
    for i in range(num_pred):
        pr, pc = int(pred_positions[i, 0]), int(pred_positions[i, 1])
        
        # Draw stamina bar above predator
        if pred_max_stamina[i] > 0:
            stamina_pct = max(0.0, min(1.0, float(pred_stamina[i]) / float(pred_max_stamina[i])))
            # Bar dimensions scaled by grid size
            bar_w = 0.6 * scale_factor
            bar_h = 0.1 * scale_factor
            bar_x = pc - bar_w / 2
            bar_y = pr - 0.4 * scale_factor # Position above icon
            
            # Background
            ax_grid.add_patch(plt.Rectangle((bar_x, bar_y), bar_w, bar_h, color='#F1F3F5', alpha=0.7, ec='none'))
            # Foreground (Stamina) - Use orange for stamina
            ax_grid.add_patch(plt.Rectangle((bar_x, bar_y), bar_w * stamina_pct, bar_h, color='#FD7E14', alpha=0.9, ec='none'))

        if pr == ar and pc == ac:
            predator_here = True
        else:
            draw_icon(ax_grid, pr, pc, 'predator', zoom=0.045)
    
    # Draw Agent (handling overlaps)
    if predator_here:
         draw_icon(ax_grid, ar, ac, 'agent_predator', zoom=0.055)
    elif any_danger_here:
         draw_icon(ax_grid, ar, ac, 'agent_danger', zoom=0.048)
    elif any_food_here:
         draw_icon(ax_grid, ar, ac, 'agent_food', zoom=0.048)
    else:
         draw_icon(ax_grid, ar, ac, 'agent', zoom=0.035)
    
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
        
        num_sensors = len(sensory_data)
        top_y, bottom_y = 0.85, 0.10
        available_h = top_y - bottom_y
        slot_h = available_h / num_sensors
        
        # Use fixed coordinates and spacing (much faster than tight_layout)
        for i, sensor in enumerate(sensory_data):
            y_center = top_y - (i * slot_h) - (slot_h / 2)
            name = sensor['name']
            vector = np.array(sensor.get('vector', []))
            
            # Label - Fixed color for better visibility
            ax_sensory.text(0.1, y_center + 0.08, name.upper(), color='#868E96', fontsize=7, fontweight='bold', transform=ax_sensory.transAxes)
            
            # Container for the sensor bar(s)
            slot_x = 0.1
            slot_w = 0.8
            bar_h = min(0.06, slot_h * 0.4)
            
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
                ax_sensory.text(0.1, y_center - 0.02, val_text, color=color, fontsize=8, weight='bold', transform=ax_sensory.transAxes, fontfamily='monospace')

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
