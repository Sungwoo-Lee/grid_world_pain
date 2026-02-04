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


def render_jax_state(state, params, episode=None, step=None, dpi=100, icon_scale=1.0):
    """
    Render a JAX EnvState to an RGB numpy array.
    
    Args:
        state: JAX EnvState dataclass
        params: JAX EnvParams dataclass
        episode: Optional episode number for display
        step: Optional step number for display
        dpi: Resolution (default 100)
        icon_scale: Scale factor for icons
        
    Returns:
        numpy array of shape (H, W, 3) representing the RGB frame
    """
    icons = _load_icons()
    
    # Colors
    bg_color = '#FFFFFF'
    text_color = '#343A40'
    grid_lines = '#DEE2E6'
    
    # Grid dimensions
    height = int(params.height)
    width = int(params.width)
    
    # Figure setup
    fig = plt.figure(figsize=(8, 6), dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax_grid = fig.add_subplot(gs[0])
    ax_stats = fig.add_subplot(gs[1])
    
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
            markers = {'agent': ('o', '#339AF0'), 'food': ('D', '#40C057'), 
                       'danger': ('X', '#FA5252'), 'predator': ('v', '#212529')}
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
    num_pred = pred_positions.shape[0]
    
    predator_here = False
    for i in range(num_pred):
        pr, pc = int(pred_positions[i, 0]), int(pred_positions[i, 1])
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
    y_cursor -= 0.20
    satiation = float(state.satiation)
    max_satiation = float(params.max_satiation)
    sat_pct = max(0, min(1, satiation / max_satiation)) if max_satiation > 0 else 0
    
    ax_stats.text(0.1, y_cursor, f"Satiation:", color='#495057', fontsize=9, transform=ax_stats.transAxes, fontweight='bold')
    ax_stats.text(0.9, y_cursor, f"{satiation:.0f}/{max_satiation:.0f}", color='#495057', fontsize=9, transform=ax_stats.transAxes, ha='right')
    
    # Bar background
    bar_y = y_cursor - 0.06
    ax_stats.add_patch(plt.Rectangle((0.1, bar_y), 0.8, 0.04, facecolor='#E9ECEF', edgecolor='#ADB5BD', transform=ax_stats.transAxes, lw=0.5))
    # Bar fill
    ax_stats.add_patch(plt.Rectangle((0.1, bar_y), 0.8 * sat_pct, 0.04, facecolor='#40C057', transform=ax_stats.transAxes))
    
    # Injury bar
    y_cursor -= 0.18
    injury = float(state.injury_level)
    max_injury = float(params.max_injury)
    inj_pct = max(0, min(1, injury / max_injury)) if max_injury > 0 else 0
    
    ax_stats.text(0.1, y_cursor, f"Injury:", color='#495057', fontsize=9, transform=ax_stats.transAxes, fontweight='bold')
    ax_stats.text(0.9, y_cursor, f"{injury:.0f}/{max_injury:.0f}", color='#495057', fontsize=9, transform=ax_stats.transAxes, ha='right')
    
    # Bar background
    bar_y = y_cursor - 0.06
    ax_stats.add_patch(plt.Rectangle((0.1, bar_y), 0.8, 0.04, facecolor='#E9ECEF', edgecolor='#ADB5BD', transform=ax_stats.transAxes, lw=0.5))
    # Bar fill (red for injury)
    ax_stats.add_patch(plt.Rectangle((0.1, bar_y), 0.8 * inj_pct, 0.04, facecolor='#FA5252', transform=ax_stats.transAxes))
    
    # Action legend
    y_cursor -= 0.22
    ax_stats.text(0.5, y_cursor, "Actions: ↑↓←→", color='#868E96', fontsize=9, ha='center', transform=ax_stats.transAxes)
    
    # Grid info
    y_cursor -= 0.12
    ax_stats.text(0.5, y_cursor, f"Grid: {height}x{width}", color='#868E96', fontsize=8, ha='center', transform=ax_stats.transAxes)
    
    # Render to array
    plt.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    s, (w, h) = canvas.print_to_buffer()
    image = np.frombuffer(s, dtype='uint8').reshape((int(h), int(w), 4))
    
    # Convert RGBA to RGB
    rgb = image[:, :, :3]
    
    plt.close(fig)
    
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
