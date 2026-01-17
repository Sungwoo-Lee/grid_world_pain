"""
Visualization utilities for GridWorld Pain environment.
Includes Q-table plotting and video rendering logic.
"""

import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def plot_q_table(q_table, save_path, food_pos=None):
    """
    Visualizes the Q-table policy and values at different satiation levels.
    """
    # Dispatch to conventional plotter if q_table is 3D
    if len(q_table.shape) == 3:
        return plot_q_table_conventional(q_table, save_path, food_pos)
        
    # Dispatch to 4D plotter if q_table is 5D (H, W, Sat, Health, Actions) or 4D depending on how we count
    # agent.py initialization: (height, width, max_sat+2, max_health+2, 5) -> 5 dimensions
    if len(q_table.shape) == 5:
        return plot_q_table_health(q_table, save_path, food_pos)

    # Ensure output directory exists
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # q_table shape: (height, width, satiation_dim, actions)
    height, width, sat_dim, _ = q_table.shape
    
    # Define representative slices to visualize
    max_satiation = sat_dim - 2
    
    slices = [
        max_satiation // 4,       # Low Satiation (Hungry)
        max_satiation // 2,       # Mid Satiation
        int(max_satiation * 0.9)  # High Satiation (Full)
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, sat_level in enumerate(slices):
        ax = axes[idx]
        
        # Extract 2D Q-table for this satiation
        q_slice = q_table[:, :, sat_level, :]
        
        # Value function: Max Q over actions
        v_values = np.max(q_slice, axis=2)
        
        # Policy: Argmax Q
        policy = np.argmax(q_slice, axis=2)
        
        # Normalize Value function for visualization
        v_min, v_max = np.min(v_values), np.max(v_values)
        if v_max > v_min:
            v_norm = (v_values - v_min) / (v_max - v_min)
        else:
            v_norm = np.zeros_like(v_values)
            
        # Plot Heatmap
        cax = ax.imshow(v_norm, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
        
        labels = ["Low", "Mid", "High"]
        ax.set_title(f"{labels[idx]} Satiation (Sat={sat_level})")
        ax.set_xticks(np.arange(width))
        ax.set_yticks(np.arange(height))
        
        # Overlay Arrows for Policy
        for r in range(height):
            for c in range(width):
                action = policy[r, c]
                arrow_char = ['\u2191', '\u2192', '\u2193', '\u2190', '\u2022'][action]
                ax.text(c, r, arrow_char, ha='center', va='center', color='white', fontsize=12, weight='bold')
        
        # Mark Food Location
        if food_pos is not None:
            fr, fc = food_pos
            ax.text(fc, fr, 'F', ha='center', va='center', color='lime', fontsize=20, weight='bold', path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])
                
    # Add colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(cax, cax=cbar_ax, label='Max Q-Value (Normalized)')
    
    plt.suptitle("Learned Policy & Value at Different Satiation Levels", fontsize=16)
    plt.savefig(save_path)
    plt.close(fig)

def plot_q_table_health(q_table, save_path, food_pos=None):
    """
    Visualizes representative slices of a 5D Q-table (Height, Width, Sat, Health, Actions) with fancy styling.
    """
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        
    height, width, sat_dim, health_dim, _ = q_table.shape
    max_satiation = sat_dim - 2
    max_health = health_dim - 2
    
    # Slices
    sat_slices = [max_satiation // 4, max_satiation // 2, int(max_satiation * 0.9)]
    sat_labels = ["Low Sat", "Mid Sat", "High Sat"]
    
    health_slices = [max_health // 4, int(max_health * 0.9)] 
    health_labels = ["Injured", "Healthy"]
    
    rows = len(health_slices)
    cols = len(sat_slices)
    
    # Global Font Settings for aesthetics
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)
    fig.patch.set_facecolor('#F8F9FA') # Soft grey background for the whole figure
    
    for r_idx, h_level in enumerate(health_slices):
        for c_idx, s_level in enumerate(sat_slices):
            ax = axes[r_idx, c_idx]
            
            # Extract 2D Q-table
            q_slice = q_table[:, :, s_level, h_level, :]
            
            v_values = np.max(q_slice, axis=2)
            policy = np.argmax(q_slice, axis=2)
            
            # Normalize
            v_min, v_max = np.min(v_values), np.max(v_values)
            if v_max > v_min:
                v_norm = (v_values - v_min) / (v_max - v_min)
            else:
                v_norm = np.zeros_like(v_values)
            
            # Heatmap with white lines to separate grid cells
            cax = ax.imshow(v_norm, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
            
            # Grid lines (white for separation)
            ax.set_xticks(np.arange(width) - 0.5, minor=True)
            ax.set_yticks(np.arange(height) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
            ax.tick_params(which="minor", bottom=False, left=False)
            
            # Title with consistent padding
            title_str = f"{health_labels[r_idx]}\n(Health={h_level})" if c_idx == 0 else ""
            if r_idx == 0:
                ax.set_title(f"{sat_labels[c_idx]}\n(Satiation={s_level})", fontsize=12, fontweight='bold', color='#495057')
            
            if c_idx == 0:
                ax.set_ylabel(f"{health_labels[r_idx]}\n(Health={h_level})", fontsize=12, fontweight='bold', color='#495057')

            # Clean Axes
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Arrows with path effects for visibility
            arrow_shadow = [PathEffects.withStroke(linewidth=2, foreground='black', alpha=0.5)]
            
            for r in range(height):
                for c in range(width):
                    val = v_norm[r, c]
                    action = policy[r, c]
                    # Arrow color dynamic: White for dark cells, Black for light cells (simple heuristic)
                    arrow_color = 'white' if val < 0.7 else 'black'
                    
                    arrow_char = ['\u2191', '\u2192', '\u2193', '\u2190', '\u2022'][action]
                    text = ax.text(c, r, arrow_char, ha='center', va='center', color=arrow_color, fontsize=14, weight='bold')
                    text.set_path_effects(arrow_shadow)
            
            # Food marker
            if food_pos is not None:
                fr, fc = food_pos
                ax.text(fc, fr, 'F', ha='center', va='center', color='#51CF66', fontsize=22, weight='bold', 
                        path_effects=[PathEffects.withStroke(linewidth=4, foreground='white')])

    # Colorbar
    # Create an axes for colorbar
    # fig.colorbar(cax, ax=axes, orientation='vertical', fraction=0.02, pad=0.04, label="Normalized Q-Value")
    cb = fig.colorbar(cax, ax=axes.ravel().tolist(), orientation='vertical', shrink=0.6, pad=0.02)
    cb.set_label('Max Q-Value (Normalized)', fontsize=12, labelpad=10)
    cb.outline.set_visible(False)
    
    plt.suptitle("Learned Policy: Health vs Satiation", fontsize=18, fontweight='bold', color='#343A40', y=1.02)
    plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='#F8F9FA')
    plt.close(fig)

def plot_q_table_conventional(q_table, save_path, food_pos=None):
    """
    Visualizes a 3D Q-table (Conventional Mode).
    """
    # Ensure output directory exists
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    height, width, _ = q_table.shape
    fig, ax = plt.subplots(figsize=(7, 6))
    
    v_values = np.max(q_table, axis=2)
    policy = np.argmax(q_table, axis=2)
    
    # Normalize
    v_min, v_max = np.min(v_values), np.max(v_values)
    if v_max > v_min:
        v_norm = (v_values - v_min) / (v_max - v_min)
    else:
        v_norm = np.zeros_like(v_values)
        
    cax = ax.imshow(v_norm, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title("Learned Policy & Value (Conventional Mode)")
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    
    for r in range(height):
        for c in range(width):
            action = policy[r, c]
            arrow_char = ['\u2191', '\u2192', '\u2193', '\u2190', '\u2022'][action]
            ax.text(c, r, arrow_char, ha='center', va='center', color='white', fontsize=12, weight='bold')
    
    if food_pos is not None:
        fr, fc = food_pos
        ax.text(fc, fr, 'G', ha='center', va='center', color='lime', fontsize=20, weight='bold', path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')])
        
    fig.colorbar(cax, ax=ax, label='Max Q-Value (Normalized)')
    plt.savefig(save_path)
    plt.close(fig)

def save_video(frames, output_path, fps=5):
    """
    Saves a list of RGB frames as an MP4 video.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Saved video to {output_path}")

def plot_learning_curves(history_csv_path, output_dir, max_steps=None, milestones=None):
    """
    Plots learning curves from a training history CSV file with a professional fancy theme.
    """
    if not os.path.exists(history_csv_path):
        print(f"Warning: {history_csv_path} not found. Skipping learning curves plot.")
        return

    import pandas as pd
    try:
        df = pd.read_csv(history_csv_path)
    except Exception as e:
        print(f"Error reading history CSV: {e}")
        return

    if df.empty:
        print("Warning: History CSV is empty. Skipping learning curves plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "learning_curves.png")

    # --- Fancy Theme Configuration ---
    # Colors (Vibrant & Professional)
    reward_color_main = '#4C6EF5'  # Royal Blue
    reward_color_raw = '#A5D8FF'   # Lighter Blue
    steps_color_main = '#20C997'   # Teal
    steps_color_raw = '#C3FAE8'    # Lighter Teal
    milestone_color = '#FA5252'    # Soft Red for badges
    bg_color = '#F8F9FA'           # Soft Gray background
    
    # Global Settings
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 3 Subplots: Reward, Steps, Epsilon
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, facecolor=bg_color)
    
    # --- Plot 1: Cumulative Reward ---
    window = max(1, len(df) // 100)
    ax1.set_facecolor(bg_color)
    ax1.plot(df['episode'], df['reward'], color=reward_color_raw, alpha=0.5, label='Raw Reward', linewidth=1)
    ax1.plot(df['episode'], df['reward'].rolling(window=window).mean(), color=reward_color_main, linewidth=2.5, label=f'Moving Average (n={window})')
    
    ax1.set_ylabel("Cumulative Reward", fontsize=12, fontweight='bold', color='#495057')
    ax1.set_title("Training Performance: Reward Evolution", loc='left', fontsize=14, fontweight='bold', pad=15, color='#212529')
    ax1.legend(frameon=True, facecolor='white', framealpha=0.8)
    ax1.grid(True, linestyle='--', alpha=0.3, color='#ADB5BD')
    
    # --- Plot 2: Survival Steps ---
    ax2.set_facecolor(bg_color)
    ax2.plot(df['episode'], df['steps'], color=steps_color_raw, alpha=0.5, label='Raw Steps', linewidth=1)
    ax2.plot(df['episode'], df['steps'].rolling(window=window).mean(), color=steps_color_main, linewidth=2.5, label=f'Moving Average (n={window})')
    
    ax2.set_ylabel("Survival Steps", fontsize=12, fontweight='bold', color='#495057')
    ax2.set_title("Training Performance: Survival Capacity", loc='left', fontsize=14, fontweight='bold', pad=15, color='#212529')
    ax2.legend(frameon=True, facecolor='white', framealpha=0.8)
    ax2.grid(True, linestyle='--', alpha=0.3, color='#ADB5BD')
    
    if max_steps:
        ax2.set_ylim(0, max_steps * 1.1)  # Give some headroom for labels

    # --- Plot 3: Epsilon Decay ---
    ax3.set_facecolor(bg_color)
    if 'epsilon' in df.columns:
        epsilon_color = '#FCC419' # Yellow/Orange
        ax3.plot(df['episode'], df['epsilon'], color=epsilon_color, linewidth=2, label='Epsilon')
        ax3.fill_between(df['episode'], 0, df['epsilon'], color=epsilon_color, alpha=0.1)
    else:
        ax3.text(0.5, 0.5, "Epsilon data not found in history", ha='center', va='center', color='#868E96')
        
    ax3.set_xlabel("Episode", fontsize=12, fontweight='bold', color='#495057')
    ax3.set_ylabel("Epsilon (Exploration)", fontsize=12, fontweight='bold', color='#495057')
    ax3.set_title("Exploration Strategy: Epsilon Decay", loc='left', fontsize=14, fontweight='bold', pad=15, color='#212529')
    ax3.legend(frameon=True, facecolor='white', framealpha=0.8)
    ax3.grid(True, linestyle='--', alpha=0.3, color='#ADB5BD')
    ax3.set_ylim(0, 1.1)

    # --- Vertical Milestone Lines & Badges ---
    if milestones:
        # Get Y limits for badge placement
        y1_min, y1_max = ax1.get_ylim()
        
        for ep, pct in milestones.items():
            # Add vertical line through all subplots
            for ax in [ax1, ax2, ax3]:
                ax.axvline(x=ep, color=milestone_color, linestyle=':', alpha=0.6, linewidth=1.5)
            
            # Add "Badge" label at the top of the first plot
            # Badge Background (box)
            ax1.text(ep, y1_max * 0.95, f" {pct}% ", color='white', fontsize=9, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=milestone_color, edgecolor='none', alpha=0.9))

    # Clean up spines (Standard feature for "Fancy" plots)
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#DEE2E6')
        ax.spines['bottom'].set_color('#DEE2E6')
        ax.tick_params(colors='#495057')

    # Add a main Dashboard title
    plt.suptitle("Interoceptive AI Training Dashboard", fontsize=20, fontweight='bold', y=0.98, color='#343A40')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, facecolor=bg_color, dpi=150)
    plt.close(fig)
    print(f"Saved fancy learning curves to {save_path}")

if __name__ == "__main__":
    print("This module is a utility library and should not be run directly.")
    

def visualize_activations(activations, target_width):
    """
    Visualizes neural network activations as a stacked heatmap image with a professional dark theme.
    Respects the order of the `activations` dictionary (assumed Input -> Output).
    """
    if not activations:
        return None
        
    # Maintain order from dict (Python 3.7+ guarantees insertion order, hooks fire in execution order)
    # Filter and process
    experiments = {}
    for k, v in activations.items():
        if v is None: continue
        vals = np.squeeze(v)
        if vals.ndim == 0: vals = np.expand_dims(vals, 0)
        # Flatten multi-dim layers (Conv2d: C*H*W) for 1D visualization
        # Alternatively, for Conv2d, we could show mean across spatial dims? 
        # But flattening preserves all info.
        if vals.ndim > 1:
            vals = vals.flatten()
        experiments[k] = vals

    n_layers = len(experiments)
    if n_layers == 0:
        return None

    # Style Configuration
    plt.rcParams['font.family'] = 'sans-serif'
    # text colors
    text_color = '#E0E0E0'
    label_color = '#B0B0B0'
    bg_color = '#1A1A1A'
    
    # Dimensions
    dpi = 100
    fig_width = target_width / dpi
    
    # Height calculation: Base overhead + per-layer height
    # Give enough space for colorbar and labels
    row_height = 0.6
    header_height = 0.5
    fig_height = header_height + (n_layers * row_height)
    
    # Grid: [Main Heatmap] [Spacer] [Colorbar]
    # Width ratios: 90% Heatmap, 2% space, 3% Cbar, 5% margin
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor=bg_color)
    gs = fig.add_gridspec(n_layers, 3, width_ratios=[30, 1, 1], wspace=0.1, hspace=0.5, 
                          top=1.0 - (header_height/fig_height), bottom=0.05, left=0.15, right=0.95)
    
    fig.suptitle("Neural Network Activity (Input \u2192 Output)", color=text_color, fontsize=12, fontweight='bold', y=0.98)
    
    # Iterate in order
    for idx, (name, vals) in enumerate(experiments.items()):
        ax_map = fig.add_subplot(gs[idx, 0])
        ax_cbar = fig.add_subplot(gs[idx, 2])
        
        # Prepare Data
        if vals.size > 0:
             img = np.expand_dims(vals, 0)
             
             # Determine visual range
             # Robust min/max to avoid outliers washing out details?
             # For now, absolute min/max
             vmin, vmax = vals.min(), vals.max()
             
             # Heatmap
             # Use a professional colormap: 'magma' or 'mako' or 'viridis'
             # 'coolwarm' for diverging if values are +/-? 
             # Neural activations (ReLU) are 0+, Tanh are +/-.
             cmap = 'magma' if vmin >= 0 else 'coolwarm'
             
             im = ax_map.imshow(img, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
             
             # Colorbar
             cbar = plt.colorbar(im, cax=ax_cbar, orientation='vertical')
             cbar.ax.tick_params(labelsize=6, colors=label_color, width=0.5)
             cbar.outline.set_visible(False)
             
             # Min/Max labels on colorbar are automatic, but let's ensure readability
             # Maybe reduce ticks to just min/max/0
             ticks = [vmin, vmax]
             if vmin < 0 < vmax: ticks.append(0)
             ticks = np.unique(ticks)
             cbar.set_ticks(ticks)
             cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

        else:
             ax_map.text(0.5, 0.5, "Empty", color=label_color, ha='center', va='center')
             ax_cbar.axis('off')

        # Styling Ax Map
        ax_map.set_yticks([])
        ax_map.set_xticks([])
        for spine in ax_map.spines.values():
            spine.set_visible(False)
            
        # Label (Layer Name + Shape)
        label_str = f"{name}\n{vals.shape}"
        ax_map.set_ylabel(label_str, rotation=0, ha='right', va='center', fontsize=7, color=text_color, labelpad=10)

    # Render
    canvas = FigureCanvas(fig)
    canvas.draw()
    
    # Buffer
    s, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(s, dtype='uint8').reshape((int(height), int(width), 4))
    
    # RGB
    image = image[:, :, :3]
    plt.close(fig)
    
    return image


def combine_frame_and_activations(game_frame, activation_frame):
    """
    Stacks game frame and activation frame vertically.
    Adjusts activation frame width to match game frame if needed.
    """
    if activation_frame is None:
        return game_frame
        
    g_h, g_w, _ = game_frame.shape
    a_h, a_w, _ = activation_frame.shape
    
    if g_w != a_w:
        # Resize activation frame is complex without cv2/PIL
        # Simplest: Crop or Pad.
        # But we generated activation frame with target_width=g_w usually.
        # But due to DPI rounding, might be off by few pixels.
        
        # Simple nearest-neighbor resize or just pad/crop
        if a_w > g_w:
            activation_frame = activation_frame[:, :g_w, :]
        else:
            # Pad with black
            pad = np.zeros((a_h, g_w - a_w, 3), dtype=game_frame.dtype)
            activation_frame = np.concatenate([activation_frame, pad], axis=1)
            
    # Vertical Stack
    return np.concatenate([game_frame, activation_frame], axis=0)

