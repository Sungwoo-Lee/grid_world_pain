"""
Visualization utilities for GridWorld Pain environment.
Includes Q-table plotting and video rendering logic.
"""

import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects

def plot_q_table(q_table, save_path, food_pos=None):
    """
    Visualizes the Q-table policy and values at different satiation levels.
    """
    # Dispatch to conventional plotter if q_table is 3D
    if len(q_table.shape) == 3:
        return plot_q_table_conventional(q_table, save_path, food_pos)

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
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, facecolor=bg_color)
    
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
    
    ax2.set_xlabel("Episode", fontsize=12, fontweight='bold', color='#495057')
    ax2.set_ylabel("Survival Steps", fontsize=12, fontweight='bold', color='#495057')
    ax2.set_title("Training Performance: Survival Capacity", loc='left', fontsize=14, fontweight='bold', pad=15, color='#212529')
    ax2.legend(frameon=True, facecolor='white', framealpha=0.8)
    ax2.grid(True, linestyle='--', alpha=0.3, color='#ADB5BD')
    
    if max_steps:
        ax2.set_ylim(0, max_steps * 1.1)  # Give some headroom for labels

    # --- Vertical Milestone Lines & Badges ---
    if milestones:
        # Get Y limits for badge placement
        y1_min, y1_max = ax1.get_ylim()
        
        for ep, pct in milestones.items():
            # Add vertical line through both subplots
            for ax in [ax1, ax2]:
                ax.axvline(x=ep, color=milestone_color, linestyle=':', alpha=0.6, linewidth=1.5)
            
            # Add "Badge" label at the top of the first plot
            # Badge Background (box)
            ax1.text(ep, y1_max * 0.95, f" {pct}% ", color='white', fontsize=9, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor=milestone_color, edgecolor='none', alpha=0.9))

    # Clean up spines (Standard feature for "Fancy" plots)
    for ax in [ax1, ax2]:
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
    print("Use train.py for training and evaluation.py for generating artifacts.")
