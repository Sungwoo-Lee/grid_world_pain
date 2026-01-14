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

def plot_learning_curves(history_csv_path, output_dir):
    """
    Plots learning curves from a training history CSV file.
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot Episode Rewards
    window = max(1, len(df) // 100)
    ax1.plot(df['episode'], df['reward'], alpha=0.3, color='blue', label='Raw Reward')
    ax1.plot(df['episode'], df['reward'].rolling(window=window).mean(), color='darkblue', linewidth=2, label=f'Moving Average (n={window})')
    ax1.set_ylabel("Cumulative Reward")
    ax1.set_title("Training Performance: Reward per Episode")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Plot Episode Steps
    ax2.plot(df['episode'], df['steps'], alpha=0.3, color='green', label='Raw Steps')
    ax2.plot(df['episode'], df['steps'].rolling(window=window).mean(), color='darkgreen', linewidth=2, label=f'Moving Average (n={window})')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Survival Steps")
    ax2.set_title("Training Performance: Steps per Episode")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved learning curves to {save_path}")

if __name__ == "__main__":
    print("This module is a utility library and should not be run directly.")
    print("Use train.py for training and evaluation.py for generating artifacts.")
