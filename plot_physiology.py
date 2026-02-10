import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse
import numpy as np

def plot_physiology(stats_file, output_path=None):
    df = pd.read_csv(stats_file)
    
    # Create a figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True, 
                             gridspec_kw={'height_ratios': [1.2, 1, 1, 1, 1]})
    (ax1, ax2, ax3, ax4, ax5) = axes
    
    # --- Aesthetics ---
    plt.rcParams['font.family'] = 'sans-serif'
    title_style = {'size': 16, 'weight': 'bold', 'color': '#212529'}
    label_style = {'size': 12, 'weight': 'semibold', 'color': '#495057'}
    grid_style = {'linestyle': '--', 'alpha': 0.4, 'linewidth': 0.8}

    # --- 1. Top Panel: Digestion & Survival ---
    ax1.plot(df['step'], df['satiation'], label='Satiation (Short-term)', color='#40C057', lw=3)
    ax1.plot(df['step'], df['nutrition'], label='Nutrition (Long-term)', color='#228BE6', lw=3, linestyle='--')
    ax1.set_ylabel('Percentage (%)', **label_style)
    ax1.set_title('I. PHYSIOLOGICAL STATE: NUTRITION & SATIATION', **title_style, loc='left')
    ax1.legend(loc='upper right', frameon=True, fontsize=10)
    ax1.grid(True, **grid_style)
    ax1.set_ylim(-5, 105)
    
    # Annotate "Death at 200" if applicable
    if df['nutrition'].iloc[-1] <= 1.0:
         ax1.axvline(df['step'].iloc[-1], color='#FA5252', linestyle=':', lw=2)
         ax1.text(df['step'].iloc[-1], 20, ' STARVATION', color='#FA5252', weight='bold', verticalalignment='center')

    # --- 2. Second Panel: Injury & Streak-based Recovery ---
    ax2.plot(df['step'], df['injury'], label='Injury level', color='#FA5252', lw=3)
    ax2.set_ylabel('Intensity', **label_style)
    
    ax2_streak = ax2.twinx()
    ax2_streak.step(df['step'], df['rest_streak'], label='Rest Streak', color='#15AABF', where='post', alpha=0.4)
    ax2_streak.fill_between(df['step'], df['rest_streak'], color='#15AABF', alpha=0.1)
    ax2_streak.set_ylabel('Rest Streak (Steps)', **label_style)
    ax2_streak.tick_params(axis='y', labelcolor='#15AABF')
    
    # Mark injury events
    if 'event_damage' in df.columns:
        damage_steps = df[df['event_damage'] > 0]
        ax2.scatter(damage_steps['step'], [df['max_injury'].iloc[0] * 0.9 if 'max_injury' in df.columns else 90]*len(damage_steps), 
                    marker='v', color='#F03E3E', s=30, alpha=0.6, label='Damage Taken')

    ax2.set_title('II. INJURY DYNAMICS & RECOVERY ACCELERATION', **title_style, loc='left')
    ax2.grid(True, **grid_style)

    # --- 3. Third Panel: Motivation (Drive Decomposition) ---
    if 'drive_hunger' in df.columns:
        ax3.plot(df['step'], df['drive_hunger'], color='#FAB005', lw=2, label='Hunger Drive')
        ax3.plot(df['step'], df['drive_injury'], color='#FA5252', lw=2, label='Injury Drive')
        ax3.plot(df['step'], df['drive'], color='#be4bdb', lw=3, alpha=0.9, label='Total Drive')
        ax3.fill_between(df['step'], df['drive'], color='#be4bdb', alpha=0.08)
        ax3.set_ylabel('Drive Scalar', **label_style)
        ax3.set_title('III. MOTIVATION: HUNGER VS. INJURY DRIVE DECOMPOSITION', **title_style, loc='left')
        ax3.legend(loc='upper left', fontsize=10, ncol=3)
        ax3.grid(True, **grid_style)
    
    # --- 4. Fourth Panel: Reward Streams ---
    if 'reward_homeostatic' in df.columns:
        ax4.bar(df['step'], df['reward_homeostatic'], color='#40C057', alpha=0.6, label='Homeostatic (Relief)', width=0.8)
        ax4.bar(df['step'], df['reward_extrinsic'], color='#FAB005', alpha=0.8, label='Extrinsic (Food)', width=0.8)
        ax4.set_ylabel('Reward', **label_style)
        ax4.set_title('IV. REWARD SIGNALS: INTERNAL VS. EXTERNAL REINFORCEMENT', **title_style, loc='left')
        ax4.legend(loc='upper right', fontsize=10)
        ax4.grid(True, **grid_style)

    # --- 5. Bottom Panel: Physical Geometry (Distances) ---
    if 'dist_to_food' in df.columns:
        ax5.plot(df['step'], df['dist_to_food'], color='#FAB005', lw=2, label='Dist to nearest Food')
        ax5.plot(df['step'], df['dist_to_pred'], color='#212529', lw=2, label='Dist to nearest Predator', linestyle=':')
        ax5.set_ylabel('Euclidean Distance', **label_style)
        ax5.set_xlabel('Environment Steps', **label_style)
        ax5.set_title('V. SPATIAL GEOMETRY: PROXIMITY TO GOALS & THREATS', **title_style, loc='left')
        ax5.legend(loc='upper right', fontsize=10)
        ax5.grid(True, **grid_style)
        ax5.set_ylim(0, 30)

    # Footer
    info_str = f"File: {os.path.basename(stats_file)} | Created: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}"
    fig.text(0.5, 0.01, info_str, ha='center', fontsize=10, color='#868E96', fontfamily='monospace')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"Full Spectrum Analysis Plot saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, help="Path to stats CSV file")
    parser.add_argument("--out", type=str, default="behavior_full_spectrum.png", help="Output path for plot")
    args = parser.parse_args()
    
    if args.csv:
        plot_physiology(args.csv, args.out)
    else:
        # Search for latest results
        stats_files = glob.glob("results/JAX_Sandbox/*/stats/ep_1_stats.csv")
        if stats_files:
            latest = max(stats_files, key=os.path.getmtime)
            print(f"Analyzing latest behavioral data: {latest}")
            plot_physiology(latest, args.out)
        else:
            print("No enriched stats found. Run a sandbox episode with record_stats: true.")
