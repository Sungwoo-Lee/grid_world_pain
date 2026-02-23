import os, glob, argparse, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns

# Silence Pandas fragmentation warning
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def main():
    parser = argparse.ArgumentParser(description="Analyze agent actions from evaluation stats.")
    parser.add_argument("--results_dir", type=str, help="Path to the results directory (e.g., results/JAX_RecurrentPPO/ID)")
    parser.add_argument("--checkpoint", type=int, help="Checkpoint number to analyze")
    parser.add_argument("--model_type", type=str, default="JAX_RecurrentPPO", help="Model type (default: JAX_RecurrentPPO)")
    parser.add_argument("--model_id", type=str, help="Model ID")
    parser.add_argument("--output", type=str, default="analysis/action_scatter.png", help="Output path for the plot")
    parser.add_argument("--title", type=str, help="Custom title for the plot")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    
    args = parser.parse_known_args()[0]

    # Priority logic for directory finding:
    # 1. results_dir and checkpoint provided
    # 2. model_type, model_id, and checkpoint provided
    # 3. Hardcoded defaults (for manual runs)
    
    if args.results_dir and args.checkpoint:
        stats_path_pattern = f"{args.results_dir}/stats/{args.checkpoint}/*ep_stats.csv"
    elif args.model_id and args.checkpoint:
        stats_path_pattern = f"./results/{args.model_type}/{args.model_id}/stats/{args.checkpoint}/*ep_stats.csv"
    else:
        # Fallback to hardcoded defaults if nothing else is provided
        MODEL_TYPE = "JAX_RecurrentPPO"
        MODEL_ID = "20260221-154412_rppoNMN_128env_Multip_G40"
        CHECKPOINT = 9000030
        stats_path_pattern = f"./results/{MODEL_TYPE}/{MODEL_ID}/stats/{CHECKPOINT}/*ep_stats.csv"

    if not args.quiet:
        print(f"Loading stats data from: {stats_path_pattern}")
    files = sorted(glob.glob(stats_path_pattern))
    
    if not files:
        if not args.quiet:
            print("No stats files found.")
        return

    all_stats = []
    for file in files:
        filename = os.path.basename(file)
        # Extract episode number from filename (e.g., 000001ep_stats.csv or ep_1_stats.csv)
        match = re.search(r'(\d+)ep', filename)
        if match:
            episode_num = int(match.group(1))
        else:
            # Fallback for old format ep_1_stats.csv
            match = re.search(r'ep_(\d+)', filename)
            if match:
                episode_num = int(match.group(1))
            else:
                if not args.quiet:
                    print(f"Warning: Could not extract episode number from {filename}. Skipping.")
                continue
        
        df = pd.read_csv(file)
        df['episode'] = episode_num
        all_stats.append(df)

    # Concatenate all dataframes
    if not all_stats:
        if not args.quiet:
            print("No valid data frames to process.")
        return
        
    full_stats = pd.concat(all_stats, ignore_index=True)

    # Sort by episode and step
    full_stats = full_stats.sort_values(['episode', 'step'])

    if not args.quiet:
        print(f"Loaded {len(files)} files.")
        print(f"Total rows in concatenated dataframe: {len(full_stats)}")
    
    # --- Scatter Plot for Rest and Eat Actions ---
    if not args.quiet:
        print("\nGenerating scatter plot for 'Rest' and 'Eat' actions...")

    # Filter for Rest and Eat actions
    actions_to_plot = ["Rest", "Eat"]
    if 'action' not in full_stats.columns:
        if not args.quiet:
            print("Error: 'action' column not found in stats.")
        return
        
    filtered_stats = full_stats[full_stats['action'].isin(actions_to_plot)].copy()

    if filtered_stats.empty:
        if not args.quiet:
            print(f"Warning: No actions matching {actions_to_plot} found. Plotting all actions instead.")
        filtered_stats = full_stats.copy()
        hue_param = 'action'
        palette_param = None
    else:
        if not args.quiet:
            print(f"Found {len(filtered_stats)} rows with actions: {actions_to_plot}")
        hue_param = 'action'
        palette_param = {'Eat': 'green', 'Rest': 'blue'}
        
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    sns.scatterplot(
        data=filtered_stats,
        x='satiation',
        y='injury',
        hue=hue_param,
        palette=palette_param,
        alpha=0.6,
        s=50,
        edgecolor='w',
        linewidth=0.5
    )
    
    title_str = f"Satiation vs Injury Level (Actions: {', '.join(actions_to_plot)})"
    if args.title:
        title_str = f"{args.title}\n{title_str}"
    plt.title(title_str)
    plt.xlabel("Satiation")
    plt.ylabel("Injury Level")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    output_path = args.output
    plt.savefig(output_path)
    if not args.quiet:
        print(f"Scatter plot saved to: {output_path}")

        # Display counts
        print("\nAction counts in filtered data:")
        print(filtered_stats['action'].value_counts())

if __name__ == "__main__":
    main()
