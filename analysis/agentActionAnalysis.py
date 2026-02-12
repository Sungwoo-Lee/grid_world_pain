## This script load stats data from the results directory and analyze the agent's actions.
## Use model type, id, and checkpoint to load the stats data.
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading stats data from the results directory...")

MODEL_TYPE = "JAX_RecurrentPPO"
MODEL_ID = "20260212-202012_rppo_128env_gru"
CHECKPOINT = 5000009

files = sorted(glob.glob(f"./results/{MODEL_TYPE}/{MODEL_ID}/stats/{CHECKPOINT}/ep*.csv"))

if not files:
    print("No stats files found.")
    exit()

all_stats = []
for file in files:
    # Extract episode number from filename (e.g., ep_1_stats.csv)
    filename = os.path.basename(file)
    try:
        episode_num = int(filename.split('_')[1])
    except (IndexError, ValueError):
        print(f"Warning: Could not extract episode number from {filename}. Skipping.")
        continue
    
    df = pd.read_csv(file)
    df['episode'] = episode_num
    all_stats.append(df)

# Concatenate all dataframes
full_stats = pd.concat(all_stats, ignore_index=True)

# Sort by episode and step
full_stats = full_stats.sort_values(['episode', 'step'])

print(f"Loaded {len(files)} files.")
print(f"Total rows in concatenated dataframe: {len(full_stats)}")
print("\nFirst 5 rows:")
print(full_stats.head())
print("\nLast 5 rows:")
print(full_stats.tail())

print("\nUnique episode numbers:")
print(sorted(full_stats['episode'].unique()))

# --- Scatter Plot for Rest and Eat Actions ---
print("\nGenerating scatter plot for 'Rest' and 'Eat' actions...")

# Filter for Rest and Eat actions
# Using 'Rest' and 'Eat' based on previous inspection
actions_to_plot = ["Rest", "Eat"]
filtered_stats = full_stats[full_stats['action'].isin(actions_to_plot)].copy()

if filtered_stats.empty:
    print(f"No actions matching {actions_to_plot} found in the data.")
else:
    print(f"Found {len(filtered_stats)} rows with actions: {actions_to_plot}")
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    sns.scatterplot(
        data=filtered_stats,
        x='satiation',
        y='injury',
        hue='action',
        palette={'Eat': 'green', 'Rest': 'blue'},
        alpha=0.6,
        s=50,
        edgecolor='w',
        linewidth=0.5
    )
    
    plt.title(f"Satiation vs Injury Level (Actions: {', '.join(actions_to_plot)})")
    plt.xlabel("Satiation")
    plt.ylabel("Injury Level")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    output_path = "analysis/action_scatter.png"
    plt.savefig(output_path)
    print(f"Scatter plot saved to: {output_path}")

    # Display counts
    print("\nAction counts in filtered data:")
    print(filtered_stats['action'].value_counts())
