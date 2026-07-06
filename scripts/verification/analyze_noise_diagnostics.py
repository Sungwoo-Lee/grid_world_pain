"""Post-evaluation noise diagnostics analysis.

Usage:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/analyze_noise_diagnostics.py results/JAX_RecurrentPPO/<run>/stats/<checkpoint>/

Reads eval CSVs with true_* and obs_* columns, computes per-modality:
  - noise_std (should match configured sigma)
  - noise_mean (should be ~0, unbiased)
  - SNR (signal_std / noise_std)
  - clip_frac (fraction of steps where noise was clipped)
"""
import sys
import os
import glob
import pandas as pd
import numpy as np

MODALITY_PAIRS = [
    ("Injury",                  ["obs_intero_injury"],        ["true_intero_injury"]),
    ("Nutrition",               ["obs_intero_nutrition"],     ["true_intero_nutrition"]),
    ("Satiation",               ["obs_intero_satiation"],     ["true_intero_satiation"]),
    ("Interoceptive Nociception", ["obs_intero_nociception"], ["true_intero_nociception"]),
    ("Extero Nociception",      ["obs_noc"],                  ["true_noc"]),
    ("Location",                ["obs_loc_r", "obs_loc_c"],   ["true_loc_r", "true_loc_c"]),
]
# Olfaction, Collision, Visual, Proprioception have variable dims — detected dynamically

EXPECTED_SIGMA = {
    "Injury": 0.05, "Nutrition": 0.10, "Satiation": 0.10,
    "Interoceptive Nociception": 0.1,
    "Extero Nociception": 0.01, "Olfaction": 0.15, "Collision": 0.01,
    "Proprioception": 0.05, "Visual": 0.05, "Location": 0.01,
}


def analyze_stats_dir(stats_dir):
    csv_files = sorted(glob.glob(os.path.join(stats_dir, "*ep_stats.csv")))
    if not csv_files:
        print(f"No CSV files found in {stats_dir}")
        return

    # Check first file for true_* columns
    sample = pd.read_csv(csv_files[0], nrows=1)
    true_cols = [c for c in sample.columns if c.startswith("true_")]
    if not true_cols:
        print("No true_* columns found — is testing.record_true_observations set to true in eval config?")
        return

    # Load all episodes
    dfs = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(csv_files)} episodes, {len(df)} total steps\n")

    # Detect variable-dim modalities from column names
    obs_olf_cols = sorted([c for c in df.columns if c.startswith("obs_olf_")])
    true_olf_cols = sorted([c for c in df.columns if c.startswith("true_olf_")])
    obs_coll_cols = sorted([c for c in df.columns if c.startswith("obs_coll_")])
    true_coll_cols = sorted([c for c in df.columns if c.startswith("true_coll_")])
    obs_vis_cols = sorted([c for c in df.columns if c.startswith("obs_vis_")])
    true_vis_cols = sorted([c for c in df.columns if c.startswith("true_vis_")])
    obs_prop_cols = sorted([c for c in df.columns if c.startswith("obs_prop_")])
    true_prop_cols = sorted([c for c in df.columns if c.startswith("true_prop_")])

    all_pairs = list(MODALITY_PAIRS)
    if obs_olf_cols and true_olf_cols:
        all_pairs.append(("Olfaction", obs_olf_cols, true_olf_cols))
    if obs_coll_cols and true_coll_cols:
        all_pairs.append(("Collision", obs_coll_cols, true_coll_cols))
    if obs_vis_cols and true_vis_cols:
        all_pairs.append(("Visual", obs_vis_cols, true_vis_cols))
    if obs_prop_cols and true_prop_cols:
        all_pairs.append(("Proprioception", obs_prop_cols, true_prop_cols))

    print(f"{'Modality':<22} {'σ_expected':>10} {'noise_std':>10} {'noise_mean':>10} "
          f"{'SNR':>8} {'clip_frac':>10} {'Status'}")
    print("-" * 90)

    for name, obs_cols, true_cols in all_pairs:
        if not all(col in df.columns for col in obs_cols + true_cols):
            continue
            
        noised = df[obs_cols].values.flatten()
        true = df[true_cols].values.flatten()
        noise = noised - true

        noise_std = np.std(noise)
        noise_mean = np.mean(noise)
        signal_std = np.std(true)
        snr = signal_std / noise_std if noise_std > 1e-8 else float('inf')

        # Clipping: where noise was applied but observation didn't change
        # (conservative: count where |noise| < 1e-8 but true != noised boundary)
        clip_frac = np.mean(np.abs(noise) < 1e-8)  # Fraction with zero effective noise

        expected_sig = EXPECTED_SIGMA.get(name, "?")
        if isinstance(expected_sig, float):
            ratio = noise_std / expected_sig if expected_sig > 0 else float('inf')
            # state_dependent noise (Injury, Interoceptive Nociception) will have
            # higher/lower std than a constant alpha, but let's use a wider range
            if name in ("Injury", "Interoceptive Nociception"):
                status = "OK (SD)" if 0.5 <= ratio <= 2.0 else f"CHECK ({ratio:.2f}x)"
            else:
                status = "OK" if 0.8 <= ratio <= 1.2 else f"MISMATCH ({ratio:.2f}x)"
        else:
            status = "?"

        print(f"{name:<22} {expected_sig:>10} {noise_std:>10.5f} {noise_mean:>+10.5f} "
              f"{snr:>8.2f} {clip_frac:>10.4f} {status}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_noise_diagnostics.py <stats_dir>")
        sys.exit(1)
    analyze_stats_dir(sys.argv[1])
