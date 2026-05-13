#!/usr/bin/env python3
"""motif_cluster.py — Offline M7 motif clustering for behavior-measure toolkit v1.

Reads eval-rollout outputs (produced by scripts/eval_rollout.py), computes 10
handcrafted features per threat-window, k-means clusters into k motif groups,
and writes analysis artefacts.

Usage
-----
    python scripts/motif_cluster.py \\
        --eval-root <output-root>/<run_tag>/<checkpoint_step>  \\
        --config <env-config-yaml>  \\
        [--n-clusters N]  \\
        [--seed S]  \\
        [--quiet]

Dependencies
------------
    sklearn:    pip install scikit-learn
    pandas:     pip install pandas pyarrow
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.config_loader import Config, load_behavior_measure_cfg, BehaviorMeasureCfg


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _entropy(counts):
    """Shannon entropy (bits) from a count array."""
    counts = np.array(counts, dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_window_features(ep, window_start, window_end, triggering_class, triggering_tag, bm_cfg):
    """Compute the 10 BM-v1 motif features for a single threat-window slice.

    Parameters
    ----------
    ep : dict
        Per-episode arrays loaded from .npz.
    window_start, window_end : int
        Inclusive slice indices.
    triggering_class : str
        "predator" or "rabbit".
    triggering_tag : str
        Tag string (e.g., "pred_0", "neut_1").
    bm_cfg : BehaviorMeasureCfg

    Returns
    -------
    dict[str, float] — keys = 10 feature names.
    """
    sl = slice(window_start, window_end + 1)
    pos = ep["agent_pos"][sl]          # [W, 2]
    actions = ep["action"][sl]         # [W]
    ate = ep["ate_food"][sl]           # [W] bool
    in_bush = ep["agent_in_bush"][sl]  # [W] bool
    noci = ep.get("nociception", np.zeros(len(actions)))[sl]

    dp = ep["dist_per_predator"]
    dn = ep["dist_per_neutral"]

    W = len(actions)
    if W == 0:
        return {k: 0.0 for k in _FEATURE_NAMES}

    # Pick correct distance array for the triggering instance
    tag_idx = -1
    if triggering_class == "predator" and dp.ndim == 2 and dp.shape[1] > 0:
        try:
            tag_idx = int(triggering_tag.split("_")[-1])
        except (ValueError, IndexError):
            tag_idx = 0
        tag_idx = min(tag_idx, dp.shape[1] - 1)
        threat_dists = dp[sl, tag_idx]
    elif triggering_class == "rabbit" and dn.ndim == 2 and dn.shape[1] > 0:
        try:
            tag_idx = int(triggering_tag.split("_")[-1])
        except (ValueError, IndexError):
            tag_idx = 0
        tag_idx = min(tag_idx, dn.shape[1] - 1)
        threat_dists = dn[sl, tag_idx]
    else:
        threat_dists = np.full(W, np.nan)

    # 1. net_displacement — L2 between start and end positions
    disp = pos[-1].astype(float) - pos[0].astype(float)
    net_displacement = float(np.linalg.norm(disp))

    # 2. path_length — sum of step-to-step L1 distances
    if W > 1:
        diffs = np.diff(pos.astype(float), axis=0)
        path_length = float(np.sum(np.abs(diffs).sum(axis=1)))
    else:
        path_length = 0.0

    # 3. threat_distance_change_rate — mean Δdist per step
    if not np.all(np.isnan(threat_dists)) and W > 1:
        valid_mask = ~np.isnan(threat_dists)
        d_arr = threat_dists[valid_mask]
        if len(d_arr) > 1:
            threat_distance_change_rate = float(np.mean(np.diff(d_arr)))
        else:
            threat_distance_change_rate = 0.0
    else:
        threat_distance_change_rate = 0.0

    # 4. min_threat_distance
    if not np.all(np.isnan(threat_dists)):
        min_threat_distance = float(np.nanmin(threat_dists))
    else:
        min_threat_distance = float(bm_cfg.cue_radius)

    # 5. bush_occupancy_fraction
    bush_occupancy_fraction = float(np.mean(in_bush))

    # 6. eat_events_per_window
    eat_events_per_window = float(np.sum(ate))

    # 7. action_entropy (over 5 possible actions: 0-4)
    n_actions = 5
    counts = np.array([np.sum(actions == a) for a in range(n_actions)], dtype=float)
    action_entropy = _entropy(counts)

    # 8. mode_action_fraction — fraction of steps taken as the modal action
    if W > 0:
        mode_action = int(np.bincount(actions, minlength=n_actions).argmax())
        mode_action_fraction = float(np.sum(actions == mode_action)) / W
    else:
        mode_action_fraction = 0.0

    # 9. stay_in_place_fraction — fraction of steps where position didn't change
    if W > 1:
        pos_changes = np.any(np.diff(pos, axis=0) != 0, axis=1)
        stay_in_place_fraction = float(np.mean(~pos_changes))
    else:
        stay_in_place_fraction = 1.0

    # 10. drive_injury_change — change in nociception across the window
    if len(noci) >= 2:
        drive_injury_change = float(noci[-1]) - float(noci[0])
    else:
        drive_injury_change = 0.0

    return {
        "net_displacement": net_displacement,
        "path_length": path_length,
        "threat_distance_change_rate": threat_distance_change_rate,
        "min_threat_distance": min_threat_distance,
        "bush_occupancy_fraction": bush_occupancy_fraction,
        "eat_events_per_window": eat_events_per_window,
        "action_entropy": action_entropy,
        "mode_action_fraction": mode_action_fraction,
        "stay_in_place_fraction": stay_in_place_fraction,
        "drive_injury_change": drive_injury_change,
    }


_FEATURE_NAMES = [
    "net_displacement", "path_length", "threat_distance_change_rate",
    "min_threat_distance", "bush_occupancy_fraction", "eat_events_per_window",
    "action_entropy", "mode_action_fraction", "stay_in_place_fraction",
    "drive_injury_change",
]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Offline M7 motif clustering (behavior-measure toolkit v1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--eval-root", required=True,
                        help="Path to a single eval-rollout result dir "
                             "(e.g., results/eval/<run_tag>/<ckpt_step>/).")
    parser.add_argument("--config", required=True,
                        help="Path to the training config YAML (reads behavior_measures.*).")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="Override behavior_measures.motif_kmeans_k.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override behavior_measures.motif_kmeans_seed.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # --- sklearn check ---
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import silhouette_score
    except ImportError:
        print("ERROR: scikit-learn is required. Install: pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas + pyarrow are required. Install: pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)

    config = Config.load_yaml(args.config)
    bm_cfg = load_behavior_measure_cfg(config)
    if bm_cfg is None:
        # Fallback with defaults
        bm_cfg = BehaviorMeasureCfg(
            enabled=True, cue_radius=3.0, obs_window=5, eval_n_episodes=10,
            eval_seeds=tuple(range(10)), eval_policy_mode="deterministic",
            eval_max_steps=500, eval_obs_noise="training", motif_window_K=7,
            motif_features=tuple(_FEATURE_NAMES), motif_kmeans_k=6,
            motif_kmeans_seed=42, motif_standardise="zscore_pooled",
            eval_output_root="results/eval",
        )

    n_clusters = args.n_clusters or bm_cfg.motif_kmeans_k
    seed = args.seed or bm_cfg.motif_kmeans_seed
    features_to_use = list(bm_cfg.motif_features)
    standardise = bm_cfg.motif_standardise

    eval_root = Path(args.eval_root)
    episodes_dir = eval_root / "episodes"
    windows_dir = eval_root / "windows"
    motifs_dir = eval_root / "motifs"
    motifs_dir.mkdir(exist_ok=True)

    # --- Load episodes ---
    ep_files = sorted(episodes_dir.glob("*.npz"))
    if not ep_files:
        print(f"ERROR: no .npz episode files found in {episodes_dir}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"[motif_cluster] Loading {len(ep_files)} episode(s)...", flush=True)
    episodes = []
    for f in ep_files:
        ep = dict(np.load(f, allow_pickle=False))
        episodes.append(ep)

    # --- Load threat-onset index ---
    parquet_path = windows_dir / "threat_onsets.parquet"
    json_path = windows_dir / "threat_onsets.json"
    if parquet_path.exists():
        df_onsets = pd.read_parquet(parquet_path)
    elif json_path.exists():
        with open(json_path) as f:
            onsets = json.load(f)
        df_onsets = pd.DataFrame(onsets) if onsets else pd.DataFrame()
    else:
        print(f"ERROR: no threat_onsets.parquet or .json found in {windows_dir}", file=sys.stderr)
        sys.exit(1)

    if len(df_onsets) == 0:
        print("WARNING: no threat-onset events; writing empty outputs.", file=sys.stderr)
        result = {"n_windows": 0, "n_clusters": n_clusters, "silhouette_mean": float("nan")}
        with open(motifs_dir / "silhouette.json", "w") as f:
            json.dump(result, f, indent=2)
        sys.exit(0)

    if not args.quiet:
        print(f"[motif_cluster] {len(df_onsets)} threat-onset windows to featurise.", flush=True)

    # --- Featurise windows ---
    rows = []
    for _, onset_row in df_onsets.iterrows():
        ep_idx = int(onset_row["episode_idx"])
        if ep_idx >= len(episodes):
            continue
        ep = episodes[ep_idx]
        feats = compute_window_features(
            ep,
            int(onset_row["window_start_t"]),
            int(onset_row["window_end_t"]),
            str(onset_row["triggering_class"]),
            str(onset_row["triggering_tag"]),
            bm_cfg,
        )
        row = {**onset_row.to_dict(), **feats}
        rows.append(row)

    if not rows:
        print("WARNING: featurisation produced no rows.", file=sys.stderr)
        sys.exit(0)

    df_feat = pd.DataFrame(rows)

    # --- Standardise ---
    X = df_feat[features_to_use].values.astype(float)
    # Replace any remaining NaN with column means
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    if standardise in ("zscore_pooled", "zscore_per_agent"):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    # --- K-Means ---
    actual_k = min(n_clusters, len(rows))
    if actual_k < n_clusters:
        warnings.warn(
            f"Fewer windows ({len(rows)}) than clusters ({n_clusters}); "
            f"reducing to {actual_k} clusters.", stacklevel=2
        )

    km = KMeans(n_clusters=actual_k, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)

    if not args.quiet:
        print(f"[motif_cluster] K-Means done: {actual_k} clusters.", flush=True)

    # Silhouette
    if actual_k >= 2 and len(rows) > actual_k:
        sil_mean = float(silhouette_score(X_scaled, labels))
        per_cluster_sil = []
        from sklearn.metrics import silhouette_samples
        sil_vals = silhouette_samples(X_scaled, labels)
        for k in range(actual_k):
            mask = labels == k
            per_cluster_sil.append(float(sil_vals[mask].mean()) if mask.any() else float("nan"))
    else:
        sil_mean = float("nan")
        per_cluster_sil = [float("nan")] * actual_k

    # Warn on degenerate clusters (< 1% of windows)
    for k in range(actual_k):
        frac = float(np.sum(labels == k)) / len(labels)
        if frac < 0.01:
            warnings.warn(f"Cluster {k} has only {frac:.1%} of windows (< 1% threshold).", stacklevel=2)

    # --- Write outputs ---
    # feature_vectors.parquet
    df_feat["cluster_label"] = [f"cluster_{lbl}" for lbl in labels]
    df_feat.to_parquet(motifs_dir / "feature_vectors.parquet", index=False)

    # cluster_assignments.parquet
    assign_cols = [c for c in df_feat.columns if c in
                   ["episode_idx", "t_star", "triggering_class", "triggering_tag",
                    "window_start_t", "window_end_t", "cluster_label"]]
    df_feat[assign_cols].to_parquet(motifs_dir / "cluster_assignments.parquet", index=False)

    # cluster_centroids.npy — [k, num_features] in z-scored space
    np.save(motifs_dir / "cluster_centroids.npy", km.cluster_centers_)

    # motif_distribution.json
    distribution = {}
    for k in range(actual_k):
        frac = float(np.sum(labels == k)) / len(labels)
        distribution[f"cluster_{k}"] = round(frac, 6)
    with open(motifs_dir / "motif_distribution.json", "w") as f:
        json.dump(distribution, f, indent=2)

    # silhouette.json
    sil_data = {
        "silhouette_mean": sil_mean if not np.isnan(sil_mean) else None,
        "per_cluster": [v if not np.isnan(v) else None for v in per_cluster_sil],
        "n_windows": len(labels),
        "n_clusters": actual_k,
    }
    with open(motifs_dir / "silhouette.json", "w") as f:
        json.dump(sil_data, f, indent=2)

    # exemplars.json — 5 nearest centroid windows per cluster
    exemplars = {}
    for k in range(actual_k):
        mask = np.where(labels == k)[0]
        if len(mask) == 0:
            exemplars[f"cluster_{k}"] = []
            continue
        centroid = km.cluster_centers_[k]
        dists = np.linalg.norm(X_scaled[mask] - centroid, axis=1)
        top5 = mask[np.argsort(dists)[:5]].tolist()
        exemplars[f"cluster_{k}"] = top5
    with open(motifs_dir / "exemplars.json", "w") as f:
        json.dump(exemplars, f, indent=2)

    if not args.quiet:
        print(f"[motif_cluster] Outputs written to: {motifs_dir}", flush=True)
        print(f"[motif_cluster] Silhouette: {sil_mean:.3f}" if not np.isnan(sil_mean) else "[motif_cluster] Silhouette: N/A", flush=True)
        print(f"[motif_cluster] Distribution: {distribution}", flush=True)


if __name__ == "__main__":
    main()
