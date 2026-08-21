#!/usr/bin/env python
"""trajectory_glm.py — factor analysis of bush use from a trajectory store.

Reads a store written by scripts/eval/traj_collect/collect_trajectories.py, computes
total bush-dwell steps per episode from the per-step table, joins the realised
per-episode environment draws, and fits a binomial GLM plus per-factor conditional means.

WHY THE OUTCOME IS A RATE, NOT A COUNT
--------------------------------------
Total bush-dwell steps is the intuitive outcome and it reverses the sign on the
strongest factor in the environment. As predator detection range rises 1->7, the
agent hides in a LARGER share of its episode (20%->57%) but for FEWER total steps
(49->26), because the episode collapses 243->45 steps. Modelling counts without
accounting for episode length measures survival, not behaviour. Hence Binomial
with episode length as the trial count. Episode length is reported separately as
an outcome in its own right.

Usage
-----
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
        scripts/analysis/trajectory_glm.py \
        --store results/trajectories/<run>/<ckpt>/<env_fp>/ \
        --run   results/JAX_RecurrentPPO/<run> \
        --out   results/analysis/trajectory_glm/
"""
import argparse, glob, json, sys, time
from pathlib import Path
import numpy as np, yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import Config                      # noqa: E402
from src.environment.config_loader import load_env_params  # noqa: E402


def bush_dwell_per_episode(store):
    """Sum agent_in_bush per episode. Reads 2 of 36 columns — the columnar win."""
    import pyarrow.parquet as pq
    dwell, t0 = {}, time.time()
    files = sorted(glob.glob(store + "steps_*.parquet"))
    for i, f in enumerate(files):
        tb = pq.read_table(f, columns=["episode_seed", "agent_in_bush"])
        s = tb.column("episode_seed").to_numpy()
        b = tb.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(np.int32)
        u, inv = np.unique(s, return_inverse=True)
        for k, v in zip(u, np.bincount(inv, weights=b).astype(np.int64)):
            dwell[int(k)] = int(v)
        if i % 50 == 0:
            print(f"  shard {i}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    return dwell


def build_design(store, run, dwell):
    import pyarrow.parquet as pq
    p = load_env_params(Config(yaml.safe_load(open(f"{run}/models/config.yaml"))))
    PRED = list(p.predator_indices)
    res_dmg = np.asarray(p.res_damage)
    HIDE = [i for i in range(res_dmg.shape[0]) if res_dmg[i, 1] > 0]
    BUSH = [i for i in range(p.obs_hides_agent.shape[0]) if bool(p.obs_hides_agent[i])]

    cols = ["episode_seed", "length", "animal_active", "animal_detect_sampled",
            "animal_attack_delay_sampled", "animal_attack_range_sampled",
            "animal_max_stamina_sampled", "obs_active", "res_allocated"]
    tb = pq.read_table(sorted(glob.glob(store + "episodes_*.parquet")), columns=cols)
    arr = lambda c, dt=None: np.array(tb.column(c).to_pylist(), dtype=dt) if dt \
        else np.array(tb.column(c).to_pylist())
    seed = tb.column("episode_seed").to_numpy()
    act = arr("animal_active", bool)
    return dict(
        seed=seed, length=tb.column("length").to_numpy().astype(np.int64),
        bush_steps=np.array([dwell.get(int(s), 0) for s in seed], np.int64),
        pact=act[:, PRED], n_pred=act[:, PRED].sum(1),
        n_bush=arr("obs_active", bool)[:, BUSH].sum(1),
        n_hide=arr("res_allocated", bool)[:, HIDE].sum(1),
        det=arr("animal_detect_sampled")[:, PRED],
        adly=arr("animal_attack_delay_sampled")[:, PRED],
        arng=arr("animal_attack_range_sampled")[:, PRED],
        stam=arr("animal_max_stamina_sampled")[:, PRED],
    )


def fit_glm(X, L, B, mask, label):
    import statsmodels.api as sm
    from scipy import stats
    Xc = sm.add_constant(X.values, has_constant="add")
    m = sm.GLM(np.column_stack([B[mask], (L - B)[mask]]), Xc,
               family=sm.families.Binomial()).fit()
    disp = m.pearson_chi2 / m.df_resid            # overdispersion: hiding comes in bouts
    se = m.bse * np.sqrt(disp)
    pbar = B[mask].sum() / L[mask].sum()
    print(f"\n=== {label} ===\nn = {mask.sum():,}   overdispersion = {disp:.1f} "
          f"(SEs scaled)\n{'term':22}{'coef':>10}{'se':>10}{'z':>9}{'Δ rate':>13}")
    print("-" * 64)
    out = {}
    for nm, c, s in zip(["(intercept)"] + list(X.columns), m.params, se):
        me = c * pbar * (1 - pbar) * 100
        print(f"{nm:22}{c:>10.4f}{s:>10.4f}{c/s:>9.1f}"
              f"{('' if nm=='(intercept)' else f'{me:+.2f} pp'):>13}")
        out[nm] = dict(coef=float(c), se=float(s), z=float(c/s), delta_pp=float(me))
    return out, float(disp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    store = a.store if a.store.endswith("/") else a.store + "/"
    Path(a.out).mkdir(parents=True, exist_ok=True)

    d = build_design(store, a.run.rstrip("/"), bush_dwell_per_episode(store))
    L, B, npred = d["length"], d["bush_steps"], d["n_pred"]
    import pandas as pd

    res = {}
    res["all"], res["disp_all"] = fit_glm(
        pd.DataFrame({"n_predators": npred, "n_bushes": d["n_bush"],
                      "n_ambush_pred": d["n_hide"]}), L, B,
        np.ones(len(L), bool), "all episodes")

    one = npred == 1
    r, slot = np.arange(one.sum()), np.argmax(d["pact"][one], axis=1)
    res["one_pred"], res["disp_one"] = fit_glm(pd.DataFrame({
        "detection_range": d["det"][one][r, slot],
        "attack_delay": d["adly"][one][r, slot],
        "attack_range": d["arng"][one][r, slot],
        "max_stamina": d["stam"][one][r, slot],
        "n_bushes": d["n_bush"][one], "n_ambush_pred": d["n_hide"][one],
    }), L, B, one, "one-predator episodes (traits unambiguous)")

    json.dump(res, open(Path(a.out) / "glm.json", "w"), indent=1)
    np.savez_compressed(Path(a.out) / "design.npz", **d)
    print(f"\nwrote {a.out}/glm.json and design.npz")


if __name__ == "__main__":
    main()
