#!/usr/bin/env python
"""What makes a trained agent hide? Episode-level factor analysis over a trajectory store.

Run-agnostic: the entity/resource/obstacle slot layout is derived from the run's own saved
config, so this works for any training run whose trajectories have been collected with
`scripts/eval/traj_collect/`.

Outcome
-------
Fraction of the agent's CHOSEN steps (t >= 1) spent in a bush. Row t=0 is the random spawn
state; it is excluded from the outcome and used only as an exogenous starting condition.

Regressor blocks
----------------
exogenous   : rerolled by the environment BEFORE the agent acts -> causally identified
consequence : produced during the episode -> reported as association only

Model
-----
Quasi-binomial GLM (binomial on the rate; SEs scaled by the Pearson overdispersion). Effects
are reported as percentage-point change in dwell per +1 SD of the regressor so that factors
on different natural scales are directly comparable.

Usage
-----
  python scripts/analysis/hiding_drivers.py --run results/JAX_RecurrentPPO/<RUN> --out <dir>
  python scripts/analysis/hiding_drivers.py --run <RUN> --stage aggregate   # cache only
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np, pyarrow.parquet as pq, yaml

INJ_EDGES = [1e-9, 25.0, 50.0]          # -> bins: 0, (0,25), [25,50), [50,inf)
NUT_EDGES = [25.0, 50.0, 75.0]
NEAR_D    = 2                            # Chebyshev radius for "an animal is near"


# ----------------------------------------------------------------- layout ----
def slot_layout(cfg: dict) -> dict:
    """Derive slot indices from the run's saved config. No environment rebuild."""
    env = cfg["environment"]
    pred, neu, s = [], [], 0
    for e in env["entities"]:
        n = e["count_high"]
        (pred if e["class"] == "predator" else neu).extend(range(s, s + n)); s += n
    bush, rock, s = [], [], 0
    for o in env["obstacles"]:
        n = o["count_high"]
        (bush if o.get("hides_agent") else rock).extend(range(s, s + n)); s += n
    food, amb, s = [], [], 0
    for r in env["resources"]:
        n = r["count_high"]
        (amb if max(r.get("damage", [0, 0])) > 0 else food).extend(range(s, s + n)); s += n
    return dict(pred=pred, neutral=neu, bush=bush, rock=rock, food=food, ambush=amb,
                n_animal=s if False else len(pred) + len(neu))


def find_store(run: str, checkpoint: str | None) -> str:
    tag = os.path.basename(run.rstrip("/"))
    pat = f"results/trajectories/{tag}/{checkpoint or '*'}/*/"
    hits = sorted(glob.glob(pat))
    if not hits:
        raise SystemExit(f"no trajectory store matches {pat}")
    if len(hits) > 1 and checkpoint is None:
        raise SystemExit("multiple checkpoints found; pass --checkpoint:\n  " +
                         "\n  ".join(hits))
    return hits[0]


def listcol(col, width):
    """Fixed-width parquet list column -> (n, width) array, without to_pylist()."""
    ch = col.chunks if hasattr(col, "chunks") else [col]
    return np.concatenate([c.flatten().to_numpy(zero_copy_only=False)
                           for c in ch]).reshape(-1, width)


# -------------------------------------------------------------- aggregate ----
def aggregate(store: str, lay: dict, verbose=True) -> dict:
    """One sweep of the step table -> per-episode arrays. Assumes (and asserts) that
    episodes are shard-aligned, seeds contiguous, and rows sorted by seed."""
    epf = sorted(glob.glob(store + "episodes_*.parquet"))
    ep = pq.read_table(epf, columns=["episode_seed", "length", "termination_reason",
                                     "animal_active", "obs_active", "res_allocated",
                                     "animal_detect_sampled", "animal_attack_delay_sampled",
                                     "animal_attack_range_sampled", "animal_max_stamina_sampled",
                                     "animal_property_sampled"])
    o = np.argsort(ep.column("episode_seed").to_numpy())
    seed = ep.column("episode_seed").to_numpy()[o]
    seed0, nep = int(seed[0]), len(seed)
    if seed.max() - seed0 + 1 != nep:
        raise SystemExit("episode seeds are not contiguous; this reader assumes they are")
    na = len(lay["pred"]) + len(lay["neutral"])
    L = lambda c, dt=np.float64: np.array(ep.column(c).to_pylist(), dtype=dt)[o]
    act, oa, ra = L("animal_active", bool), L("obs_active", bool), L("res_allocated", bool)
    prop = L("animal_property_sampled").reshape(nep, na, -1)
    P, R = lay["pred"], lay["neutral"]
    mean_over = lambda X, M: np.where(M.sum(1) > 0, (X * M).sum(1) / np.maximum(M.sum(1), 1), np.nan)
    pa, rb = act[:, P], act[:, R]

    A = dict(seed=seed, term=ep.column("termination_reason").to_numpy(zero_copy_only=False)[o],
             n_pred=pa.sum(1).astype(float), n_rab=rb.sum(1).astype(float),
             n_bush=oa[:, lay["bush"]].sum(1).astype(float),
             n_rock=oa[:, lay["rock"]].sum(1).astype(float),
             n_food=ra[:, lay["food"]].sum(1).astype(float),
             n_ambush=ra[:, lay["ambush"]].sum(1).astype(float),
             pred_detect=mean_over(L("animal_detect_sampled")[:, P], pa),
             pred_delay=mean_over(L("animal_attack_delay_sampled")[:, P], pa),
             pred_range=mean_over(L("animal_attack_range_sampled")[:, P], pa),
             pred_stamina=mean_over(L("animal_max_stamina_sampled")[:, P], pa),
             pred_predatorness=mean_over(prop[:, P, 1] - prop[:, P, 2], pa),
             rab_predatorness=mean_over(prop[:, R, 1] - prop[:, R, 2], rb))

    z = lambda: np.zeros(nep)
    G = {k: z() for k in ["n_rows", "n_steps", "bush_steps", "inj0", "nut0", "arow0", "acol0",
                          "inj_sum", "inj_max", "nut_sum", "dmg_sum", "n_ate", "n_rest",
                          "d_bush0", "d_pred0", "n_pred_near", "n_rab_near"]}
    IB, IBb = np.zeros((nep, 4)), np.zeros((nep, 4))
    NB, NBb = np.zeros((nep, 4)), np.zeros((nep, 4))
    cols = ["episode_seed", "t", "agent_in_bush", "injury_level", "nutrition", "damage",
            "ate_food", "rested", "agent_row", "agent_col", "obs_row", "obs_col",
            "animal_row", "animal_col"]
    nobs = oa.shape[1]
    files = sorted(glob.glob(store + "steps_*.parquet"))
    t0 = time.time()
    for fi, f in enumerate(files):
        tb = pq.read_table(f, columns=cols)
        sd = tb.column("episode_seed").to_numpy(); t = tb.column("t").to_numpy()
        gi = sd - seed0
        st = np.flatnonzero(t == 0); gidx = gi[st]
        if not np.array_equal(gidx, np.arange(gidx[0], gidx[0] + len(gidx))):
            raise SystemExit(f"{f}: episodes are not shard-aligned/contiguous")
        num = lambda c: tb.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
        bu, inj, nut = num("agent_in_bush"), num("injury_level"), num("nutrition")
        ar, ac = num("agent_row"), num("agent_col")
        red = lambda a: np.add.reduceat(a, st)
        G["n_rows"][gidx] += np.diff(np.append(st, len(t)))
        G["inj_sum"][gidx] += red(inj); G["nut_sum"][gidx] += red(nut)
        G["dmg_sum"][gidx] += red(num("damage"))
        G["n_ate"][gidx] += red(num("ate_food")); G["n_rest"][gidx] += red(num("rested"))
        G["inj_max"][gidx] = np.maximum.reduceat(inj, st)
        G["inj0"][gidx], G["nut0"][gidx] = inj[st], nut[st]
        G["arow0"][gidx], G["acol0"][gidx] = ar[st], ac[st]

        AR, AC = listcol(tb.column("animal_row"), na), listcol(tb.column("animal_col"), na)
        near = (np.maximum(np.abs(AR - ar[:, None]), np.abs(AC - ac[:, None])) <= NEAR_D) \
               & act[gi]
        pn = near[:, P].any(1); rn = near[:, R].any(1) & ~pn

        m = t >= 1
        li = gi[m] - gidx[0]; nb_ = len(gidx)
        bc = lambda w=None: np.bincount(li, weights=w, minlength=nb_)
        G["n_steps"][gidx] += bc(); G["bush_steps"][gidx] += bc(bu[m])
        G["n_pred_near"][gidx] += bc(pn[m].astype(float))
        G["n_rab_near"][gidx] += bc(rn[m].astype(float))
        ib = np.digitize(inj[m], INJ_EDGES); nbn = np.digitize(nut[m], NUT_EDGES)
        for M, W, Bn in ((IB, None, ib), (IBb, bu[m], ib), (NB, None, nbn), (NBb, bu[m], nbn)):
            M[gidx] += np.bincount(li * 4 + Bn, weights=W,
                                   minlength=nb_ * 4).reshape(-1, 4)
        OR, OC = listcol(tb.column("obs_row"), nobs), listcol(tb.column("obs_col"), nobs)
        a0r, a0c = ar[st][:, None], ac[st][:, None]
        ob = oa[gidx][:, lay["bush"]]
        db = np.where(ob, np.maximum(np.abs(OR[st][:, lay["bush"]] - a0r),
                                     np.abs(OC[st][:, lay["bush"]] - a0c)), np.inf)
        G["d_bush0"][gidx] = db.min(1)
        pact = act[gidx][:, P]
        dp = np.where(pact, np.maximum(np.abs(AR[st][:, P] - a0r),
                                       np.abs(AC[st][:, P] - a0c)), np.inf)
        G["d_pred0"][gidx] = dp.min(1)
        if verbose and fi % 25 == 0:
            print(f"  shard {fi}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

    assert np.allclose(IB.sum(1), G["n_steps"]) and np.allclose(NB.sum(1), G["n_steps"])
    assert (G["bush_steps"] <= G["n_steps"]).all()
    assert np.allclose(G["n_rows"], G["n_steps"] + 1), "expected exactly one reset row/episode"
    return dict(**A, **G, IB=IB, IBb=IBb, NB=NB, NBb=NBb)


# --------------------------------------------------------------------- fit ----
def fit_glms(D: dict, out_dir: str):
    import pandas as pd, statsmodels.api as sm
    from scipy import stats
    L, Y = D["n_steps"], D["bush_steps"]
    ns = np.maximum(L, 1)
    EXO = {"start_injury": D["inj0"], "start_nutrition": D["nut0"],
           "n_predators": D["n_pred"], "n_rabbits": D["n_rab"], "n_bushes": D["n_bush"],
           "n_rocks": D["n_rock"], "n_food": D["n_food"], "n_ambush_predators": D["n_ambush"],
           "spawn_dist_to_bush": D["d_bush0"]}
    EXO_P = {"pred_detection_range": D["pred_detect"], "pred_attack_delay": D["pred_delay"],
             "pred_attack_range": D["pred_range"], "pred_max_stamina": D["pred_stamina"],
             "pred_smell_predatorness": D["pred_predatorness"],
             "spawn_dist_to_predator": np.where(np.isfinite(D["d_pred0"]), D["d_pred0"], np.nan)}
    EXO_R = {"rab_smell_predatorness": D["rab_predatorness"]}
    END = {"frac_time_injured": D["IB"][:, 1:].sum(1) / ns,
           "frac_time_inj_severe": D["IB"][:, 3] / ns,
           "mean_injury": (D["inj_sum"] - D["inj0"]) / ns, "peak_injury": D["inj_max"],
           "frac_time_low_nutrition": (D["NB"][:, 0] + D["NB"][:, 1]) / ns,
           "mean_nutrition": (D["nut_sum"] - D["nut0"]) / ns,
           "total_damage_taken": D["dmg_sum"], "eat_rate": D["n_ate"] / ns,
           "rest_rate": D["n_rest"] / ns, "episode_length": L.astype(float),
           "frac_time_predator_near": D["n_pred_near"] / ns,
           "frac_time_rabbit_near": D["n_rab_near"] / ns}

    def fit(X, keep, label):
        k = keep & np.isfinite(X.to_numpy()).all(1)
        Xc = sm.add_constant(X[k], has_constant="add")
        y = np.column_stack([Y[k], (L - Y)[k]])
        m = sm.GLM(y, Xc, family=sm.families.Binomial()).fit()
        disp = m.pearson_chi2 / m.df_resid
        se = m.bse * np.sqrt(disp); z = m.params / se
        pbar = Y[k].sum() / L[k].sum(); s = pbar * (1 - pbar) * 100
        sd = np.r_[1.0, X[k].std().to_numpy()]
        null = sm.GLM(y, np.ones((int(k.sum()), 1)), family=sm.families.Binomial()).fit()
        return pd.DataFrame({"model": label, "term": Xc.columns, "n": int(k.sum()),
                             "coef": m.params, "se": se, "z": z,
                             "p": 2 * stats.norm.sf(np.abs(z)),
                             "dpp_per_unit": m.params * s, "dpp_per_sd": m.params * sd * s,
                             "pseudo_r2": 1 - m.deviance / null.deviance,
                             "overdispersion": disp})

    ALL = np.ones(len(L), bool); P1 = D["n_pred"] == 1; R1 = D["n_rab"] == 1
    uni = []
    for nm, v in {**EXO, **EXO_P, **EXO_R, **END}.items():
        keep = P1 if nm in EXO_P else (R1 if nm in EXO_R else ALL)
        blk = "consequence" if nm in END else "exogenous"
        r = fit(pd.DataFrame({nm: v}), keep & np.isfinite(v), "univariate").iloc[1:]
        uni.append(r.assign(block=blk))
    uni = pd.concat(uni).sort_values("dpp_per_sd", key=abs, ascending=False)

    multi = pd.concat([
        fit(pd.DataFrame(EXO), ALL, "M1 exogenous, all episodes"),
        fit(pd.DataFrame({**EXO, **EXO_P}).drop(columns=["n_predators"]), P1,
            "M2 exogenous + predator traits, 1-predator episodes"),
        fit(pd.DataFrame({**EXO, **EXO_P, **EXO_R}).drop(columns=["n_predators", "n_rabbits"]),
            P1 & R1, "M3 + rabbit smell, 1 predator + 1 rabbit"),
        fit(pd.DataFrame({**EXO, **END}), ALL, "M4 exogenous + consequences")])
    os.makedirs(out_dir, exist_ok=True)
    uni.to_csv(f"{out_dir}/univariate.csv", index=False)
    multi.to_csv(f"{out_dir}/multivariate.csv", index=False)
    return uni, multi


def cross_tabs(D: dict, out_dir: str) -> dict:
    """Within-episode / conditional views that the episode-level GLM cannot express."""
    L = np.maximum(D["n_steps"], 1)
    inj = {"labels": ["injury=0", "injury 0-25", "injury 25-50", "injury>=50"],
           "steps": D["IB"].sum(0).tolist(),
           "dwell": (100 * D["IBb"].sum(0) / np.maximum(D["IB"].sum(0), 1)).tolist()}
    nut = {"labels": ["nutr<25", "nutr 25-50", "nutr 50-75", "nutr>=75"],
           "steps": D["NB"].sum(0).tolist(),
           "dwell": (100 * D["NBb"].sum(0) / np.maximum(D["NB"].sum(0), 1)).tolist()}
    prox = {"predator_near": float(D["n_pred_near"].sum()),
            "rabbit_near": float(D["n_rab_near"].sum()),
            "neither": float(D["n_steps"].sum() - D["n_pred_near"].sum() - D["n_rab_near"].sum())}
    res = {"injury_bins": inj, "nutrition_bins": nut, "proximity_steps": prox,
           "overall_dwell": float(100 * D["bush_steps"].sum() / D["n_steps"].sum()),
           "mean_survival_steps": float(D["n_steps"].mean()),
           "episodes": int(len(D["n_steps"]))}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(f"{out_dir}/summary.json", "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="results/<ALGO>/<RUN_DIR>")
    ap.add_argument("--checkpoint", default=None, help="checkpoint step (default: the only one)")
    ap.add_argument("--out", default=None, help="output dir (default results/analysis/hiding_drivers/<tag>)")
    ap.add_argument("--cache", default=None, help="npz cache path for the aggregation pass")
    ap.add_argument("--stage", choices=["aggregate", "fit", "all"], default="all")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(f"{a.run}/models/config.yaml"))
    lay = slot_layout(cfg)
    store = find_store(a.run, a.checkpoint)
    tag = os.path.basename(a.run.rstrip("/"))
    out = a.out or f"results/analysis/hiding_drivers/{tag}"
    cache = a.cache or f"{out}/aggregate.npz"
    print(f"run   {a.run}\nstore {store}\nout   {out}")
    print(f"slots predators={lay['pred']} neutrals={lay['neutral']} "
          f"bushes={len(lay['bush'])} rocks={len(lay['rock'])} "
          f"food={len(lay['food'])} ambush={len(lay['ambush'])}")

    if a.stage in ("aggregate", "all") and not os.path.exists(cache):
        D = aggregate(store, lay)
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(cache, **D)
        print(f"cached -> {cache}")
    if a.stage == "aggregate":
        return
    D = dict(np.load(cache, allow_pickle=True))
    s = cross_tabs(D, out)
    print(f"\nepisodes {s['episodes']:,}  mean survival {s['mean_survival_steps']:.1f} steps  "
          f"overall dwell {s['overall_dwell']:.2f}%")
    uni, multi = fit_glms(D, out)
    print("\n=== univariate, ranked by |Δ percentage-points per +1 SD| ===")
    print(f"{'factor':28}{'block':13}{'n':>10}{'Δpp/SD':>10}{'Δpp/unit':>11}{'pseudoR2':>10}")
    for _, r in uni.iterrows():
        print(f"{r.term:28}{r.block:13}{r.n:>10,}{r.dpp_per_sd:>+10.2f}"
              f"{r.dpp_per_unit:>+11.3f}{r.pseudo_r2:>10.4f}")
    for label, grp in multi.groupby("model", sort=False):
        g = grp[grp.term != "const"]
        print(f"\n=== {label}  (n={int(grp.n.iloc[0]):,}, "
              f"overdispersion {grp.overdispersion.iloc[0]:.1f}, "
              f"pseudo-R2 {grp.pseudo_r2.iloc[0]:.4f}) ===")
        for _, r in g.iterrows():
            print(f"  {r.term:28}{r.dpp_per_sd:>+9.2f} pp/SD{r.dpp_per_unit:>+10.3f} pp/unit")
    print(f"\nwritten: {out}/univariate.csv, {out}/multivariate.csv, {out}/summary.json")


if __name__ == "__main__":
    main()
