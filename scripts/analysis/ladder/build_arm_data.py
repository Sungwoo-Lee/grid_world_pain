"""One sweep of one arm's trajectory store -> the aggregates every ladder figure reads.

WHY THIS EXISTS. Each ladder figure asks a different question, but they all need the same three
expensive things: how often the agent was in a bush, how far the nearest predator and the nearest
rabbit were when it decided, and what wound it woke up with. Re-deriving those separately per
figure would mean fourteen more passes over a billion rows. So this script does the sweep once
and writes a small file; the figure scripts each own their analysis and none of them touch parquet.

WHAT IT COMPUTES.
  * per episode: how long it lasted, how it ended, how many of its steps were spent in a bush,
    the randomised starting wound and hunger, and the odour strength drawn for the predators and
    for the rabbits in that world.
  * binned over steps: bush occupancy against nearest-predator distance and against
    nearest-rabbit distance, split by the randomised starting wound and, separately, by the
    wound the agent was actually carrying at the moment it decided.

TWO CONVENTIONS THAT MATTER.
  1. The row at t=0 is the world as handed to the agent, not a step it took. It is excluded from
     every numerator and denominator, and the action that produced row t was chosen while looking
     at row t-1 - so every predictor here is read off the PREVIOUS row.
  2. `body.random_start_injury` is true, so the wound at t=0 was assigned by the environment
     rather than earned. Splitting by it is a causal contrast. Splitting by the wound the agent is
     carrying mid-episode is not - a hurt agent got hurt by doing something - and both are computed
     so the figures can show the difference instead of asserting it.

Usage:  python scripts/analysis/ladder/build_arm_data.py [--arms A_baseline V5_sharp]
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ladder as L

STEP_COLS = ["episode_seed", "t", "agent_in_bush", "injury_level", "nutrition", "damage",
             "ate_food", "agent_row", "agent_col", "animal_row", "animal_col"]
EP_COLS = ["episode_seed", "length", "termination_reason", "animal_active",
           "animal_property_sampled"]
EARLY = 25          # steps before the randomised starting wound has washed out


def listcol(col, width):
    ch = col.chunks if hasattr(col, "chunks") else [col]
    return np.concatenate([c.flatten().to_numpy(zero_copy_only=False)
                           for c in ch]).reshape(-1, width)


def smell_channels(cfg):
    """The two odour channels that separate predators from rabbits - derived, not assumed."""
    ent = cfg["environment"]["entities"]
    pm = np.mean([e["properties"] for e in ent if e["class"] == "predator"], axis=0)
    nm = np.mean([e["properties"] for e in ent if e["class"] != "predator"], axis=0)
    d = np.asarray(pm) - np.asarray(nm)
    a, b = int(np.argmax(d)), int(np.argmin(d))
    if a == b or d[a] <= 0 or d[b] >= 0:
        raise SystemExit("this config does not separate predator and rabbit odour")
    return a, b


def build(arm: str, run: str, verbose=True) -> dict:
    cfg = L.arm_config(run)
    lay = L.slot_layout(cfg)
    P, R = lay["pred"], lay["neutral"]
    na = len(P) + len(R)
    ch1, ch2 = smell_channels(cfg)
    store = L.arm_store(arm)

    ep = pq.read_table(sorted(glob.glob(store + "episodes_*.parquet")), columns=EP_COLS)
    o = np.argsort(ep.column("episode_seed").to_numpy())
    seed = ep.column("episode_seed").to_numpy()[o]
    seed0, nep = int(seed[0]), len(seed)
    if seed.max() - seed0 + 1 != nep:
        raise SystemExit(f"{arm}: episode seeds are not contiguous")
    length = ep.column("length").to_numpy()[o].astype(np.float64)
    term = ep.column("termination_reason").to_numpy(zero_copy_only=False)[o].astype(int)
    act = listcol(ep.column("animal_active"), na)[o]
    flat = np.concatenate([c.flatten().to_numpy(zero_copy_only=False)
                           for c in ep.column("animal_property_sampled").chunks])
    if flat.size % (nep * na):
        raise SystemExit(f"{arm}: odour property array is not (episodes x animals x channels)")
    prop = flat.reshape(nep, na, -1)[o]
    pa, rb = act[:, P], act[:, R]
    mean_over = lambda X, M: np.where(M.sum(1) > 0, (X * M).sum(1) / np.maximum(M.sum(1), 1), np.nan)
    pred_olf = mean_over(prop[:, P, ch1] + prop[:, P, ch2], pa)
    rab_olf = mean_over(prop[:, R, ch1] + prop[:, R, ch2], rb)
    # A THIRD of all episodes contain no predator, and a third contain no rabbit. Those episodes
    # have no "distance to the nearest predator" and no "how strongly the predator smelled", and
    # they must be dropped from the grids that condition on those quantities. Leaving them in is
    # not a rounding issue: np.digitize files every NaN into the TOP bin and np.clip files every
    # inf into the FARTHEST distance bin, so the no-predator episodes - which hide far less,
    # because nothing is hunting them - silently become the reference group that everything else
    # is compared against.
    has_p, has_r = pa.sum(1) > 0, rb.sum(1) > 0
    n_pred, n_rab = pa.sum(1).astype(np.int32), rb.sum(1).astype(np.int32)

    # ---- accumulators -------------------------------------------------------
    z1, z2 = lambda n: np.zeros(n), lambda a, b: np.zeros((a, b))
    E = {k: np.zeros(nep) for k in ["n_steps", "bush_steps", "inj0", "nut0", "dmg", "n_ate",
                                    "bush_early", "steps_early"]}
    G = {
        "pd_bush": z2(L.DIST_MAX, 4), "pd_tot": z2(L.DIST_MAX, 4),      # x start injury
        "rd_bush": z2(L.DIST_MAX, 4), "rd_tot": z2(L.DIST_MAX, 4),
        "pdc_bush": z2(L.DIST_MAX, 4), "pdc_tot": z2(L.DIST_MAX, 4),    # x carried injury
        "rdc_bush": z2(L.DIST_MAX, 4), "rdc_tot": z2(L.DIST_MAX, 4),
        "dw_inj": z1(4), "dwt_inj": z1(4),                               # all steps
        "dw_early": z1(4), "dwt_early": z1(4),                           # first EARLY steps
        "dw_nut": z1(4), "dwt_nut": z1(4),
        "dw_carried": z1(4), "dwt_carried": z1(4),
        "dmg_inj": z1(4),                                                # damage taken per bin
    }
    OLF = {"rab_bush": z2(4, 4), "rab_tot": z2(4, 4),                    # odour quartile x injury
           "pred_bush": z2(4, 4), "pred_tot": z2(4, 4)}

    ib_ep = np.digitize(np.zeros(nep), L.INJ_EDGES)   # filled after the first shard sets inj0
    files = sorted(glob.glob(store + "steps_*.parquet"))
    t0 = time.time()
    for fi, f in enumerate(files):
        tb = pq.read_table(f, columns=STEP_COLS)
        sd = tb.column("episode_seed").to_numpy(); t = tb.column("t").to_numpy()
        gi = sd - seed0
        st = np.flatnonzero(t == 0); gidx = gi[st]
        if not np.array_equal(gidx, np.arange(gidx[0], gidx[0] + len(gidx))):
            raise SystemExit(f"{f}: episodes are not shard-aligned")
        n = len(t)
        ends = np.append(st[1:], n)
        estart = np.repeat(st, ends - st)

        num = lambda c: tb.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
        bu, inj, nut = num("agent_in_bush"), num("injury_level"), num("nutrition")
        ar, ac = num("agent_row"), num("agent_col")
        anr = listcol(tb.column("animal_row"), na).astype(np.float64)
        anc = listcol(tb.column("animal_col"), na).astype(np.float64)

        E["inj0"][gidx] = inj[st]; E["nut0"][gidx] = nut[st]
        red = lambda a: np.add.reduceat(a, st)
        E["n_steps"][gidx] += np.diff(np.append(st, n)) - 1        # t=0 is not a step
        E["bush_steps"][gidx] += red(bu) - bu[st]
        E["dmg"][gidx] += red(num("damage")); E["n_ate"][gidx] += red(num("ate_food"))

        # nearest active predator / rabbit, in chebyshev steps, on every row
        d = np.maximum(np.abs(anr - ar[:, None]), np.abs(anc - ac[:, None]))
        live = act[gi]                                              # per-row active mask
        dm = np.where(live, d, np.inf)
        dpred = dm[:, P].min(1); drab = dm[:, R].min(1)

        # the agent chose row t's action while looking at row t-1
        cur = np.arange(n)
        ok = cur > estart
        prev = cur[ok] - 1
        y = bu[ok]
        g = gi[ok]
        dpb = np.clip(dpred[prev], 1, L.DIST_MAX).astype(int) - 1
        drb = np.clip(drab[prev], 1, L.DIST_MAX).astype(int) - 1
        ib = np.digitize(E["inj0"][g], L.INJ_EDGES)                 # randomised, causal
        cb = np.digitize(inj[prev], L.INJ_EDGES)                    # carried, associational
        nb = np.digitize(E["nut0"][g], L.INJ_EDGES)

        hp, hr = has_p[g], has_r[g]          # only episodes that actually contain one
        np.add.at(G["pd_bush"], (dpb[hp], ib[hp]), y[hp])
        np.add.at(G["pd_tot"],  (dpb[hp], ib[hp]), 1.0)
        np.add.at(G["rd_bush"], (drb[hr], ib[hr]), y[hr])
        np.add.at(G["rd_tot"],  (drb[hr], ib[hr]), 1.0)
        np.add.at(G["pdc_bush"], (dpb[hp], cb[hp]), y[hp])
        np.add.at(G["pdc_tot"],  (dpb[hp], cb[hp]), 1.0)
        np.add.at(G["rdc_bush"], (drb[hr], cb[hr]), y[hr])
        np.add.at(G["rdc_tot"],  (drb[hr], cb[hr]), 1.0)
        np.add.at(G["dw_inj"], ib, y);   np.add.at(G["dwt_inj"], ib, 1.0)
        np.add.at(G["dw_nut"], nb, y);   np.add.at(G["dwt_nut"], nb, 1.0)
        np.add.at(G["dw_carried"], cb, y); np.add.at(G["dwt_carried"], cb, 1.0)
        np.add.at(G["dmg_inj"], ib, num("damage")[ok])

        early = ok & (cur <= estart + EARLY)
        ye, ge = bu[early], gi[early]
        ibe = np.digitize(E["inj0"][ge], L.INJ_EDGES)
        np.add.at(G["dw_early"], ibe, ye); np.add.at(G["dwt_early"], ibe, 1.0)
        np.add.at(E["bush_early"], ge, ye); np.add.at(E["steps_early"], ge, 1.0)

        if fi == 0:
            edges_r = np.nanquantile(rab_olf, [.25, .5, .75])
            edges_p = np.nanquantile(pred_olf, [.25, .5, .75])
        mr, mp = has_r[ge], has_p[ge]
        rq = np.digitize(rab_olf[ge][mr], edges_r)
        pq_ = np.digitize(pred_olf[ge][mp], edges_p)
        np.add.at(OLF["rab_bush"], (rq, ibe[mr]), ye[mr])
        np.add.at(OLF["rab_tot"],  (rq, ibe[mr]), 1.0)
        np.add.at(OLF["pred_bush"], (pq_, ibe[mp]), ye[mp])
        np.add.at(OLF["pred_tot"],  (pq_, ibe[mp]), 1.0)

        if verbose and (fi % 15 == 14 or fi == len(files) - 1):
            print(f"  {arm}: shard {fi+1}/{len(files)}  {time.time()-t0:5.0f}s", flush=True)

    if not np.allclose(E["n_steps"], length):
        raise SystemExit(f"{arm}: step count disagrees with the episode table's length column")

    os.makedirs(L.OUT_ROOT, exist_ok=True)
    np.savez_compressed(f"{L.OUT_ROOT}/{arm}_episodes.npz",
                        seed=seed, length=length, term=term,
                        bush_steps=E["bush_steps"], n_steps=E["n_steps"],
                        inj0=E["inj0"], nut0=E["nut0"], dmg=E["dmg"], n_ate=E["n_ate"],
                        bush_early=E["bush_early"], steps_early=E["steps_early"],
                        n_pred=n_pred, n_rab=n_rab,
                        pred_olf=pred_olf, rab_olf=rab_olf)

    out = {"arm": arm, "run": run, "store": store, "n_episodes": int(nep),
           "sensory": L.sensory_summary(cfg),
           "mean_survival": float(length.mean()),
           "bush_dwell_pct": float(100 * E["bush_steps"].sum() / E["n_steps"].sum()),
           "term_pct": {L.TERM_NAMES.get(k, str(k)): float(100 * np.mean(term == k))
                        for k in sorted(set(term.tolist()))},
           "odour_edges": {"rabbit": edges_r.tolist(), "predator": edges_p.tolist()},
           "episodes_with_a_predator": float(has_p.mean()),
           "episodes_with_a_rabbit": float(has_r.mean()),
           "grids": {k: v.tolist() for k, v in G.items()},
           "odour": {k: v.tolist() for k, v in OLF.items()}}
    L.save_json(arm, out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()
    runs = L.arm_runs()
    for arm in (a.arms or L.ARM_ORDER):
        t = time.time()
        r = build(arm, runs[arm])
        print(f"{arm:20} survival {r['mean_survival']:6.1f}  bush dwell "
              f"{r['bush_dwell_pct']:5.1f}%   ({time.time()-t:.0f}s)", flush=True)
