#!/usr/bin/env python3
"""collect_arm_data.py - the sensor-ladder arm sweep, ported onto core/scan.

A faithful port of `scripts/analysis/ladder/build_arm_data.py`. It must reproduce that script's
output exactly - the gate is `scripts/analysis/core/golden.py` against
`results/_golden_prerefactor_20260904/`, and anything it does not reproduce bit-for-bit (integer
fields) or inside rtol=1e-12 (floats) is a defect in this file, not a licence to adjust the gate.

What changed: the shard loop, the episode-boundary bookkeeping, the seed-contiguity and
shard-alignment asserts, and the step-count cross-check all moved into `core`. What did NOT change:
every accumulation below, which is the study's own arithmetic and is deliberately still ordinary
NumPy in the study's own folder.

Two conventions are preserved exactly rather than tidied, because tidying them would move published
numbers:
  * `bush_steps` excludes the reset row; `dmg` and `n_ate` include it. Both forms are asked for
    explicitly.
  * the odour quartile edges are computed from the FIRST shard only, then reused for every later
    shard. That is what the original does, so that is what this does.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis", "core"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis", "ladder"))
os.chdir(ROOT)

import env as ENV                    # noqa: E402
import scan as SCAN                  # noqa: E402
import store as STORE                # noqa: E402
import _ladder as L                  # noqa: E402  - study facts: arm order, bin edges, run paths

EARLY = 25
EP_COLS = ["episode_seed", "length", "termination_reason", "animal_active",
           "animal_property_sampled"]
STEP_COLS = ["episode_seed", "t", "agent_in_bush", "injury_level", "nutrition", "damage",
             "ate_food", "agent_row", "agent_col", "animal_row", "animal_col"]


def build(arm: str, run: str, verbose: bool = True) -> dict:
    cfg = L.arm_config(run)
    lay = ENV.slot_layout(cfg)
    P, R = lay["pred"], lay["neutral"]
    na = lay["n_animal"]
    ch1, ch2 = ENV.smell_channels(cfg)

    st = STORE.open_run(L.arm_stores(arm), EP_COLS)
    nep, seed0 = st.n_episodes, st.seed0

    length = st.episode("length", np.float64)
    term = st.episode("termination_reason").astype(int)
    act = st.episode_list("animal_active", na)
    prop = st.episode_property("animal_property_sampled", na)

    pa, rb = act[:, P], act[:, R]
    mean_over = lambda X, M: np.where(M.sum(1) > 0, (X * M).sum(1) / np.maximum(M.sum(1), 1), np.nan)
    pred_olf = mean_over(prop[:, P, ch1] + prop[:, P, ch2], pa)
    rab_olf = mean_over(prop[:, R, ch1] + prop[:, R, ch2], rb)
    # A THIRD of episodes contain no predator and a third no rabbit. They have no "distance to the
    # nearest predator" and no "how strongly it smelled", so they must be dropped from any grid
    # conditioned on those. Left in, np.digitize files every NaN into the TOP bin and np.clip files
    # every inf into the FARTHEST distance bin, and the no-predator episodes - which hide far less,
    # nothing hunting them - silently become the reference group everything is compared against.
    has_p, has_r = pa.sum(1) > 0, rb.sum(1) > 0
    n_pred, n_rab = pa.sum(1).astype(np.int32), rb.sum(1).astype(np.int32)

    z1, z2 = lambda n: np.zeros(n), lambda a, b: np.zeros((a, b))
    E = {k: np.zeros(nep) for k in ["n_steps", "bush_steps", "inj0", "nut0", "dmg", "n_ate",
                                    "bush_early", "steps_early"]}
    G = {
        "pd_bush": z2(L.DIST_MAX, 4), "pd_tot": z2(L.DIST_MAX, 4),
        "rd_bush": z2(L.DIST_MAX, 4), "rd_tot": z2(L.DIST_MAX, 4),
        "pdc_bush": z2(L.DIST_MAX, 4), "pdc_tot": z2(L.DIST_MAX, 4),
        "rdc_bush": z2(L.DIST_MAX, 4), "rdc_tot": z2(L.DIST_MAX, 4),
        "dw_inj": z1(4), "dwt_inj": z1(4),
        "dw_early": z1(4), "dwt_early": z1(4),
        "dw_nut": z1(4), "dwt_nut": z1(4),
        "dw_carried": z1(4), "dwt_carried": z1(4),
        "dmg_inj": z1(4),
    }
    OLF = {"rab_bush": z2(4, 4), "rab_tot": z2(4, 4),
           "pred_bush": z2(4, 4), "pred_tot": z2(4, 4)}
    edges = {}

    def collect(fr, acc, fi):
        gi, gidx = fr.episode_id, fr.episodes
        bu = fr.raw("agent_in_bush"); inj = fr.raw("injury_level"); nut = fr.raw("nutrition")
        ar, ac = fr.raw("agent_row"), fr.raw("agent_col")
        anr = fr.list_raw("animal_row", na).astype(np.float64)
        anc = fr.list_raw("animal_col", na).astype(np.float64)

        E["inj0"][gidx] = fr.at_initial(inj); E["nut0"][gidx] = fr.at_initial(nut)
        E["n_steps"][gidx] += fr.steps_per_episode
        E["bush_steps"][gidx] += fr.per_episode_sum(bu)
        E["dmg"][gidx] += fr.per_episode_sum_with_initial(fr.raw("damage"))
        E["n_ate"][gidx] += fr.per_episode_sum_with_initial(fr.raw("ate_food"))

        d = np.maximum(np.abs(anr - ar[:, None]), np.abs(anc - ac[:, None]))
        live = act[gi]
        dm = np.where(live, d, np.inf)
        dpred = dm[:, P].min(1); drab = dm[:, R].min(1)

        prev = fr.prev                       # the row the action was chosen on
        y = bu[fr.is_step]
        g = fr.episode_of_step
        dpb = np.clip(dpred[prev], 1, L.DIST_MAX).astype(int) - 1
        drb = np.clip(drab[prev], 1, L.DIST_MAX).astype(int) - 1
        ib = np.digitize(E["inj0"][g], L.INJ_EDGES)
        cb = np.digitize(inj[prev], L.INJ_EDGES)
        nb = np.digitize(E["nut0"][g], L.INJ_EDGES)

        hp, hr = has_p[g], has_r[g]
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
        np.add.at(G["dmg_inj"], ib, fr.raw("damage")[fr.is_step])

        early = fr.is_step & (np.arange(fr.n) <= fr.estart + EARLY)
        ye, ge = bu[early], gi[early]
        ibe = np.digitize(E["inj0"][ge], L.INJ_EDGES)
        np.add.at(G["dw_early"], ibe, ye); np.add.at(G["dwt_early"], ibe, 1.0)
        np.add.at(E["bush_early"], ge, ye); np.add.at(E["steps_early"], ge, 1.0)

        if fi == 0:
            edges["r"] = np.nanquantile(rab_olf, [.25, .5, .75])
            edges["p"] = np.nanquantile(pred_olf, [.25, .5, .75])
        mr, mp = has_r[ge], has_p[ge]
        rq = np.digitize(rab_olf[ge][mr], edges["r"])
        pq_ = np.digitize(pred_olf[ge][mp], edges["p"])
        np.add.at(OLF["rab_bush"], (rq, ibe[mr]), ye[mr])
        np.add.at(OLF["rab_tot"],  (rq, ibe[mr]), 1.0)
        np.add.at(OLF["pred_bush"], (pq_, ibe[mp]), ye[mp])
        np.add.at(OLF["pred_tot"],  (pq_, ibe[mp]), 1.0)

    SCAN.sweep(st, STEP_COLS, collect, verbose=verbose, label=arm)

    seed = st.seeds
    os.makedirs(L.OUT_ROOT, exist_ok=True)
    np.savez_compressed(f"{L.OUT_ROOT}/{arm}_episodes.npz",
                        seed=seed, length=length, term=term,
                        bush_steps=E["bush_steps"], n_steps=E["n_steps"],
                        inj0=E["inj0"], nut0=E["nut0"], dmg=E["dmg"], n_ate=E["n_ate"],
                        bush_early=E["bush_early"], steps_early=E["steps_early"],
                        n_pred=n_pred, n_rab=n_rab,
                        pred_olf=pred_olf, rab_olf=rab_olf)

    out = {"arm": arm, "run": run, "stores": L.arm_stores(arm), "n_episodes": int(nep),
           "seed_range": [int(seed.min()), int(seed.max())],
           "sensory": L.sensory_summary(cfg),
           "mean_survival": float(length.mean()),
           "bush_dwell_pct": float(100 * E["bush_steps"].sum() / E["n_steps"].sum()),
           "term_pct": {L.TERM_NAMES.get(k, str(k)): float(100 * np.mean(term == k))
                        for k in sorted(set(term.tolist()))},
           "odour_edges": {"rabbit": edges["r"].tolist(), "predator": edges["p"].tolist()},
           "episodes_with_a_predator": float(has_p.mean()),
           "episodes_with_a_rabbit": float(has_r.mean()),
           "grids": {k: v.tolist() for k, v in G.items()},
           "odour": {k: v.tolist() for k, v in OLF.items()}}
    L.save_json(arm, out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="*", default=L.ARM_ORDER)
    a = ap.parse_args()
    runs = L.arm_runs()
    for arm in a.arms:
        build(arm, runs[arm])
        print(f"{arm}: scanned")
