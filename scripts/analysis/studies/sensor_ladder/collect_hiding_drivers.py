#!/usr/bin/env python3
"""collect_hiding_drivers.py - the episode-level factor sweep, ported onto core/scan.

A faithful port of `aggregate()` in `scripts/analysis/hiding_drivers.py`, gated against that
script's `aggregate.npz` by `scripts/analysis/core/golden.py`.

TWO THINGS THIS PORT DELIBERATELY DOES NOT DO.

It does not replace `hiding_drivers.py`. That script stays: `scripts/analysis/supplementary/README.md`
names it as the producer for the earlier trajectory-factors study, so "reproduce as published" fails
without it. This is a second implementation that must agree with it, not a successor.

It does not gate on the regression CSVs. They carry p-values like 2.26e-212, where a one-ulp change
in a z-statistic moves the value by ~1e-9 relative, so no tolerance over them is meaningful. The
gate is `aggregate.npz`, bit-exact, from which the CSVs follow deterministically - verified: an
identical npz gives byte-identical CSVs.

ONE DIVERGENCE, REGISTERED. `hiding_drivers.py:214` bins injury CONTEMPORANEOUSLY - the injury at
row t against the bush state at row t - where every other predictor in this analysis comes from row
t-1. That feeds injury cross-tabs published in the a01 document, so this port REPRODUCES it
bug-for-bug and gates on the reproduction. Correcting it is a separate, adjudicated change with its
own pre-declared expected diff; doing it here would move a published number under cover of a
refactor, which is exactly what the divergence register exists to prevent.
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis", "core"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis", "ladder"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis"))
os.chdir(ROOT)

import env as ENV                    # noqa: E402
import scan as SCAN                  # noqa: E402
import store as STORE                # noqa: E402
import _ladder as L                  # noqa: E402
from hiding_drivers import INJ_EDGES, NUT_EDGES, NEAR_D   # noqa: E402  - the study's own bin edges

EP_COLS = ["episode_seed", "length", "termination_reason", "animal_active", "obs_active",
           "res_allocated", "animal_detect_sampled", "animal_attack_delay_sampled",
           "animal_attack_range_sampled", "animal_max_stamina_sampled", "animal_property_sampled"]
STEP_COLS = ["episode_seed", "t", "agent_in_bush", "injury_level", "nutrition", "damage",
             "ate_food", "rested", "agent_row", "agent_col", "obs_row", "obs_col",
             "animal_row", "animal_col"]


def aggregate(roots, lay: dict, chans: tuple[int, int], verbose: bool = True) -> dict:
    st = STORE.open_run(roots, EP_COLS)
    nep = st.n_episodes
    na, P, R = lay["n_animal"], lay["pred"], lay["neutral"]

    col = lambda c, dt=np.float64: np.array(st._ep.column(c).to_pylist(), dtype=dt)[st.order]
    act, oa, ra = col("animal_active", bool), col("obs_active", bool), col("res_allocated", bool)
    prop = col("animal_property_sampled").reshape(nep, na, -1)
    mean_over = lambda X, M: np.where(M.sum(1) > 0, (X * M).sum(1) / np.maximum(M.sum(1), 1), np.nan)
    pa, rb = act[:, P], act[:, R]
    det = col("animal_detect_sampled")

    A = dict(seed=st.seeds, term=st.episode("termination_reason"),
             n_pred=pa.sum(1).astype(float), n_rab=rb.sum(1).astype(float),
             n_bush=oa[:, lay["bush"]].sum(1).astype(float),
             n_rock=oa[:, lay["rock"]].sum(1).astype(float),
             n_food=ra[:, lay["food"]].sum(1).astype(float),
             n_ambush=ra[:, lay["ambush"]].sum(1).astype(float),
             pred_detect=mean_over(det[:, P], pa),
             pred_delay=mean_over(col("animal_attack_delay_sampled")[:, P], pa),
             pred_range=mean_over(col("animal_attack_range_sampled")[:, P], pa),
             pred_stamina=mean_over(col("animal_max_stamina_sampled")[:, P], pa),
             pred_predatorness=mean_over(prop[:, P, chans[0]] - prop[:, P, chans[1]], pa),
             rab_predatorness=mean_over(prop[:, R, chans[0]] - prop[:, R, chans[1]], rb),
             pred_olf_ch1=mean_over(prop[:, P, chans[0]], pa),
             pred_olf_ch2=mean_over(prop[:, P, chans[1]], pa),
             rab_olf_ch1=mean_over(prop[:, R, chans[0]], rb),
             rab_olf_ch2=mean_over(prop[:, R, chans[1]], rb),
             pred_olf_intensity=mean_over(prop[:, P, chans[0]] + prop[:, P, chans[1]], pa),
             rab_olf_intensity=mean_over(prop[:, R, chans[0]] + prop[:, R, chans[1]], rb),
             pred_detect_max=np.where(pa.sum(1) > 0,
                 np.nanmax(np.where(pa, det[:, P], np.nan), axis=1), np.nan),
             pred_detect_min=np.where(pa.sum(1) > 0,
                 np.nanmin(np.where(pa, det[:, P], np.nan), axis=1), np.nan))

    z = lambda: np.zeros(nep)
    G = {k: z() for k in ["n_rows", "n_steps", "bush_steps", "inj0", "nut0", "arow0", "acol0",
                          "inj_sum", "inj_max", "nut_sum", "dmg_sum", "n_ate", "n_rest",
                          "d_bush0", "d_pred0", "n_pred_near", "n_rab_near"]}
    IB, IBb = np.zeros((nep, 4)), np.zeros((nep, 4))
    NB, NBb = np.zeros((nep, 4)), np.zeros((nep, 4))
    nobs = oa.shape[1]

    def collect(fr, _acc, fi):
        gi, gidx, st_rows = fr.episode_id, fr.episodes, fr.initial
        bu, inj, nut = fr.raw("agent_in_bush"), fr.raw("injury_level"), fr.raw("nutrition")
        ar, ac = fr.raw("agent_row"), fr.raw("agent_col")

        G["n_rows"][gidx] += fr.steps_per_episode + 1          # includes the reset row
        G["inj_sum"][gidx] += fr.per_episode_sum_with_initial(inj)
        G["nut_sum"][gidx] += fr.per_episode_sum_with_initial(nut)
        G["dmg_sum"][gidx] += fr.per_episode_sum_with_initial(fr.raw("damage"))
        G["n_ate"][gidx] += fr.per_episode_sum_with_initial(fr.raw("ate_food"))
        G["n_rest"][gidx] += fr.per_episode_sum_with_initial(fr.raw("rested"))
        G["inj_max"][gidx] = np.maximum.reduceat(inj, st_rows)
        G["inj0"][gidx], G["nut0"][gidx] = inj[st_rows], nut[st_rows]
        G["arow0"][gidx], G["acol0"][gidx] = ar[st_rows], ac[st_rows]

        AR, AC = fr.list_raw("animal_row", na), fr.list_raw("animal_col", na)
        near = (np.maximum(np.abs(AR - ar[:, None]), np.abs(AC - ac[:, None])) <= NEAR_D) & act[gi]
        pn = near[:, P].any(1); rn = near[:, R].any(1) & ~pn

        m = fr.is_step
        li = gi[m] - gidx[0]; nb_ = len(gidx)
        bc = lambda w=None: np.bincount(li, weights=w, minlength=nb_)
        G["n_steps"][gidx] += bc(); G["bush_steps"][gidx] += bc(bu[m])
        G["n_pred_near"][gidx] += bc(pn[m].astype(float))
        G["n_rab_near"][gidx] += bc(rn[m].astype(float))
        # CONTEMPORANEOUS by design here - see the divergence note in this file's docstring.
        ib = np.digitize(inj[m], INJ_EDGES); nbn = np.digitize(nut[m], NUT_EDGES)
        for M, W, Bn in ((IB, None, ib), (IBb, bu[m], ib), (NB, None, nbn), (NBb, bu[m], nbn)):
            M[gidx] += np.bincount(li * 4 + Bn, weights=W, minlength=nb_ * 4).reshape(-1, 4)

        OR, OC = fr.list_raw("obs_row", nobs), fr.list_raw("obs_col", nobs)
        a0r, a0c = ar[st_rows][:, None], ac[st_rows][:, None]
        ob = oa[gidx][:, lay["bush"]]
        db = np.where(ob, np.maximum(np.abs(OR[st_rows][:, lay["bush"]] - a0r),
                                     np.abs(OC[st_rows][:, lay["bush"]] - a0c)), np.inf)
        G["d_bush0"][gidx] = db.min(1)
        pact = act[gidx][:, P]
        dp = np.where(pact, np.maximum(np.abs(AR[st_rows][:, P] - a0r),
                                       np.abs(AC[st_rows][:, P] - a0c)), np.inf)
        G["d_pred0"][gidx] = dp.min(1)

    SCAN.sweep(st, STEP_COLS, collect, verbose=verbose, label="hiding_drivers")

    assert np.allclose(IB.sum(1), G["n_steps"]) and np.allclose(NB.sum(1), G["n_steps"])
    assert (G["bush_steps"] <= G["n_steps"]).all()
    assert np.allclose(G["n_rows"], G["n_steps"] + 1), "expected exactly one reset row/episode"
    return dict(**A, **G, IB=IB, IBb=IBb, NB=NB, NBb=NBb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="A_baseline")
    ap.add_argument("--out", required=True, help="scratch path for aggregate.npz")
    a = ap.parse_args()
    cfg = L.arm_config(L.arm_runs()[a.arm])
    lay = ENV.slot_layout(cfg)
    t0 = time.time()
    D = aggregate(L.arm_stores(a.arm), lay, ENV.smell_channels(cfg))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out, **D)
    print(f"{a.arm}: aggregate written to {a.out} ({time.time()-t0:.0f}s)")
