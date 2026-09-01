"""Second sweep: what happens step by step after the agent wakes up wounded.

WHY A SECOND SWEEP. `build_arm_data.py` bins over whole episodes or fixed windows, which answers
"does a wound change behaviour" but not "for how long". That distinction turned out to matter: a
reviewer pointed out that the injury effect shrinks as the outcome window is widened, which looks
like a fragile result chosen at a flattering window. It is not - the assigned wound HEALS, so the
cause disappears and the effect must go with it. Showing that requires the step-by-step trace, so
it gets its own pass.

WHAT IT ACCUMULATES, per arm, per starting-wound quarter, for the first 120 steps:
  the injury level still carried      (the body's state)
  the PERCEIVED nociception signal    (what the agent is actually given - see below)
  bush dwell                          (the response)
  nutrition and food eaten            (the cost the response incurs)

THE DISTINCTION THAT MATTERS. The agent has no sensor for its injury level - `injury_observable`
is false in every arm. What it receives is one scalar from the interoceptive nociceptor: the last
twelve injury levels, convolved with a normalised alpha kernel (tau=3), with slot 0 weighted ZERO
so the current step's injury never leaks in instantaneously. The buffer is zeroed at reset. So an
agent that wakes with an injury of 100 perceives EXACTLY NOTHING at t=0, 9% of it by step 2, 36%
by step 4, and does not feel the full wound until step 12 - by which time the wound itself has
begun to heal. Binning behaviour by the injury level therefore bins it by a quantity the agent
cannot sense; this sweep records the perceived signal alongside it so the two can be told apart.

Writes results/analysis/ladder/time_course.json. Takes about a minute per arm.
"""
from __future__ import annotations
import argparse, glob, os, sys, time
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _ladder as L

MAXT = 120
COLS = ["episode_seed", "t", "injury_level", "nutrition", "ate_food", "agent_in_bush"]


def nociception_kernel(cfg) -> np.ndarray:
    """The alpha kernel the environment convolves the injury buffer with (config_loader.py:1424)."""
    n = int(cfg["sensory"]["interoceptive_kernel_length"])
    tau = float(cfg["sensory"]["interoceptive_kernel_tau"])
    k = np.arange(n, dtype=np.float64)
    raw = (k / tau) * np.exp(1.0 - k / tau)
    return raw / raw.sum()


def perceived_nociception(inj, t, estart, kernel):
    """Rebuild the scalar the nociceptor actually hands the agent, row by row.

    Mirrors core.py:115 (roll the buffer, write the new injury at slot 0) and core.py:1112 (the
    buffer is ZEROED at reset, so the reset row's injury is never in it). The `src > estart` guard
    is strict for exactly that reason - including the reset row was a real bug once, and it leaks
    the randomised starting injury into the first steps as if the agent had felt it immediately.
    """
    idx = np.arange(len(t))
    out = np.zeros(len(t))
    for j in range(1, len(kernel)):
        src = idx - j
        ok = src > estart
        out[ok] += kernel[j] * inj[src[ok]]
    return out


def sweep(arm: str) -> dict:
    kernel = nociception_kernel(L.arm_config(L.arm_runs()[arm]))
    stores = L.arm_stores(arm)
    ep = pq.read_table(L.store_files(stores, "episodes"), columns=["episode_seed"])
    sd0 = ep.column("episode_seed").to_numpy()
    o = np.argsort(sd0); seed0 = int(sd0[o][0]); nep = len(o)
    acc = {k: np.zeros((4, MAXT)) for k in ["injury", "noci", "bush", "nutrition", "ate", "n"]}
    # bush dwell against the PERCEIVED signal, on the same 0-25-50-75-100 bins as the injury level
    dose = {"bush": np.zeros(4), "n": np.zeros(4)}
    inj0 = np.full(nep, np.nan)
    for f in L.store_files(stores, "steps"):
        tb = pq.read_table(f, columns=COLS)
        sd = tb.column("episode_seed").to_numpy(); t = tb.column("t").to_numpy()
        gi = sd - seed0
        st = np.flatnonzero(t == 0)
        num = lambda c: tb.column(c).to_numpy(zero_copy_only=False).astype(np.float64)
        inj = num("injury_level")
        inj0[gi[st]] = inj[st]
        st_all = np.flatnonzero(t == 0)
        ends = np.append(st_all[1:], len(t))
        estart = np.repeat(st_all, ends - st_all)
        noci = perceived_nociception(inj, t, estart, kernel)

        m = t < MAXT
        if np.isnan(inj0[gi[m]]).any():
            raise SystemExit(f"{arm}: a step row precedes its episode's t=0 row - shards misordered")
        ib = np.digitize(inj0[gi[m]], L.INJ_EDGES)
        np.add.at(acc["injury"], (ib, t[m]), inj[m])
        np.add.at(acc["noci"], (ib, t[m]), noci[m])

        # The perceptual dose-response, over every step of every episode (not just the first 120).
        # The PREDICTOR comes from the PREVIOUS row, as everywhere else in this analysis: the action
        # that put the agent in a bush at row t was chosen while it was feeling row t-1's signal.
        # An earlier version paired noci[t] with bush[t] - the same row - which relates the state
        # AFTER the action to the action, the same mistake the "carried wound" panel exists to
        # illustrate. It changed the published spread by 0.35 pp (the signal is a twelve-step
        # convolution, so adjacent rows barely differ), but it was the wrong convention.
        idx = np.arange(len(t))
        step = idx > estart + 1              # needs a previous row that is itself a step
        nb = np.digitize(noci[idx[step] - 1], L.INJ_EDGES)
        np.add.at(dose["bush"], nb, num("agent_in_bush")[step])
        np.add.at(dose["n"], nb, 1.0)
        np.add.at(acc["bush"], (ib, t[m]), num("agent_in_bush")[m])
        np.add.at(acc["nutrition"], (ib, t[m]), num("nutrition")[m])
        np.add.at(acc["ate"], (ib, t[m]), num("ate_food")[m])
        np.add.at(acc["n"], (ib, t[m]), 1.0)
    n = np.maximum(acc["n"], 1)
    return {k: (acc[k] / n).tolist() for k in ["injury", "noci", "bush", "nutrition", "ate"]} | \
           {"n": acc["n"].tolist(),
            "dose_bush": dose["bush"].tolist(), "dose_n": dose["n"].tolist(),
            "kernel": kernel.tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()
    import json
    # ONE FILE PER ARM. A single shared time_course.json was read-modify-written by each arm, so
    # running arms concurrently made the last writer clobber every other arm's entry - and the
    # result looked fine, because the clobbered entries were stale data from a previous run rather
    # than missing. Per-arm files make that failure structurally impossible.
    os.makedirs(L.OUT_ROOT, exist_ok=True)
    for arm in (a.arms or L.ARM_ORDER):
        t0 = time.time()
        p = f"{L.OUT_ROOT}/time_course_{arm}.json"
        json.dump(sweep(arm), open(p, "w"))
        print(f"{arm:22} done ({time.time()-t0:.0f}s) -> {p}", flush=True)
