#!/usr/bin/env python
"""Does the agent's behaviour depend on its INTERNAL STATE, and does a neuromodulator
strengthen that dependence?

Why this exists
---------------
A FiLM neuromodulator reads the agent's internal state (injury, nutrition, ...) and rescales its
perceptual features. The sharp prediction is therefore NOT "the modulated agent behaves
differently" — it is that the modulated agent's **gain on external cues should vary more with its
internal state**. That is an interaction, and it is what this tool measures.

The three measures, weakest evidence to strongest
-------------------------------------------------
1. `state_span`   — how much hiding changes across internal-state cells, with no predator nearby.
                    Pure internal-state dependence of baseline behaviour.
2. `gain_span`    — the THREAT GAIN in each internal-state cell, defined as
                    P(hide | predator near) - P(hide | no predator near) within that cell, and how
                    far that gain ranges across cells. This is the direct analogue of what a
                    modulator is supposed to do: change the gain on a percept according to state.
3. `causal_span`  — the same gain, but computed early in the episode and binned by the RANDOMISED
                    starting injury / nutrition the environment assigns before the agent acts.
                    Only this one is causally identified; 1 and 2 condition on state the agent
                    itself produced.

Internal state can be read either as the raw body variable (`--state injury`, `--state nutrition`)
or as the signal the agent actually receives (`--state felt_pain`), which for injury is a lagged,
smoothed convolution rather than the level itself. `felt_pain` is usually the right choice for
injury: the agent has no direct readout of its wound.

Usage
-----
  P=/home/vncuser/miniconda3/envs/grid_world_pain/bin/python
  # one run
  $P scripts/analysis/context_dependence.py --run results/JAX_RecurrentPPO/<RUN> \
      --store-root results/trajectories_nmn --checkpoint <STEP>
  # compare two (e.g. modulated vs not) — pass --label to name them
  $P scripts/analysis/context_dependence.py --compare A.json B.json

Run from the repo root. Re-run this against new neuromodulator checkpoints as they land; the
metrics are defined so they are comparable across runs and across worlds.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
import numpy as np, pyarrow.parquet as pq, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hiding_drivers import slot_layout, find_store, listcol   # noqa: E402

INJ_EDGES  = [1e-9, 25.0, 50.0]
NUT_EDGES  = [25.0, 50.0, 75.0]
PAIN_EDGES = [1e-9, 8.0, 18.0, 32.0]
NEAR_D     = 2
EARLY      = 25          # steps counted as "early" for the randomised-state measure
LABELS = {"injury":    ["injury 0", "0-25", "25-50", ">=50"],
          "nutrition": ["nutr <25", "25-50", "50-75", ">=75"],
          "felt_pain": ["felt 0", "0-8", "8-18", "18-32", "32+"]}


def sweep(store, lay, state, kernel, verbose=True):
    """One pass. Returns counts[state_bin, predator_near, in_bush] and the early/randomised
    version binned by the STARTING value of the same body variable."""
    na = len(lay["pred"]) + len(lay["neutral"])
    epf = sorted(glob.glob(store + "episodes_*.parquet"))
    ep = pq.read_table(epf, columns=["episode_seed", "animal_active"])
    o = np.argsort(ep.column("episode_seed").to_numpy())
    seed0 = int(ep.column("episode_seed").to_numpy()[o][0])
    act = np.array(ep.column("animal_active").to_pylist(), bool)[o]
    P = lay["pred"]
    nb = len(LABELS[state])
    C = np.zeros((nb, 2, 2)); E = np.zeros((nb, 2, 2))
    body_col = "nutrition" if state == "nutrition" else "injury_level"
    edges = {"injury": INJ_EDGES, "nutrition": NUT_EDGES, "felt_pain": PAIN_EDGES}[state]
    files = sorted(glob.glob(store + "steps_*.parquet")); t0 = time.time()
    for fi, f in enumerate(files):
        tb = pq.read_table(f, columns=["episode_seed", "t", "agent_in_bush", body_col,
                                       "agent_row", "agent_col", "animal_row", "animal_col"])
        sd = tb.column("episode_seed").to_numpy(); t = tb.column("t").to_numpy(); N = len(t)
        body = tb.column(body_col).to_numpy(zero_copy_only=False).astype(np.float64)
        bu = tb.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(np.int64)
        ar = tb.column("agent_row").to_numpy(); ac = tb.column("agent_col").to_numpy()
        st = np.flatnonzero(t == 0); ends = np.append(st[1:], N)
        estart = np.repeat(st, ends - st); idx = np.arange(N)

        if state == "felt_pain":
            # reconstruct what the agent receives: buffer of injury LEVELS, zeroed at reset, so
            # the reset row is never in it (core.py:1112 / :115), convolved with the alpha kernel
            sig = np.zeros(N)
            for j in range(1, len(kernel)):
                src = idx - j; ok = src > estart
                sig[ok] += kernel[j] * body[src[ok]]
        else:
            sig = body
        # the action producing row t was chosen on the row t-1 observation
        prev = np.zeros(N); prev[1:] = sig[:-1]; prev[idx == estart] = sig[estart[0]] * 0
        start_val = np.repeat(body[st], ends - st)

        AR = listcol(tb.column("animal_row"), na); AC = listcol(tb.column("animal_col"), na)
        near = (np.maximum(np.abs(AR - ar[:, None]), np.abs(AC - ac[:, None])) <= NEAR_D) \
               & act[sd - seed0]
        pn = near[:, P].any(1).astype(np.int64)

        m = t >= 2
        b = np.digitize(prev[m], edges)
        k = (b * 2 + pn[m]) * 2 + bu[m]
        C += np.bincount(k, minlength=nb * 4).reshape(nb, 2, 2)
        me = m & (t <= EARLY)
        be = np.digitize(start_val[me], edges)
        ke = (be * 2 + pn[me]) * 2 + bu[me]
        E += np.bincount(ke, minlength=nb * 4).reshape(nb, 2, 2)
        if verbose and fi % 50 == 0:
            print(f"  shard {fi}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    return C, E


def metrics(C, labels, min_n=2e4):
    """P(hide) with and without a predator near, per internal-state cell, plus the spans."""
    rows = []
    for i, lab in enumerate(labels):
        n0, n1 = C[i, 0].sum(), C[i, 1].sum()
        p0 = C[i, 0, 1] / n0 if n0 >= min_n else np.nan
        p1 = C[i, 1, 1] / n1 if n1 >= min_n else np.nan
        rows.append(dict(state=lab, n_calm=float(n0), n_threat=float(n1),
                         hide_calm=100 * p0, hide_threat=100 * p1,
                         gain=100 * (p1 - p0)))
    ok = [r for r in rows if np.isfinite(r["gain"])]
    span = lambda key, rs: (max(r[key] for r in rs) - min(r[key] for r in rs)) if len(rs) > 1 else np.nan
    # The zero bin ("no signal at all") is qualitatively unlike the graded bins and dominates the
    # step count, so it distorts a max-minus-min span. Report the span over the SIGNAL-carrying
    # bins separately, and a trend, which is the cleanest single number: does the threat gain rise
    # or fall as internal state increases, and how steeply per bin. Weighted by cell size.
    sig = [r for r in ok if not r["state"].endswith(" 0")]
    if len(sig) > 1:
        x = np.arange(len(sig), dtype=float)
        y = np.array([r["gain"] for r in sig])
        w = np.array([r["n_calm"] + r["n_threat"] for r in sig])
        xm = np.average(x, weights=w); ym = np.average(y, weights=w)
        trend = float(np.sum(w * (x - xm) * (y - ym)) / max(np.sum(w * (x - xm) ** 2), 1e-9))
    else:
        trend = np.nan
    return rows, {"state_span": span("hide_calm", ok), "gain_span": span("gain", ok),
                  "gain_span_signal": span("gain", sig),
                  "gain_trend_per_bin": trend,
                  "gain_min": min((r["gain"] for r in ok), default=np.nan),
                  "gain_max": max((r["gain"] for r in ok), default=np.nan),
                  "cells_used": len(ok)}


def show(rows, m, title):
    print(f"\n=== {title} ===")
    print(f"{'internal state':16}{'no predator':>13}{'predator near':>15}{'THREAT GAIN':>14}"
          f"{'n (M)':>12}")
    for r in rows:
        if not np.isfinite(r["gain"]):
            print(f"{r['state']:16}{'— too few steps —':>54}"); continue
        print(f"{r['state']:16}{r['hide_calm']:>12.1f}%{r['hide_threat']:>14.1f}%"
              f"{r['gain']:>+13.1f}{(r['n_calm']+r['n_threat'])/1e6:>12.1f}")
    print(f"  state span (baseline hiding across states) : {m['state_span']:.1f} pp")
    print(f"  GAIN span  (all states)                    : {m['gain_span']:.1f} pp"
          f"   [{m['gain_min']:.1f} .. {m['gain_max']:.1f}]")
    print(f"  GAIN span  (signal-carrying states only)   : {m['gain_span_signal']:.1f} pp")
    print(f"  GAIN TREND (pp per state bin, weighted)    : {m['gain_trend_per_bin']:+.2f}"
          f"   <- positive = more internal signal, stronger threat response")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run"); ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--store-root", default="results/trajectories")
    ap.add_argument("--state", default="felt_pain", choices=["injury", "nutrition", "felt_pain"])
    ap.add_argument("--label", default=None, help="name for this run in the output")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                    help="compare two previously written metric files")
    a = ap.parse_args()

    if a.compare:
        A = json.load(open(a.compare[0])); B = json.load(open(a.compare[1]))
        assert A["state"] == B["state"], "the two runs used different --state"
        print(f"### context dependence on {A['state']}: {A['label']}  vs  {B['label']}\n")
        for tag in ("observed", "randomised_early"):
            ra, rb = A[tag]["rows"], B[tag]["rows"]
            ma, mb = A[tag]["metrics"], B[tag]["metrics"]
            print(f"--- {tag} ---")
            print(f"{'internal state':16}{A['label'][:11]:>13}{B['label'][:11]:>13}"
                  f"{'gain A':>10}{'gain B':>10}{'Δgain':>9}")
            for x, y in zip(ra, rb):
                if not (np.isfinite(x["gain"]) and np.isfinite(y["gain"])): continue
                print(f"{x['state']:16}{x['hide_calm']:>12.1f}%{y['hide_calm']:>12.1f}%"
                      f"{x['gain']:>+10.1f}{y['gain']:>+10.1f}{y['gain']-x['gain']:>+9.1f}")
            for key, nm in (("gain_span", "GAIN SPAN  (all)"),
                            ("gain_span_signal", "GAIN SPAN  (signal)"),
                            ("gain_trend_per_bin", "GAIN TREND /bin")):
                print(f"  {nm:22} {A['label']}: {ma[key]:+.2f}    "
                      f"{B['label']}: {mb[key]:+.2f}    diff {mb[key]-ma[key]:+.2f}")
            print(f"  state span  {A['label']}: {ma['state_span']:.1f} pp    "
                  f"{B['label']}: {mb['state_span']:.1f} pp    "
                  f"difference {mb['state_span']-ma['state_span']:+.1f} pp\n")
        print("Reading it: a modulator that gates perception by internal state should show a")
        print("LARGER gain span — its response to a nearby predator should depend more on how")
        print("hurt or hungry it is. The randomised_early block is the only causally identified")
        print("half; the observed block conditions on state the agent produced itself.")
        return

    if not a.run:
        ap.error("--run is required unless --compare is used")
    cfg = yaml.safe_load(open(f"{a.run}/models/config.yaml"))
    lay = slot_layout(cfg)
    KL = int(cfg["sensory"]["interoceptive_kernel_length"])
    TAU = float(cfg["sensory"]["interoceptive_kernel_tau"])
    k = np.arange(KL, dtype=np.float64); raw = (k / TAU) * np.exp(1.0 - k / TAU)
    kernel = raw / raw.sum()
    store = find_store(a.run, a.checkpoint, a.store_root)
    label = a.label or os.path.basename(a.run.rstrip("/"))[16:]
    print(f"run   {a.run}\nstore {store}\nstate {a.state}   label {label}")
    if a.state == "felt_pain":
        print(f"kernel tau={TAU} len={KL}  (weights lag1..lag{KL-1}, current step excluded)")

    C, E = sweep(store, lay, a.state, kernel)
    labs = LABELS[a.state]
    rows_o, m_o = metrics(C, labs)
    rows_e, m_e = metrics(E, labs)
    show(rows_o, m_o, f"{label} — observed internal state (associational)")
    show(rows_e, m_e, f"{label} — RANDOMISED starting state, first {EARLY} steps (causal)")
    out = {"label": label, "run": a.run, "state": a.state, "store": store,
           "observed": {"rows": rows_o, "metrics": m_o},
           "randomised_early": {"rows": rows_e, "metrics": m_e}}
    dest = a.out or f"results/analysis/context_dependence/{label}_{a.state}.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print(f"\nwritten: {dest}")


if __name__ == "__main__":
    main()
