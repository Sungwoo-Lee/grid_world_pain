#!/usr/bin/env python3
"""aggregate_grid.py - collapse the 21 per-run analyses into the study's comparison tables.

Reads the JSON `context_dependence.py` wrote for each of the 16 grid arms and the 5 unmodulated
baselines, and emits the numbers the design pre-registered:

  Tier A   delta_B0   - the change in bush-ENTRY rate between the lowest and highest quarter of the
                        RANDOMISED starting injury. Scores a movement decision, so an agent that
                        freezes leaves the denominator instead of loading the numerator.
  Tier B   delta_rest - how much more the agent rests when it wakes wounded. Printed beside every B0
                        number because it is the effect that dwarfs it, and a B0 result read without
                        it will be misattributed.
  Tier B   prox trend - does being hurt change the response to a nearby predator.

THE CONTROL IS A BAND, NOT A NUMBER. Five unmodulated runs differing only in seed give five values
per measure; their spread is the noise floor. An arm is distinguishable from the control only if it
falls OUTSIDE that band -- and with one seed per arm even that is suggestive, not conclusive, which
is why nothing here prints the word "eliminated".
"""
from __future__ import annotations
import json, os

OUT = "results/analysis/nmn_site_grid"
CELLS = ["t1none",
         "t2enc_I", "t2enc_X", "t2enc_ALL",
         "t3rnn_I", "t3rnn_X", "t3rnn_ALL",
         "t4act_I", "t4act_X", "t4act_ALL",
         "t5crt_I", "t5crt_X", "t5crt_ALL",
         "t16quad_I", "t16quad_X", "t16quad_ALL"]
KEY = "first_25|action|any"          # the pre-registered headline variant


def load(label):
    p = f"{OUT}/{label}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def row(d):
    b0 = d["entry"]["b0"][KEY]
    b1 = d["entry"]["b1"]["first_25"]
    return dict(d_b0=b0["delta_b0"], d_b0_ci=b0["delta_b0_ci95"],
                d_rest=b1["delta_rest"],
                trend=d["randomised_early"]["metrics"]["proximity_effect_trend_per_bin"],
                pe_min=d["randomised_early"]["metrics"]["pe_min"],
                pe_max=d["randomised_early"]["metrics"]["pe_max"],
                n_ep=d["entry"]["diagnostics"]["n_episodes"],
                n_rows=d["entry"]["diagnostics"]["n_rows"])


def main():
    base = {s: load(f"baseline_s{s}") for s in (42, 43, 44, 45, 46)}
    missing = [s for s, v in base.items() if v is None]
    if missing:
        raise SystemExit(f"baselines missing: {missing} - the control band cannot be formed")
    B = {s: row(v) for s, v in base.items()}
    band = {k: (min(B[s][k] for s in B), max(B[s][k] for s in B))
            for k in ("d_b0", "d_rest", "trend")}

    import statistics as st
    stats = {}
    for k in ("d_b0", "d_rest", "trend"):
        v = [B[s_][k] for s_ in B]
        stats[k] = (st.mean(v), st.stdev(v))

    print("CONTROL - five unmodulated runs, seeds 42-46, identical configuration")
    print(f"  {'measure':<26}{'mean':>10}{'sd':>10}{'min':>10}{'max':>10}")
    for k, nm in (("d_b0", "delta_B0 (pp)"), ("d_rest", "delta_rest (pp)"),
                  ("trend", "prox-effect trend (pp)")):
        m, sd = stats[k]; lo, hi = band[k]
        print(f"  {nm:<26}{m:>10.3f}{sd:>10.3f}{lo:>10.3f}{hi:>10.3f}")

    print("""
  READ THE min-max AS A RANGE, NOT AS A CONFIDENCE INTERVAL. A sixth draw from the same
  distribution is the new minimum or maximum with probability 2/6 = 33%, so about a third of
  cells fall outside the range of five controls whether or not anything is going on. The
  z-column below is against the control mean and SD and is the number to read; even it is
  n=5 on the control side and n=1 on the cell side, so it is a screening statistic, not a test.
""")

    print(f"{'cell':<14}{'dB0 (pp)':>10}{'z':>7}{'dRest (pp)':>12}{'z':>7}"
          f"{'trend':>8}{'z':>7}{'episodes':>12}")
    cells = {}
    for c in CELLS:
        d = load(c)
        if d is None:
            print(f"{c:<14}   -- not analysed --")
            continue
        r = row(d); cells[c] = r
        z = lambda k: (r[k] - stats[k][0]) / stats[k][1] if stats[k][1] else float("nan")
        r["z_b0"], r["z_rest"], r["z_trend"] = z("d_b0"), z("d_rest"), z("trend")
        mark = "  <- the CONTROL cell" if c == "t1none" else ""
        print(f"{c:<14}{r['d_b0']:>10.3f}{z('d_b0'):>7.1f}{r['d_rest']:>12.2f}{z('d_rest'):>7.1f}"
              f"{r['trend']:>8.2f}{z('trend'):>7.1f}{r['n_ep']:>12,}{mark}")

    outside = {k: [c for c, v in cells.items() if v[k] < band[k][0] or v[k] > band[k][1]]
               for k in ("d_b0", "d_rest", "trend")}
    print("\n  cells outside the five-control range, against the 33% expected by chance:")
    for k, nm in (("d_b0", "delta_B0"), ("d_rest", "delta_rest"), ("trend", "prox trend")):
        n = len(outside[k])
        print(f"    {nm:<12}{n:>3}/16 ({100*n/16:>3.0f}%)  {'at or below chance' if n <= 6 else 'ABOVE chance'}"
              f"   {', '.join(outside[k]) or 'none'}")
    if "t1none" in outside["d_b0"] or "t1none" in outside["d_rest"]:
        print("""
    NOTE: t1none is the grid's OWN unmodulated control, configured identically to the sixteen
    arms. Its appearing 'outside the control range' is therefore a property of the range test,
    not a property of the agent -- and it is the clearest available evidence that a cell falling
    outside these bounds means nothing on its own.""")

    json.dump({"band": band, "control_mean_sd": {k: list(v) for k, v in stats.items()},
               "outside_vs_chance": outside, "baselines": {f"s{s}": B[s] for s in B}, "cells": cells,
               "headline_variant": KEY},
              open(f"{OUT}/_grid_summary.json", "w"), indent=1, default=float)
    print(f"\nwritten: {OUT}/_grid_summary.json")


if __name__ == "__main__":
    main()
