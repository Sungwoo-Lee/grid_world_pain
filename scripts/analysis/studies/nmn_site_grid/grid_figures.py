#!/usr/bin/env python3
"""grid_figures.py - the NMN grid's figures, modelled on the sensor-ladder study's structure.

The first version of this analysis was three tables of endpoint differences. That is enough to say
"nothing moved" and not enough to see WHY, or to notice when a difference is an artefact of the
endpoints chosen. These figures show the CURVES the differences were taken from.

Every figure is built from the per-run JSON that `context_dependence.py` already wrote; none of them
needs another pass over the 21 trajectory stores.

ONE CORRECTION IS BAKED IN. Delta_rest was originally computed against the `injury 0` bin, which
holds about 25 rows out of five million because the starting wound is drawn from a continuous range.
These figures recompute it against the lowest POPULATED bin, the same guard B0 uses.
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
os.chdir(ROOT)
IN  = "results/analysis/nmn_site_grid"
FIG = os.environ.get("NMN_FIG_ROOT", "docs/experiments/active/nmn_input_site_grid/figures")
MIN_N = 2000
SITES = ["t2enc", "t3rnn", "t4act", "t5crt", "t16quad"]
SLICES = ["I", "X", "ALL"]
SITE_NAME = {"t2enc": "encoder", "t3rnn": "memory", "t4act": "actor", "t5crt": "critic",
             "t16quad": "all four"}
C = {"I": "#8a4b8f", "X": "#2f6f9f", "ALL": "#2d6a4f", "ctrl": "#6f6d69", "none": "#b3322b"}
BINS = ["0-25", "25-50", ">=50"]
XT = ["0-25", "25-50", "50-100"]


def load(lab):
    p = f"{IN}/{lab}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def b0_curve(d):
    rows = d["entry"]["b0"]["first_25|action|any"]["rows"]
    return [next((r["b0"] for r in rows if r["bin"] == b), np.nan) for b in BINS]


def rest_curve(d):
    rows = d["entry"]["b1"]["first_25"]["rows"]
    return [next((r["rest_rate"] if r["n"] >= MIN_N else np.nan
                  for r in rows if r["bin"] == b), np.nan) for b in BINS]


def prox_curve(d):
    rows = d["randomised_early"]["rows"]
    return [next((r["proximity_effect"] for r in rows if r["state"] == b), np.nan) for b in BINS]


def finish(fig, name):
    os.makedirs(FIG, exist_ok=True)
    p = f"{FIG}/{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {p}")


def band(fn, ctrl):
    """min/max envelope of the five controls at each bin."""
    M = np.array([fn(c) for c in ctrl], dtype=float)
    return np.nanmin(M, 0), np.nanmax(M, 0)


def main():
    ctrl = [load(f"baseline_s{s}") for s in (42, 43, 44, 45, 46)]
    ctrl = [c for c in ctrl if c]
    none = load("t1none")
    cells = {f"{s}_{sl}": load(f"{s}_{sl}") for s in SITES for sl in SLICES}
    cells = {k: v for k, v in cells.items() if v}
    x = np.arange(3)

    # ---- FIG 1: the two curves, then both on ONE axis ----
    # The third panel exists because the first two do not share a scale, and without it the eye
    # reads a 1.5-point axis and a 35-point axis as comparable. The whole finding is that one of
    # these effects is twenty times the other, so a figure that hides the ratio argues against
    # its own caption.
    fig, ax = plt.subplots(1, 3, figsize=(16.4, 5.0),
                           gridspec_kw={"width_ratios": [1, 1, 1.05]})
    for i, (fn, ttl, yl) in enumerate(
            [(b0_curve, "The decision: does a wounded agent step INTO cover?",
              "bush-entry rate (% of open-standing steps\non which the agent moved into a bush)"),
             (rest_curve, "The by-product: does it stop moving at all?",
              "rest rate (% of steps on which\nthe agent chose to rest)")]):
        lo, hi = band(fn, ctrl)
        ax[i].fill_between(x, lo, hi, color=C["ctrl"], alpha=.25, lw=0,
                           label="five unmodulated controls (range)")
        for k, d in cells.items():
            ax[i].plot(x, fn(d), color=C[k.split("_")[1]], lw=1.0, alpha=.45)
        ax[i].plot(x, fn(none), color=C["none"], lw=2.4, label="t1none — in-grid control", zorder=5)
        ax[i].set_xticks(x); ax[i].set_xticklabels(XT)
        ax[i].set_xlabel("randomised starting injury (0-100 scale), in quarters")
        ax[i].set_ylabel(yl, fontsize=9)
        ax[i].set_title(ttl, fontsize=10, loc="left")
        ax[i].grid(alpha=.25, lw=.5)
        span = np.nanmax(hi) - np.nanmin(lo)
        ax[i].annotate(f"this axis spans {span:.1f} pp", xy=(.98, .03), xycoords="axes fraction",
                       ha="right", fontsize=8.5, color=C["none"] if i == 0 else "#3d3c3a",
                       fontweight="bold")
    # panel 3: the same two quantities, one axis, no rescaling
    for k, d in cells.items():
        ax[2].plot(x, b0_curve(d), color=C[k.split("_")[1]], lw=1.0, alpha=.4)
        ax[2].plot(x, rest_curve(d), color=C[k.split("_")[1]], lw=1.0, alpha=.4)
    ax[2].plot(x, b0_curve(none), color=C["none"], lw=2.4, zorder=5)
    ax[2].plot(x, rest_curve(none), color=C["none"], lw=2.4, zorder=5)
    ax[2].set_ylim(0, 62); ax[2].set_xticks(x); ax[2].set_xticklabels(XT)
    ax[2].set_xlabel("randomised starting injury (0-100 scale), in quarters")
    ax[2].set_ylabel("percent of steps (both quantities, one scale)", fontsize=9)
    ax[2].set_title("The same two panels, on one axis", fontsize=10, loc="left")
    ax[2].grid(alpha=.25, lw=.5)
    ax[2].annotate("resting", xy=(2, 57), fontsize=9.5, color="#3d3c3a", fontweight="bold")
    ax[2].annotate("entering cover — the flat line at the bottom", xy=(0.02, 10.5),
                   fontsize=9.5, color=C["none"], fontweight="bold")
    h = [plt.Line2D([], [], color=C[s], lw=1.6) for s in SLICES]
    ax[0].legend(handles=h + ax[0].get_legend_handles_labels()[0],
                 labels=["modulator reads body only", "reads world only", "reads everything"]
                        + ax[0].get_legend_handles_labels()[1],
                 fontsize=8, loc="upper left", framealpha=.9)
    ax[1].legend(fontsize=8, loc="upper left", framealpha=.9)
    finish(fig, "g01_decision_vs_byproduct")

    # ---- FIG 2: the interaction the whole study is about ----
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.0), sharey=True)
    lo, hi = band(prox_curve, ctrl)
    for i, (grp, ttl) in enumerate([(["I"], "Modulator reads the BODY only"),
                                    (["X"], "Modulator reads the WORLD only")]):
        ax[i].fill_between(x, lo, hi, color=C["ctrl"], alpha=.25, lw=0, label="controls (range)")
        for k, d in cells.items():
            if k.split("_")[1] not in grp: continue
            ax[i].plot(x, prox_curve(d), color=C[grp[0]], lw=1.6, marker="o", ms=3.5,
                       label=SITE_NAME[k.split("_")[0]])
        ax[i].plot(x, prox_curve(none), color=C["none"], lw=2.2, ls="--", label="t1none")
        ax[i].set_xticks(x); ax[i].set_xticklabels(XT)
        ax[i].set_xlabel("randomised starting injury (0-100 scale), in quarters")
        ax[i].set_title(ttl, fontsize=10, loc="left")
        ax[i].grid(alpha=.25, lw=.5); ax[i].legend(fontsize=8, ncol=2)
    ax[0].set_ylabel("threat response (percentage points):\nhiding with a predator near minus hiding with none", fontsize=9)
    finish(fig, "g02_interaction_body_vs_world")
    print("done")


if __name__ == "__main__":
    main()
