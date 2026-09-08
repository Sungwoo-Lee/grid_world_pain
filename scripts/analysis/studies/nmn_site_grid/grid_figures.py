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
C = {"I": "#8a4b8f", "X": "#2f6f9f", "ALL": "#2d6a4f", "ctrl": "#6f6d69", "none": "#1b1b1d"}
# t1none is drawn in INK, not the page's warn-red. Red already means "callout accent" and "tier A"
# in the surrounding page, and inside panel 3 a red annotation sat directly above a red line and
# read as its label. A data series and the page's chrome must not share a colour.
ANNO = "#3d3c3a"
# The page renders in the viewer's theme; a PNG does not. A transparent figure therefore inherits
# whichever ground the reader happens to have, and near-black ink on the dark ground is invisible
# (defect F30). Ship one opaque light card instead, which reads correctly on either ground.
PAPER = "#f8f7f5"
# Rendered text size = native size x (column width / figure width). At a 730px column a 2266px-wide
# figure scales by 0.26, so a 10pt tick label lands at 7.7px -- below the 9px floor this project
# uses for diagrams. Sizes here are chosen so the SMALLEST text clears 9px after that scaling.
plt.rcParams.update({"font.size": 21, "axes.titlesize": 22, "axes.labelsize": 20,
                     "xtick.labelsize": 19, "ytick.labelsize": 19, "legend.fontsize": 18,
                     "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
                     "axes.facecolor": "#ffffff"})
MARK = {"t2enc": "o", "t3rnn": "s", "t4act": "^", "t5crt": "D", "t16quad": "v"}
DASH = {"t2enc": (0,()), "t3rnn": (0,(5,2)), "t4act": (0,(1,1.5)),
        "t5crt": (0,(6,2,1,2)), "t16quad": (0,(3,1,1,1,1,1))}
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
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=PAPER)
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
    # hspace: a 21pt left-aligned title sits directly under the previous panel's two-line
    # x-label. At the default 0.2 they overlap. 0.55 clears both.
    fig, ax = plt.subplots(3, 1, figsize=(11.5, 21.5), gridspec_kw={"hspace": .55})
    for i, (fn, ttl, yl) in enumerate(
            [(b0_curve, "The decision: does a wounded agent step INTO cover?",
              "bush-entry rate (%)\nopen steps ending in a bush"),
             (rest_curve, "The by-product: does it stop moving at all?",
              "rest rate (%)\nsteps spent resting")]):
        lo, hi = band(fn, ctrl)
        ax[i].fill_between(x, lo, hi, color=C["ctrl"], alpha=.25, lw=0,
                           label="controls (range)")
        for k, d in cells.items():
            ax[i].plot(x, fn(d), color=C[k.split("_")[1]], lw=1.0, alpha=.45)
        ax[i].plot(x, fn(none), color=C["none"], lw=2.4, label="t1none (control)", zorder=5)
        ax[i].set_xticks(x); ax[i].set_xticklabels(XT)
        ax[i].set_xlabel("randomised starting injury (0-100 scale), in quarters")
        ax[i].set_ylabel(yl, fontsize=18)
        ax[i].set_title(ttl, fontsize=21, loc="left")
        ax[i].grid(alpha=.25, lw=.5)
        span = np.nanmax(hi) - np.nanmin(lo)
        ax[i].annotate(f"the five controls span {span:.1f} pp in total", xy=(.98, .04), xycoords="axes fraction",
                       ha="right", fontsize=19, color=ANNO,
                       fontweight="bold")
        # The legend goes upper-left, so the data must not. Lift the top of the axis until the
        # legend block has empty plot to sit on rather than covering the control band.
        b, t = ax[i].get_ylim()
        ax[i].set_ylim(b, b + (t - b) * (1.75 if i == 0 else 1.45))
    # panel 3: the same two quantities, one axis, no rescaling
    for k, d in cells.items():
        ax[2].plot(x, b0_curve(d), color=C[k.split("_")[1]], lw=1.0, alpha=.4)
        ax[2].plot(x, rest_curve(d), color=C[k.split("_")[1]], lw=1.0, alpha=.4)
    ax[2].plot(x, b0_curve(none), color=C["none"], lw=2.4, zorder=5)
    ax[2].plot(x, rest_curve(none), color=C["none"], lw=2.4, zorder=5)
    ax[2].set_ylim(0, 70); ax[2].set_xticks(x); ax[2].set_xticklabels(XT)
    ax[2].set_xlabel("randomised starting injury (0-100 scale), in quarters")
    ax[2].set_ylabel("percent of steps\nboth quantities, one scale", fontsize=18)
    ax[2].set_title("The same two quantities, on one axis", fontsize=21, loc="left")
    ax[2].grid(alpha=.25, lw=.5)
    ax[2].annotate("resting", xy=(1.95, 65), ha="right", fontsize=20, color=ANNO, fontweight="bold")
    ax[2].annotate("entering cover — the flat line along the bottom", xy=(0.02, 11.5),
                   fontsize=20, color=ANNO, fontweight="bold")
    h = [plt.Line2D([], [], color=C[s], lw=1.6) for s in SLICES]
    ax[0].legend(handles=h + ax[0].get_legend_handles_labels()[0],
                 labels=["reads body only", "reads world only", "reads everything"]
                        + ax[0].get_legend_handles_labels()[1],
                 fontsize=17, loc="upper left", ncol=2, framealpha=.94)
    ax[1].legend(fontsize=17, loc="upper left", framealpha=.94)
    finish(fig, "g01_decision_vs_byproduct")

    # ---- FIG 2: the interaction the whole study is about ----
    fig, ax = plt.subplots(2, 1, figsize=(11.5, 14.0), sharey=True, sharex=True,
                           gridspec_kw={"hspace": .42})
    lo, hi = band(prox_curve, ctrl)
    for i, (grp, ttl) in enumerate([(["I"], "Modulator reads the BODY only"),
                                    (["X"], "Modulator reads the WORLD only")]):
        ax[i].fill_between(x, lo, hi, color=C["ctrl"], alpha=.25, lw=0, label="controls (range)")
        for k, d in cells.items():
            if k.split("_")[1] not in grp: continue
            site = k.split("_")[0]
            ax[i].plot(x, prox_curve(d), color=C[grp[0]], lw=2.0, marker=MARK[site], ms=9,
                       ls=DASH[site], label=SITE_NAME[site])
        ax[i].plot(x, prox_curve(none), color=C["none"], lw=2.2, ls="--", label="t1none")
        ax[i].set_xticks(x); ax[i].set_xticklabels(XT)
        ax[i].set_title(ttl, fontsize=21, loc="left")
        ax[i].grid(alpha=.25, lw=.5)
        ax[i].legend(fontsize=16, ncol=2, framealpha=.92, loc="upper left")
    # sharex=True: only the bottom panel carries the axis label, otherwise the top panel's label
    # is drawn into the gap between the two panels and reads as a caption for neither.
    ax[1].set_xlabel("randomised starting injury (0-100 scale), in quarters")
    b, t = ax[0].get_ylim()
    ax[0].set_ylim(b, b + (t - b) * 1.40)
    for a in ax: a.set_ylabel("threat response (pp):\npredator near minus none", fontsize=19)
    finish(fig, "g02_interaction_body_vs_world")
    print("done")


if __name__ == "__main__":
    main()
