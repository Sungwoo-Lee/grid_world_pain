#!/usr/bin/env python3
"""grid_ladder_figures.py - the sensor ladder's questions, asked of the neuromodulator grid.

The grid's first figures asked only about the pre-registered headline measure. The sensor-ladder
study asks five more questions of every agent it has, and every one of them applies here:

    how long does it live, and how much of its life does it spend hidden   (ladder figure 1)
    which of the three ways of dying does it actually die of               (ladder figure 3)
    how does hiding change as a predator gets closer                       (ladder figure 4)
    does it tell a predator from a rabbit, or respond to both              (ladder figure 5)
    what does hiding cost it in survival                                   (ladder figure 15)

None of those is re-implemented here. The aggregates these figures read were produced by the
ladder's OWN arm sweep - `studies/sensor_ladder/collect_arm_data.py --manifest` - pointed at the
grid's twenty-one runs, so "the same analysis" is literally true rather than a claim about two
pieces of code that resemble each other.

INPUT   results/analysis/nmn_site_grid/ladderstyle/<cell>.json   (+ <cell>_episodes.npz)
OUTPUT  docs/experiments/active/nmn_input_site_grid/figures/g0[3-7]_*.png
"""
from __future__ import annotations
import json, os, sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "analysis", "ladder"))
os.chdir(ROOT)
import _ladder as L                                                    # noqa: E402

IN  = "results/analysis/nmn_site_grid/ladderstyle"
FIG = os.environ.get("NMN_FIG_ROOT", "docs/experiments/active/nmn_input_site_grid/figures")

SITES  = ["t2enc", "t3rnn", "t4act", "t5crt", "t16quad"]
SLICES = ["I", "X", "ALL"]
CTRL   = [f"baseline_s{s}" for s in (42, 43, 44, 45, 46)]
SITE_NAME = {"t2enc": "encoder", "t3rnn": "memory", "t4act": "actor", "t5crt": "critic",
             "t16quad": "all four"}
SLICE_NAME = {"I": "body only", "X": "world only", "ALL": "everything"}
# Same three data colours as the rest of the page, and nothing else may use them.
C = {"I": "#8a4b8f", "X": "#2f6f9f", "ALL": "#2d6a4f", "ctrl": "#6f6d69", "none": "#1b1b1d"}
PAPER = "#f8f7f5"          # opaque: see defect F30
ANNO  = "#3d3c3a"
plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.labelsize": 17,
                     "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
                     "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
                     "axes.facecolor": "#ffffff", "axes.axisbelow": True})
MARK = {"t2enc": "o", "t3rnn": "s", "t4act": "^", "t5crt": "D", "t16quad": "v"}

CELLS = [f"{s}_{sl}" for s in SITES for sl in SLICES]
ORDER = CELLS + ["t1none"] + CTRL


def load(name):
    p = f"{IN}/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def colour(name):
    if name in CTRL:  return C["ctrl"]
    if name == "t1none": return C["none"]
    return C[name.split("_")[1]]


def label(name):
    if name in CTRL:  return f"control seed {name[-2:]}"
    if name == "t1none": return "t1none (in-grid control)"
    s, sl = name.split("_")
    return f"{SITE_NAME[s]} / {SLICE_NAME[sl]}"


def finish(fig, stem):
    os.makedirs(FIG, exist_ok=True)
    p = f"{FIG}/{stem}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"  wrote {p}")


def band(vals):
    """min/max of the five unmodulated reference runs."""
    v = np.array(vals, float)
    return float(np.nanmin(v)), float(np.nanmax(v))


def hbars(ax, names, vals, xlabel, title, fmt="{:.1f}", pad=0.012):
    y = np.arange(len(names))
    ax.barh(y, vals, color=[colour(n) for n in names], height=.72)
    ax.set_yticks(y); ax.set_yticklabels([label(n) for n in names])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel); ax.set_title(title, loc="left")
    ax.grid(axis="x", alpha=.25, lw=.5)
    span = max(vals) - 0
    ax.set_xlim(0, max(vals) * 1.16)
    for i, v in enumerate(vals):
        ax.text(v + span * pad, i, fmt.format(v), va="center", fontsize=13, color=ANNO)
    # Every annotation must sit on its own bar. Read both back off the axes rather than trusting
    # the loop above: this exact pairing has silently inverted before, in the ladder's figure 1.
    bars = {round(b.get_y() + b.get_height() / 2, 3): b.get_width() for b in ax.patches}
    for t in ax.texts:
        yy = round(t.get_position()[1], 3)
        if yy not in bars or t.get_text() != fmt.format(bars[yy]):
            raise SystemExit(f"label {t.get_text()!r} does not sit on its own bar (y={yy})")


def fig3(D):
    """Survival and hiding, every cell, against the five unmodulated controls."""
    names = ORDER
    surv = [D[n]["mean_survival"] for n in names]
    hide = [D[n]["bush_dwell_pct"] for n in names]
    # NOT sharey. `hbars` calls invert_yaxis() per axis, and on a SHARED y-axis the second call
    # undoes the first, so the rows silently come out bottom-to-top while every label still reads
    # top-to-bottom; blanking the right panel's tick labels also blanks the left panel's, because
    # they are the same axis. Both bugs shipped in the first render of this figure.
    fig, ax = plt.subplots(1, 2, figsize=(16.5, 9.0))
    hbars(ax[0], names, surv, "mean survival (steps per episode)", "How long it lives", "{:.0f}")
    hbars(ax[1], names, hide, "bush hiding (% of an episode's steps in a bush)",
          "How much of its life it spends hidden", "{:.1f}")
    ax[1].set_yticks(np.arange(len(names))); ax[1].set_yticklabels([])
    for a, vals in ((ax[0], surv), (ax[1], hide)):
        lo, hi = band([v for n, v in zip(names, vals) if n in CTRL])
        a.axvspan(lo, hi, color=C["ctrl"], alpha=.16, lw=0, zorder=0)
    fig.subplots_adjust(wspace=.04)
    finish(fig, "g03_survival_and_hiding")
    return {n: (s, h) for n, s, h in zip(names, surv, hide)}


def fig4(D):
    """Which of the three ways of dying each cell actually dies of."""
    names = ORDER
    keys = ["killed by predator", "starved", "survived to time limit"]
    cols = ["#8c3b34", "#8a6d1f", "#3f6f53"]      # outcome colours: NOT the three data colours
    fig, ax = plt.subplots(figsize=(13.5, 9.0))
    y = np.arange(len(names)); left = np.zeros(len(names))
    for k, c in zip(keys, cols):
        v = np.array([D[n]["term_pct"].get(k, 0.0) for n in names])
        ax.barh(y, v, left=left, color=c, height=.72, label=k)
        for i, (l_, w) in enumerate(zip(left, v)):
            if w > 4.5:
                ax.text(l_ + w / 2, i, f"{w:.0f}", va="center", ha="center",
                        fontsize=12, color="white", fontweight="bold")
        left += v
    if not np.allclose(left, 100, atol=.05):
        raise SystemExit(f"outcome shares do not sum to 100: {left.min():.2f}..{left.max():.2f}")
    ax.set_yticks(y); ax.set_yticklabels([label(n) for n in names]); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("share of that run's 1,000,000 episodes (%)")
    ax.set_title("How each episode ended", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.09), ncol=3, frameon=False)
    ax.grid(axis="x", alpha=.25, lw=.5)
    finish(fig, "g04_how_it_ends")


def fig5(D):
    """Hiding against how close the nearest predator, and the nearest rabbit, actually is."""
    x = np.arange(L.DIST_MAX)
    fig, ax = plt.subplots(2, 1, figsize=(11.5, 13.0), sharex=True,
                           gridspec_kw={"hspace": .30})
    for i, (bk, tk, who) in enumerate([("pd_bush", "pd_tot", "predator"),
                                       ("rd_bush", "rd_tot", "rabbit")]):
        M = np.array([L.dist_curve(D[n]["grids"][bk], D[n]["grids"][tk]) for n in CTRL])
        ax[i].fill_between(x, np.nanmin(M, 0), np.nanmax(M, 0), color=C["ctrl"], alpha=.25, lw=0,
                           label="controls (range)")
        for n in CELLS:
            ax[i].plot(x, L.dist_curve(D[n]["grids"][bk], D[n]["grids"][tk]),
                       color=colour(n), lw=1.1, alpha=.5)
        ax[i].plot(x, L.dist_curve(D["t1none"]["grids"][bk], D["t1none"]["grids"][tk]),
                   color=C["none"], lw=2.4, label="t1none (control)", zorder=5)
        ax[i].set_xticks(x); ax[i].set_xticklabels(L.DIST_NAMES)
        ax[i].set_ylabel("in a bush (% of steps)")
        ax[i].set_title(f"Nearest {who}", loc="left")
        ax[i].grid(alpha=.25, lw=.5)
        b, t = ax[i].get_ylim(); ax[i].set_ylim(b, b + (t - b) * 1.28)
        ax[i].legend(loc="upper left", framealpha=.94)
    ax[1].set_xlabel("distance from the agent to the nearest one, in chebyshev steps")
    h = [plt.Line2D([], [], color=C[s], lw=1.8) for s in SLICES]
    ax[0].legend(handles=h + ax[0].get_legend_handles_labels()[0],
                 labels=[f"reads {SLICE_NAME[s]}" for s in SLICES]
                        + ax[0].get_legend_handles_labels()[1],
                 loc="upper left", ncol=2, fontsize=13, framealpha=.94)
    finish(fig, "g05_hiding_vs_distance")


def fig6(D):
    """Does it tell a predator from a rabbit? One point per run, on identical axes."""
    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    pe = {n: L.proximity_effect(D[n]["grids"]["pd_bush"], D[n]["grids"]["pd_tot"]) for n in ORDER}
    re_ = {n: L.proximity_effect(D[n]["grids"]["rd_bush"], D[n]["grids"]["rd_tot"]) for n in ORDER}
    for n in CELLS:
        ax.scatter(pe[n], re_[n], s=140, color=colour(n), marker=MARK[n.split("_")[0]],
                   edgecolor="white", lw=1.0, zorder=4)
    ax.scatter([pe[n] for n in CTRL], [re_[n] for n in CTRL], s=150, facecolor="none",
               edgecolor=C["ctrl"], lw=2.0, zorder=3, label="five unmodulated controls")
    ax.scatter(pe["t1none"], re_["t1none"], s=230, color=C["none"], marker="*", zorder=6,
               label="t1none (in-grid control)")
    # An equal-aspect square with the y=x line puts every point in one corner and spends four
    # fifths of the panel proving something the axis labels already say. Zoom to the data, mark the
    # five controls as a box, and state the ratio in words instead of drawing a line nobody is near.
    P = np.array([pe[n] for n in ORDER]); R = np.array([re_[n] for n in ORDER])
    px, rx = (P.min(), P.max()), (R.min(), R.max())
    ax.set_xlim(px[0] - 1.4, px[1] + 1.4)
    ax.set_ylim(max(0.0, rx[0] - .6), rx[1] + 1.5)
    cp = [pe[n] for n in CTRL]; cr = [re_[n] for n in CTRL]
    ax.add_patch(plt.Rectangle((min(cp), min(cr)), max(cp) - min(cp), max(cr) - min(cr),
                               facecolor=C["ctrl"], alpha=.16, edgecolor=C["ctrl"], lw=1.2,
                               ls="--", zorder=0))
    ax.annotate("the five unmodulated controls\nspan this box", xy=(np.mean(cp), max(cr) + .30),
                ha="center", fontsize=13, color=ANNO)
    ax.annotate(f"every run answers a predator about {P.mean()/R.mean():.0f}x more strongly than a\n"
                f"rabbit. The equal-response line is far above this view.",
                xy=(.02, .04), xycoords="axes fraction", fontsize=14, color=ANNO,
                fontweight="bold")
    ax.set_xlabel("response to a nearby PREDATOR (percentage points)")
    ax.set_ylabel("response to a nearby RABBIT\n(percentage points)")
    ax.set_title("Predator or rabbit \u2014 does the agent tell them apart?", loc="left")
    ax.grid(alpha=.25, lw=.5)
    h = [plt.Line2D([], [], ls="", marker=MARK[s_], color=ANNO, ms=10) for s_ in SITES]
    ax.legend(handles=h + ax.get_legend_handles_labels()[0],
              labels=[SITE_NAME[s_] for s_ in SITES] + ax.get_legend_handles_labels()[1],
              loc="upper left", ncol=2, fontsize=13, framealpha=.94)
    finish(fig, "g06_predator_vs_rabbit")
    return pe, re_


def fig7(D, sh):
    """What hiding costs. One point per run."""
    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    for n in ORDER:
        h, s = sh[n][1], sh[n][0]
        ax.scatter(h, s, s=150 if n != "t1none" else 240, color=colour(n),
                   marker="*" if n == "t1none" else
                          ("o" if n in CTRL else MARK[n.split("_")[0]]),
                   edgecolor="white", lw=1.0, zorder=4)
    H = np.array([sh[n][1] for n in ORDER]); S = np.array([sh[n][0] for n in ORDER])
    m, b = np.polyfit(H, S, 1)
    xs = np.linspace(H.min(), H.max(), 2)
    r = float(np.corrcoef(H, S)[0, 1])
    ax.plot(xs, m * xs + b, ls="--", lw=1.4, color=ANNO, alpha=.7, zorder=1)
    ax.annotate(f"least-squares fit: {m:+.1f} steps of survival\nper extra point of hiding "
                f"(r = {r:+.2f}, n = {len(ORDER)})",
                xy=(.03, .05), xycoords="axes fraction", fontsize=14, color=ANNO)
    ax.set_xlabel("bush hiding (% of an episode's steps spent in a bush)")
    ax.set_ylabel("mean survival (steps per episode)")
    ax.set_title("What hiding costs", loc="left")
    ax.grid(alpha=.25, lw=.5)
    hs = [plt.Line2D([], [], ls="", marker=MARK[s_], color=ANNO, ms=9) for s_ in SITES] \
       + [plt.Line2D([], [], ls="", marker="o", color=C[sl], ms=9) for sl in SLICES] \
       + [plt.Line2D([], [], ls="", marker="o", color=C["ctrl"], ms=9),
          plt.Line2D([], [], ls="", marker="*", color=C["none"], ms=13)]
    ls_ = [SITE_NAME[s_] for s_ in SITES] + [f"reads {SLICE_NAME[sl]}" for sl in SLICES] \
        + ["unmodulated control", "t1none"]
    ax.legend(hs, ls_, loc="upper right", ncol=2, fontsize=12, framealpha=.94)
    finish(fig, "g07_price_of_hiding")
    return m, r


def main():
    D = {}
    for n in ORDER:
        d = load(n)
        if d is None:
            raise SystemExit(f"missing aggregate for {n}: run collect_arm_data.py --manifest first")
        D[n] = d
    print(f"{len(D)} cells loaded from {IN}")
    sh = fig3(D)
    fig4(D)
    fig5(D)
    pe, re_ = fig6(D)
    m, r = fig7(D, sh)

    print(f"\n{'cell':26}{'survival':>10}{'hiding':>9}{'pred':>8}{'rab':>8}")
    for n in ORDER:
        print(f"{n:26}{sh[n][0]:>10.1f}{sh[n][1]:>8.1f}%{pe[n]:>+8.1f}{re_[n]:>+8.1f}")
    cs = [sh[n][0] for n in CTRL]; ch = [sh[n][1] for n in CTRL]
    print(f"\ncontrols: survival {np.mean(cs):.1f} +- {np.std(cs, ddof=1):.1f} "
          f"({min(cs):.1f}..{max(cs):.1f});  hiding {np.mean(ch):.2f} +- {np.std(ch, ddof=1):.2f}")
    print(f"price of hiding: {m:+.1f} steps per point, r = {r:+.2f}")


if __name__ == "__main__":
    main()
