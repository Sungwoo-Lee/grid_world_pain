#!/usr/bin/env python3
"""compare_estimators.py - the one properly replicated comparison in this study.

Every other comparison in the neuromodulator grid has one seed per cell. This one does not: the
five unmodulated reference runs were trained twice, once under Monte-Carlo returns and once under
normalised GAE, on the same day, from the same five seeds, from configs that differ in exactly one
substantive key (`agent.return_mode`). Five against five, one variable.

WHY THE CONTRAST MUST BE DECOMPOSED. The headline number is "response to a nearby predator",
defined as hiding when one is 1-2 cells away minus hiding when the nearest is 6 or more away. That
difference is 5.5 percentage points larger under GAE_NORM, which reads as "answers a predator more
strongly" and is wrong: the near term does not move (p = 0.38). The whole difference is in the FAR
term - the GAE_NORM agent sits in cover much less when nothing is hunting it. Its hiding is more
SELECTIVE, not stronger, and a figure that plots only the contrast hides which half moved.

INPUT   results/analysis/{nmn_site_grid,nmn_gaenorm_grid}/ladderstyle/<cell>.json
OUTPUT  docs/experiments/active/nmn_input_site_grid/figures_gae/g08_estimator_effect.png
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
import _ladder as L                                                     # noqa: E402

MC  = "results/analysis/nmn_site_grid/ladderstyle"
GAE = "results/analysis/nmn_gaenorm_grid/ladderstyle"
FIG = os.environ.get("NMN_FIG_ROOT", "docs/experiments/active/nmn_input_site_grid/figures_gae")
CTRL = [f"baseline_s{s}" for s in (42, 43, 44, 45, 46)]
PAPER = "#f8f7f5"; ANNO = "#3d3c3a"
# Two estimators are not two input slices, so they may not use the page's three data colours -- and
# "not the same colour" is not the test. A first attempt used a blue-grey that measured only 13.5
# CIE76 from `--extero`, the blue that means "reads the world only" on this page: distinguishable in
# a swatch, and read as "blue" by anyone who met the legend three thousand pixels earlier. Every hue
# on the page is spent, so the answer is the neutral ramp the register prescribes -- and these two
# neutrals are the ones already used for the outcome categories in g04, so the page carries one
# neutral pair rather than inventing a second.
C_MC, C_GAE = "#3c4650", "#8d99a6"
plt.rcParams.update({"font.size": 17, "axes.titlesize": 19, "axes.labelsize": 17,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 15,
                     "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
                     "axes.facecolor": "#ffffff", "axes.axisbelow": True})


def nearfar(root, name, bush="pd_bush", tot="pd_tot"):
    d = json.load(open(f"{root}/{name}.json"))
    B = np.array(d["grids"][bush], float); T = np.array(d["grids"][tot], float)
    n = 100 * B[list(L.NEAR_BINS)].sum() / T[list(L.NEAR_BINS)].sum()
    f = 100 * B[list(L.FAR_BINS)].sum() / T[list(L.FAR_BINS)].sum()
    return n, f


def main():
    near_mc  = [nearfar(MC,  n)[0] for n in CTRL]; far_mc  = [nearfar(MC,  n)[1] for n in CTRL]
    near_gae = [nearfar(GAE, n)[0] for n in CTRL]; far_gae = [nearfar(GAE, n)[1] for n in CTRL]

    fig, ax = plt.subplots(figsize=(11.0, 7.6))
    groups = [("a predator 1\u20132 cells away", near_mc, near_gae),
              ("no predator within 6 cells", far_mc, far_gae)]
    w = 0.30
    for gi, (lab, a, b) in enumerate(groups):
        for oi, (vals, col, nm) in enumerate(((a, C_MC, "Monte-Carlo returns"),
                                              (b, C_GAE, "normalised GAE"))):
            x = gi + (oi - .5) * w * 1.25
            ax.bar(x, np.mean(vals), width=w, color=col, zorder=2,
                   label=nm if gi == 0 else None)
            # the five seeds themselves, so the reader sees the spread the bar is an average of
            ax.scatter(np.full(len(vals), x), vals, s=40, facecolor="white",
                       edgecolor=ANNO, lw=1.1, zorder=4)
    # a mid-grey bar and a dark-grey bar read as one series if nothing separates them
    for b in ax.patches: b.set_edgecolor(ANNO); b.set_linewidth(1.0)
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("share of those steps spent in a bush (%)")
    ax.set_title("Where the estimator's effect actually is", loc="left")
    ax.set_ylim(0, 58); ax.grid(axis="y", alpha=.25, lw=.5)
    ax.legend(loc="upper right", framealpha=.94)
    d_near = np.mean(near_gae) - np.mean(near_mc)
    d_far  = np.mean(far_gae)  - np.mean(far_mc)
    ax.annotate(f"{d_near:+.1f} pp\nnot distinguishable\n(p = 0.38)", xy=(0, 52), ha="center",
                fontsize=15, color=ANNO)
    ax.annotate(f"{d_far:+.1f} pp\nthe whole effect\n(p = 0.004)", xy=(1, 30), ha="center",
                fontsize=15, color=ANNO, fontweight="bold")
    os.makedirs(FIG, exist_ok=True)
    p = f"{FIG}/g08_estimator_effect.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    from PIL import Image
    a = np.asarray(Image.open(p).convert("RGB")).astype(int)
    ink = (np.abs(a - np.array([248, 247, 245])).sum(2) > 40)
    for side, strip in (("left", ink[:, :3]), ("right", ink[:, -3:]),
                        ("top", ink[:3, :]), ("bottom", ink[-3:, :])):
        if strip.sum():
            raise SystemExit(f"g08: ink in the {side} margin - something is clipped")
    print(f"  wrote {p}")
    print(f"  near: MC {np.mean(near_mc):.2f}%  GAE {np.mean(near_gae):.2f}%  ({d_near:+.2f} pp)")
    print(f"  far : MC {np.mean(far_mc):.2f}%  GAE {np.mean(far_gae):.2f}%  ({d_far:+.2f} pp)")


if __name__ == "__main__":
    main()
