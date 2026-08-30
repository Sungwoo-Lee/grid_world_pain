"""House plotting style for the sensor-ladder figures.

Every figure in this folder obeys the same four rules, because the previous round of figures
broke all four at least once:
  1. both axes carry a label that names the quantity AND its unit
  2. long arm names go on the y-axis of a horizontal bar chart, never rotated under an x-axis
  3. a second y-axis is never used - two quantities on different scales get two panels
  4. a cell with too little support is drawn as a visible gap, never as a silent zero
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import _ladder as L

DPI = 200
INK, MUTED, GRID = "#1c1c1e", "#6b6b70", "#dcdce0"
# poorest senses -> richest, so colour carries the ladder itself
LADDER_CMAP = plt.get_cmap("viridis")
THREAT, HARMLESS = "#b3322b", "#2f6f9f"          # predator vs rabbit, used consistently
LOWINJ, HIGHINJ = "#8fb8d8", "#7a2438"           # light wound vs heavy wound

plt.rcParams.update({
    "figure.dpi": DPI, "savefig.dpi": DPI, "font.size": 9,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white",
})


def arm_colors(arms=None):
    arms = arms or L.ARM_ORDER
    return {a: LADDER_CMAP(i / max(len(arms) - 1, 1) * 0.88) for i, a in enumerate(arms)}


def arm_ylabels(arms=None):
    """`V4_blur05 - reference agent` : the config name first, its meaning after."""
    arms = arms or L.ARM_ORDER
    return [f"{a}  -  {L.ARM_LABEL[a][0]}" for a in arms]


def finish(fig, path, tight=True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"written: {path}")
    return path


def hbar_axis(ax, arms, values, colors, xlabel, note=None):
    """Horizontal bars with the arm names spelled out - the layout that cannot overlap."""
    y = np.arange(len(arms))[::-1]
    ax.barh(y, values, color=[colors[a] for a in arms], height=0.72, edgecolor="none")
    ax.set_yticks(y); ax.set_yticklabels(arm_ylabels(arms), fontsize=8)
    ax.set_ylabel("sensor-ladder arm  (poorest senses at the bottom)")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    if note:
        ax.set_title(note, fontsize=9, color=MUTED, loc="left", pad=8)
    return y


def label_points(ax, xs, ys, labels, colors, min_gap=0.052, dx=0.035, fontsize=7.2):
    """Annotate scatter points so no two labels collide.

    Fourteen arms land in a tight cluster on several of these figures, and plain offset text
    overlaps to the point of being unreadable. This lays the labels out in axes-fraction space:
    points on the left half get their label to the right and vice versa, then within each side the
    labels are pushed apart vertically until they clear each other, with a leader line back to the
    point. No dependency on adjustText, which is not installed in this environment.
    """
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    fx, fy = np.array([ax.transLimits.transform((x, y)) for x, y in zip(xs, ys)]).T
    right = fx < 0.5
    for side in (True, False):
        idx = np.flatnonzero(right == side)
        if not len(idx):
            continue
        idx = idx[np.argsort(fy[idx])]
        pos = fy[idx].copy()
        for i in range(1, len(pos)):                       # push up from the bottom
            pos[i] = max(pos[i], pos[i - 1] + min_gap)
        over = pos[-1] - 1.0
        if over > 0:                                       # then push back down if it overflowed
            pos -= over
            for i in range(len(pos) - 2, -1, -1):
                pos[i] = min(pos[i], pos[i + 1] - min_gap)
        for k, i in enumerate(idx):
            tx = fx[i] + dx if side else fx[i] - dx
            ax.annotate(labels[i], xy=(xs[i], ys[i]), xycoords="data",
                        xytext=(np.clip(tx, 0.01, 0.99), np.clip(pos[k], 0.01, 0.99)),
                        textcoords="axes fraction", fontsize=fontsize, color=INK,
                        va="center", ha="left" if side else "right",
                        arrowprops=dict(arrowstyle="-", lw=0.55, color=MUTED,
                                        shrinkA=0.5, shrinkB=2.5))


def outward_label(ax, value, y, pad, fmt="{:+.1f}", color=None, fontsize=7.2):
    """Put a bar's number outside the bar, on whichever side the bar points."""
    x = value + (pad if value >= 0 else -pad)
    ax.text(x, y, fmt.format(value), va="center",
            ha="left" if value >= 0 else "right", fontsize=fontsize, color=color or INK)
