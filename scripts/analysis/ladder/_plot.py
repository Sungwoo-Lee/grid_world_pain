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

DPI = 260
INK, MUTED, GRID = "#1c1c1e", "#6b6b70", "#dcdce0"
# poorest senses -> richest, so colour carries the ladder itself
LADDER_CMAP = plt.get_cmap("viridis")
# ONE MEANING PER COLOUR, ACROSS EVERY FIGURE. An earlier draft used red and blue for five
# different things - sign of a change, predator, "cannot resolve identity", heavy wound, and cause
# of death - twice within a single image. A reader who learns a colour on one figure must not be
# punished for carrying it to the next, so the mapping is fixed here and nowhere else:
THREAT, HARMLESS = "#b3322b", "#2f6f9f"          # predator (red) vs rabbit (blue). ALWAYS.
WOUND_LO, WOUND_HI = "#c3b3d4", "#54346e"        # wound level: light purple -> deep purple
GROUP_YES, GROUP_NO = "#2d6a4f", "#c9762e"       # sight resolves identity (green) or not (orange)
NEUTRAL = "#6d8595"                              # a bar whose colour carries nothing but its sign
ACCENT = GROUP_YES                               # alias: the page accent is the same green

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
    y = np.arange(len(arms))
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


def group_lines(ax, x, curves: dict, grp: dict, spotlight=(), lw_thin=0.9, lw_bold=2.5,
                label_end=False, alpha=0.30):
    """Draw one line per arm, coloured by the two-group split rather than a 14-step ramp.

    Register entry F12: more than about six series cannot be told apart by colour alone, and a
    continuous ramp is worst of all, because the arms a reader is asked to compare are usually
    ADJACENT in the ramp. Colour instead by the distinction the argument turns on, draw everything
    else thin, and name only the series the prose discusses.

    `curves` maps arm -> y values, `grp` maps arm -> True/False (resolves identity), `spotlight`
    names the arms to draw bold. Returns the legend handles.
    """
    import matplotlib.pyplot as _plt
    for a, y in curves.items():
        c = GROUP_YES if grp[a] else GROUP_NO
        if a in spotlight:
            ax.plot(x, y, lw=lw_bold, color=c, marker="o", ms=4.0, zorder=3)
            if label_end:
                ax.annotate(f" {a}", (x[-1], y[-1]), fontsize=7.2, color=c, va="center",
                            xytext=(4, 0), textcoords="offset points", zorder=4)
        else:
            ax.plot(x, y, lw=lw_thin, color=c, alpha=alpha, zorder=2)
    return [_plt.Line2D([], [], color=GROUP_YES, lw=2.2,
                        label="sight resolves WHAT it sees  (9 arms)"),
            _plt.Line2D([], [], color=GROUP_NO, lw=2.2,
                        label="sight cannot resolve WHAT it sees  (5 arms)"),
            _plt.Line2D([], [], color=MUTED, lw=2.4, marker="o", ms=4,
                        label="thick + named = an arm the text discusses")]


def stagger_end_labels(ax, points, min_gap_frac=0.055):
    """Place end-of-line labels so they cannot overprint each other.

    `points` is a list of (x, y, text, colour). Three lines that finish at nearly the same height
    printed their names on top of one another, which read as garble.
    """
    if not points:
        return
    pts = sorted(points, key=lambda p: p[1])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * min_gap_frac
    ys = [p[1] for p in pts]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    over = ys[-1] - hi
    if over > 0:
        ys = [y - over for y in ys]
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - gap)
    for (x, y0, txt, col), y in zip(pts, ys):
        ax.annotate(txt, xy=(x, y0), xytext=(x + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.02, y),
                    fontsize=7.2, color=col, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=MUTED, shrinkA=0, shrinkB=2))


def assert_labels_fit(fig, axes, slack=1.04):
    """Refuse to write a figure whose axis labels are wider than the panel they belong to.

    Matplotlib silently lets an xlabel run past its axes and off the canvas: in a multi-panel
    figure two neighbouring labels then print through each other, and the outermost one is clipped
    at the edge. Nothing warns, and the defect lives inside the PNG where a DOM-based layout
    checker cannot see it. This measures each label's rendered width against its own panel and
    raises, in the same read-back-and-assert spirit the bar annotations already use.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for ax in np.atleast_1d(axes).ravel():
        panel = ax.get_window_extent(renderer=r).width
        for which, art in (("xlabel", ax.xaxis.label), ("title", ax.title)):
            if not art.get_text():
                continue
            w = max(art.get_window_extent(renderer=r).width, 1.0)
            if w > panel * slack:
                bad.append(f"{which} is {w:.0f}px wide in a {panel:.0f}px panel: "
                           f"{art.get_text().splitlines()[0][:60]!r}")
    if bad:
        raise SystemExit("axis text does not fit its panel -\n  " + "\n  ".join(bad) +
                         "\nShorten it, or split it across more lines.")
