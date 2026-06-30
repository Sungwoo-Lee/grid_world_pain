"""Polished heatmap renderer for behavior-measure summary matrices (journal style).

Shared by the behavior_measures analysis scripts. Renders rows=experiment x
cols=criterion with: per-column semantically-appropriate color (sequential for
magnitude metrics, diverging-at-0 for signed metrics), card-style cells, two-tier
mean/std annotation, column-group headers, row-group separators, and PNG+PDF output.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# Helvetica/Arial-equivalent for academic figures, with safe fallback.
for _f in ("Nimbus Sans", "Liberation Sans", "DejaVu Sans"):
    try:
        plt.rcParams["font.family"] = _f; break
    except Exception:
        continue
plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0})

SEQ = sns.color_palette("crest", as_cmap=True)
DIV = plt.cm.RdBu_r
MUTED = "#6b7280"


def _text_color(rgba):
    r, g, b = rgba[:3]
    return "white" if (0.299 * r + 0.587 * g + 0.114 * b) < 0.55 else "#222222"


def _fmt(mu, sd):
    if np.isnan(mu):
        return None, None
    m = f"{mu:.2f}" if abs(mu) < 10 else f"{mu:.0f}"
    s = f"±{sd:.2f}" if abs(sd) < 10 else f"±{sd:.0f}"
    return m, s


def render(mean, std, row_labels, col_labels, *, signed_cols=(), col_groups=None,
          row_group_bounds=(), title="", caption="", out_png=None, out_pdf=None):
    """mean/std: (nrow,ncol) arrays (NaN allowed). signed_cols: indices using the
    diverging-at-0 map. col_groups: list of (name,[col idxs]). row_group_bounds:
    row indices where a separator is drawn ABOVE the row."""
    nrow, ncol = mean.shape
    # per-column facecolors
    face = np.zeros((nrow, ncol, 4))
    for j in range(ncol):
        col = mean[:, j]
        good = col[~np.isnan(col)]
        if j in signed_cols and good.size and good.min() < 0 < good.max():
            norm = mcolors.TwoSlopeNorm(vmin=good.min(), vcenter=0.0, vmax=good.max())
            for i in range(nrow):
                face[i, j] = (0.93, 0.93, 0.93, 1) if np.isnan(col[i]) else DIV(norm(col[i]))
        else:
            lo, hi = (good.min(), good.max()) if good.size else (0, 1)
            for i in range(nrow):
                t = 0.5 if hi == lo else (col[i] - lo) / (hi - lo)
                face[i, j] = (0.93, 0.93, 0.93, 1) if np.isnan(col[i]) else SEQ(t)

    fig_w, fig_h = 1.55 * ncol + 2.6, 0.74 * nrow + 2.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, ncol); ax.set_ylim(0, nrow); ax.invert_yaxis()
    ax.set_aspect("auto"); ax.axis("off")

    pad = 0.06  # gap -> card look
    for i in range(nrow):
        for j in range(ncol):
            fc = face[i, j]
            ax.add_patch(FancyBboxPatch((j + pad, i + pad), 1 - 2 * pad, 1 - 2 * pad,
                                        boxstyle="round,pad=0,rounding_size=0.06",
                                        linewidth=0, facecolor=fc, mutation_aspect=1))
            m, s = _fmt(mean[i, j], std[i, j])
            if m is None:
                ax.text(j + 0.5, i + 0.5, "–", ha="center", va="center", color=MUTED, fontsize=11)
                continue
            tc = _text_color(fc)
            ax.text(j + 0.5, i + 0.40, m, ha="center", va="center", color=tc, fontsize=10, fontweight="medium")
            sc = "white" if tc == "white" else MUTED
            ax.text(j + 0.5, i + 0.72, s, ha="center", va="center",
                    color=sc, fontsize=6.8, alpha=0.85 if tc == "white" else 1.0)

    # column labels (bottom)
    for j, lab in enumerate(col_labels):
        ax.text(j + 0.5, nrow + 0.18, lab, ha="center", va="top", fontsize=8.6, linespacing=1.15)
    # row labels (left)
    for i, lab in enumerate(row_labels):
        ax.text(-0.12, i + 0.5, lab, ha="right", va="center", fontsize=8.6)

    # column-group headers + separators
    if col_groups:
        for name, idxs in col_groups:
            c0, c1 = min(idxs), max(idxs) + 1
            ax.plot([c0 + 0.04, c1 - 0.04], [-0.30, -0.30], color="#9aa0a6", lw=1.1, clip_on=False)
            ax.text((c0 + c1) / 2, -0.46, name, ha="center", va="bottom",
                    fontsize=9.2, fontweight="bold", color="#374151")
        for name, idxs in col_groups[1:]:
            x = min(idxs)
            ax.plot([x, x], [0, nrow], color="#ffffff", lw=4, clip_on=False)
            ax.plot([x, x], [0, nrow], color="#cdd2d8", lw=1.4, clip_on=False)
    # row-group separators
    for r in row_group_bounds:
        ax.plot([0, ncol], [r, r], color="#cdd2d8", lw=1.4, clip_on=False)

    if title:
        fig.suptitle(title, x=0.5, y=0.99, fontsize=13, fontweight="bold", ha="center")
    if caption:
        import textwrap
        wrapped = "\n".join(textwrap.wrap(caption, width=120))
        ax.text(0.0, nrow + 1.35, wrapped, ha="left", va="top",
                fontsize=7.8, color=MUTED, linespacing=1.5)
    if out_png:
        fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
