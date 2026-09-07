"""FIGURE 4 - Does the agent hide more when a predator is close, and does that need eyes?

QUESTION. Hiding is only defensive if it is triggered by the threat. This figure plots bush
occupancy against how far the nearest predator was when the agent chose its move, one curve per
arm. A flat curve means the agent hides on a schedule; a rising-toward-zero curve means it hides
in response to something it perceived.

HOW IT IS COMPUTED. For every step, the distance from the agent to the nearest ACTIVE predator is
measured in chebyshev steps (the number of moves a king would need, since the agent moves
diagonally too). The action that produced step t was chosen while the agent was looking at step
t-1, so the distance is read off the PREVIOUS row and the bush occupancy off the current one.
Distances of 8 or more are pooled into one bin. A bin with fewer than 1,000 steps is left as a
gap rather than drawn as a noisy point.

WHAT IT CANNOT SHOW. Predator distance is not randomised - a predator is close partly because of
where the agent went. This curve is therefore descriptive. The causal claims in this report come
from the randomised starting wound (Figures 8-11) and the randomised odour draw (Figure 12).
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

D = L.load_all(); arms = L.ARM_ORDER
x = np.arange(1, L.DIST_MAX + 1)
GRP = {a: L.resolves_identity(D[a]["sensory"]) for a in arms}
# The four the prose names. Everything else is background, drawn thin so the shape of the two
# families is still visible without fourteen near-identical colours competing for attention.
SPOT = {"A_baseline": "smell, no direction", "B_olf_only": "smell with direction",
        "V4_blur05": "reference agent", "V5_sharp": "sharp sight"}

fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=True)
for j, (key, ttl) in enumerate([("pd", "Nearest PREDATOR - a real threat"),
                                ("rd", "Nearest RABBIT - harmless by construction")]):
    ends = []   # in the rabbit panel three named lines finish within a few points of each other
    for a in arms:
        g = D[a]["grids"]
        y = L.dist_curve(g[f"{key}_bush"], g[f"{key}_tot"])
        c = PL.GROUP_YES if GRP[a] else PL.GROUP_NO
        if a in SPOT:
            ax[j].plot(x, y, lw=2.4, color=c, marker="o", ms=4.2, zorder=3)
            ends.append((x[-1], y[-1], f" {a}", c))
        else:
            ax[j].plot(x, y, lw=1.0, color=c, alpha=0.34, zorder=2)
    ax[j].set_title(ttl, fontsize=10, color=PL.INK, loc="left", pad=8)
    ax[j].set_xlabel("distance to the nearest animal when it decided\n"
                     "chebyshev steps - the moves a chess king would need\n"
                     "8 = eight or more")
    ax[j].set_xticks(x); ax[j].set_xticklabels(L.DIST_NAMES)
    ax[j].set_xlim(0.7, L.DIST_MAX + 1.6)
    PL.stagger_end_labels(ax[j], ends)
ax[0].set_ylabel("bush hiding  (% of those steps spent in a bush)")
h = [plt.Line2D([], [], color=PL.GROUP_YES, lw=2.2, label=L.GROUP_LABEL[True] + "  (9 arms)"),
     plt.Line2D([], [], color=PL.GROUP_NO, lw=2.2, label=L.GROUP_LABEL[False] + "  (5 arms)"),
     plt.Line2D([], [], color=PL.MUTED, lw=2.4, marker="o", ms=4,
                label="thick + named = an arm the text discusses; thin = the other ten")]
ax[0].legend(handles=h, loc="lower center", bbox_to_anchor=(1.03, 1.10), ncol=1, fontsize=8.3)
POP = L.population()
pd_used = sum(int(np.asarray(D[a]["grids"]["pd_tot"], float).sum()) for a in arms)
rd_used = sum(int(np.asarray(D[a]["grids"]["rd_tot"], float).sum()) for a in arms)
L.record_samples("lad04_threat_distance_curve", [
    dict(what="step rows, predator panel", used=pd_used, total=POP["steps"],
         note="excludes episodes containing no predator, and each episode's first step, "
              "which has no previous row to read the distance from"),
    dict(what="step rows, rabbit panel", used=rd_used, total=POP["steps"],
         note="same, for episodes containing no rabbit")])

PL.assert_labels_fit(fig, ax)
PL.finish(fig, f"{L.FIG_ROOT}/lad04_threat_distance_curve.png")
print(f"{'arm':22}" + "".join(f"{d:>7}" for d in L.DIST_NAMES) + "   (predator, % in bush)")
for a in arms:
    g = D[a]["grids"]
    print(f"{a:22}" + "".join(f"{v:>7.1f}" for v in L.dist_curve(g["pd_bush"], g["pd_tot"])))
