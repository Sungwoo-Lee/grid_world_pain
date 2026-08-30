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
from the randomised starting wound (Figures 6-8) and the randomised odour draw (Figure 9).
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

D = L.load_all(); arms = L.ARM_ORDER; col = PL.arm_colors()
x = np.arange(1, L.DIST_MAX + 1)

fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.4), sharex=True)
for j, (key, ttl, c0) in enumerate([
        ("pd", "Nearest PREDATOR - a real threat", PL.THREAT),
        ("rd", "Nearest RABBIT - harmless by construction", PL.HARMLESS)]):
    for a in arms:
        g = D[a]["grids"]
        y = L.dist_curve(g[f"{key}_bush"], g[f"{key}_tot"])
        ax[j].plot(x, y, marker="o", ms=3.2, lw=1.6, color=col[a],
                   label=f"{a} - {L.ARM_LABEL[a][0]}")
    ax[j].set_title(ttl, fontsize=10, color=PL.INK, loc="left", pad=8)
    ax[j].set_xlabel("distance from agent to the nearest animal when it decided\n"
                     "(chebyshev steps; 8 = eight or more)")
    ax[j].set_xticks(x); ax[j].set_xticklabels(L.DIST_NAMES)
ax[0].set_ylabel("bush dwell  (% of those steps spent in a bush)")
ax[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.4,
             title="sensor-ladder arm", title_fontsize=8)
lo = min(np.nanmin(L.dist_curve(D[a]["grids"][f"{k}_bush"], D[a]["grids"][f"{k}_tot"]))
         for a in arms for k in ("pd", "rd"))
hi = max(np.nanmax(L.dist_curve(D[a]["grids"][f"{k}_bush"], D[a]["grids"][f"{k}_tot"]))
         for a in arms for k in ("pd", "rd"))
for a_ in ax: a_.set_ylim(max(0, lo - 3), hi + 3)
PL.finish(fig, f"{L.FIG_ROOT}/lad04_threat_distance_curve.png")
print(f"{'arm':22}" + "".join(f"{d:>7}" for d in L.DIST_NAMES) + "   (predator, % in bush)")
for a in arms:
    g = D[a]["grids"]
    print(f"{a:22}" + "".join(f"{v:>7.1f}" for v in L.dist_curve(g["pd_bush"], g["pd_tot"])))
