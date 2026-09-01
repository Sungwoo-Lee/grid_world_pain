"""FIGURE 3 - What actually kills each agent.

QUESTION. Survival alone hides the trade-off. An agent that hides constantly does not get eaten,
it starves; an agent that forages constantly does not starve, it gets eaten. This figure splits
every arm's 1,000,000 episodes into the three ways an episode can end, so the reader can see WHICH
failure each sensory setting buys down and which one it buys up.

HOW IT IS COMPUTED. The `termination_reason` column of the episode table, counted per arm.
1 = reached the step limit alive, 2 = starved, 4 = killed by a predator. The three shares sum to
100% by construction.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

D = L.load_all(); arms = L.ARM_ORDER
order = ["killed by predator", "starved", "survived to time limit"]
cols = {"killed by predator": PL.THREAT, "starved": "#c9a227",
        "survived to time limit": "#4a7c59"}
M = np.array([[D[a]["term_pct"].get(k, 0.0) for k in order] for a in arms])

fig, ax = plt.subplots(figsize=(10.2, 5.6))
y = np.arange(len(arms))
left = np.zeros(len(arms))
for j, k in enumerate(order):
    ax.barh(y, M[:, j], left=left, color=cols[k], height=0.72, edgecolor="white",
            linewidth=0.6, label=k)
    for i in range(len(arms)):
        if M[i, j] > 6:
            ax.text(left[i] + M[i, j] / 2, y[i], f"{M[i,j]:.0f}%", va="center", ha="center",
                    fontsize=7.5, color="white")
    left += M[:, j]
ax.set_yticks(y); ax.set_yticklabels(PL.arm_ylabels(arms), fontsize=8)
ax.set_ylabel("sensor-ladder arm  (poorest senses at the bottom)")
ax.set_xlabel("share of the arm's 1,000,000 episodes  (%)")
ax.set_xlim(0, 100); ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8.5)
P = L.population()
L.record_samples("lad03_how_it_ends", [
    dict(what="episodes classified by outcome", used=P["episodes"], total=P["episodes"],
         note="every episode ends exactly one of the three ways, so the shares sum to 100%")])

PL.finish(fig, f"{L.FIG_ROOT}/lad03_how_it_ends.png")
print(f"{'arm':22}" + "".join(f"{k[:14]:>17}" for k in order))
for a, row in zip(arms, M):
    print(f"{a:22}" + "".join(f"{v:>16.1f}%" for v in row))
