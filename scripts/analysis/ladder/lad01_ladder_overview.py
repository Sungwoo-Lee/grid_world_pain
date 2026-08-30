"""FIGURE 1 - What does each sense buy the agent?

QUESTION. Fourteen agents were trained on the same world with the same seed and differ only in
what they can sense. If a sense matters, the agents that have it should live longer. This figure
puts survival and bush dwell side by side for all fourteen, in ladder order.

WHY BOTH PANELS. Survival is the project's performance measure. Bush dwell - the share of an
episode's steps spent standing in a bush - is the behaviour we are trying to explain. Showing
them together is the whole point: an agent can raise one by sacrificing the other, because a bush
is safe but has no food in it.

HOW IT IS COMPUTED. Survival is the mean episode length over 300,000 evaluation episodes.
Bush dwell is (bush steps) / (steps), pooled over every episode of the arm; the t=0 row is the
world as handed to the agent, not a step it took, so it is in neither the numerator nor the
denominator.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

D = L.load_all()
arms = L.ARM_ORDER
surv = [D[a]["mean_survival"] for a in arms]
dwell = [D[a]["bush_dwell_pct"] for a in arms]
col = PL.arm_colors()

fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
PL.hbar_axis(ax[0], arms, surv, col, "mean survival  (steps per episode)",
             "How long it lives")
ax[0].set_xlim(0, max(surv) * 1.12)
for i, v in enumerate(surv):
    ax[0].text(v + 4, len(arms) - 1 - i, f"{v:.0f}", va="center", fontsize=8, color=PL.INK)

PL.hbar_axis(ax[1], arms, dwell, col, "bush dwell  (% of an episode's steps spent in a bush)",
             "How much it hides")
ax[1].set_ylabel("")
ax[1].set_xlim(0, max(dwell) * 1.15)
for i, v in enumerate(dwell):
    ax[1].text(v + 0.3, len(arms) - 1 - i, f"{v:.1f}%", va="center", fontsize=8, color=PL.INK)

PL.finish(fig, f"{L.FIG_ROOT}/lad01_ladder_overview.png")
print(f"\n{'arm':22}{'survival':>10}{'bush dwell':>13}")
for a, s, d in zip(arms, surv, dwell):
    print(f"{a:22}{s:>10.1f}{d:>12.1f}%")
