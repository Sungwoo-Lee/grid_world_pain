"""FIGURE 5 - Can the agent tell a predator from a rabbit?

QUESTION. A rabbit cannot hurt the agent. If an arm hides just as hard for a nearby rabbit as for a
nearby predator, that arm is not discriminating - it is reacting to "an animal is near" and paying
for it in lost foraging time. This figure asks which sensory settings buy the ability to tell the
two apart.

WHAT THE ANSWER TURNS OUT TO BE. The arms split into two groups, and the split is not
sight-versus-no-sight. It is whether the agent's sight can resolve WHAT it is looking at. An agent
with a 13-cell visual field carrying eight appearance channels shows no rabbit response at all. An
agent whose visual field carries a single channel - it sees THAT something is there but not WHAT -
falls back into the same false alarm as an agent with no useful sight whatsoever.

HOW IT IS COMPUTED. For each arm and each animal class, the proximity effect is
    P(in bush | nearest animal 1-2 cells away) - P(in bush | nearest animal 6+ cells away)
in percentage points, over all 300,000 episodes. Distance is chebyshev (the moves a king would
need) and is read off the row BEFORE the step, since that is the observation the action was chosen
on. Counts are pooled before the ratio is taken, so a distance bin with more steps carries more
weight. The left panel shows both classes with a line joining them, so the length of the line is
the arm's discrimination. The right panel isolates the rabbit response, which is the part that is
pure waste.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

D = L.load_all(); arms = L.ARM_ORDER
P = np.array([L.proximity_effect(D[a]["grids"]["pd_bush"], D[a]["grids"]["pd_tot"]) for a in arms])
R = np.array([L.proximity_effect(D[a]["grids"]["rd_bush"], D[a]["grids"]["rd_tot"]) for a in arms])

# Does the arm's visual field carry enough channels to tell a predator from a rabbit at range?
def resolves_identity(a):
    s = D[a]["sensory"]
    return s["visual_sensor_range"] >= 2 and s["visual_vector_size"] > 1

grp = np.array([resolves_identity(a) for a in arms])
CY, CN = "#2f6f9f", "#b3322b"

fig, ax = plt.subplots(1, 2, figsize=(13.4, 6.0), sharey=True,
                       gridspec_kw={"width_ratios": [1.25, 1]})
y = np.arange(len(arms))[::-1]
ax[0].hlines(y, R, P, color=PL.GRID, lw=2.4, zorder=1)
ax[0].scatter(R, y, s=54, color=PL.HARMLESS, zorder=3, label="nearest animal is a RABBIT (harmless)")
ax[0].scatter(P, y, s=54, color=PL.THREAT, zorder=3, label="nearest animal is a PREDATOR (a real threat)")
ax[0].axvline(0, color=PL.INK, lw=1)
ax[0].set_yticks(y); ax[0].set_yticklabels(PL.arm_ylabels(arms), fontsize=8)
ax[0].set_ylabel("sensor-ladder arm  (poorest senses at the bottom)")
ax[0].set_xlabel("hiding triggered by a nearby animal  (percentage points)\n"
                 "bush dwell at 1-2 cells minus at 6+ cells")
ax[0].grid(axis="y", visible=False)
ax[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=1, fontsize=8.5)
for i in range(len(arms)):
    ax[0].text(P[i] + 1.2, y[i], f"gap {P[i]-R[i]:.0f}", va="center", fontsize=7, color=PL.MUTED)
ax[0].set_xlim(min(R.min(), 0) - 4, P.max() + 9)

ax[1].barh(y, R, color=[CY if g else CN for g in grp], height=0.72, edgecolor="none")
ax[1].axvline(0, color=PL.INK, lw=1)
ax[1].set_xlabel("FALSE ALARM: hiding triggered by a nearby rabbit\n"
                 "(percentage points; above zero = wasted hiding)")
ax[1].grid(axis="y", visible=False)
m = max(abs(R)) * 1.55
ax[1].set_xlim(-m, m)
for i in range(len(arms)):
    PL.outward_label(ax[1], R[i], y[i], m * 0.025)
h = [plt.Rectangle((0, 0), 1, 1, color=CY, label="sight resolves WHAT it sees\n(visual range 2, 8 appearance channels)"),
     plt.Rectangle((0, 0), 1, 1, color=CN, label="sight cannot resolve WHAT it sees\n(range < 2, or a single channel)")]
ax[1].legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=8, ncol=1)
PL.finish(fig, f"{L.FIG_ROOT}/lad05_discrimination.png")
print(f"{'arm':22}{'predator':>10}{'rabbit':>9}{'gap':>8}   resolves identity")
for i, a in enumerate(arms):
    print(f"{a:22}{P[i]:>10.1f}{R[i]:>9.1f}{P[i]-R[i]:>+8.1f}   {grp[i]}")
