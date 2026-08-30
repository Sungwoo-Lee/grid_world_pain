"""FIGURE 14 - Two different ways to get the injury result wrong.

QUESTION. "Does injury make the agent hide?" has three plausible-looking answers in this data, and
two of them are wrong. This figure puts all three side by side, because a reader who saw only one
would come away believing something false - and because the two mistakes have DIFFERENT causes that
are easy to conflate.

PANEL A is the honest measurement. The wound is the one the environment ASSIGNED at t=0, drawn
uniformly at random before the agent acts, and the outcome is measured over the episode's first 25
steps, while that assigned dose is still largely intact. Waking up wounded makes the agent hide
more.

PANEL B changes ONE thing: the same randomised wound, but the outcome averaged over the whole
episode. The effect reverses. Nothing about the cause changed - only the window. The reversal is a
weighting artefact: hiding is much commoner early in an episode than late, and a lightly wounded
agent goes on to have a LONGER episode (270 steps against 255), so its whole-episode average is
diluted by more late, low-hiding steps. The reversal says something about episode length, not about
what a wound does.

PANEL C changes the variable instead: the wound the agent was CARRYING at the moment it decided,
mid-episode. This is the number that comes for free from any trajectory log, and it is the most
misleading of the three, because a mid-episode wound is a CONSEQUENCE of behaviour. The agent is
carrying a big wound precisely because it was out in the open near a predator - which is also where
the bushes are not. This panel measures where the agent WAS and reports it as what the agent DECIDED.

ALL THREE PANELS SHARE ONE Y-SCALE. They did not in an earlier draft, and the effect was to make
panel B's reversal - which is worth well under one percentage point - look as large as panel A's
real effect, in a figure whose whole purpose is to show that the three answers are not equally good.

TAKE-AWAY. Panel A is the number this report uses. B and C are shown so that the choice is visible
rather than asserted, and so that a future reader who reproduces one of them knows why it differs.

HOW IT IS COMPUTED. All three use identical bins - four equal quarters of injury on 0-100 - and the
identical outcome, bush dwell with the t=0 row excluded. They differ only in which injury number a
step is filed under and over which steps the average is taken.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

D = L.load_all(); arms = L.ARM_ORDER; col = PL.arm_colors()
x = np.arange(4)

early, whole = {}, {}
for a in arms:
    z = np.load(f"{L.OUT_ROOT}/{a}_episodes.npz")
    ib = np.digitize(z["inj0"], L.INJ_EDGES)
    early[a] = np.array([100 * z["bush_early"][ib == k].sum() / z["steps_early"][ib == k].sum()
                         for k in range(4)])
    whole[a] = np.array([100 * z["bush_steps"][ib == k].sum() / z["n_steps"][ib == k].sum()
                         for k in range(4)])
carried = {a: L.rate(D[a]["grids"]["dw_carried"], D[a]["grids"]["dwt_carried"]) for a in arms}

# cross-check panel A against the grid the sweep accumulated independently
for a in arms:
    g = L.rate(D[a]["grids"]["dw_early"], D[a]["grids"]["dwt_early"])
    if not np.allclose(g, early[a], atol=1e-6):
        raise SystemExit(f"{a}: the per-episode and per-step early windows disagree")

fig, ax = plt.subplots(1, 3, figsize=(15.6, 5.6), sharey=True)
panels = [
    (early, "A.  ASSIGNED wound, first 25 steps\nthe honest measurement",
     "wound the environment handed it at t=0  (0-100)"),
    (whole, "B.  ASSIGNED wound, whole episode\nsame cause, diluted window - effect reverses",
     "wound the environment handed it at t=0  (0-100)"),
    (carried, "C.  CARRIED wound, whole episode\ndifferent variable - a consequence, not a cause",
     "wound it was carrying when it decided  (0-100)")]
for j, (dat, ttl, xl) in enumerate(panels):
    for a in arms:
        ax[j].plot(x, dat[a], marker="o", ms=4, lw=1.7, color=col[a],
                   label=f"{a} - {L.ARM_LABEL[a][0]}")
    ax[j].set_xticks(x); ax[j].set_xticklabels(L.INJ_NAMES)
    ax[j].set_xlabel(xl)
    ax[j].set_ylabel("bush dwell  (% of those steps spent in a bush)")
    ax[j].set_title(ttl, fontsize=9.3, loc="left", pad=8)
ax[2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.4,
             title="sensor-ladder arm", title_fontsize=8)
PL.finish(fig, f"{L.FIG_ROOT}/lad14_window_and_variable.png")
print(f"{'arm':22}{'A assigned/early':>18}{'B assigned/whole':>18}{'C carried/whole':>18}"
      "   (quarter 4 minus quarter 1, percentage points)")
for a in arms:
    print(f"{a:22}{early[a][3]-early[a][0]:>+18.2f}{whole[a][3]-whole[a][0]:>+18.2f}"
          f"{carried[a][3]-carried[a][0]:>+18.2f}")
