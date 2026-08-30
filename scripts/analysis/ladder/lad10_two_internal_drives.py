"""FIGURE 10 - Two competing internal drives: a wound says hide, hunger says forage.

QUESTION. The agent carries two internal states that pull in opposite directions. A wound argues
for staying in cover; an empty stomach argues for leaving it, because a bush contains no food.
Both are handed to the agent at random at the start of every episode (`random_start_injury` and
`random_start_nutrition` are both true), so both can be tested causally and on the same footing.
Which of the two actually steers the behaviour, and does the answer depend on what the agent can
sense?

WHY BOTH PANELS USE THE FIRST 25 STEPS. Figure 8 shows that averaging over the whole episode
reverses the sign of the injury effect, because a lightly-wounded agent goes on to have a longer
episode and its average is diluted by late, low-hiding steps. That artefact would hit the hunger
panel too. Restricting both panels to the window in which the assigned dose is still largely intact
is what makes them comparable to each other AND to Figure 6.

Note the two panels do not share a y-range: the hunger effect is several times the size of the
wound effect, and forcing one scale would flatten the wound panel into a line. Compare the spans
printed in the table, not the visual steepness.

HOW IT IS COMPUTED. Episodes are split into four equal quarters of the assigned value (0-25, 25-50,
50-75, 75-100). Within each quarter, bush dwell is (bush steps) / (steps) pooled over the first 25
steps of every episode in it, with the t=0 row excluded from both.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

arms = L.ARM_ORDER; col = PL.arm_colors(); x = np.arange(4)
nut, inj = {}, {}
for a in arms:
    z = np.load(f"{L.OUT_ROOT}/{a}_episodes.npz")
    be, se = z["bush_early"], z["steps_early"]
    for dst, key in ((nut, "nut0"), (inj, "inj0")):
        b = np.digitize(z[key], L.INJ_EDGES)
        dst[a] = np.array([100 * be[b == k].sum() / se[b == k].sum() for k in range(4)])

fig, ax = plt.subplots(1, 2, figsize=(12.6, 5.4))
for j, (dat, ttl, xl) in enumerate([
        (nut, "HUNGER - assigned at random at the start of the episode",
         "how well fed the agent woke up  (nutrition, 0-100)\nleft = woke up starving"),
        (inj, "WOUND - also assigned at random at the start of the episode",
         "how wounded the agent woke up  (injury level, 0-100)\nleft = woke up unhurt")]):
    for a in arms:
        ax[j].plot(x, dat[a], marker="o", ms=4, lw=1.7, color=col[a],
                   label=f"{a} - {L.ARM_LABEL[a][0]}")
    ax[j].set_xticks(x); ax[j].set_xticklabels(L.INJ_NAMES)
    ax[j].set_xlabel(xl)
    ax[j].set_ylabel("bush dwell over the episode's first 25 steps\n(% of those steps spent in a bush)")
    ax[j].set_title(ttl, fontsize=9.5, loc="left", pad=8)
ax[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.4,
             title="sensor-ladder arm", title_fontsize=8)
PL.finish(fig, f"{L.FIG_ROOT}/lad10_two_internal_drives.png")
print(f"{'arm':22}{'hunger span':>13}{'wound span':>13}   (percentage points, quarter 4 - quarter 1)")
for a in arms:
    print(f"{a:22}{nut[a][3]-nut[a][0]:>+13.2f}{inj[a][3]-inj[a][0]:>+13.2f}")
