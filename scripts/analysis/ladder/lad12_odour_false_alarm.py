"""FIGURE 12 - The smell of a rabbit, and whether a wound makes it scarier.

QUESTION. Each animal's odour is drawn fresh for every episode, independently of everything else.
That makes odour strength a second randomised handle - a clean one, because the agent cannot
choose what a rabbit smells like. If bush dwell rises with the RABBIT odour draw, the agent is
responding to a cue that carries no danger at all. This figure asks whether it does, whether that
depends on what the arm can see, and whether waking up wounded makes it worse.

WHY THIS IS THE SHARPEST TEST IN THE REPORT. The predator panel is what a competent agent should
do: hide more when the world happens to contain strong-smelling predators. The rabbit panel is
pure false alarm - and because the draw is random, no confound can produce it.

HOW IT IS COMPUTED. Within each arm, episodes are split into quartiles of the mean odour intensity
drawn for its rabbits (and separately, its predators); intensity is the sum of the two olfactory
channels that separate the two classes, with the channels derived from the run's own config rather
than assumed. Episodes containing no predator (or no rabbit) are excluded from the corresponding
panel - they have no odour draw at all, and letting them fall into a bin would quietly make
"episodes with no predator in them" the comparison group.

WHY THE PREDATOR PANEL DIPS AT THE LOUDEST QUARTILE. That dip is real behaviour, not an artefact.
Intensity is the SUM of the two odour channels while what marks an animal as a predator is their
DIFFERENCE, so an episode whose predators smell very loudly has both channels near their ceiling
and the difference is squeezed toward zero. Measured on this run: mean predator-ness is 0.12 in the
loudest quartile against 0.18-0.22 in the other three. The loudest predators are the least
distinguishable ones, and the agent responds to them less. Bush dwell is pooled over each episode's first 25 steps. Solid lines are episodes
that began nearly unhurt (start wound 0-25), dashed lines those that began badly wounded (75-100);
the gap between a pair is how much the wound amplified the response to a harmless smell.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

D = L.load_all(); arms = L.ARM_ORDER
x = np.arange(4)
QN = ["weakest\nquarter", "2nd", "3rd", "strongest\nquarter"]
GRP = {a: L.resolves_identity(D[a]["sensory"]) for a in arms}

def curve(a, kind, ib):
    """One arm's bush-dwell curve across the four odour quartiles, at one starting-wound level."""
    o = D[a]["odour"]
    b = np.asarray(o[f"{kind}_bush"], float)[:, ib]
    t_ = np.asarray(o[f"{kind}_tot"], float)[:, ib]
    return np.where(t_ >= 1000, 100 * b / np.maximum(t_, 1), np.nan)

def group_curve(kind, ib, want):
    """The two groups' curves, pooling COUNTS across their arms rather than averaging rates.

    Averaging fourteen arms' percentages would give a 1,000,000-episode arm the same weight as one
    contributing a tenth as many steps to a bin. Pooling the counts weights each arm by the
    evidence it actually carries.
    """
    b = np.zeros(4); t_ = np.zeros(4)
    for a in arms:
        if GRP[a] != want:
            continue
        o = D[a]["odour"]
        b += np.asarray(o[f"{kind}_bush"], float)[:, ib]
        t_ += np.asarray(o[f"{kind}_tot"], float)[:, ib]
    return 100 * b / t_

fig, ax = plt.subplots(1, 2, figsize=(12.8, 5.8), sharey=True)
for j, (kind, ttl, xl) in enumerate([
        ("rab", "RABBIT odour - carries no danger, so any rise is a false alarm",
         "how strongly this episode's RABBITS happened to smell\n(quartile of the randomised odour draw)"),
        ("pred", "PREDATOR odour - a real cue (control panel)",
         "how strongly this episode's PREDATORS happened to smell\n(quartile of the randomised odour draw)")]):
    for a in arms:                                    # every arm, faint, for spread
        c = PL.GROUP_YES if GRP[a] else PL.GROUP_NO
        ax[j].plot(x, curve(a, kind, 0), lw=0.9, color=c, alpha=0.26, zorder=1)
        ax[j].plot(x, curve(a, kind, 3), lw=0.9, ls="--", color=c, alpha=0.26, zorder=1)
    for want, c in ((True, PL.GROUP_YES), (False, PL.GROUP_NO)):   # the two group curves, bold
        ax[j].plot(x, group_curve(kind, 0, want), lw=2.6, color=c, marker="o", ms=5, zorder=3)
        ax[j].plot(x, group_curve(kind, 3, want), lw=2.2, ls="--", color=c, marker="s", ms=4.4,
                   zorder=3)
    ax[j].set_xticks(x); ax[j].set_xticklabels(QN, fontsize=8.4)
    ax[j].set_xlabel(xl)
    ax[j].set_title(ttl, fontsize=9.8, loc="left", pad=8)
ax[0].set_ylabel("bush dwell over the episode's first 25 steps\n(% of those steps spent in a bush)")
h = [plt.Line2D([], [], color=PL.GROUP_YES, lw=2.6, label=L.GROUP_LABEL[True] + "  (9 arms)"),
     plt.Line2D([], [], color=PL.GROUP_NO, lw=2.6, label=L.GROUP_LABEL[False] + "  (5 arms)"),
     plt.Line2D([], [], color=PL.MUTED, lw=2.4, marker="o", ms=4.6,
                label="solid = episodes that began nearly unhurt  (start wound 0-25)"),
     plt.Line2D([], [], color=PL.MUTED, lw=2.0, ls="--", marker="s", ms=4.2,
                label="dashed = episodes that began badly wounded  (start wound 75-100)")]
ax[0].legend(handles=h, loc="lower center", bbox_to_anchor=(1.03, 1.10), ncol=2, fontsize=8.2)
fig.text(0.5, -0.03, "Bold lines pool the counts within each group; the faint lines behind them are "
         "the fourteen individual arms.", ha="center", fontsize=8, color=PL.MUTED)
PL.finish(fig, f"{L.FIG_ROOT}/lad12_odour_false_alarm.png")
print(f"{'arm':22}{'rabbit slope':>14}{'  (unhurt)':>12}{'rabbit slope':>14}{'  (wounded)':>13}"
      f"{'predator slope':>16}")
for a in arms:
    r0, r3 = curve(a, "rab", 0), curve(a, "rab", 3)
    p0 = curve(a, "pred", 0)
    print(f"{a:22}{r0[3]-r0[0]:>+13.2f}{'':>12}{r3[3]-r3[0]:>+13.2f}{'':>13}{p0[3]-p0[0]:>+15.2f}")
