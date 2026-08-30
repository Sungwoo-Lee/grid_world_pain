"""FIGURE 6 - Which features of the world drive hiding, and does that change with the senses?

QUESTION. Figures 1-9 each isolate one thing. This one steps back: across every feature of the
world that is randomised before the agent acts, which ones move bush dwell, and by how much - in
each of the fourteen arms at once. If sight and smell matter, the map should change shape down the
ladder, not just change intensity.

WHAT IS IN IT. Only EXOGENOUS features: how many predators, rabbits, bushes, rocks, food patches
and ambush predators the world was given, how far the agent spawned from the nearest bush, and the
wound and hunger it was handed. Every one of these is drawn by the environment at reset, before the
agent has done anything, so none of them can be a consequence of its behaviour. Features that ARE
consequences - how much it ate, how long it survived, how injured it got - are deliberately absent;
they belong to a different question and would swamp this one.

HOW IT IS COMPUTED. A quasi-binomial regression per arm on the episode-level bush-dwell rate
(bush steps out of steps), all nine features entered together so each is adjusted for the others.
Standard errors are scaled by the Pearson overdispersion, which runs 13-27 here - without that
scaling every p-value in the table would be meaningless. The number plotted is the effect of moving
that feature by one standard deviation, converted to percentage points of bush dwell. Red = hides
more, blue = hides less. Cells are directly comparable across arms because all fourteen replayed
the same 300,000 worlds, so each feature has the same spread in every column.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csv, numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

MODEL = "M1 exogenous, all episodes"
PRETTY = {
    "start_injury": "wound it woke up with",
    "start_nutrition": "how well fed it woke up",
    "n_predators": "number of predators",
    "n_rabbits": "number of rabbits",
    "n_bushes": "number of bushes",
    "n_rocks": "number of rocks",
    "n_food": "number of food patches",
    "n_ambush_predators": "number of ambush predators",
    "spawn_dist_to_bush": "distance it spawned from a bush",
}
GLM_ROOT = "results/analysis/lad"

arms = [a for a in L.ARM_ORDER if os.path.exists(f"{GLM_ROOT}/{a}/multivariate.csv")]
if len(arms) < len(L.ARM_ORDER):
    print(f"note: {len(L.ARM_ORDER)-len(arms)} arm(s) have no GLM output yet: "
          f"{[a for a in L.ARM_ORDER if a not in arms]}")

terms = list(PRETTY)
M = np.full((len(terms), len(arms)), np.nan)
for j, a in enumerate(arms):
    rows = [r for r in csv.DictReader(open(f"{GLM_ROOT}/{a}/multivariate.csv"))
            if r["model"] == MODEL]
    d = {r["term"]: float(r["dpp_per_sd"]) for r in rows}
    for i, t in enumerate(terms):
        if t in d:
            M[i, j] = d[t]

v = np.nanmax(np.abs(M))
fig, ax = plt.subplots(figsize=(0.92 * len(arms) + 4.8, 0.52 * len(terms) + 4.0))
im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
ax.set_xticks(range(len(arms)))
ax.set_xticklabels(arms, fontsize=7.8, rotation=38, ha="right", rotation_mode="anchor")
ax.set_yticks(range(len(terms))); ax.set_yticklabels([PRETTY[t] for t in terms], fontsize=8.5)
ax.set_xlabel("sensor-ladder arm  (poorest senses on the left)")
ax.set_ylabel("feature of the world, all randomised before the agent acts")
ax.grid(False)
for i in range(len(terms)):
    for j in range(len(arms)):
        if np.isfinite(M[i, j]):
            ax.text(j, i, f"{M[i,j]:+.1f}", ha="center", va="center", fontsize=6.6,
                    color="white" if abs(M[i, j]) > v * 0.55 else PL.INK)
cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.028)
cb.set_label("effect on bush dwell of moving this feature by one standard deviation\n"
             "(percentage points; red = hides more, blue = hides less)", fontsize=8)
ax.set_title("Adjusted for all the other features in the same regression. The number printed in "
             "each cell is the exact value,\nso rows that saturate the colour scale can still be "
             "read and compared.", fontsize=9, color=PL.MUTED, loc="left", pad=10)
PL.finish(fig, f"{L.FIG_ROOT}/lad06_world_factor_map.png")
print(f"{'feature':32}" + "".join(f"{a[:9]:>10}" for a in arms))
for i, t in enumerate(terms):
    print(f"{PRETTY[t]:32}" + "".join(f"{x:>+10.2f}" for x in M[i]))
