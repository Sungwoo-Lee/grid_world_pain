"""FIGURE 15 - What hiding costs, and why more of it is not better.

QUESTION. Hiding is not free. A bush has no food in it, so every step spent in cover is a step not
spent eating - and starvation kills roughly a third of all episodes in every arm. This figure puts
the trade-off on one pair of axes: how much each arm hides, what that does to how much it eats, and
where it lands on survival.

WHY IT MATTERS FOR THE LADDER. If hiding were simply good, the best arms would be the ones that hide
most. They are not. The relationship between bush dwell and survival across the fourteen arms is the
single clearest statement of what the senses are actually for: not to make the agent hide more, but
to let it hide at the RIGHT moments and forage the rest of the time.

HOW IT IS COMPUTED. One point per arm, from that arm's 300,000 episodes. Bush dwell is bush steps
over steps. Eating rate is `ate_food` events per step. Survival is mean episode length. All three
pool over the same episodes, and every arm saw the same 300,000 worlds.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, matplotlib.pyplot as plt
import _ladder as L, _plot as PL

arms = L.ARM_ORDER; col = PL.arm_colors()
dwell, surv, eat = [], [], []
for a in arms:
    z = np.load(f"{L.OUT_ROOT}/{a}_episodes.npz")
    dwell.append(100 * z["bush_steps"].sum() / z["n_steps"].sum())
    eat.append(100 * z["n_ate"].sum() / z["n_steps"].sum())
    surv.append(float(z["length"].mean()))
dwell, surv, eat = map(np.array, (dwell, surv, eat))

fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.5))
for j, (yv, yl, ttl) in enumerate([
        (surv, "mean survival  (steps per episode)",
         "More hiding goes with SHORTER life across the ladder"),
        (eat, "eating rate  (food eaten per 100 steps)",
         "and the reason is food: every step in cover is a step not eating")]):
    ax[j].scatter(dwell, yv, s=70, c=[col[a] for a in arms], zorder=3,
                  edgecolor="white", linewidth=0.8)
    b, a0 = np.polyfit(dwell, yv, 1)
    xs = np.linspace(dwell.min(), dwell.max(), 20)
    ax[j].plot(xs, b * xs + a0, ls="--", lw=1, color=PL.MUTED, zorder=1)
    pad = (dwell.max() - dwell.min()) * 0.16
    ax[j].set_xlim(dwell.min() - pad, dwell.max() + pad)
    yp = (yv.max() - yv.min()) * 0.10
    ax[j].set_ylim(yv.min() - yp, yv.max() + yp)
    PL.label_points(ax[j], dwell, yv, arms, col, min_gap=0.064)
    r = np.corrcoef(dwell, yv)[0, 1]
    ax[j].set_xlabel("bush dwell  (% of an episode's steps spent in a bush)")
    ax[j].set_ylabel(yl)
    ax[j].set_title(f"{ttl}\n(across the 14 arms: r = {r:+.2f}, dashed line is the fit)",
                    fontsize=9.3, loc="left", pad=8)
PL.finish(fig, f"{L.FIG_ROOT}/lad15_price_of_hiding.png")
print(f"{'arm':22}{'bush dwell':>12}{'survival':>11}{'eat/100 steps':>15}")
for a, d, s, e in zip(arms, dwell, surv, eat):
    print(f"{a:22}{d:>11.1f}%{s:>11.1f}{e:>15.2f}")
