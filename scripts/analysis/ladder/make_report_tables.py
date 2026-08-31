"""Emit every numeric table in the sensor-ladder report as markdown.

The report's prose is written by hand; its numbers are not. Transcribing fourteen rows of results
into a document by hand is exactly how a report ends up disagreeing with the figures beside it, so
every table in `sensor_ladder.md` is generated here and pasted in whole. Re-run after any change to
the underlying data and diff the output against the document.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import _ladder as L

D = L.load_all(); arms = L.ARM_ORDER
runs = L.arm_runs()
out = []
w = out.append

w("### TABLE 1 - the fourteen arms\n")
w("| arm | what it can sense | smell grid | sight range | blur scale | blur anisotropy | "
  "appearance channels | value mode | sight blocked by |")
w("|---|---|---|---|---|---|---|---|---|")
for a in arms:
    s = D[a]["sensory"]; cfg = L.arm_config(runs[a]); bs = L.blocks_sight(cfg)
    w(f"| `{a}` | {L.ARM_LABEL[a][0]} | {s['olfactory_grid_range']} | "
      f"{s['visual_sensor_range']} | {s['visual_blur_radial_scale'] or 'off'} | "
      f"{s['visual_blur_anisotropy'] or '-'} | {s['visual_vector_size']} | "
      f"{s['visual_value_mode']} | {', '.join(bs) if bs else 'nothing'} |")

w("\n### TABLE 2 - outcome per arm\n")
w("| arm | mean survival (steps) | bush dwell (%) | killed (%) | starved (%) | "
  "reached the limit (%) | food per 100 steps |")
w("|---|---|---|---|---|---|---|")
for a in arms:
    z = np.load(f"{L.OUT_ROOT}/{a}_episodes.npz"); t = D[a]["term_pct"]
    w(f"| `{a}` | {D[a]['mean_survival']:.1f} | {D[a]['bush_dwell_pct']:.1f} | "
      f"{t.get('killed by predator',0):.1f} | {t.get('starved',0):.1f} | "
      f"{t.get('survived to time limit',0):.1f} | "
      f"{100*z['n_ate'].sum()/z['n_steps'].sum():.2f} |")

w("\n### TABLE 3 - single-variable sensor changes\n")
pairs = sorted(L.ARM_REFERENCE.items(), key=lambda p: arms.index(p[0]))
w("| change | arm | compared with | survival (steps) | bush dwell (pp) |")
w("|---|---|---|---|---|")
for a, r in pairs:
    w(f"| {L.ARM_LABEL[a][1]} | `{a}` | `{r}` | "
      f"{D[a]['mean_survival']-D[r]['mean_survival']:+.1f} | "
      f"{D[a]['bush_dwell_pct']-D[r]['bush_dwell_pct']:+.1f} |")

w("\n### TABLE 4 - the rabbit false alarm, four independent measures\n")
w("| arm | sight resolves identity | proximity to a rabbit (pp) | rabbit odour slope (pp) | "
  "rabbit odour, adjusted (pp) | one SD more rabbits (pp) |")
w("|---|---|---|---|---|---|")
import csv
def glm(a, model, term):
    p = f"results/analysis/lad/{a}/multivariate.csv"
    for r in csv.DictReader(open(p)):
        if r["model"] == model and r["term"] == term:
            return float(r["dpp_per_sd"])
    return float("nan")
for a in arms:
    s = D[a]["sensory"]
    ident = s["visual_sensor_range"] >= 2 and s["visual_vector_size"] > 1
    g = D[a]["grids"]; o = D[a]["odour"]
    b = np.asarray(o["rab_bush"], float).sum(1); t = np.asarray(o["rab_tot"], float).sum(1)
    c = 100 * b / t
    w(f"| `{a}` | {'yes' if ident else 'no'} | "
      f"{L.proximity_effect(g['rd_bush'], g['rd_tot']):+.1f} | {c[3]-c[0]:+.2f} | "
      f"{glm(a,'M3 + rabbit smell, 1 predator + 1 rabbit','rab_olf_intensity'):+.2f} | "
      f"{glm(a,'M1 exogenous, all episodes','n_rabbits'):+.2f} |")

w("\n### TABLE 5 - does the split actually separate the two groups, or do they overlap?\n")
w("| measure | largest value among the nine that resolve identity | smallest among the five that "
  "do not | separation | closest of the nine |")
w("|---|---|---|---|---|")
def _slope(a):
    o = D[a]["odour"]; b = np.asarray(o["rab_bush"], float).sum(1); t = np.asarray(o["rab_tot"], float).sum(1)
    c = 100 * b / t; return c[3] - c[0]
MEAS = {"how much it hides when a rabbit is near":
            lambda a: L.proximity_effect(D[a]["grids"]["rd_bush"], D[a]["grids"]["rd_tot"]),
        "response to a strong rabbit smell": _slope,
        "rabbit smell, adjusted for the world":
            lambda a: glm(a, "M3 + rabbit smell, 1 predator + 1 rabbit", "rab_olf_intensity"),
        "one SD more rabbits in the world":
            lambda a: glm(a, "M1 exogenous, all episodes", "n_rabbits")}
def _ident(a):
    s2 = D[a]["sensory"]; return s2["visual_sensor_range"] >= 2 and s2["visual_vector_size"] > 1
for nm, f in MEAS.items():
    yes = {a: f(a) for a in arms if _ident(a)}; no = {a: f(a) for a in arms if not _ident(a)}
    my, mn = max(yes.values()), min(no.values())
    who = [a for a, v in yes.items() if v == my][0]
    w(f"| {nm} | {my:+.2f} | {mn:+.2f} | {mn-my:+.2f} | `{who}` |")

w("\n### TABLE 6 - what a randomised starting wound does\n")
w("| arm | bush dwell, first 25 steps (pp per full wound range) | "
  "shift in rabbit proximity (pp) | shift in predator proximity (pp) | "
  "wound amplifies rabbit odour (pp) | wound amplifies predator odour (pp) |")
w("|---|---|---|---|---|---|")
def sl(a, kind, ib):
    o = D[a]["odour"]
    b = np.asarray(o[f"{kind}_bush"], float)[:, ib]; t = np.asarray(o[f"{kind}_tot"], float)[:, ib]
    c = np.where(t >= 1000, 100 * b / np.maximum(t, 1), np.nan)
    return c[3] - c[0]
for a in arms:
    g = D[a]["grids"]
    e = L.rate(g["dw_early"], g["dwt_early"])
    dr = L.proximity_effect(g["rd_bush"], g["rd_tot"], (3,)) - L.proximity_effect(g["rd_bush"], g["rd_tot"], (0,))
    dp = L.proximity_effect(g["pd_bush"], g["pd_tot"], (3,)) - L.proximity_effect(g["pd_bush"], g["pd_tot"], (0,))
    w(f"| `{a}` | {e[3]-e[0]:+.2f} | {dr:+.2f} | {dp:+.2f} | "
      f"{sl(a,'rab',3)-sl(a,'rab',0):+.2f} | {sl(a,'pred',3)-sl(a,'pred',0):+.2f} |")

w("\n### TABLE 9 - the two internal drives, first 25 steps\n")
w("| arm | hunger: change in bush dwell (pp, signed) | wound: change in bush dwell (pp, signed) | ratio |")
w("|---|---|---|---|")
for a in arms:
    z = np.load(f"{L.OUT_ROOT}/{a}_episodes.npz")
    be, se = z["bush_early"], z["steps_early"]
    sp = {}
    for key in ("nut0", "inj0"):
        b = np.digitize(z[key], L.INJ_EDGES)
        c = np.array([100 * be[b == k].sum() / se[b == k].sum() for k in range(4)])
        sp[key] = c[3] - c[0]
    r = sp["nut0"] / sp["inj0"] if sp["inj0"] > 0 else float("nan")
    w(f"| `{a}` | {sp['nut0']:+.2f} | {sp['inj0']:+.2f} | "
      f"{('%.1fx' % r) if np.isfinite(r) else 'wound effect is negative'} |")

w("\n### TABLE 7 - the three readings of the injury question (Figure 14)\n")
w("| arm | A: assigned wound, first 25 steps (pp) | B: assigned wound, whole episode (pp) | "
  "C: carried wound, whole episode (pp) |")
w("|---|---|---|---|")
for a in arms:
    z = np.load(f"{L.OUT_ROOT}/{a}_episodes.npz")
    ib = np.digitize(z["inj0"], L.INJ_EDGES)
    e = np.array([100 * z["bush_early"][ib == k].sum() / z["steps_early"][ib == k].sum() for k in range(4)])
    h = np.array([100 * z["bush_steps"][ib == k].sum() / z["n_steps"][ib == k].sum() for k in range(4)])
    c = L.rate(D[a]["grids"]["dw_carried"], D[a]["grids"]["dwt_carried"])
    w(f"| `{a}` | {e[3]-e[0]:+.2f} | {h[3]-h[0]:+.2f} | {c[3]-c[0]:+.2f} |")

w("\n### TABLE 8 - how long an episode lasts, by the wound the agent woke up with\n")
w("| arm | started 0-25 | started 25-50 | started 50-75 | started 75-100 | difference |")
w("|---|---|---|---|---|---|")
for a in arms:
    z = np.load(f"{L.OUT_ROOT}/{a}_episodes.npz")
    ib = np.digitize(z["inj0"], L.INJ_EDGES)
    m = [z["length"][ib == k].mean() for k in range(4)]
    w(f"| `{a}` | {m[0]:.1f} | {m[1]:.1f} | {m[2]:.1f} | {m[3]:.1f} | {m[3]-m[0]:+.1f} |")

w("\n### TABLE 10 - the wound's effect through time (Figure 9)\n")
try:
    TC = L.load_time_course()
except SystemExit:
    TC = None
if TC:
    w("| arm | peak extra hiding (pp) | at step | extra hiding by step 60 (pp) | "
      "worst nutrition gap |")
    w("|---|---|---|---|---|")
    for a in arms:
        if a not in TC: continue
        b = 100 * (np.asarray(TC[a]["bush"], float)[3, :100] - np.asarray(TC[a]["bush"], float)[0, :100])
        n = np.asarray(TC[a]["nutrition"], float)[3, :100] - np.asarray(TC[a]["nutrition"], float)[0, :100]
        w(f"| `{a}` | {b.max():+.2f} | {int(np.argmax(b))} | {b[60]:+.2f} | {n.min():+.1f} |")
else:
    w("_(run build_time_course.py to generate this table)_")

print("\n".join(out))
