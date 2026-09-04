# Sensor-ladder analysis scripts

Everything behind the sensor-ladder report
([`sensor_ladder.md`](../../../docs/experiments/active/sensor_ladder/sensor_ladder.md)).

## What the ladder is

Fourteen agents, trained on **one** environment with **one** seed, differing only in what they can
sense. Each arm then replayed the **same 300,000 evaluation episodes** (`--seed-base 1000000`), so
two arms met identical predators, identical food, and identical randomised starting wounds. Any
difference between them is the sensor change and nothing else.

## Layout

| file | what it owns |
|---|---|
| `_ladder.py` | arm list, arm labels, which arm is a single-variable step from which, bin edges, store discovery |
| `_plot.py` | the house figure style, and the four layout rules every figure obeys |
| `../studies/sensor_ladder/collect_arm_data.py` | the main sweep, now on `core/scan`; writes `results/analysis/ladder/<arm>.json` + `<arm>_episodes.npz` |
| `../studies/sensor_ladder/collect_time_course.py` | a second, step-by-step sweep for Figure 9; writes `time_course_<arm>.json` |
| `lad01…lad15_*.py` | one figure each — question, method, and plot, all in the one file. The number in the filename IS the figure number in the report |
| `make_report_tables.py` | every numeric table in the report, as markdown |
| `build_artifact.py` | assembles the shareable HTML page from the template + real figures + generated tables |
| `run_all.py` | rebuild every figure |

## Running it

```bash
P=/home/vncuser/miniconda3/envs/grid_world_pain/bin/python
$P scripts/analysis/studies/sensor_ladder/collect_arm_data.py     # ~60-95 s per arm, 14 arms
$P scripts/analysis/studies/sensor_ladder/collect_time_course.py  # ~2.5 min per arm
$P scripts/analysis/ladder/run_all.py              # seconds
$P scripts/analysis/ladder/make_report_tables.py   # the report's tables
$P scripts/analysis/ladder/build_artifact.py       # the HTML page
```

Figures 6 and 7 additionally require `scripts/analysis/hiding_drivers.py` to have been run per arm,
since they plot its regression output.

Figures land in `docs/experiments/active/sensor_ladder/figures/`.

## The two conventions every script obeys

1. **The `t=0` row is not a step.** It is the world as handed to the agent. It is in no numerator
   and no denominator. (Getting this wrong once produced 5,568 episodes with more successes than
   trials.)
2. **Predictors come from the previous row.** The action that produced row `t` was chosen while the
   agent was looking at row `t-1`, so anything the agent conditioned on is read off `t-1`.

## What is causal here and what is not

`body.random_start_injury` and `body.random_start_nutrition` are **true** in this environment, and
each animal's odour is drawn fresh per episode. Those three are assigned before the agent acts, so
splitting on them supports a causal claim. Distance to a predator is **not** randomised — the agent
chose where to walk — so the distance curves are descriptive. `lad08` exists specifically to show
how far apart the two readings land.

## Three traps this analysis fell into

Recorded because each produced a plausible-looking wrong number, and each is easy to repeat.

1. **A third of episodes contain no predator, and a third no rabbit.** `numpy.digitize` files every
   `NaN` into the TOP bin and `numpy.clip` files every `inf` into the FARTHEST distance bin, so
   leaving those episodes in quietly made "there is no predator" the comparison group. Every
   accumulator that conditions on an animal existing now carries a `has_p` / `has_r` mask.
2. **A reference pairing drifted to two variables.** `R1_range1` was compared against `V4_blur05`,
   which differs from it in visual range AND blur, inflating the range effect fourfold.
   `check_single_variable_pairs()` now asserts the claim the figure makes, and `settings()`
   collapses conditional keys so an inert value cannot read as a third difference.
3. **The window was doing more work than the variable.** The injury effect reverses sign between a
   25-step and a whole-episode window. It is not fragile — the wound heals by step 28, so the cause
   genuinely ends — but any number quoted for a transient internal state is a statement about a
   window. `lad09` exists to make that explicit rather than leave it to a reader to discover.
