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
| `build_arm_data.py` | the **one** sweep of the parquet stores; writes `results/analysis/ladder/<arm>.json` + `<arm>_episodes.npz` |
| `lad01…lad09_*.py` | one figure each — question, method, and plot, all in the one file |
| `run_all.py` | rebuild every figure |

## Running it

```bash
P=/home/vncuser/miniconda3/envs/grid_world_pain/bin/python
$P scripts/analysis/ladder/build_arm_data.py          # ~60-90 s per arm, 14 arms
$P scripts/analysis/ladder/run_all.py                 # seconds
```

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
