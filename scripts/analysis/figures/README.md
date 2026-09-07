# Per-figure analysis scripts

One figure, one script. Each is self-contained and reviewable on its own: it opens with the
question it answers, states what it conditions on and what it knows to be biased, reads a
trajectory store, writes a single JSON to `results/analysis/figures/`, and prints a table.

This structure exists because the previous arrangement — a handful of scripts each computing
several unrelated things — hid two real defects. A figure's code was silently deleted during an
unrelated edit and the figure rendered as an empty table for several revisions; and two panels went
blank when a display label was renamed but the data key was not. Both are structurally impossible
here: a missing script is a missing output file, and `build_figure_data.py` fails loudly rather
than emitting a partial dataset.

## Running

```bash
P=/home/vncuser/miniconda3/envs/grid_world_pain/bin/python
$P scripts/analysis/figures/fig07_proximity.py                 # one figure
$P scripts/analysis/figures/run_all.py                         # every figure, in order
$P scripts/analysis/figures/build_figure_data.py               # merge JSONs for the artifact
```

Run from the repository root. Every script takes `--run`, `--checkpoint`, `--store-root` and
`--quiet`, and defaults to the a01 run analysed in the write-up. The slot layout is derived from
each run's own saved config, so the scripts work on any collected run.

## The figures

| script | figure | question it answers |
|---|---|---|
| `verify_environment_parity.py` | 1 | does today's environment reproduce the one this agent trained in? |
| `fig02_factor_ranking.py` | 2 | which randomised world factors move bush hiding, ranked on a common scale |
| `fig03_dose_response.py` | 3 | the shape of each factor's effect, on dwell and on survival together |
| `fig04_olfactory_ladders.py` | 4 | what happens as a predator's, and a rabbit's, odour becomes more predator-like |
| `fig05_consequence_chain.py` | 5 | does the odour response cost food, and life? |
| `fig06_response_targeting.py` | 6 | is the response aimed at the misleading animal, or diffuse? |
| `fig07_proximity.py` | 7 | how much does simple proximity explain, lagged to defeat reverse causation |
| `fig08_peri_damage.py` | 8 | behaviour time-locked to a damage event |
| `fig09_nociception_by_origin.py` | 9 | same signal level, different origin — signal or evidence? |
| `fig10_dwell_by_injury_time.py` | 10 | is the wound-dwell link just a proxy for elapsed time? |
| `fig11_near_lethal.py` | 11 | what happens as the wound approaches fatal |
| `fig12_dwell_by_nutrition_time.py` | 12 | the same time check for hunger |
| `fig13_ten_agents.py` | 13 | how much do ten independently trained agents differ? |
| `fig14_variance_decomposition.py` | 14 | world vs agent vs their interaction, decomposed exactly |
| `fig15_nociception_all_agents.py` | 15 | is the nociception-dwell coupling a property of the task? |
| `fig16_modulator_ratio.py` | 16 | does the neuromodulator change reliance on the true cue? |
| `fig17_modulator_variance.py` | 17 | how much does the modulator explain, across four matched pairs |
| `fig18_internal_state_dependence.py` | 18 | does the response to a nearby predator depend on internal state? |
| `fig19_internal_state_table.py` | 19 | the same, as a single number per pair |

## Conventions every script follows

- **The outcome is bush hiding**: bush steps over episode steps. The `t=0` row is the initial state,
  not a step, and is excluded from both.
- **Anything the agent conditioned on is read from the previous row**, because the action producing
  row `t` was chosen on the row `t-1` observation.
- **Nociception is reconstructed, not read.** The store does not save the observation vector, so
  scripts that need the received signal rebuild it from `injury_level` using the run's own kernel
  settings. `_common.reconstruct_nociception` is the single implementation; do not re-derive it.
- **Causal or associational is stated in the docstring**, never left to the reader. Randomised at
  reset means causal; produced during the episode means associational.
- **Known biases are named in the docstring**, including ones that make the result weaker.

## Supplementary

`../supplementary/` holds the exploratory passes that preceded these — the store audit that found
three untested factors, the environment parity harness, the identification probes. They are kept
for provenance and are not run-agnostic. Anything of lasting value has been promoted here.
