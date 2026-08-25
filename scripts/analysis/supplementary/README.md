# Supplementary passes for the hiding-drivers analysis

`scripts/analysis/hiding_drivers.py` produces the factor ranking, the multivariate models and
the same-step cross-tabs. The scripts here produce the remaining evidence in
[`a01_hiding_drivers.md`](../../../docs/experiments/active/trajectory_factors/a01_hiding_drivers.md).
Each is a single sweep of the step table for one specific question.

| Script | Question it answers | Where it appears |
|---|---|---|
| `prox_full.py` | Hiding given a predator / rabbit / neither nearby, same-step **and lagged** | Finding 1 |
| `curves.py` | Dose-response of hiding and survival against every randomised factor | Ranking, findings 2-4 |
| `falsealarm.py` | Is the rabbit-scent response *aimed* at the rabbit, or diffuse? | Finding 2 |
| `crux.py` | Injury gradient conditioned on predator proximity and recent damage | Finding 3 |
| `injwin.py` | Start-injury effect inside the window where the wound is still live, unconditional | Finding 3 |
| `timectrl.py` | Hiding by elapsed-time bin x injury / nutrition, to test temporal confounding | Findings 3-4 |
| `mech.py` | Does hiding block eating? Injury gradient time-controlled | Finding 4 |
| `window.py` | Truncation robustness: every effect recomputed on a fixed early window | Method |
| `spawn.py` | Spawn distance to food and to map centre | Ranking |
| `noci.py` | Reconstructs the **perceived** pain signal; per-step start-injury decomposition; hiding by perceived pain x predator proximity | Finding 3 |
| `noci2.py` | Perceived-vs-actual dissociation, peri-damage-event dynamics, rising-vs-falling pain | Interoceptive pain |

**Not run-agnostic.** Unlike `hiding_drivers.py`, these hardcode the a01 store glob, the slot
indices and the seed base (1,000,000-1,999,999). They are archived for reproducibility of this
specific analysis, not as general tools. Generalising one means deriving its slot layout from
the saved config the way `hiding_drivers.py:slot_layout` does.

**On the pain signal.** `noci.py` and `noci2.py` reconstruct the agent's interoceptive
nociception from the per-step `injury_level` column, because the store does not record the
observation vector itself. The reconstruction mirrors `src/environment/core.py:115` (buffer of
injury levels, rolled each step, zeroed at reset) and `sensor.py:sense_interoceptive_nociception`
(alpha kernel, tau and length read from the run's config). If either changes, these scripts
must change with them.

**Known wart, deliberately preserved.** `timectrl.py` bins by the *contemporaneous* nutrition
at each step, not by the randomised starting value. An earlier draft of the write-up misread
its first-time-bin row as evidence that randomised hunger has no immediate effect; it does not
show that, because the binning variable is not the randomised draw. `injwin.py` computes the
correct estimator. See [Review response](../../../docs/experiments/active/trajectory_factors/a01_hiding_drivers.md#review-response).

Run from the repo root with the project interpreter:
`/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/analysis/supplementary/<script>.py`
