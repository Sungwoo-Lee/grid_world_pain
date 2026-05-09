---
title: "SameProp Round 2.5 — full-budget re-launch with per-tag distance disambiguation"
topic: hypervigilance
status: planned
created: 2026-05-09
last_updated: 2026-05-09
phase: 1
wandb_tag: "hypervigilance-round25"
develop_link: "../../../develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md"
supersedes: []
---

# SameProp Round 2.5 — full-budget re-launch with per-tag distance disambiguation

> **Status**: PLANNED — configs finalised, per-tag metrics shipped (commit `0a73613` + bug fix `6d3d382`), waiting on user authorization to launch.
> **Date**: 2026-05-09
> **Author**: experiment-designer (§§0–8); experiment-analyzer to fill §§9–11 post-hoc.
> **Related**:
> - Predecessor (truncated to ~5 % of budget): [`sameprop_round2_design.md`](sameprop_round2_design.md). Round 2.5 succeeds Round 2; the truncated record stays as-is.
> - Round 1 baseline (cross-round comparator): [`round1_relog_baseline_analysis.md`](round1_relog_baseline_analysis.md).
> - Per-tag distance logging plan that resolves the Round-2 ambiguity: [`docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md`](../../../develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md) (status: implemented, verification: pass).
> - Channels memo (priors on which signal carries class identity): [`docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`](../../../develop/active/hypervigilance/sameprop_discriminating_channels.md).
> - Study-level recap of the week's hypervigilance arc: [`docs/experiments/summaries/20260509_1552_sameprop_rabbit_avoidance_study.md`](../../summaries/20260509_1552_sameprop_rabbit_avoidance_study.md).

---

## 0. What this round is testing — plain-English entry point

The hypervigilance study is asking a simple question: when a predator and a neutral rabbit emit *identical* smells in this grid world, can the agent still tell them apart and stay further from the predator? In Round 1 the answer looked like yes — the agent ended up about 0.6 grid-cells further from predators than from rabbits — but a confound was flagged after the fact: rabbits and food shared the same two corners, so the agent might just have been chasing food and incidentally coming closer to rabbits. Round 2 was designed to break that coincidence with two cells. Cell C moves food into the predator-only corners, so any remaining rabbit-vs-predator gap can no longer come from food. Cell A1 strips the predator's hunting behaviour and locks it to one corner that already contains a rabbit, so rabbit and predator look kinematically identical and live in the same place — making class recognition the only way the agent can distinguish them.

Round 2 was started but stopped by the user at about 5 % of its 10-million-episode budget after only ~3.5 hours, which left both cells in early-learning territory and produced a deeper problem: the *aggregated* per-class distance metrics (averaged over rabbit instances) cannot tell apart "the agent recognises predators specifically" from "the agent simply learned to never visit the dangerous corner." Both behaviours produce the same number for `MeanDistRabbit` and `MeanDistPredator`. **Round 2.5 re-launches the same two cells at the full 10-million-episode horizon, but now with per-instance "tagged" distance metrics** (one number per labelled rabbit and per labelled predator, e.g. `Episode/MeanDistRabbit_TL` for the top-left rabbit). With those metrics in place, the smoking-gun test for Cell A1 becomes a direct comparison between the agent's distance to the top-left rabbit and its distance to the top-left predator, who share the same corner under matched smells. If those two numbers diverge, the agent recognises the predator class; if they agree and survival looks healthy, the agent is just camping the safe corner. Round 2.5 also fans both per-tag values out for Cell C as a cross-check on whether the food-decoupling effect generalises across both rabbit corners or is being driven by one. One seed per cell, on node 106, both GPUs.

## 1. Research Question

**Q (cell A1, primary):** Under matched olfactory `properties [0,1,0,0,0]` for predator and rabbit, with the predator's HUNT mode disabled and patrol area shrunk to TL `[[1,1],[5,5]]` so the predator co-occupies a quadrant with the TL rabbit, does the agent at full convergence (10 M episodes, last-10 % window 9.0–10.0 M) maintain a *class-conditional* distance gap between the same-corner predator and the same-corner rabbit (`MeanDistPredator_TL > MeanDistRabbit_TL`), or does it converge on a *location-conditional* avoidance strategy (predator-corner camping with no class discrimination, evidenced by `MeanDistPredator_TL ≈ MeanDistRabbit_TL` at high survival)?

**Q (cell C, confound resolution):** With food spawn areas decoupled from rabbit spawn areas (food in TR + BL, rabbits in TL + BR; predator full-grid), does the Round-1 effect `MeanDistRabbit < MeanDistPredator` persist at full convergence, and does it appear *symmetrically* across the two rabbit instances (per-tag cross-check `MeanDistRabbit_TL` and `MeanDistRabbit_BR` both below `MeanDistPredator_full`), or only on one tag (artefact of single-quadrant aversion)?

Verdicts on both cells are pre-registered in §4 against confirmation/refutation thresholds keyed to the per-tag metrics shipped in commit `0a73613` (verified `6d3d382`). A clean falsification of either cell does not depend on the other.

> **H₁(A1) — agent recognises the predator class.** `MeanDistPredator_TL − MeanDistRabbit_TL ≥ 1.0` cells in the 9.0–10.0 M window AND `PredatorHits < RabbitHits − 1.5/ep` AND survival `Episode/Steps` not driven entirely by `Term_MaxSteps` (i.e., the agent is not pure-camping). The agent stays meaningfully farther from the same-corner predator than from the same-corner rabbit.
>
> **H₀(A1) — agent camps the safe corner.** `|MeanDistPredator_TL − MeanDistRabbit_TL| ≤ 0.3` cells AND survival ≥ 470/500 dominated by `Term_MaxSteps` (the §5 "spatial segregation" failure mode confirmed). Both same-corner entities are treated identically because the agent does not visit TL — class discrimination is moot.
>
> **H₁(C) — class discrimination survives food-decoupling.** Aggregate Δ ≡ `MeanDistPredator − MeanDistRabbit` ≥ 0.3 cells AND ΔH ≡ `RabbitHits − PredatorHits` ≥ 1.5/ep, AND per-tag cross-check shows both `MeanDistRabbit_TL` and `MeanDistRabbit_BR` below `MeanDistPredator_full` (no single-quadrant artefact).
>
> **H₀(C) — Round-1 was the food confound.** `|Δ| ≤ 0.1` AND `|ΔH| ≤ 1.0/ep`, with both per-tag rabbit distances close to `MeanDistPredator_full`.

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Cell C — decoupleFood | Cell A1 — passivePredator |
|---|---|---|
| Food spawn quadrants | TR + BL (decoupled from rabbits) | TL + BR (Round-1 default) |
| Predator `detection_range` | 5 (Round-1 default) | **0** |
| Predator `hunt_stamina_threshold` | 0.7 (Round-1 default) | **1.1** (HUNT unreachable) |
| Predator `patrol_area` | `[[1,1],[10,10]]` (full grid) | **`[[1,1],[5,5]]`** (matches rabbit TL) |
| Predator `spawn_area` | `[[1,1],[10,10]]` | **`[[1,1],[5,5]]`** |
| Predator `tag` | `"full"` | `"TL"` |

All other fields fixed at Round-1 baseline (`01-interoNocicept_sameProp.yaml`). Rabbit `tag` values are `"TL"` and `"BR"` in **both** cells (geometry-matched to the rabbit `spawn_area` in each row).

### 2.2 Controlled Variables

- Olfactory `properties` for predator + rabbit: `[0.0, 1.0, 0.0, 0.0, 0.0]` (the sameProp condition itself).
- `properties_std`: zero for all entities (no per-step olfactory noise).
- `perceptual_noise.enabled: false`.
- 4 hiding_predators in 4 quadrants, properties `[0,0,0,0,0]`.
- Rabbits: count 1 per `neutral_animals` entry, total 2 (TL + BR), move_interval 1, properties `[0,1,0,0,0]`.
- Bushes: 5 in TR, 5 in BL.
- Rocks: 3 in each of 4 quadrants, damage `[1, 5]`.
- Sensors: olfaction radius 20, decay 2.0, vector_size 5; visual range 0; nociception enabled; intero-noc enabled (tau 3.0, kernel 12); proprioception on; injury_observable false; nutrition_observable false.
- Body: full Round-1 homeostatic reward, max_steps 500, metabolic_cost 1.0.
- Agent: `configs/models/recurrent_ppo.yaml` (RPPO, identical to Round 1 + Round 2).
- Step budget: **10,000,000 episodes** (full Round-1 plan).
- Parallel envs: 128.
- Per-tag distance metrics live (`Episode/MeanDistRabbit_<tag>`, `Episode/MeanDistPredator_<tag>`) at all 5 WandB sites — verified at commit `6d3d382`.
- Existing aggregated metrics (`Episode/MeanDistRabbit`, `Episode/MeanDistPredator`, `Episode/RabbitHits`, `Episode/PredatorHits`, `Episode/Steps`, `Episode/Term_*`, `Episode/MeanDistFood`, `Episode/MeanDistHidingPredator`, `Episode/HidingPredatorHits`, `Episode/FoodEaten`, `Episode/Reward`) preserved unchanged for cross-round comparability.

### 2.3 Confounds & Limitations

| Confound | Affected | Severity | Mitigation |
|---|---|---|---|
| n = 1 seed per cell | both cells | **Medium-High** | User-chosen 2-GPU configuration. Per the project's `feedback_launch_manifest.md` rule, single-seed verdicts require the per-cell signal to exceed the §4 thresholds **by a wide margin (≥ 2x)** to be elevated above provisional. Borderline outcomes route to Round 2.6 with seed 44 (Cell C) and seed 45 (Cell A1) on the next GPU rotation — see §5. |
| Cross-round seed pairing — Cell C uses seed 42 (matches R1-seed42 `rg5nl1ov`), Cell A1 uses seed 43 (matches R1-seed43 `6ks4bjbq`) | both cells | Low | Same RNG-pair as Round 1, lets §4.5 pairwise comparisons run under matched seed initialisation. Same scheme as the truncated Round 2; consistency across the two rounds is the priority. |
| Cell A1 keeps predator damage on contact | A1 | Low | Intentional — extero-noc 0.9 + visual ch.5 at contact remain available so post-contact teaching can still inform the GRU. The question is whether class-conditional avoidance survives without HUNT, not whether the predator is harmless. |
| Cell A1 predator confined to TL (one of two rabbit quadrants) | A1 | **By design** — this is the disambiguation primitive | Co-occupancy with TL rabbit is the experiment. The new per-tag metrics make this disambiguation tractable: `MeanDistPredator_TL` vs `MeanDistRabbit_TL` directly tests class vs location. |
| `random_start_pos: true` | both | Low | Same condition as Round 1 + Round 2; cross-round comparability preserved. |
| Cell C still shares quadrants with hiding_predators in all 4 corners | C | Low | Hiding predators have `properties [0,0,0,0,0]` — no olfactory leak. Identical to Round 1 + Round 2. |
| Round 2.5 is the *re-launch* of the truncated Round 2; under-trained Round-2 numbers in `0u266oj5` / `27svrmhv` should NOT be merged with Round 2.5 numbers as if they were one population | both | Low | Different `wandb-tag` (`hypervigilance-round25` vs `hypervigilance-round2`) keeps the WandB groups distinct; the analyzer's pairwise comparisons read Round 2.5 only. |

## 3. Launch Manifest

**Hardware**: Node 106, 2 GPUs (cuda:0, cuda:1). 1 cell per GPU, 1 seed per cell. Total: 2 runs.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | planned | C — decoupleFood | `hypervigilance-round25-C-seed42_n106_gpu0` | hypervigilance | prod | 42 | 106 | cuda:0 | — | — | — |
| 2 | planned | A1 — passivePredator | `hypervigilance-round25-A1-seed43_n106_gpu1` | hypervigilance | prod | 43 | 106 | cuda:1 | — | — | — |

### 3.1 Configs to Produce (designer-only, pre-launch)

| Run | Config (env) | Config (agent) |
|-----|--------------|----------------|
| 1 | `configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml` | `configs/models/recurrent_ppo.yaml` |
| 2 | `configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml` | `configs/models/recurrent_ppo.yaml` |

**Configs are reused unchanged from the truncated Round 2.** Both already carry the `tag:` fields per the per-tag metrics implementation (commit `0a73613`). Tag↔spawn_area correctness was verified by the senior-developer in the per-tag plan's verification report and re-checked against the live YAML for this design (Cell A1: rabbits TL+BR, predator TL co-located with TL rabbit ✓; Cell C: rabbits TL+BR, predator full ✓).

### 3.2 Launch Commands (for `training-runner` reference)

Both invoked via `run_command.py` with the standard project Python interpreter. The commands the runner will issue (modulo the `train_command-agent.sh` template) are:

```bash
# Run 1 — Cell C — node 106 cuda:0 — seed 42
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml \
  --agent_config configs/models/recurrent_ppo.yaml \
  --episodes 10000000 \
  --num-envs 128 \
  --seed 42 \
  --device cuda:0 \
  --log-interval 50 \
  --wandb-group hypervigilance \
  --wandb-job-type prod \
  --wandb-name "hypervigilance-round25-C-seed42_n106_gpu0" \
  --tag "hypervigilance-round25-C-seed42_n106_gpu0"

# Run 2 — Cell A1 — node 106 cuda:1 — seed 43
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml \
  --agent_config configs/models/recurrent_ppo.yaml \
  --episodes 10000000 \
  --num-envs 128 \
  --seed 43 \
  --device cuda:1 \
  --log-interval 50 \
  --wandb-group hypervigilance \
  --wandb-job-type prod \
  --wandb-name "hypervigilance-round25-A1-seed43_n106_gpu1" \
  --tag "hypervigilance-round25-A1-seed43_n106_gpu1"
```

**Tag-vs-Round-2 distinction**: prefix `hypervigilance-round25-` (not `-round2-`) keeps the WandB group cleanly separable from the truncated Round 2 records.

**Wallclock estimate**: at the Round-2 SPS of ~41–43 k (§9.1 of the predecessor doc), 10 M episodes ≈ 75 h per cell. Total wallclock ~75 h on node 106 (cells run in parallel). The runner should plan for an uninterrupted ~3-day window or rely on checkpoint-and-resume if the run is interrupted.

## 4. Pre-Registered Analysis Plan

### 4.1 Primary metrics

Computed on the **last-10 % window** of training (episodes ≈9.0M – 10.0M), mean across all 128 envs of the history records in the window.

| Metric | Cell C | Cell A1 |
|---|---|---|
| `Episode/MeanDistRabbit_TL` (cells, L2) | primary cross-check | **primary disambiguator** |
| `Episode/MeanDistRabbit_BR` (cells, L2) | primary cross-check | secondary (BR control rabbit) |
| `Episode/MeanDistPredator_TL` (cells, L2) | n/a (predator tag is `full`) | **primary disambiguator** |
| `Episode/MeanDistPredator_full` (cells, L2) | primary | n/a (predator tag is `TL`) |
| `Episode/MeanDistRabbit` (aggregated, cells) | primary | primary |
| `Episode/MeanDistPredator` (aggregated, cells) | primary | primary |
| `Episode/RabbitHits` (count / ep) | primary | primary |
| `Episode/PredatorHits` (count / ep) | primary | primary |
| `Episode/Steps` (survival) | secondary (project headline) | **primary contamination check** (saturation at 470+ flags §5 row-2) |
| `Episode/Term_MaxSteps` | secondary | **primary contamination check** (≥0.85 + Steps ≥ 470 ⇒ camping) |
| Δ ≡ MeanDistPredator − MeanDistRabbit (aggregated) | derived primary | derived primary |
| ΔH ≡ RabbitHits − PredatorHits | derived primary | derived primary |
| Δ_TL ≡ MeanDistPredator_TL − MeanDistRabbit_TL | n/a | **DERIVED PRIMARY DISAMBIGUATOR** |

### 4.2 Confirmation / refutation thresholds — Cell A1 (the disambiguation cell)

The smoking-gun comparison is `Episode/MeanDistRabbit_TL` vs `Episode/MeanDistPredator_TL` — both entities live in the same TL quadrant under matched smells. The Round-2 §4.3 thresholds (keyed to aggregated `MeanDistRabbit` / `MeanDistPredator`) are superseded for Cell A1 by the per-tag table below.

| Outcome | Per-tag thresholds | Aggregated cross-check | Implies |
|---|---|---|---|
| **H₁(A1) confirmed — class-conditional avoidance** | `Δ_TL ≡ MeanDistPredator_TL − MeanDistRabbit_TL ≥ 1.0` cells **AND** `PredatorHits < RabbitHits − 1.5/ep` | survival NOT dominated by `Term_MaxSteps` (i.e., `Term_MaxSteps ≤ 0.7` OR Steps `<` 470) | The agent stays meaningfully farther from the same-corner predator than from the same-corner rabbit — class-level avoidance via post-contact teaching (visual ch.5 + extero-noc + GRU) survives the HUNT ablation. |
| **H₀(A1) confirmed — location-conditional avoidance (corner-camping)** | `\|Δ_TL\| ≤ 0.3` cells | survival ≥ 470/500 **AND** `Term_MaxSteps ≥ 0.85` (§5 row-2 contamination signature) | The two TL-tagged metrics agree because the agent treats both same-corner entities identically — i.e., never visits TL. Class discrimination is moot; the agent survives by camping. |
| **Ambiguous** | 0.3 < \|Δ_TL\| < 1.0 | any | Schedule Round 2.6 with seed 44 + the §7 second-tier metric (per-rabbit time-fraction-near-instance). |
| **Inverted** | Δ_TL ≤ −0.3 cells | any | Agent stays *closer* to the same-corner predator than to the same-corner rabbit. Surprising; would imply the GRU has learned an attraction toward the (now-passivated) predator. Treat as null result requiring a Round 3 cell. |

### 4.3 Confirmation / refutation thresholds — Cell C (the food-decoupling cell)

Round 2's §4.2 thresholds are retained with the same numerical bounds, augmented with a per-tag cross-check that distinguishes uniform avoidance from single-quadrant artefact.

| Outcome | Aggregated thresholds | Per-tag cross-check | Implies |
|---|---|---|---|
| **H₁(C) confirmed — decoupling did not erase the gap** | Δ ≥ 0.3 cells **AND** ΔH ≥ 1.5/ep | both `MeanDistRabbit_TL` and `MeanDistRabbit_BR` < `MeanDistPredator_full` | Class discrimination is real under sameProp and survives food-decoupling. Movement signature / visual+extero teaching is the channel. |
| **H₀(C) confirmed — Round-1 was the food confound** | \|Δ\| ≤ 0.1 **AND** \|ΔH\| ≤ 1.0/ep | both `MeanDistRabbit_*` keys close to `MeanDistPredator_full` (within 0.2 cells) | Round-1's apparent rabbit avoidance was food spillover. Sub-question for Round 3. |
| **Inverted (Round-2-truncated signal extrapolated)** | Δ ≤ −0.3 cells | per-tag check distinguishes "agent avoids both rabbit quadrants" (`MeanDistRabbit_TL` and `MeanDistRabbit_BR` both > `MeanDistPredator_full`) from "agent only avoids one quadrant due to a learned aversion" (one high, one low) | The decoupling forced rabbits to corners the agent learned were unreachable / unrewarding. The per-tag breakdown discriminates "bilateral rabbit avoidance" (compatible with food-only camping in TR+BL) from "asymmetric rabbit avoidance" (alternative learned aversion). Either reading needs Round 3. |
| **Ambiguous** | 0.1 < \|Δ\| < 0.3 | any | Add seed 44 in Round 2.6. |

### 4.4 Temporal evolution checks (mandatory, per project convention)

For each cell, partition training into 10 equal episode-count windows. Plot:
1. `Episode/MeanDistRabbit_<tag>` for each rabbit tag and `Episode/MeanDistPredator_<tag>` per window (overlay).
2. `Episode/RabbitHits` and `Episode/PredatorHits` per window (overlay).
3. `Episode/Steps` and `Episode/Term_MaxSteps` per window.

**A cell is taken seriously only if its primary-metric verdict is stable across the last 3 windows** (windows 8, 9, 10 — i.e., the last 30 % of training). The Round-2 truncated read failed this rule for Cell C (Δ was monotonically moving across the only 3 available windows); Round 2.5 has the budget to apply this rule meaningfully.

### 4.5 Cross-cell + cross-round contrasts

| Comparison | What it isolates |
|---|---|
| C (R2.5, seed 42) vs Round-1 R1-seed42 (`rg5nl1ov`) | Effect of food-quadrant decoupling, holding seed = 42. |
| A1 (R2.5, seed 43) vs Round-1 R1-seed43 (`6ks4bjbq`) | Effect of HUNT ablation + patrol-area match, holding seed = 43. |
| A1 (R2.5) `MeanDistRabbit_TL` vs `MeanDistPredator_TL` | **The within-quadrant disambiguation that Round 2.5 was launched to do.** |
| A1 (R2.5) `MeanDistRabbit_TL` vs `MeanDistRabbit_BR` | Internal control: does the agent treat the two same-class rabbits differently because one shares TL with the predator? Answers "is the discrimination class-level or quadrant-level." |
| C (R2.5) `MeanDistRabbit_TL` vs `MeanDistRabbit_BR` | Symmetric-or-asymmetric rabbit avoidance — required to interpret the Round-2-truncated inverted Δ if it persists. |
| R2.5 vs the truncated R2 (`0u266oj5`, `27svrmhv`) | Convergence check — does the early-window signal in R2 truncated match windows 1–3 of R2.5? Useful sanity but NOT primary. |

### 4.6 Random-policy baseline (anchor)

Round-1 analysis pre-computed: uniform-random `MeanDistPredator` ≈ 4.684 on the 10×10 grid; `MeanDistRabbit` (averaged over 2 rabbits) ≈ 4.68 likewise. For per-tag rabbits: random baseline ≈ 4.68 per tag (one rabbit ≈ uniform-random distance over the grid). For Cell A1's TL-restricted predator: random baseline depends on agent's spatial distribution but is bounded above by ~5–6 cells (the average distance from a uniform-grid agent to the centre of TL `[3, 3]`). A cell whose `MeanDistPredator_TL` and `MeanDistRabbit_TL` both sit within ±0.3 of the random-policy baseline has not learned anything specific.

## 5. Failure-Mode Catalog (pre-decided)

Re-uses Round 2's §5 catalog, augmented with per-tag-specific failure modes.

| Failure | Resolution |
|---|---|
| **(R2 row 1) Training instability** (NaN, value explosion) | Refutes the run, not the hypothesis. Re-launch with same seed; if it recurs, surface to `senior-developer`. |
| **(R2 row 2 — escalated for R2.5) Cell A1: agent never approaches predator (spatial segregation)** — agent camps BR food/rabbit corner, never visits TL. | **Now diagnosable directly** via the per-tag metrics. Signature: `Δ_TL ≤ 0.3 cells` (TL rabbit and TL predator equally distant) + `Steps ≥ 470` + `Term_MaxSteps ≥ 0.85`. This is **H₀(A1) confirmed** under §4.2 — a real verdict, not an uninformative ambiguity. The §9.5-truncated-Round-2 reading "uninformative because the agent never visits TL" is upgraded by Round 2.5 to "informative — the agent has chosen location-conditional safety." |
| **(NEW for R2.5) Cell A1: per-tag rabbit metrics diverge but agent never enters TL anyway** (`MeanDistRabbit_TL` is itself stuck near the random-baseline ≈ 4.68 because the agent literally never enters TL) | Diagnose via the BR control: if `MeanDistRabbit_BR` is much smaller than `MeanDistRabbit_TL` (e.g., < 2.5 vs > 4.5), AND `Term_MaxSteps ≥ 0.85`, the agent is BR-camping. Verdict: H₀(A1) confirmed (§4.2 row 2) — the per-tag gap is location-driven. |
| **(NEW for R2.5) Per-tag NaN** — `MeanDistRabbit_TL` itself is NaN/missing because the agent never enters TL during a window | The metric is a per-step mean over the *full episode*, not over "steps when agent is in TL", so the agent's TL-avoidance does not produce NaN — the metric reports the average L2 distance from the agent to the TL rabbit each step regardless of which quadrant the agent is in. NaN would only arise if `num_neutral = 0` (config error) or `ep_length = 0` (no episodes complete in the window — not possible at 10 M episodes). If observed, escalate as a logging bug. |
| **(NEW for R2.5) Per-tag fan-out emits zeros** | The senior-developer's verification (per-tag plan §"Required fix") found a Site-1 RPPO bug that emitted `MeanDistRabbit_*=0`; the fix is in commit `6d3d382`. If Round 2.5 logs show identical zeros at iter 1 of either run, halt the launch and re-verify the fix is in the deployed code. |
| **(R2 row 3 — refined for R2.5) Cell C: food-camping in TR+BL prevents rabbit visits** — `MeanDistRabbit` looks artificially high and ΔH near zero | Per-tag cross-check from §4.3: if both `MeanDistRabbit_TL` and `MeanDistRabbit_BR` are > 4.5 (above random baseline) AND `MeanDistFood < 2.0`, the agent is food-camping in the predator-only quadrants. Verdict: H₀(C) confirmed in the structural sense — the food-coupling *was* the load-bearing R1 mechanism. |
| **(R2 row 4) Saturation: `Episode/Steps` plateaus at ≈340 (Round-1 ceiling) without further differentiation** | At 10 M episodes, the §4.4 stability rule should reach steady state. A stable plateau at 340 across the last 3 windows is a real null. Unlikely to need Round 3 budget extension if §4 thresholds aren't met. |
| **(R2 row 5 — refined for R2.5) Seed-dependent noise: 1 seed per cell** | Cell C verdict required by §4.3 to exceed thresholds by ≥ 2x for elevation above provisional. Cell A1 verdict required by §4.2 likewise. Borderline outcomes route to a Round 2.6 with **seed 44 (Cell C)** + **seed 45 (Cell A1)** on the next available 2-GPU rotation. The §4.5 cross-round contrast against R1's 2-seed-agreed verdict provides additional anchoring at single-seed budget. |

## 6. Predicted Outcomes (pre-registration)

Designer's prior on the outcomes — recorded so the post-hoc reading does not adapt:

- **Cell C (most likely outcome — prior pulled from §9 of the truncated Round 2):** Truncated Round 2 showed Δ = −0.43 (rabbits *farther* than predators) at windows 1–3 (0–0.5 M ep), monotonically widening. At full convergence, the prediction is that this inverted signal **stabilises rather than reverts**: Δ in 9.0–10.0 M window will be in the range **−0.6 to −0.2 cells**, with the per-tag check showing `MeanDistRabbit_TL` and `MeanDistRabbit_BR` both > `MeanDistPredator_full` (bilateral rabbit avoidance). Reading: H₀(C) confirmed in spirit (Round-1 effect was food-driven) plus a *new* finding (the agent learned active rabbit avoidance once the food incentive to visit rabbit corners was removed). 60 % probability.
  Alternative (40 %): Δ drifts back toward 0 over training and ends in the H₀(C) band (\|Δ\| ≤ 0.1) — the agent stops avoiding rabbits once it learns the new food map, and the per-tag check shows both rabbit tags within 0.2 of `MeanDistPredator_full`.

- **Cell A1 (most likely outcome):** Truncated Round 2 had `Steps ≈ 482, Term_MaxSteps = 0.89, MeanDistPredator (aggregated) = 6.45, MeanDistRabbit = 2.58` — the §5 row-2 contamination signature in full force. At convergence, the prediction is that this stabilises rather than transitions: per-tag `Δ_TL ≈ 0` (within 0.3) AND survival saturated at ≈ 480/500 with `Term_MaxSteps ≥ 0.85`. Reading: **H₀(A1) confirmed — the agent learned location-conditional safety, not class recognition.** 70 % probability.
  Alternative (30 %): given another 9.5 M episodes the agent eventually starts approaching the TL quadrant (food incentive in TL + BR after ~2 M ep risk-tolerance growth) and the per-tag gap opens — `Δ_TL ≥ 1.0`, `Term_MaxSteps` falls below 0.7. H₁(A1) confirmed.

If both predictions are correct, the paper-level reading of the sameProp study becomes: **under matched olfactory properties, RPPO produces survival-driven location-conditional avoidance, not class-conditional avoidance** — Round 1's apparent rabbit-vs-predator distance gap was the joint product of (a) food-quadrant collinearity (Cell C result) and (b) movement-signature asymmetry filtered through location learning (Cell A1 result), neither of which constitutes class-level olfactory or extero-noc discrimination.

## 7. Metrics Requested (deferred — only escalate if Round 2.5 is borderline)

The per-tag metrics shipped in commit `0a73613` (verified `6d3d382`) cover the disambiguation Round 2.5 needs. The metrics below would help only if Round 2.5 lands in the §4.2 / §4.3 ambiguous bands.

| Subfield | Content |
|---|---|
| **Metric (1) — second-tier disambiguator for borderline Cell A1** | `Episode/TimeFractionNearTL_Rabbit`, `Episode/TimeFractionNearTL_Predator` — fraction of episode steps the agent spent within 1 cell of each TL-tagged entity instance. |
| **Why now** | If Cell A1 lands in the ambiguous `0.3 < Δ_TL < 1.0` band, distinguishing "agent lingers near TL rabbit but ducks the TL predator on co-occupancy events" from "agent's TL transits are too brief to differentiate" requires a per-instance time-near metric. Aggregated `MeanDistRabbit_TL` already tells us the *average* distance; this metric tells us the *engagement profile*. |
| **Where it'd live** | `src/environment/core.py` per-step `info` — boolean mask `dist_per_neutral < 1.0` summed and divided by ep_length. Shape `[num_*]`; same fan-out pattern as `dist_per_neutral`. |
| **Cost** | Cheap — boolean indicator + reduce. |

| Subfield | Content |
|---|---|
| **Metric (2) — quadrant occupancy** | `Episode/QuadrantOccupancy_{TL,TR,BL,BR}` — fraction of episode steps the agent spends in each quadrant. (Original Round-2 §7 item 2 — deferred in the per-tag implementation as redundant for the disambiguation use case.) |
| **Why now** | If both Cell C and Cell A1 land in the inverted/contamination bands, quadrant occupancy directly visualises "the agent never enters TL/BR" without depending on per-entity proxies. Useful for Round 3 design; not required for Round 2.5 verdicts. |
| **Where it'd live** | `src/environment/core.py` per-step occupancy counter, surfaced via `info`. |
| **Cost** | Cheap — 4 boolean masks + reduce. |

If Round 2.5 lands in any §5 NEW row's failure-mode signature, escalate one of these to a `feature-workflow` plan before Round 2.6 launches. **Round 2.5 does NOT block on these.**

## 8. Launchable Status

**Configs**: ready ✓ (no edits needed)
- `configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml` — verified rabbits TL/BR, predator full
- `configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml` — verified rabbits TL/BR, predator TL (matches TL rabbit's quadrant — the disambiguation primitive)

**Code state**: ready ✓
- Per-tag metrics shipped (commit `0a73613`), Site-1 RPPO bug fixed (commit `6d3d382`), full verification pass (per-tag plan §"Verification Report").
- All 5 WandB sites emit `Episode/MeanDistRabbit_<tag>`, `Episode/MeanDistPredator_<tag>`. Aggregated keys preserved.

**Pre-flight gates** (must pass before runner launches):
1. ⏸ `env-config-auditor` — modality order, mandatory keys, `properties`/`property` typo check, predator-side keys consistent. Configs are reused from Round 2 (which passed audit), so this is a quick re-audit, not a fresh audit.
2. ⏸ User authorization — explicit "go" after auditor sign-off.

**Runner handoff**: `training-runner` will fill the actual columns of §3 (Status, Launched at, WandB run ID, Log path) at launch time. Per the diary protocol, runner must call `diary training-start --tag … --node 106 --gpu cuda:0/cuda:1 --cell C/A1 --wandb <id> --doc docs/experiments/active/hypervigilance/sameprop_round25_design.md` for each run.

---

## 9. Results
*(to be filled by `experiment-analyzer` post-training, against the §4 thresholds)*

## 10. Analysis
*(to be filled by `experiment-analyzer` post-training)*

## 11. Conclusions
*(to be filled by `experiment-analyzer` post-training, with §6 prior recorded so post-hoc reading does not adapt)*

---

## Appendix

### A. Cell mapping table (channel ablations)

| Round-1 channel rank (`sameprop_discriminating_channels.md`) | R2.5 cell that ablates it | Configured how |
|---|---|---|
| §4 Movement / temporal signature in olfaction (predator HUNT tracks `agent_pos`; rabbit jitters) | Cell A1 | `hunt_stamina_threshold: 1.1` (rested_enough never true) + `detection_range: 0` (HUNT condition unreachable) |
| §4 Patrol-area asymmetry (predator full-grid; rabbit quadrant) | Cell A1 | `patrol_area: [[1,1],[5,5]]` and `spawn_area: [[1,1],[5,5]]` (predator now matches rabbit TL); per-tag `MeanDistPredator_TL` directly comparable to `MeanDistRabbit_TL` |
| §1 Olfactory instantaneous shape (indistinguishable when properties match) | NOT ablated | properties remain `[0,1,0,0,0]` for both — that is the *condition under test* |
| §3 Visual ch.5 vs ch.7 at colocation | NOT ablated (intentionally, for both cells) | `visual_sensor_enabled: true`, range 0 — present in both A1 and C; isolates the *contribution* of post-contact teaching |
| §2 Extero-noc 0.9 at contact | NOT ablated | `extero_nociception` active for predator hits in both A1 and C |
| **NEW: food-quadrant collinearity** (analyzer flag, R1) | **Cell C** | food → TR + BL only; rabbits remain in TL + BR; food and rabbits share NO quadrant |
| **NEW: per-tag disambiguation primitive** (R2.5) | both cells | `tag` fields on rabbit (TL/BR) and predator (TL or full); per-instance `Episode/MeanDist*_<tag>` keys |

### B. Config diff vs Round 2 (= Round 2.5)

**None.** Round 2.5 reuses the Round 2 configs unchanged. The per-tag fields on those configs were added in commit `0a73613` (per-tag metrics implementation) — the same configs were nominally used for the truncated Round 2 launch on 2026-05-08, but per-tag values for that round were already enabled before SIGINT (so the truncated WandB runs `0u266oj5` and `27svrmhv` carry per-tag keys too — useful as an early-window sanity reference for Round 2.5).

### C. Single-seed escalation policy (Round 2.6 contingency)

If Round 2.5 lands in §4.2 / §4.3 *ambiguous* bands, or if a cell's signal is borderline relative to its threshold (within 25 % of the threshold value), schedule Round 2.6:

| Round 2.6 run | Cell | Tag | Seed | Node/GPU |
|---|---|---|---|---|
| 1 | C — decoupleFood | `hypervigilance-round26-C-seed44_<n>_<gpu>` | 44 | next-available 2-GPU rotation |
| 2 | A1 — passivePredator | `hypervigilance-round26-A1-seed45_<n>_<gpu>` | 45 | same node, second GPU |

This contingency is pre-registered: do not retrofit it post-hoc to "rescue" a result that landed inside a non-ambiguous band but with a sign or magnitude the designer did not predict in §6.

### D. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-05-09 | Initial pre-registered design for Round 2.5; 2 cells × 1 seed each on node 106 cuda:0/cuda:1; per-tag metrics live (commits `0a73613` + `6d3d382`); thresholds re-keyed to per-tag for Cell A1, augmented with per-tag cross-check for Cell C | experiment-designer |
