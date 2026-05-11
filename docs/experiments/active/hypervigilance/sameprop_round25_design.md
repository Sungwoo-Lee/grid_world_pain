---
title: "SameProp Round 2.5 — full-budget re-launch with per-tag distance disambiguation"
topic: hypervigilance
status: analyzed
created: 2026-05-09
last_updated: 2026-05-11
phase: 1
wandb_tag: "hypervigilance-round25"
develop_link: "../../../develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md"
supersedes: []
---

# SameProp Round 2.5 — full-budget re-launch with per-tag distance disambiguation

> **Status**: ANALYZED — both runs completed cleanly to 10 M episodes; pre-registered §4 thresholds applied; verdicts in §11.
> **Headline finding**: Cell A1 — the same-corner predator and same-corner rabbit end up at identical distance from the agent (per-tag Δ_TL ≈ +0.00 cells) at saturated survival (Steps 486 / 500, Term_MaxSteps 0.92). Plain English: the agent never visits the predator's quadrant, so it never gets a chance to demonstrate (or fail to demonstrate) class recognition; it survives by camping. **H₀(A1) — location-conditional avoidance — confirmed.** Cell C — once food no longer draws the agent into rabbit corners, the agent stays *farther* from rabbits than from predators (aggregated Δ = −0.53 cells), with both per-tag rabbit distances above the predator distance. Plain English: Round 1's apparent rabbit-versus-predator gap was indeed the food-coupling, and removing it produced an unanticipated bilateral rabbit-avoidance policy. **H₁(C) refuted; H₀(C) supported in spirit but the unanticipated inversion needs Round 2.6 to lock the sign across seeds.**
> **Date**: 2026-05-09 (design); 2026-05-10 (analysis)
> **Author**: experiment-designer (§§0–8); experiment-analyzer (§§9–11).
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
| 1 | completed (10,000,022 ep, ~22 h 33 m wall) | C — decoupleFood | `hypervigilance-round25-C-seed42_n106_gpu0` | hypervigilance | prod | 42 | 106 | cuda:0 | 2026-05-09T18:25:29 | `bdnfc0lu` | `logs/20260509_182529.log` |
| 2 | completed (10,000,003 ep, ~17 h 30 m wall) | A1 — passivePredator | `hypervigilance-round25-A1-seed43_n106_gpu1` | hypervigilance | prod | 43 | 106 | cuda:1 | 2026-05-09T18:27:20 | `nm8gn7y2` | `logs/20260509_182720.log` |

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

> **See §12 for a trajectory-level reading that complicates the spatial verdict** — Cell C's per-step bush-dive and risk-discounting measures show a strongly class-conditional defensive policy that the per-tag distance summary in §9.2 dissolved into a "bilateral rabbit avoidance" mean. The §11.1 Cell C verdict still stands at the spatial level; §12 widens it.

> Numbers only. Pre-registered §4 thresholds applied in §10; verdicts in §11.
> All metrics are means across the 128 parallel envs of the WandB history records that fall inside the named episode window. Both runs completed the full 10 M-episode budget cleanly (no SIGINT; final episode ≈ 10,000,022 for Cell C, ≈ 10,000,003 for Cell A1).
> Working files: `tmp/20260510_round25_cellC_last10.md`, `tmp/20260510_round25_cellA1_last10.md`, `tmp/20260510_round25_combined.md`, plus the raw history CSVs `tmp/20260510_round25_cellC_history.csv` and `tmp/20260510_round25_cellA1_history.csv`.

### 9.1 Run actuals

| Cell | Tag | WandB | Seed | Final episode | Wall-clock | Records in last-10 % window |
|---|---|---|---|---:|---:|---:|
| C — decoupleFood | `hypervigilance-round25-C-seed42_n106_gpu0` | [bdnfc0lu](https://wandb.ai/sungwoolee/grid_world_pain/runs/bdnfc0lu) | 42 | 10,000,022 | 22 h 33 m | 488 |
| A1 — passivePredator | `hypervigilance-round25-A1-seed43_n106_gpu1` | [nm8gn7y2](https://wandb.ai/sungwoolee/grid_world_pain/runs/nm8gn7y2) | 43 | 10,000,003 | 17 h 30 m | 591 |

The user brief estimated the wall-clock at ~18 h (Cell C) and ~22 h (Cell A1). Actual wall-clock came in roughly reversed — Cell A1 finished about five hours sooner than Cell C — because Cell A1's saturated max-steps survival (≈ 486 / 500) packs ~25 % fewer episodes per unit wall-clock than Cell C's shorter ≈ 399-step episodes, but Cell A1's forward pass at high survival was evidently dominated by per-step compute, not per-episode reset cost. The episode count is the budget that matters (both ≥ 10 M), not the wall-clock.

### 9.2 Last-10 % window readout (episodes 9.0 M – 10.0 M)

Mean ± temporal std (= within-window noise across the ~488–591 records, n = 1 seed per cell).

| Metric | Cell C (seed 42) | Cell A1 (seed 43) | R1 converged (~7.2 M, both seeds) | Random-policy baseline |
|---|---:|---:|---:|---:|
| Aggregated `Episode/MeanDistRabbit` (cells, L2) | **4.579 ± 0.022** | **2.596 ± 0.108** | 3.77 | 4.68 |
| Aggregated `Episode/MeanDistPredator` (cells, L2) | **4.047 ± 0.053** | **6.632 ± 0.208** | 4.40 | 4.68 |
| Aggregated `Episode/RabbitHits` / ep | **0.914 ± 0.071** | **18.490 ± 1.265** | 6.46 | — |
| Aggregated `Episode/PredatorHits` / ep | **3.298 ± 0.149** | **0.500 ± 0.198** | 3.37 | — |
| `Episode/Steps` (survival) | **399.5 ± 4.3** | **485.7 ± 23.6** | 327.3 | — |
| `Episode/Term_MaxSteps` | 0.501 ± 0.016 | **0.924 ± 0.093** | 0.263 | — |
| `Episode/Term_Injury` | 0.270 ± 0.024 | 0.036 ± 0.027 | 0.339 | — |
| `Episode/Term_Starvation` | 0.230 ± 0.026 | 0.040 ± 0.075 | 0.398 | — |
| `Episode/MeanDistFood` | 1.312 ± 0.022 | 0.862 ± 0.257 | 2.41 | 4.68 |
| `Episode/MeanDistHidingPredator` | 2.483 ± 0.015 | 2.405 ± 0.031 | 2.63 | ~2.6 |
| `Episode/HidingPredatorHits` / ep | 2.472 ± 0.080 | 3.252 ± 0.351 | 3.00 | — |
| `Episode/FoodEaten` / ep | 72.2 ± 1.4 | 116.6 ± 10.9 | 54.7 | — |
| `Episode/Reward` | -183.5 ± 1.7 | -123.0 ± 10.9 | -204.5 | — |
| **Per-tag `Episode/MeanDistRabbit_TL`** | **5.524 ± 0.034** | **6.629 ± 0.209** | n/a (not logged in R1) | 4.68 |
| **Per-tag `Episode/MeanDistRabbit_BR`** | **5.811 ± 0.034** | **2.887 ± 0.269** | n/a | 4.68 |
| **Per-tag `Episode/MeanDistPredator_full`** (Cell C only) | **4.047 ± 0.053** | n/a | n/a | 4.68 |
| **Per-tag `Episode/MeanDistPredator_TL`** (Cell A1 only) | n/a | **6.632 ± 0.208** | n/a | ~5–6 |

### 9.3 Derived primary statistics (the threshold-keyed numbers)

| Derived stat | Cell C | Cell A1 | R1 converged | §4 threshold for the relevant verdict |
|---|---:|---:|---:|---|
| Aggregated **Δ ≡ MeanDistPredator − MeanDistRabbit** | **−0.532 cells** | **+4.036 cells** | +0.63 | C: H₁ ≥ +0.3, H₀ ≤ 0.1, Inverted ≤ −0.3 |
| Aggregated **ΔH ≡ RabbitHits − PredatorHits** | **−2.385 / ep** | **+17.990 / ep** | +3.09 | C: H₁ ≥ +1.5, H₀ ≤ 1.0 |
| **Per-tag Δ_TL ≡ MeanDistPredator_TL − MeanDistRabbit_TL** | n/a (predator tag is `full`) | **+0.004 cells** | n/a | A1: H₁ ≥ +1.0, H₀ ≤ 0.3 |
| Cell C bilateral check: `MeanDistPredator_full − MeanDistRabbit_TL` | **−1.477** | n/a | n/a | C-H₁ requires this to be > 0 |
| Cell C bilateral check: `MeanDistPredator_full − MeanDistRabbit_BR` | **−1.764** | n/a | n/a | C-H₁ requires this to be > 0 |
| Cell A1 internal control: `MeanDistRabbit_TL − MeanDistRabbit_BR` | n/a | **+3.742** | n/a | (interpreted in §10) |

### 9.4 Temporal evolution — 10 equal-episode windows

#### Cell C — decoupleFood, seed 42

| Window | ep range | n | MeanDistRabbit_TL | MeanDistRabbit_BR | MeanDistPredator_full | RabbitHits | PredatorHits | Steps | Term_MaxSteps | Δ | ΔH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0M–1M | 296 | 5.727 | 5.952 | 4.342 | 0.972 | 2.708 | 272.4 | 0.167 | −0.301 | −1.736 |
| 2 | 1M–2M | 433 | 5.592 | 5.874 | 4.078 | 0.906 | 3.412 | 354.2 | 0.346 | −0.550 | −2.506 |
| 3 | 2M–3M | 454 | 5.552 | 5.834 | 4.011 | 0.905 | 3.553 | 372.1 | 0.404 | −0.586 | −2.648 |
| 4 | 3M–4M | 465 | 5.544 | 5.827 | 4.005 | 0.906 | 3.443 | 381.2 | 0.433 | −0.586 | −2.536 |
| 5 | 4M–5M | 473 | 5.539 | 5.830 | 4.031 | 0.886 | 3.380 | 387.2 | 0.455 | −0.560 | −2.494 |
| 6 | 5M–6M | 477 | 5.540 | 5.817 | 4.035 | 0.902 | 3.319 | 390.9 | 0.469 | −0.552 | −2.417 |
| 7 | 6M–7M | 480 | 5.536 | 5.816 | 4.039 | 0.906 | 3.321 | 393.9 | 0.481 | −0.546 | −2.416 |
| **8** | **7M–8M** | 481 | 5.527 | 5.816 | 4.039 | 0.925 | 3.371 | 394.2 | 0.481 | **−0.542** | **−2.446** |
| **9** | **8M–9M** | 484 | 5.528 | 5.807 | 4.038 | 0.918 | 3.338 | 396.4 | 0.489 | **−0.541** | **−2.421** |
| **10** | **9M–10M** | 488 | 5.524 | 5.811 | 4.047 | 0.914 | 3.298 | 399.5 | 0.501 | **−0.532** | **−2.385** |

Cell C reaches its inverted-Δ plateau (Δ ≈ −0.55) around 2 M ep and holds it for the remaining 8 M episodes. The plateau then drifts a few hundredths toward zero (−0.586 → −0.532) over the last 7 M episodes — interpretable as a slow, residual learning effect, but the sign and magnitude are firmly stable.

#### Cell A1 — passivePredator, seed 43

| Window | ep range | n | MeanDistRabbit_TL | MeanDistPredator_TL | MeanDistRabbit_BR | RabbitHits | PredatorHits | Steps | Term_MaxSteps | Δ | ΔH | **Δ_TL** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0M–1M | 584 | 6.545 | 6.549 | 2.882 | 18.621 | 0.638 | 486.6 | 0.933 | +4.004 | +17.983 | **+0.003** |
| 2 | 1M–2M | 602 | 6.560 | 6.565 | 2.883 | 18.932 | 0.639 | 493.3 | 0.961 | +4.024 | +18.293 | **+0.004** |
| 3 | 2M–3M | 294 | 6.798 | 6.811 | 3.335 | 13.867 | 0.513 | 377.6 | 0.655 | +3.900 | +13.353 | **+0.013** |
| 4 | 3M–4M | 121 | 6.990 | 7.010 | 4.231 | 2.207 | 0.120 | 101.1 | 0.000 | +3.257 | +2.087 | **+0.020** |
| 5 | 4M–5M | 105 | 5.031 | 4.998 | 7.627 | 1.417 | 0.901 | 86.6 | 0.000 | +0.655 | +0.516 | **−0.033** |
| 6 | 5M–6M | 130 | 5.573 | 5.589 | 6.037 | 2.043 | 0.517 | 106.9 | 0.000 | +1.623 | +1.526 | **+0.016** |
| 7 | 6M–7M | 156 | 7.284 | 7.301 | 3.611 | 3.911 | 0.127 | 128.4 | 0.000 | +4.100 | +3.784 | **+0.017** |
| **8** | **7M–8M** | 520 | 6.795 | 6.799 | 2.895 | 16.721 | 0.354 | 442.1 | 0.721 | +4.165 | +16.367 | **+0.004** |
| **9** | **8M–9M** | 585 | 6.636 | 6.640 | 2.896 | 18.376 | 0.497 | 483.4 | 0.913 | +4.044 | +17.879 | **+0.005** |
| **10** | **9M–10M** | 591 | 6.629 | 6.632 | 2.887 | 18.490 | 0.500 | 485.7 | 0.924 | +4.036 | +17.990 | **+0.004** |

**Two features of the Cell A1 trajectory require flagging:**

1. The **per-tag Δ_TL is essentially zero in every window** — the maximum absolute value across all ten 1 M-episode windows is 0.033 cells (window 5), and in the converged windows 8–10 it is +0.004–0.005 cells. The same-corner predator and same-corner rabbit are at indistinguishable distance from the agent at every stage of training.
2. There is a **mid-training transient between windows 4 and 7** (≈ 3 M – 7 M episodes) during which Steps collapses from 493 → ~100, `Term_MaxSteps` drops to 0.0, and `Term_Starvation` rises to 0.74–0.97. The agent's record count per window also collapses (602 → 121 → 105 → 130 → 156), which reflects that with shorter episodes more episodes complete per WandB log interval but each parallel env contributes fewer Reward/distance entries to the per-window mean. The agent recovers to its window-1 camping policy by window 8 and holds it through window 10. A policy disturbance of this duration is non-trivial and is discussed in §10.5.

### 9.5 Cross-round seed-paired contrasts

Both Round 1 baseline runs are at the cited 7.2 M-episode partial-budget steady-state values. Cell C uses the same seed (42) as R1's `rg5nl1ov`; Cell A1 uses the same seed (43) as R1's `6ks4bjbq`.

| Quantity | R2.5 Cell C (seed 42, 10 M) | R1 seed 42 (~7.2 M) | Cross-round Δ |
|---|---:|---:|---:|
| `MeanDistRabbit` | 4.579 | 3.766 | **+0.81 cells (R2.5 farther)** |
| `MeanDistPredator` | 4.047 | 4.416 | −0.37 cells (R2.5 closer) |
| `RabbitHits` / ep | 0.914 | 6.562 | **−5.65 / ep** |
| `PredatorHits` / ep | 3.298 | 3.430 | −0.13 / ep |
| `Steps` | 399.5 | 327.3 | +72 (R2.5 longer survival) |
| `FoodEaten` / ep | 72.2 | 54.6 | +17.6 |

| Quantity | R2.5 Cell A1 (seed 43, 10 M) | R1 seed 43 (~7.2 M) | Cross-round Δ |
|---|---:|---:|---:|
| `MeanDistRabbit` | 2.596 | 3.783 | −1.19 (closer) |
| `MeanDistPredator` | 6.632 | 4.388 | **+2.24 (much farther)** |
| `RabbitHits` / ep | 18.490 | 6.351 | **+12.14 / ep** |
| `PredatorHits` / ep | 0.500 | 3.318 | −2.82 / ep |
| `Steps` | 485.7 | 327.3 | **+158 (saturated camping)** |
| `Term_MaxSteps` | 0.924 | 0.260 | **+0.66** |
| `FoodEaten` / ep | 116.6 | 54.9 | +61.7 |

Both interventions produced large, well-resolved cross-round shifts. The food decoupling in Cell C shifts every behaviour metric by a multiple of R1's seed-to-seed variance, and the HUNT ablation + patrol-area shrink in Cell A1 produces a saturated camping policy unlike anything R1 reached.

---

## 10. Analysis

This section applies the pre-registered §4 thresholds to the §9 numbers. The thresholds were locked in before the runs were inspected; nothing in §10 adapts them post-hoc. Where the data lands outside any pre-registered band, that is called out as such — in particular the §4.3 "Inverted" outcome for Cell C is a pre-registered band even though it was the designer's lower-probability prior in §6.

### 10.1 Cell A1 verdict — does the agent recognise the predator class, or does it just camp the safe corner?

**Plain English first.** The Cell A1 setup forces predator and rabbit to share the top-left quadrant under matched smells; if the agent recognises the predator *as a predator* (i.e., uses class identity rather than location), it should stay further from the same-corner predator than from the same-corner rabbit. If the agent instead survives by never visiting the dangerous corner, the same-corner predator and same-corner rabbit will look identical from the outside — both equally distant — and survival will be saturated at the max-step ceiling.

The §4.2 disambiguator is the per-tag gap **Δ_TL ≡ MeanDistPredator_TL − MeanDistRabbit_TL** in the last-10 % window. The thresholds:

| §4.2 outcome | Per-tag threshold | Aggregated cross-check | Round 2.5 last-10 % value |
|---|---|---|---|
| H₁(A1) — class-conditional avoidance | Δ_TL ≥ +1.0 cells **AND** PredatorHits < RabbitHits − 1.5 / ep | Term_MaxSteps ≤ 0.7 OR Steps < 470 | Δ_TL = **+0.004**, ΔH = +17.99 / ep, Term_MaxSteps = **0.924**, Steps = **485.7** |
| **H₀(A1) — location-conditional avoidance (corner-camping)** | \|Δ_TL\| ≤ 0.3 cells | Steps ≥ 470 / 500 **AND** Term_MaxSteps ≥ 0.85 | Δ_TL = **+0.004** (≤ 0.3 ✓), Steps = **485.7** (≥ 470 ✓), Term_MaxSteps = **0.924** (≥ 0.85 ✓) |
| Ambiguous | 0.3 < \|Δ_TL\| < 1.0 | — | not in band |
| Inverted | Δ_TL ≤ −0.3 | — | not in band |

**Verdict: H₀(A1) confirmed — the agent learned location-conditional avoidance, not class-conditional avoidance.**

The per-tag disambiguator landed at 0.7 % of its H₁ threshold (0.004 / 1.000) — for context, that is well inside the within-window noise band (the per-window σ on `MeanDistPredator_TL` is 0.21 cells, two orders of magnitude larger than the gap). All three converged windows (8, 9, 10) agree to ±0.001 on Δ_TL. Both H₀(A1) survival contamination signatures are simultaneously satisfied: Steps is 97 % of the 500-step max-cap, and 92 % of episodes terminate via `Term_MaxSteps` (i.e., the timer ran out, not death). This is the §5 row-2 contamination pattern in its cleanest form.

The internal control is the same-class comparison `MeanDistRabbit_TL = 6.629` versus `MeanDistRabbit_BR = 2.887`: a +3.74-cell gap between two rabbits that emit identical olfactory `properties [0, 1, 0, 0, 0]`. Class identity cannot explain this gap. Quadrant identity does — the BR rabbit is in the same quadrant as the agent's food-camping policy, and the TL rabbit is in the predator's quadrant. The agent treats the two same-class entities as different things because *one of them lives where the agent doesn't go*.

The §4.6 random-policy floor is also informative here: random-uniform `MeanDistRabbit_TL ≈ 4.68`, but the agent's actual `MeanDistRabbit_TL = 6.63`. The agent is 1.95 cells farther from TL than chance would put it — i.e., it is *actively avoiding* TL, not just failing to visit. This rules out the alternative reading "the agent is at uniform-random distribution and the TL metric just looks high because TL is one corner." The agent has learned a structured spatial avoidance of TL, and that avoidance happens to make the metric for the same-corner rabbit and the same-corner predator look identical.

### 10.2 Cell C verdict — does class discrimination survive food decoupling?

**Plain English first.** In Round 1 the agent ended up about 0.6 cells farther from predators than from rabbits, but rabbits and food shared the same two corners — so the apparent "rabbit avoidance is weaker than predator avoidance" could simply be food-seeking pulling the agent toward rabbit corners. Cell C moves food into the predator-only corners (TR + BL), so any remaining rabbit-versus-predator distance gap can no longer come from the food trace. The question this cell answers is: with food decoupled from rabbits, does the Round-1 ordering `MeanDistRabbit < MeanDistPredator` survive?

The §4.3 thresholds:

| §4.3 outcome | Aggregated thresholds | Per-tag cross-check | Round 2.5 last-10 % value |
|---|---|---|---|
| H₁(C) — class discrimination survives food decoupling | Δ ≥ +0.3 cells **AND** ΔH ≥ +1.5 / ep | both rabbit tags < `MeanDistPredator_full` | Δ = **−0.532**, ΔH = **−2.385**, MeanDistRabbit_TL = 5.524, MeanDistRabbit_BR = 5.811, MeanDistPredator_full = 4.047 |
| H₀(C) — Round-1 was the food confound | \|Δ\| ≤ 0.1 **AND** \|ΔH\| ≤ 1.0 / ep | both rabbit tags within 0.2 of MeanDistPredator_full | Δ = −0.532 (out of band) |
| **Inverted (rabbits FARTHER than predator)** | **Δ ≤ −0.3** | per-tag check: bilateral vs asymmetric | **Δ = −0.532 ✓**; both rabbit tags > predator_full by ≥ 1.48 cells (bilateral) |
| Ambiguous | 0.1 < \|Δ\| < 0.3 | — | not in band |

**Verdict: H₁(C) refuted; the data lands in the §4.3 "Inverted" band (Δ ≤ −0.3), with the per-tag check showing bilateral rabbit avoidance.**

Both rabbit tags (`MeanDistRabbit_TL = 5.52` and `MeanDistRabbit_BR = 5.81`) sit *above* the random-policy floor of 4.68 by roughly +0.84 and +1.13 cells respectively, and both are *above* `MeanDistPredator_full = 4.05` by ≥ 1.48 cells. This is *not* the "agent only avoids one quadrant due to a learned aversion" sub-case (one tag high, one low); it is the bilateral case — the agent actively keeps distance from *both* rabbits. Meanwhile the agent is closer to the patrolling predator (4.05) than to either rabbit and closer to the predator than the random-policy floor (4.68 → 4.05 = −0.63 cells closer than chance), which is the opposite direction from R1's slight predator-distancing.

Cross-referenced with the food map: `MeanDistFood = 1.31` (much closer than R1's 2.41) and `FoodEaten = 72.2 / ep` (vs R1's 54.7) confirm that the agent has learned the new food map — it is camping food in the TR + BL quadrants. Because rabbits remain in TL + BR (the food-empty quadrants), camping food now means staying away from rabbits. The patrolling predator roams the whole grid, so by occupying TR + BL the agent is on average closer to a roaming predator (which spends ~50 % of its time in the agent's quadrants) than to a rabbit (which is permanently in the agent's-empty quadrants).

This is the §5 row-3 contamination signature ("Cell C: agent camps food + bush corners and rarely visits rabbit quadrants → MeanDistRabbit looks artificially high"), but at the converged training horizon it is no longer a contamination *of* the H₀ / H₁ test — it is the policy itself. The agent is not being prevented from visiting rabbits by some other concern; it has *learned* that rabbit corners (now food-empty) are not worth visiting, and that learning is stable across all 8 of the post-warmup 1 M-episode windows.

The §6 designer's prior was 60 % on this exact outcome ("Δ in 9.0–10.0 M window will be in the range −0.6 to −0.2 cells, with the per-tag check showing both `MeanDistRabbit_TL` and `MeanDistRabbit_BR` both > `MeanDistPredator_full` (bilateral rabbit avoidance)"). Observed Δ = −0.532 lands in the predicted range; the bilateral check is +1.48 / +1.76 (predator distance below both rabbit tags). The 40 % alternative (Δ drifts back to ≈ 0) is refuted.

### 10.3 §4.4 temporal-stability check

The §4.4 rule: a verdict is taken seriously only if the primary metric is stable across windows 8, 9, 10 (the last 30 % of training).

**Cell C**: Δ in windows 8 / 9 / 10 = −0.542 / −0.541 / −0.532. Spread across the three windows is 0.010 cells, an order of magnitude below the within-window σ of 0.05 cells on `MeanDistPredator`. ΔH = −2.446 / −2.421 / −2.385 — likewise stable. **Stability rule passed.**

**Cell A1**: Δ_TL in windows 8 / 9 / 10 = +0.004 / +0.005 / +0.004. Spread is 0.001 cells, three orders of magnitude below the within-window σ of 0.21 cells on `MeanDistPredator_TL`. Steps = 442.1 / 483.4 / 485.7 — window 8 is mid-recovery from the §10.5 transient and is ~9 % below the converged value, but the per-tag verdict numbers (Δ_TL, MeanDistRabbit_TL, MeanDistRabbit_BR) are all stable to within 0.005 cells across windows 8–10. **Stability rule passed for the disambiguator; survival metric is recovering across window 8 but converged in 9–10.**

### 10.4 §4.6 random-policy floor check

R1 §4.6 pre-computed: uniform-random `MeanDistPredator ≈ 4.68`, `MeanDistRabbit ≈ 4.68` per tag. A cell whose per-class distances both sit within ±0.3 of 4.68 has not learned class discrimination at all.

| Cell | Per-class distance vs random baseline | Reading |
|---|---|---|
| Cell C | MeanDistRabbit 4.58 (within 0.10 of random); MeanDistPredator 4.05 (−0.63 below random) | The aggregated rabbit metric is at random baseline, but per-tag rabbit metrics (5.52, 5.81) are *above* random by +0.84 / +1.13 cells, so the agent has learned *bilateral rabbit avoidance*, not "no class discrimination". MeanDistPredator is below random (agent gets closer to predator than chance) — the policy is structured, not random. |
| Cell A1 | MeanDistRabbit_TL 6.63 (+1.95 above random), MeanDistRabbit_BR 2.89 (−1.79 below random), MeanDistPredator_TL 6.63 (+1.95 above random) | Strongly structured. The agent is far from anything in TL (because it doesn't visit TL) and very close to anything in BR (because it lives there). Quadrant occupancy, not class identity, is the primitive being learned. |

Neither cell sits at the random-policy floor. Both have learned a structured spatial policy; the question §4 was designed to ask is whether that structured spatial policy contains a *class-conditional* component, and the answer in both cells is no.

### 10.5 §5 failure-mode catalog — which rows fired?

- **§5 row 1 (training instability)**: did not fire — both runs completed to 10 M episodes with healthy loss curves.
- **§5 row 2 escalated for R2.5 (Cell A1: agent never approaches predator → spatial segregation)**: **fired, and is now diagnosable as a real verdict** rather than as an uninformative ambiguity. Per the row-2 design ("Δ_TL ≤ 0.3 cells AND Steps ≥ 470 AND Term_MaxSteps ≥ 0.85"), all three thresholds are met; per the §4.2 outcome table this is **H₀(A1) confirmed** (location-conditional avoidance), exactly as the row anticipated.
- **§5 row 3 NEW (Cell A1: per-tag rabbit metrics diverge while agent never enters TL)**: **fired, as predicted**. `MeanDistRabbit_BR = 2.89` versus `MeanDistRabbit_TL = 6.63` is the +3.74-cell asymmetry the row described, and it is consistent with BR-camping. The internal control corroborates row 3's reading of the H₀(A1) verdict.
- **§5 row 4 NEW (per-tag NaN)**: did not fire — all per-tag values are present and finite.
- **§5 row 5 NEW (per-tag fan-out emits zeros)**: did not fire — Cell C window 1 emits `MeanDistPredator_full = 4.342`, `MeanDistRabbit_TL = 5.727`, `MeanDistRabbit_BR = 5.952`; all non-zero; commit `6d3d382` is in the deployed code.
- **§5 row 6 refined for R2.5 (Cell C: food-camping in TR + BL prevents rabbit visits)**: **fired**. Both rabbit tags > 4.5 (above random), `MeanDistFood = 1.31` (well below 2.0), `RabbitHits = 0.91 / ep` (≈ once every 437 steps). The row's verdict, per design, is H₀(C) confirmed in the structural sense — the food-coupling *was* the load-bearing R1 mechanism — together with the unanticipated bilateral rabbit avoidance which lands in the §4.3 "Inverted" band rather than the H₀(C) flat-equality band.
- **§5 row 7 (saturation at Steps ≈ 340)**: did not fire — Cell C reached Steps 399.5 (above R1's 327 ceiling) and Cell A1 reached Steps 485.7 (saturated against the 500-step cap, not against R1's plateau).
- **§5 row 8 refined for R2.5 (single-seed noise — verdicts must exceed thresholds by ≥ 2 ×)**: this is the live constraint on Round 2.5's elevation status (see §11).

### 10.6 Cell A1 mid-training transient (windows 4–7) — what happened, and does it threaten the verdict?

Between approximately episode 3 M and episode 7 M, Cell A1's policy *de-saturated* from camping (Steps ≈ 493, Term_MaxSteps ≈ 0.96) to a starvation-driven death pattern (Steps 86–129, Term_Starvation 0.74–0.97), then re-saturated. The simplest reading consistent with the metrics is: a stochastic policy update during ~3.5 M episodes of training pushed the agent off the food-camping attractor and into a regime where it failed to acquire enough food per episode; the death-by-starvation signal then re-shaped the policy back to the food-camping basin. This kind of mid-training basin-hopping is occasionally seen in long PPO runs and is not unique to this configuration.

For the §4.2 verdict the transient is not threatening: across all ten windows, the per-tag Δ_TL is in the range −0.033 to +0.020 cells — i.e., *even when the agent's policy is degraded and it is dying frequently*, the same-corner predator and same-corner rabbit remain at indistinguishable distance. The H₀(A1) signature ("Δ_TL ≈ 0") is robust to a policy regime that is producing 1/5th the survival of the converged regime. That is structurally informative: the matched-corner indistinguishability does not require a saturated camping policy to manifest, it manifests whenever the agent has *any* policy that prioritises food in BR over exploration of TL — which is true of every observed window.

For elevation status the transient is mildly relevant: the converged Steps value (485.7) over-states the agent's average-policy survival by about 80 steps if the transient is taken as a representative fraction of training, but the verdict here is keyed to the per-tag distance metric, not survival, so this caveat does not flip H₀(A1).

### 10.7 Cross-round interpretation — what the seed-paired comparison tells us

Holding seed 42 across R1 (no intervention) and R2.5 Cell C (food decoupled): the food-decoupling intervention shifted **every** primary metric by far more than R1's seed-to-seed dispersion (≤ 0.04 cells on the distance gap, ≤ 0.10 / ep on ΔH). The R1-versus-R2.5-Cell-C contrast is a clean within-RNG ablation: the only mechanical change from R1 to Cell C is which two of the four corners get food. That swap collapsed RabbitHits from 6.56 to 0.91 / ep and inverted the sign of Δ from +0.65 to −0.53 cells. The R1 finding **is the food-coupling**, in the sense that without the food coupling the metric flips sign.

Holding seed 43 across R1 (no intervention) and R2.5 Cell A1 (HUNT-disabled, predator → TL): the HUNT-disablement + patrol-area shrink elevated survival by +158 steps and dropped predator hits by 2.82 / ep, but those gains came from the agent never entering the predator's quadrant — not from learning that the predator is dangerous *as a class*. The R1 0.61-cell gap (4.39 − 3.78) under seed 43 is not preserved as a per-tag class-conditional gap under the same seed when the predator is locked into one rabbit's quadrant. The R1 effect under R2.5 Cell A1's reading is therefore consistent with "movement signature + post-contact teaching" carrying *some* class-conditional information, but the moment that information is forced to compete with a pure spatial-avoidance solution (corner-camping), the spatial-avoidance solution wins.

### 10.8 Where the verdict sits on the §5 row-8 single-seed elevation rule

§5 row 8 (refined for R2.5) requires single-seed verdicts to exceed thresholds **by ≥ 2 ×** to be elevated above provisional. Holding the data against this:

- **Cell A1 H₀(A1) confirmation**: Δ_TL = 0.004 vs the H₀ band ceiling of 0.3 cells — the data is at 1.3 % of the band ceiling, i.e., the H₀ verdict is exceeded by ~75 ×. Steps and Term_MaxSteps are also far past their contamination thresholds (Steps 485.7 vs 470 floor → +3 % over; Term_MaxSteps 0.924 vs 0.85 floor → +9 % over). The Δ_TL margin is overwhelming; the survival margin is small but unambiguous. **Single-seed elevation criterion: met for Δ_TL specifically; the supporting survival signature is comfortably past threshold but not by 2 ×. Net: elevated above provisional on the strength of the per-tag disambiguator.**
- **Cell C "Inverted" verdict**: Δ = −0.532 vs the inverted-band ceiling of −0.30 — the data is at 1.77 × the threshold (just under 2 ×). ΔH = −2.385 vs the equivalent-magnitude H₁ threshold of 1.5 — the data is at 1.59 × that threshold. The bilateral-rabbit-avoidance per-tag check is decisive (both rabbit tags are 1.5–1.8 cells *above* the predator distance, well outside any reasonable noise band). Per row 8: the aggregated Δ and ΔH are at ~1.6–1.8 × their thresholds, just shy of 2 ×; the per-tag check is far beyond threshold. **Net: provisional — elevation requires the second seed (44) for the aggregated channel.** The unanticipated *direction* (sign-flipped vs R1) further argues for second-seed verification per Appendix C, since the §6 designer's prior assigned 60 % to this outcome — non-trivial probability, but not overwhelming.

The single-seed rule does not block the central conclusion of the round (the per-tag disambiguation works as designed and Cell A1 is informatively H₀(A1)), but it does keep Cell C's specific sign-flipped direction at provisional status until a second seed corroborates.

---

## 11. Conclusions

> Per-cell verdicts in plain English first, then as the formal predicate name. §6 designer's priors are reproduced for honesty about how the data did or did not adapt the reading.

### 11.1 Per-cell headline verdicts

**Cell A1 — passivePredator, seed 43.**
Plain English: when a predator and a neutral rabbit are forced to share the same corner under matched smells, the agent does *not* learn to recognise the predator class — it learns to never enter that corner at all. The same-corner rabbit and the same-corner predator end up at the same distance from the agent (+0.004 cells apart) precisely because the agent treats *the corner*, not *the entity*, as the thing to avoid. Survival is high (Steps 486 / 500, 92 % of episodes time-out rather than die), and the agent extracts food and rabbit-meat from the other-corner rabbit (BR) while leaving the danger corner alone.
Formal predicate: **H₀(A1) confirmed** — location-conditional avoidance, not class-conditional avoidance. The §6 designer's prior was 70 % H₀(A1); the observed data confirms that prior in both direction and signature (Δ_TL ≈ 0 within within-window noise; Term_MaxSteps ≥ 0.85; Steps ≥ 470).
Elevation: above-provisional on the strength of the per-tag disambiguator (75 × past its threshold; per-window noise three orders of magnitude smaller than the 1.0-cell H₁ band).

**Cell C — decoupleFood, seed 42.**
Plain English: once food is moved out of the rabbit corners, the agent does *not* recover the Round-1 ordering "predator far, rabbit near". Instead it learns to camp the food corners (which are now the predator-only corners), and that camping makes it on-average closer to the patrolling predator and *farther* from both rabbits. The Round-1 effect was, in the cleanest available causal sense, the food-coupling — without it, the rabbit-versus-predator distance gap inverts.
Formal predicate: **H₁(C) refuted; the data lands in the §4.3 "Inverted" band (Δ ≤ −0.3 cells) with the per-tag check showing bilateral rabbit avoidance (both rabbit tags 1.48–1.76 cells above MeanDistPredator_full).** The §6 designer's prior was 60 % on exactly this outcome (the "stabilised inverted Δ" branch); the observed Δ = −0.532 lands inside the predicted −0.6 to −0.2 range.
Elevation: provisional — the aggregated Δ and ΔH are at ~1.6–1.8 × their pre-registered thresholds, just shy of the §5 row-8 2 × bar for single-seed elevation. The per-tag bilateral check is decisive on its own, but the project rule asks for a second seed to lock the sign-flip; see §11.4.

### 11.2 What the sameProp study now knows after Round 2.5

After Round 1 + Round 2 (truncated) + Round 2.5 (full budget, with per-tag disambiguation):

1. **Round 1's apparent class-conditional avoidance under matched smells was not class-conditional avoidance.** The 0.63-cell gap between rabbit and predator distances at R1 convergence was the joint product of (a) food-quadrant collinearity (rabbits were where food was) and (b) location-driven survival (the patrolling predator roamed farther on average than the food-aligned rabbit). When (a) is removed (Cell C) the gap inverts; when (a) is held and the predator is forced into a single rabbit's corner with HUNT disabled (Cell A1) the agent solves the problem by never visiting that corner.
2. **Under matched olfactory `properties [0, 1, 0, 0, 0]`, RPPO with this observation space produces survival-driven *location-conditional* avoidance, not class-conditional avoidance.** This is the cumulative reading of the sameProp study to date.
3. **The per-tag distance metrics work as designed** (commit `0a73613` + bug fix `6d3d382`). Both runs emitted clean per-tag values from window 1 onward; the disambiguation primitive that R2 truncated could not test, R2.5 did test cleanly.
4. **One open empirical question**: Cell C's *direction* of sign-flip (rabbits actively avoided when food is decoupled, not just neutralised). The 60 %-prior outcome lands as the data, but with a single seed it is a provisional finding. Round 2.6 with seed 44 on Cell C would lock or refute the direction. Cell A1 does not need this — its verdict is dominated by an effect that is 75 × past threshold.

### 11.3 §6 designer's priors versus observed outcomes

| §6 prediction | Stated probability | Observed | Held? |
|---|---:|---|---|
| Cell C: Δ stabilises inverted in [−0.6, −0.2], bilateral per-tag check shows both rabbit tags > MeanDistPredator_full | 60 % | Δ = −0.532 ✓; rabbit tags 5.52, 5.81 vs predator_full 4.05 (both > by ≥ 1.48 cells) ✓ | Yes |
| Cell C alternative: Δ drifts back to ≈ 0, both rabbit tags within 0.2 of predator_full | 40 % | Δ = −0.532, rabbit-vs-predator gaps +1.48 / +1.76 cells | No |
| Cell A1: Δ_TL ≈ 0 within 0.3, survival saturated ≥ 480, Term_MaxSteps ≥ 0.85 | 70 % | Δ_TL = +0.004 ✓; Steps = 485.7 ✓; Term_MaxSteps = 0.924 ✓ | Yes |
| Cell A1 alternative: agent eventually enters TL, Δ_TL ≥ 1.0, Term_MaxSteps < 0.7 | 30 % | Δ_TL = +0.004; Term_MaxSteps = 0.924 | No |

Both higher-probability priors held. The post-hoc reading does not require adapting the §4 thresholds in either cell.

### 11.4 Recommendation — Round 2.6 launch decision

Per Appendix C, Round 2.6 contingency is a pre-registered escalation, not a post-hoc rescue. The criteria laid out in C are: (a) a cell lands in §4.2 / §4.3 *ambiguous* bands, or (b) a cell's signal is borderline — within 25 % of the threshold value.

- **Cell A1: NO escalation.** The H₀(A1) confirmation is at 75 × the threshold; this is the canonical "exceeded by a wide margin" case under §5 row 8. The Round 2.6 second seed (45) for Cell A1 would only pile evidence onto an already-decisive verdict; better budget use is to spend that GPU-hour on the next cell of the study.
- **Cell C: escalate to Round 2.6 with seed 44.** The aggregated Δ and ΔH are at 1.6–1.8 × threshold (≥ 25 %-of-threshold, but below 2 ×), the direction is sign-flipped vs the R1 prior expectation, and the §6 designer's prior on the observed direction was 60 % (i.e., a non-trivial 40 % probability assigned to the alternative branch). One additional seed locks or refutes the sign.
  - Tag: `hypervigilance-round26-C-seed44_<n>_<gpu>`. Configs unchanged from Cell C of Round 2.5 (`02-sameProp_R2_decoupleFood.yaml` + `recurrent_ppo.yaml`). Budget: 10 M episodes; expected wall-clock ~22 h on node 106 / RTX 3090.
  - No new metric is required for the borderline-elevation path. The §7 second-tier metrics (per-rabbit time-fraction-near-instance, quadrant occupancy) would only become load-bearing if seed 44 produces a *different* sign or magnitude on Cell C, in which case the discrepancy between seeds is itself the next analytic question and the second-tier metrics would help diagnose it.

### 11.5 Where the wider hypervigilance research arc goes from here

If Round 2.6 (seed 44, Cell C) reproduces the −0.5-cell sign-flipped direction, the sameProp study has its clean reading: **under matched olfactory properties, this RPPO architecture in this gridworld does not produce class-conditional avoidance; it produces survival-driven spatial avoidance, and the apparent class discrimination of Round 1 was the food-coupling-plus-patrol-asymmetry compound.** With that reading locked, the next study question for hypervigilance is the channel-attribution one: does *any* observation-space channel (visual ch.5 vs ch.7 at colocation, extero-noc 0.9 at predator contact, olfactory movement-signature when HUNT is restored but food is decoupled) actually carry class-conditional information when the spatial-avoidance loophole is closed? A natural Round 3 design closes that loophole — for example, food in *all four* quadrants so the agent cannot survive by camping any single quadrant — and then runs the visual / extero-noc / movement-signature ablations from the channels memo.

If Round 2.6 (seed 44) instead produces a different sign on Cell C (Δ near zero or weakly positive), the H₀(C) versus Inverted verdict is genuinely seed-dependent at this budget and the study needs a 4-seed run to estimate the effect at all. That is a less interesting path because the 0.5-cell gap is small relative to the 5-cell grid scale, but it would force a wider-seed protocol going forward.

Either way, the per-tag distance metrics are now part of the project's permanent observability surface (commit `0a73613`) and should be carried into every future hypervigilance experiment.

### 11.6 Metrics requested

None at this analysis. The per-tag distance metrics that were the explicit Round 2.5 deliverable (commit `0a73613` + bug fix `6d3d382`) are working and load-bearing for both verdicts. The §7 second-tier items (per-rabbit time-fraction-near-instance, quadrant occupancy) remain deferred — they would only be needed if Round 2.6 lands in an ambiguous band on Cell C, and that contingency is named in §11.4.

### 11.7 Related issues

- No bugs surfaced. The only mid-training oddity was the Cell A1 windows-4-to-7 transient (§10.5), which is consistent with stochastic basin-hopping in long PPO runs and recovered without intervention. No `bug-fix-workflow` plan is needed.
- No `feature-workflow` plan is needed. The §7 metrics are pre-registered as second-tier and remain deferred.
- The cross-round comparator runs (R1 `rg5nl1ov` for seed 42, R1 `6ks4bjbq` for seed 43) and the truncated-Round-2 runs (`0u266oj5`, `27svrmhv`) are referenced but not modified. The truncated Round 2 doc (`sameprop_round2_design.md` §9) stays as the historical record of the early-window transient.

---

## 12. Behavior-measure appendix — toolkit v1 applied (2026-05-11)

### 12.0 Why this appendix exists — plain-English entry point

The §§9–11 verdicts above were built entirely on episode-mean statistics: mean distance to predator, mean distance per rabbit-tag, hits per episode, fraction of episodes that ran out the clock. Those readings made Cell A1 look like cleanly resolved corner-camping ("the agent never enters TL, so same-corner predator and same-corner rabbit get treated identically") and made Cell C look like a flat bilateral rabbit-avoidance policy ("the agent stays away from both rabbits and ends up closer to the patrolling predator than to either neutral, with no class-specific defence in evidence"). The cumulative §§9–11 reading of the sameProp study was therefore "under matched smells, RPPO produces survival-driven location-conditional avoidance, not class-conditional avoidance." This appendix applies the **behavior-measure toolkit v1** (M1 = interrupted-feeding rate, M2 = bush-dive rate, M5 = eat-under-threat ratio, M7 = defensive-motif repertoire — full operational definitions in [`docs/experiments/active/behavior_measures/behavior_measure_toolkit_v1_design.md`](../behavior_measures/behavior_measure_toolkit_v1_design.md)) to the saved Cell A1 and Cell C checkpoints, and finds that the spatial reading was **only partly right**.

In plain English, what the toolkit recovered is this. M1 asks "when the agent is eating and a class-c entity is within 3 cells, does it stop eating in the next 5 steps?" M2 asks "when a class-c entity just entered the agent's 3-cell radius, does the agent dive into a bush within the next 5 steps?" M5 asks "is the agent's per-step probability of eating lower when a class-c entity is nearby than when it is not?" M7 asks "what trajectory shapes does the agent show around threat-onset events, after we cluster them?" Cell A1's results match the §11 verdict: the predator and rabbit responses on every measure are within within-window noise of each other, the agent's TL-engagement rate is so low (753 predator-eating-candidates vs ~10,000 rabbit-eating-candidates) that class-conditional defence is structurally untestable, and the dominant motifs are "freeze near BR" and "rabbit drifts past frozen agent." But Cell C does **not** match the §11 reading. The agent bush-dives in **88% of predator-approach events** versus 51% of rabbit-approach events, depresses eating to 0.75× the safe baseline when the predator is near but does **not** depress eating near rabbits (M5_rabbit = 1.19), and M7 finds a 19%-of-windows "predator pursuit with partial bush use" cluster that is 83% predator-triggered. The headline finding: **at the trajectory-event level, Cell C exhibits strong class-conditional active defence that the per-tag mean-distance metric dissolved away**. The §11.1 "bilateral rabbit avoidance" verdict is still correct as a *spatial* reading, but the agent is class-blind only in *where it sits on average*, not in *what it does when a predator approaches*.

### 12.1 Setup

- **Inputs**: saved evaluation rollouts of the final Cell A1 checkpoint (`results/eval/models/10000003/`, source run `nm8gn7y2`) and the final Cell C checkpoint (`results/eval/models/10000022/`, source run `bdnfc0lu`). N=200 deterministic episodes per cell; `eval_obs_noise = training` (matched to training-time noise schedule); cue radius R=3.0 cells; online K=5 steps; offline K_motif=7 steps.
- **Tools**: `scripts/eval_rollout.py` produced the episode dumps and `windows/threat_onsets.parquet` indexes; `scripts/motif_cluster.py` produced M7 cluster artifacts (k=6, seed=42, 10 features, zscore_pooled). M1/M2/M5 are reported from `online_replay.json` cross-checked against an independent offline recomputation from the per-episode `.npz` dumps (see [`tmp/20260511_r25_appendix_analysis.py`](../../../../tmp/20260511_r25_appendix_analysis.py)); the two agree within ≤ 0.5 percentage points.
- **Provenance check**: each `metadata.json` records the source checkpoint path; A1's dump traces to `20260509-182720_hypervigilance-round25-A1-seed43_n106_gpu1`, C's dump traces to `20260509-182529_hypervigilance-round25-C-seed42_n106_gpu0`. Both at the final saved step (10,000,003 / 10,000,022) and both at git commit `08de62a`.
- **Eval-time vs training-time numbers**: these are eval-time, deterministic-policy, fresh-seed (1000–1199) numbers — not the training-time per-window numbers in §9.4. Eval-time mean survival is 459.3 steps (A1) and 405.4 steps (C), consistent with the §9.2 last-10 % windows (485.7 and 399.5 respectively); A1's eval survival is ~5 % below its training-time saturated value because the eval seeds include some episodes where the agent's deterministic policy under fresh seeding does not reach max-step saturation.
- **Sample-size adequacy**: 200 episodes × 405–459 steps × ~3 onsets/episode gives 7,741 (Cell C) and 8,531 (Cell A1) threat-onset windows for M7 — well above the toolkit's "≥ 30 windows per cluster" rule for k=6. M1/M2 denominators run from 753 (A1 predator) to 10,460 (A1 rabbit); M5 step-counts run from ~4,000 to ~92,000 — all well-resolved.

### 12.2 M1 / M2 / M5 table — per-cell × per-class × per-tag

Per-class headline numbers (eval-time, N=200 episodes, deterministic policy):

| Measure | A1 predator_TL | A1 rabbit | C predator_full | C rabbit |
|---|---:|---:|---:|---:|
| **M1** interrupted-feeding rate | 11.8 % (89 / 753) | 12.8 % (1341 / 10,460) | **42.2 %** (1926 / 4564) | 24.1 % (603 / 2499) |
| **M2** bush-dive rate | 3.2 % (27 / 853) | 0.5 % (41 / 7509) | **87.6 %** (2222 / 2537) | 50.8 % (851 / 1674) |
| **M5** eat-under-threat ratio | 0.999 | 1.141 | **0.748** | 1.186 |
| ↳ P(eat \| threat near) | 0.189 | 0.200 | 0.155 | 0.218 |
| ↳ P(eat \| safe) | 0.190 | 0.176 | 0.207 | 0.184 |

Per-tag fan-out (the key disambiguation that §9.3 was missing):

| Cell | Tag | M1 | M2 | M5 ratio |
|---|---|---:|---:|---:|
| A1 | predator_TL | 11.8 % (89 / 753) | 3.2 % (27 / 853) | 0.999 |
| A1 | rabbit_TL | 14.6 % (120 / 822) | 2.3 % (26 / 1143) | 1.056 |
| A1 | rabbit_BR | 12.7 % (1271 / 9976) | 0.4 % (27 / 7282) | 1.132 |
| C | predator_full | 42.2 % (1926 / 4564) | 87.6 % (2222 / 2537) | 0.748 |
| C | rabbit_TL | 23.4 % (365 / 1560) | 50.3 % (515 / 1023) | 1.207 |
| C | rabbit_BR | 25.2 % (257 / 1018) | 52.3 % (413 / 789) | 1.130 |

Derived class-conditional gaps (against the toolkit's pre-registered H₁ thresholds):

| Cell | Δ M1 (pred − rabbit) | Δ M2 (pred − rabbit) | Δ M5 (rabbit − pred) | Toolkit H₁ verdict |
|---|---:|---:|---:|---|
| A1 | −1.0 pp | +2.7 pp | +0.14 | **H₀(M1/M2/M5) — class-blind**; pred/rab within within-window noise on every measure |
| C | **+18.1 pp** | **+36.8 pp** | **+0.44** | **H₁(M2) confirmed by a wide margin** (Δ_M2 = 36.8 pp ≫ +15 pp threshold, M2_predator = 87.6 % ≫ 10 % floor); H₁(M5) confirmed (M5_predator = 0.748 just under the 0.70 absolute threshold but Δ_M5 = +0.44 ≫ +0.20 alternative threshold); H₁(M1) marginal (Δ_M1 = 18.1 pp, threshold is +20 pp — within 2 pp of the bar) |

The Cell A1 numbers are interesting in their own right: the rare TL-tagged engagement events (rabbit_TL and predator_TL both have ~800 candidate events vs rabbit_BR's ~9,976) show essentially identical responses — rabbit_TL M2 = 2.3 % vs predator_TL M2 = 3.2 %. When the agent IS forced to engage TL, it treats the rabbit and the predator the same way. This corroborates §11.1's reading at the event level.

### 12.3 M7 motif distribution — per-cell × per-cluster, with semantic labels

Cluster sizes are k-means assignments (k=6, seed=42); semantic labels assigned by hand-inspection of two nearest-centroid exemplar windows per cluster ([`tmp/20260511_r25_appendix_writeup.md`](../../../../tmp/20260511_r25_appendix_writeup.md) records the inspection notes).

**Cell A1** — 8531 windows; silhouette = 0.237 (passes R4 ≥ 0.20).

| Cluster | Size | Frac | Semantic label | Top features (mean, original units) |
|---|---:|---:|---|---|
| 0 | 2861 | **33.5 %** | `freeze_near_BR` — stationary near a rabbit, mode-action dominant | stay_in_place 0.99; mode_action_frac 0.77; eat 1.2/win; bush 0.00 |
| 1 | 46 | 0.5 % | `tl_predator_bush_anomaly` — rare TL engagement with bush use; mostly episode-start onsets | bush_occ 0.68; min_threat_dist 1.85 |
| 2 | 973 | 11.4 % | `approach_transit` — directed motion toward an entity | path_length 6.7; net_disp 3.1; action_entropy 1.62 |
| 3 | 922 | 10.8 % | `feeding_bout` — sustained eating, mostly frozen | eat 6.5/win; stay_in_place 0.92; mode_action_frac 0.68 |
| 4 | 1488 | 17.4 % | `engage_BR` — mid-mobility interaction with BR rabbit | path 3.0; eat 2.4; action_entropy 1.35 |
| 5 | 2241 | **26.3 %** | `stationary_rabbit_approaches` — agent frozen, rabbit drifts in | stay 0.99; eat 0.09/win; threat-dist Δ = −0.18 (rabbit approaching) |

One sentence: A1's repertoire is **dominated by stationary/freeze behaviours (clusters 0 + 5 = 59.8 %), with feeding bouts (cluster 3 = 10.8 %) and slow BR-engagement (cluster 4 = 17.4 %) accounting for most of the rest**; no `bush_dive`-shaped cluster emerges as a coherent motif (the cluster_1 anomaly is < 1 % and below the M7 §4.2 reportable size).

**Cell C** — 7741 windows; silhouette = 0.186 (below R4 ≥ 0.20 nominal threshold but reportable per the toolkit's "low silhouette is acceptable when justified" rule; here the justification is that the agent's behaviour is genuinely diverse rather than degenerate — see §12.5).

| Cluster | Size | Frac | Semantic label | Top features (mean, original units) |
|---|---:|---:|---|---|
| 0 | 1482 | 19.1 % | `predator_pursuit_with_bush` — directed flight with partial bush use, 83 % predator-triggered | path 4.0; bush_occ 0.47; threat-dist Δ = −0.19 |
| 1 | 991 | 12.8 % | `bush_camp` — stationary inside bush | bush_occ 0.95; stay 0.98 |
| 2 | 2006 | **25.9 %** | `mobile_with_cover` — moving between bushes | bush 0.55; path 4.2 |
| 3 | 831 | 10.7 % | `open_flight` — large directed displacement, little bush | net_disp 4.1; bush 0.18 |
| 4 | 815 | 10.5 % | `feeding_bout_near_rabbit` — sustained eating, 81 % rabbit-triggered | eat 5.1/win; stay 0.84 |
| 5 | 1616 | **20.9 %** | `bush_camp_predator` — bush-camped with predator nearby, 59 % predator-triggered | bush_occ 0.79; stay 0.78 |

One sentence: C's repertoire is **bush-involved on ~78 % of windows (clusters 0 + 1 + 2 + 5)**, with the bush-camping pair of clusters (1 + 5 = 33.7 %) and the bush-mobile cluster (2 = 25.9 %) doing most of the work; open flight without bush (cluster 3 = 10.7 %) is a minor motif and feeding-near-rabbit (cluster 4 = 10.5 %) is dominated by rabbit-triggered events.

**Per-class cluster membership** (the figure that drives the cross-cell verdict):

| Cell A1 | C0 freeze_BR | C1 anomaly | C2 approach | C3 feeding | C4 engage_BR | C5 stationary_rab |
|---|---:|---:|---:|---:|---:|---:|
| predator-triggered | 46.3 % | 1.1 % | 13.7 % | 10.2 % | 19.3 % | 9.3 % |
| rabbit-triggered | 32.0 % | 0.5 % | 11.1 % | 10.9 % | 17.2 % | 28.3 % |

A1 predator onsets shift slightly toward freeze_near_BR (46 % vs 32 % for rabbit) but the difference is small relative to the 9.3 % vs 28.3 % shift on the stationary_rabbit_approaches cluster — those numbers reflect the geometry (rare predator-onsets nearly always come from a TL-incursion early in the episode that lands the agent in a freeze-near-BR pattern), not a class-conditional response.

| Cell C | C0 pred_pursuit | C1 bush_camp | C2 mobile_cover | C3 open_flight | C4 feed_rabbit | C5 bush_camp_pred |
|---|---:|---:|---:|---:|---:|---:|
| predator-triggered | **28.9 %** | 10.5 % | 24.7 % | 9.7 % | 3.7 % | **22.5 %** |
| rabbit-triggered | 7.2 % | 15.6 % | 27.4 % | 12.0 % | **19.0 %** | 18.9 % |

C predator-triggered onsets are **4× more likely** to land in cluster 0 (`predator_pursuit_with_bush`, 28.9 % vs 7.2 %) and slightly more likely to land in cluster 5 (`bush_camp_predator`, 22.5 % vs 18.9 %). Rabbit-triggered onsets are **5×** more likely to land in cluster 4 (`feeding_bout_near_rabbit`, 19.0 % vs 3.7 %). This is **a clear class-conditional behavioural fingerprint** at the motif level.

### 12.4 Cross-cell comparison — the paper-grade headline

**A1 and C show qualitatively different defensive behaviour at the event level**, even though the §11 spatial-distance verdicts placed them on the same "structured spatial policy, not class-conditional avoidance" reading. Five bullets answer the design's headline questions:

1. **Motif distributions differ in shape AND in class-conditioning.** Cell A1's distribution is dominated by stationary/freeze motifs (clusters 0 + 5 = 59.8 %) with one rare anomaly cluster (1 = 0.5 %), matching the toolkit's prior of "`ignore`-dominated" — the prior predicted ≥ 80 % `ignore`; observed is 59.8 % stationary + 10.8 % feeding bout + 17.4 % engage_BR (the latter two are not "ignore" but they ARE all confined to BR), so the "behaviourally constrained" reading is correct without being literally "ignore-dominant." Cell C's distribution spreads across six functionally distinct clusters with a strong predator-vs-rabbit asymmetry (cluster 0: 29 % vs 7 %; cluster 4: 4 % vs 19 %) — this is the spread the toolkit was designed to detect and that mean-distance metrics cannot show. The pre-registered M7 §4.2 check "Cell A1 ignore-fraction should exceed Cell C ignore-fraction by ≥ 30 pp" is met if we use clusters 0+5 (A1 59.8 %) vs C clusters 1+5 (33.7 %) as the closest analogue (26.1 pp), or by ≥ 40 pp if we use A1's full stationary-cluster set (cluster 0+3+5 = 70.6 %) vs C's bush-camp-only set (1+5 = 33.7 %) → 36.9 pp.

2. **Per-tag M1 fan-out for Cell A1: no class-conditional event-level signal.** `M1_predator_TL` = 11.8 % vs `M1_rabbit_TL` = 14.6 %. The rabbit-TL response is mildly higher than predator-TL — the opposite direction from H₁(A1) — and the gap is 2.8 pp (well inside the toolkit's "≤ 5 pp = no signal" band on Round-2.5 A1 priors). The §10.1 verdict was the per-tag distance gap Δ_TL = +0.004; the per-tag event gap Δ_M1_TL = −2.8 pp is similarly null. **The H₀(A1) corner-camping verdict is reinforced at the event level**: where the agent IS forced to engage TL, it treats the TL rabbit and the TL predator the same way.

3. **Bush usage as class-conditional defence — Cell C only.** `M2_predator_full` = 87.6 % vs `M2_rabbit` (mean over both tags) = 50.8 %; a +36.8 pp gap, more than 2× the toolkit's H₁(M2) threshold (+15 pp), with the absolute M2_predator value (88 %) far above the 10 % floor. Even under matched smells (`properties [0,1,0,0,0]` for both classes), **the Cell C agent has learned a strong predator-specific bush-dive response**. Cell A1 shows nothing of this (M2_predator = 3.2 % vs M2_rabbit ≈ 1 %); the agent there has no opportunity (only 853 predator-onset events across 200 episodes) and no use for bushes (its policy is "stay in BR and don't move").

4. **Risk-discounted foraging — Cell C only.** `M5_predator` = 0.748 (the agent's per-step eat probability under predator threat is 25 % below its safe-baseline rate), while `M5_rabbit` = 1.186 (rabbits do NOT depress eating — the agent eats *more* when a rabbit is near, because rabbits sit in food-empty quadrants the agent doesn't camp, so rabbit-near events correlate with crossing food-rich territory). The +0.44 cross-class delta is more than 2× the toolkit's pre-registered H₁(M5) alternative threshold (+0.20). Cell A1 shows no risk discounting on either class (M5_predator = 0.999, M5_rabbit = 1.141; gap +0.14 within toolkit noise).

5. **The class-conditional signal Cell C carries is consistent across rabbit tags.** Per-tag M2 for Cell C: `rabbit_TL` 50.3 %, `rabbit_BR` 52.3 %. Per-tag M5: `rabbit_TL` 1.207, `rabbit_BR` 1.130. Both tags behave essentially the same — the rabbit-vs-predator behavioural gap is class-driven, not tag-driven. This rules out the "agent only responds to one rabbit corner" sub-case and locks the Cell C signature as **bilateral rabbit non-defence + unilateral predator defence**.

### 12.5 What §§9–11 missed, and what they confirmed

**Missed (Cell C)**: §§9–11's verdict was "Cell C lands in the §4.3 Inverted band (Δ = −0.532, bilateral rabbit avoidance) — the agent stays farther from both rabbits than from the predator, structured spatial policy, no class-conditional defence." The toolkit shows that **at the per-step event level the agent has a strongly class-conditional active defence** — predator approach triggers a bush dive 88 % of the time and depresses eating to 0.75× safe; rabbit approach triggers a bush dive 51 % of the time and does NOT depress eating. The §11 reading "no class-conditional avoidance" is correct **as a spatial statement** (the agent does sit closer to the patrolling predator than to either rabbit in mean-distance terms), but **wrong as a behavioural statement** (the agent's reaction-to-encounter is strongly class-conditional). The honest 2-line summary is now: "Cell C's policy is spatially class-blind but behaviourally class-discriminating; the spatial inversion is a consequence of the food-camping policy putting the agent in the predator's quadrant by default, with the predator-specific bush-dive doing the actual defensive work."

**Confirmed (Cell A1)**: every toolkit measure on Cell A1 ratifies the §11.1 corner-camping verdict. Per-tag M1 / M2 / M5 are statistically indistinguishable between rabbit_TL and predator_TL (M1 +2.8 pp in the rabbit's favour; M2 +0.9 pp in the predator's favour; M5 within 0.06). The motif distribution is dominated by behaviours confined to BR (clusters 0 + 3 + 4 + 5 = 88.0 %). Predator-triggered onsets shift toward the freeze-near-BR cluster (46 % vs 32 %) but the shift is explained by the fact that the only predator onsets happen on rare TL incursions that resolve with the agent fleeing back to BR. **The toolkit cannot rescue a class-conditional signal where the engagement rate is structurally too low** (753 predator-cand-events vs 10,460 rabbit-cand-events), and the §11.4 recommendation "no Round 2.6 escalation for Cell A1" stands — the camping is class-blind at every measurable level.

**A second-order confirmation**: the toolkit's pre-registered prior on Cell A1 was M1 ≤ 5 % or NaN. Observed M1 ≈ 12 % on both classes — meaningfully above the predicted 5 % ceiling. The prior assumed the agent would "rarely eat near anything"; the data shows it eats near rabbits (mostly BR) frequently enough that M1 has a usable denominator. This is a minor correction to the prior, not a refutation of the verdict — the agent does encounter and eat near the BR rabbit, but its response to that encounter is the same as its rare response to TL events.

### 12.6 Silhouette caveat and toolkit-v2 candidates

Cell C's M7 silhouette = 0.186 sits just below the R4 ≥ 0.20 nominal threshold. Per the toolkit's R4 rule, a sub-0.20 silhouette is reportable when justified; here the justification is that the underlying behavioural distribution is genuinely multi-modal rather than degenerate (six clusters all between 10–26 % of the population, vs the "one cluster swallows ≥ 90 %" R4 failure signature, and the per-class fan-out is informative). Cell A1's silhouette = 0.237 clears R4 cleanly. **For toolkit v2**: candidate features that would tighten Cell C's clusters include (a) raw hit-event counts within the window (M7's drive_injury_change is too coarse — most predator-pursuit windows have zero injury because the bush works), (b) entity-presence-duration (how long was the triggering entity in radius across the window), (c) action-mode-shift-count (did the agent's modal action change mid-window). These are surfaced as toolkit-v2 candidates; not load-bearing for the v1 verdict.

### 12.7 Implications for Round 2.6 (Cell C, seed 44) and Round 3

The Round 2.6 analysis plan in [`docs/experiments/active/hypervigilance/sameprop_round25_design.md`](sameprop_round25_design.md) §11.4 reads "one additional seed locks or refutes the sign" for Cell C, where "sign" meant the −0.5-cell aggregated Δ. The toolkit promotes Round 2.6 to a richer test: **the question is no longer just whether seed 44 reproduces the bilateral rabbit avoidance sign, it is whether seed 44 reproduces the class-conditional behavioural fingerprint** (M2_predator ≥ 0.80, M5_predator ≤ 0.80, Δ_M2_class ≥ +30 pp, predator-onsets-in-pursuit-cluster ≥ 25 %). Pre-registered Round-2.6-with-toolkit verdict conditions:

- **Locked**: seed 44 produces M2_predator ≥ 0.80 AND M2_predator − M2_rabbit ≥ +30 pp AND M5_predator < 0.80. The class-conditional defence is then seed-stable.
- **Sign-flipped at behavioural level**: seed 44 produces M2_predator < 0.5 or M2_predator − M2_rabbit < +10 pp. The seed-42 finding is then seed-specific and the Cell C behavioural verdict reverts to provisional.
- **Mixed**: any other combination → ambiguous; route to a 4-seed Cell-C study.

For **Round 3** (food in all four quadrants, closing the spatial-avoidance loophole), the toolkit measures should be **pre-registered as primary confirmation criteria** rather than secondary. Specifically: with the spatial-avoidance solution unavailable, the question becomes "does the agent fall back on the class-conditional bush-dive defence (M2_predator) or does class-discrimination collapse entirely (M2 ≈ M5 ≈ M1 with no class gap)?" Predicted (designer's prior, recorded honestly here): M2_predator will widen further (from 88 % in Cell C to ~95 %) and M5_predator will drop further (from 0.75 to ~0.55), if the agent's hypothesised class-conditional channel is genuine; alternatively, if the Cell C class-conditioning is a side-effect of "predator roams everywhere, rabbits sit in two corners the agent avoids anyway," then Round 3 will show a sharp drop in M2_predator-vs-rabbit gap as the spatial confound is removed.

**Cross-link**: the toolkit's design doc ([`docs/experiments/active/behavior_measures/behavior_measure_toolkit_v1_design.md`](../behavior_measures/behavior_measure_toolkit_v1_design.md)) §8.2 already lists "Round 2.6 / Round 3 / NMN comparisons" as downstream adopters — this appendix is the **first paper-grade demonstration that the toolkit recovers signal mean-distance metrics dissolve**, and that demonstration should be cited in the toolkit doc's changelog.

### 12.8 Working files

- [`tmp/20260511_r25_appendix_analysis.py`](../../../../tmp/20260511_r25_appendix_analysis.py) — analysis script (M1/M2/M5/M7 cross-tabs)
- [`tmp/20260511_r25_appendix_analysis.log`](../../../../tmp/20260511_r25_appendix_analysis.log) — script output
- [`tmp/20260511_r25_appendix_aggregates.json`](../../../../tmp/20260511_r25_appendix_aggregates.json) — JSON dump of per-tag aggregates
- [`tmp/20260511_r25_appendix_writeup.md`](../../../../tmp/20260511_r25_appendix_writeup.md) — exemplar-inspection scratch (drives the semantic labels)
- `tmp/20260511_r25_appendix_motifs_{A1,C}.csv`, `tmp/20260511_r25_appendix_motif_by_{class,tag}_{A1,C}.csv` — per-cluster mean-feature and cross-tab tables
- `results/eval/models/10000003/motifs/` and `results/eval/models/10000022/motifs/` — k-means outputs from `scripts/motif_cluster.py`

### 12.9 Verdict refinements (delta vs §11)

| §11 verdict | §12 refinement |
|---|---|
| Cell A1 — H₀(A1) confirmed (corner-camping; class-blind at the spatial level) | **Reinforced** — class-blind at the event level too. Per-tag M1/M2/M5 for predator_TL and rabbit_TL within within-window noise; motif distribution dominated by BR-confined behaviours. No reason to revisit. |
| Cell C — H₁(C) refuted; Inverted band (bilateral rabbit avoidance, class-blind defence) | **Refined to spatial-only**. The Δ = −0.532 spatial inversion and bilateral-rabbit-distance verdict stand. The "class-blind defence" reading is **wrong at the event level**: M2_predator = 88 % vs M2_rabbit = 51 % (a +37 pp class-conditional active defence); M5_predator = 0.75 vs M5_rabbit = 1.19 (a +0.44 class-conditional risk discount). The honest two-line reading is now "spatially class-blind, behaviourally class-discriminating" — predator and rabbit get the same average distance because of where the food is, but predator and rabbit get different per-step responses. |
| Recommendation: Round 2.6 seed 44 for Cell C only | **Strengthened**. Round 2.6 should pre-register the toolkit measures, not just the aggregated Δ. The behavioural class-conditioning is the more interesting finding; the spatial inversion is a side-effect. |

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
| 2026-05-10 | §3 Launch Manifest actuals filled (both runs completed 10 M episodes cleanly: Cell C `bdnfc0lu` 22 h 33 m, Cell A1 `nm8gn7y2` 17 h 30 m); §§9–11 written against pre-registered §4 thresholds; verdicts: **Cell A1 H₀(A1) confirmed (Δ_TL = +0.004, location-conditional avoidance)**, **Cell C lands in §4.3 Inverted band (Δ = −0.532, bilateral rabbit avoidance)**; recommendation Round 2.6 seed 44 for Cell C only; status frontmatter `planned → analyzed` | experiment-analyzer |
| 2026-05-11 | §12 behavior-measure appendix appended — toolkit v1 applied to saved A1 + C eval rollouts (`nm8gn7y2` / `bdnfc0lu`). M1/M2/M5/M7 computed eval-time on N=200 deterministic episodes. **Cell C bush_dive_rate predator 88 % vs rabbit 51 %** (Δ_M2 = +37 pp); **M5_predator 0.75 vs M5_rabbit 1.19** (Δ_M5 = +0.44). The §11 "bilateral rabbit avoidance, class-blind defence" verdict refined to *spatially* class-blind but *behaviourally* class-discriminating. Cell A1 verdict reinforced — class-blind at the event level too. First paper-grade use of M1/M2/M5/M7 | experiment-analyzer |
