---
title: "SameProp Round 2.6 — seed-lock of Round 2.5 with behavior-toolkit measures as primary"
topic: hypervigilance
status: planned
created: 2026-05-12
last_updated: 2026-05-16T13:21
phase: 2
wandb_tag: "hypervigilance-round26"
supersedes: []
---

# SameProp Round 2.6 — seed-lock of Round 2.5 with behavior-toolkit measures as primary

> **Status**: PLANNED — configs need a small `behavior_measures:` block backfill before the runner can launch (see §2.3 and §8). Auditor pass pending.
> **Author**: experiment-designer
> **Date**: 2026-05-12
> **Headline**: This is the seed-lock round. Round 2.5 single-seed verdicts on Cells C (decoupleFood, seed 42) and A1 (passivePredator, seed 43) need one more seed each to clear the project's elevation bar; the behavior-measure toolkit v1 (`verification_status: pass`) supplies the primary verdict criteria, with the Round-2.5 mean-distance measures retained as secondary cross-round comparators.
> **Related**:
> - Predecessor (single-seed result for both cells, with §12 toolkit appendix): [`sameprop_round25_design.md`](sameprop_round25_design.md).
> - Behavior-toolkit design doc (operational definitions of M1, M2, M5, M7): [`behavior_measure_toolkit_v1_design.md`](../behavior_measures/behavior_measure_toolkit_v1_design.md).
> - Toolkit implementation plan (`verification_status: pass`): [`behavior_measure_toolkit_v1_plan.md`](../../../develop/active/behavior/behavior_measure_toolkit_v1_plan.md).
> - The class-discriminating-defence finding R2.6 is partly confirming: [`docs/memory/memories/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md`](../../../../docs/memory/memories/hypervigilance/20260512_1428_sameprop_class_discriminating_defence_event_level.md).
> - Round 1 baseline (for cross-round anchoring of the new primary criteria): [`round1_relog_baseline_analysis.md`](round1_relog_baseline_analysis.md).

---

## 0. What this round is testing — plain-English entry point

Round 2.5 closed last week with two single-seed verdicts on the sameProp study, where a predator and a neutral rabbit emit identical smells and the question is whether the agent learns to tell them apart. In Cell C — where food was moved to the predator-only corners so the agent could no longer use food-seeking as a shortcut — the agent's average distance to rabbits actually became larger than its average distance to the predator (the "inverted" sign), and at the trajectory level the behavior-measure toolkit revealed that the same agent was diving into bushes 88% of the time when the predator approached but only 51% of the time for rabbits, and suppressing eating to 0.75× near the predator versus 1.19× near rabbits. The headline reframing was that the agent IS class-discriminating, but at the **event level** (defensive responses to threat encounters), not the **spatial-trajectory level** (mean distances over the whole episode). In Cell A1 — where the predator was stripped of its hunting behaviour and locked to one corner that already contained a rabbit — the agent gave up on the threat entirely and survived by camping the safe opposite corner (survival 486/500 steps; per-tag distance to the same-corner predator and same-corner rabbit was identical to four decimals).

Both findings come from one seed each. The project rule is that single-seed verdicts on marginal effects need a second seed before they leave "provisional." Round 2.6 runs each cell once more — Cell C on seed 44, Cell A1 on seed 45 — at the full ten-million-episode budget on node 106. The primary verdict criteria are upgraded: the toolkit's bush-dive rate and eat-under-threat ratio (M2 and M5) become the locked confirmation thresholds for Cell C; for Cell A1, the seed-lock question is whether corner-camping is a generic basin or a seed-specific local minimum that a different RNG might escape. Mean distances are retained as secondary cross-round comparators against Round 2.5 and Round 1. Designer's prior: ~80% Cell C reproduces the event-level discrimination; ~70% Cell A1 reproduces the corner-camping basin.

## 1. Research Question

Round 2.6 pre-registers four falsifiable predicates — one per cell, primary verdict (event-level for C, behavioural-stability for A1) plus the seed-stability of the Round-2.5 result.

> **H₁(C-event) — class-discriminating defence is seed-stable.** At convergence (last-10% window, eval-rollout protocol of [`behavior_measure_toolkit_v1_design.md`](../behavior_measures/behavior_measure_toolkit_v1_design.md) §3), seed-44 Cell C produces `M2_BushDiveRate_predator ≥ 0.80` AND `Δ_M2_class ≡ M2_predator − M2_rabbit ≥ +30 pp` AND `M5_EatUnderThreatRatio_predator < 0.80`. Per-tag rabbit_TL vs rabbit_BR agree within ±5 pp on M2 and ±0.10 on M5 (rules out single-instance artifacts).
>
> **H₀(C-event) — class-discriminating defence is seed-specific (Round 2.5 was a one-off).** Seed-44 Cell C produces `M2_predator < 0.50` OR `Δ_M2_class < +10 pp`. The Round-2.5 finding does not generalise; the §12 verdict reverts to provisional.
>
> **H₁(A1-stable) — corner-camping is a generic basin.** Seed-45 Cell A1 produces `Episode/Steps ≥ 470/500` AND per-tag `|Δ_TL| ≤ 0.30` cells AND `M2_BushDiveRate_*` near zero for both classes (M2_predator < 0.10 AND M2_rabbit < 0.10 — the camping signature is "agent rarely encounters threats because it rarely leaves the safe corner").
>
> **H₀(A1-escape) — corner-camping is seed-specific; seed 45 finds a different policy.** Seed-45 Cell A1 produces `Steps < 400/500` AND the agent visits TL meaningfully (`M1_predator > 0.05` — i.e., agent eats with predator nearby in TL). Under this outcome, A1's verdict is no longer about corner-camping; the analyzer reports the new policy in its own right (see §5).

Verdicts on the two cells are independent — a clean refutation of one does not depend on the other.

## 2. Experimental Design

### 2.1 What is identical to Round 2.5

This section is short by design. Round 2.6 inherits the configs, the noise preset, the env definition, the agent architecture, the step budget, and the cross-round comparability rules from Round 2.5. **See [`sameprop_round25_design.md`](sameprop_round25_design.md) §2 for the full pinned-factors table** (4 hiding_predators, 2 tagged rabbits, olfactory `properties [0,1,0,0,0]` matched on rabbit + active predator, `perceptual_noise.enabled: false`, `random_start_pos: true`, sensors, body, 128 parallel envs, max_steps 500, RPPO via `configs/models/recurrent_ppo.yaml`).

The independent variables (Cell C ablates food-quadrant coupling; Cell A1 ablates predator HUNT + restricts patrol area) are identical to Round 2.5; the configs `02-sameProp_R2_decoupleFood.yaml` and `02-sameProp_R2_passivePredator.yaml` are reused unchanged at the environment-level. Step budget is **10,000,000 episodes** per run.

### 2.2 What is different from Round 2.5

| Delta | Round 2.5 | Round 2.6 |
|---|---|---|
| Seeds | Cell C = 42 ; Cell A1 = 43 | **Cell C = 44 ; Cell A1 = 45** |
| WandB tag prefix | `hypervigilance-round25-` | `hypervigilance-round26-` |
| Primary verdict measures | Aggregated `MeanDistPredator − MeanDistRabbit` (Δ) + per-tag Δ_TL | **Toolkit M2 + M5 (event-level)** for Cell C; **toolkit M2 + Steps + per-tag Δ_TL** for Cell A1 |
| Behavior-toolkit pipeline | Run **offline** post-hoc on saved checkpoints (eval-rollout + motif_cluster scripts) | Same offline pipeline on Round 2.6 saved checkpoints. Online M1/M2/M5 keys can ride along during training if the configs carry a `behavior_measures:` block, but the primary verdict is computed offline at the final checkpoint using the same protocol that produced the R2.5 §12 numbers. |

### 2.3 Config readiness

| Config | Tag fields | `behavior_measures:` block |
|---|---|---|
| `configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml` | ✓ verified — rabbits `TL`/`BR`, predator `full` (file read 2026-05-12, lines 128, 137, 156) | **✗ MISSING — flagged in §8** |
| `configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml` | ✓ verified — rabbits `TL`/`BR`, predator `TL` co-located with TL rabbit (file read 2026-05-12, lines 148, 157, 181) | **✗ MISSING — flagged in §8** |

The env-level configs are correct as-is; the tag assignments that the per-tag pipeline depends on are unchanged from Round 2.5. The toolkit's primary verdict can be computed **offline** against the final R2.6 checkpoints using the existing `scripts/eval_rollout.py` + `scripts/motif_cluster.py` pipeline (the same two scripts that produced the R2.5 §12 numbers from saved checkpoints `10000003` and `10000022`). The offline path **does not require the configs to carry a `behavior_measures:` block at training time**, because the scripts read the env config to set up the eval rollout but the protocol parameters (R=3.0, K=5, K_motif=7, etc.) are baked into the scripts/toolkit-design defaults. **However**, the project's no-fallback-defaults rule (per `CLAUDE.md`) and the toolkit-plan §10 launch-gate explicitly require the `behavior_measures:` block to live in the env config when M1/M2/M5 are wanted as online training-time keys. See §8 for the remediation route — this designer **does not edit** the configs; remediation belongs to a developer follow-up before the runner launches if online keys are desired. If the user authorises launching with offline-only post-hoc evaluation, the existing configs are launchable as-is (Round 2.5's configs were).

## 3. Launch Manifest

**Hardware**: Node 106, 2 GPUs (cuda:0, cuda:1). 1 cell per GPU, 1 seed per cell. Total: 2 runs.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | running | C — decoupleFood | `hypervigilance-round26-C-seed44_n106_gpu0` | hypervigilance | prod | 44 | 106 | cuda:0 | 2026-05-12T17:01:10 | s3k03eua | logs/20260512_170110.log |
| 2 | running | A1 — passivePredator | `hypervigilance-round26-A1-seed45_n106_gpu1` | hypervigilance | prod | 45 | 106 | cuda:1 | 2026-05-12T17:03:45 | k08v38af | logs/20260512_170345_hypervigilance-round26-A1-seed45_n106_gpu1.log |
| 3 | running | C — decoupleFood (re-launch of Run 1; n106 node crash 2026-05-12) | `hypervigilance-round26-C-seed44_n101_gpu0_relaunch` | hypervigilance | prod | 44 | 101 | cuda:0 | 2026-05-16T13:21:04 | ja5fu5k3 | logs/20260516_132101.log |

Tag uniqueness verified against R2.5 (`hypervigilance-round25-*`) — `round26` prefix collides with neither.

### 3.1 Configs to Produce (designer-only, pre-launch)

| Run | Config (env) | Config (agent) |
|-----|--------------|----------------|
| 1 | `configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml` (reused unchanged) | `configs/models/recurrent_ppo.yaml` |
| 2 | `configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml` (reused unchanged) | `configs/models/recurrent_ppo.yaml` |

**No new configs to produce.** Both env configs are inherited from Round 2.5 with no edits. See §2.3 for the optional `behavior_measures:` block backfill if online M1/M2/M5 keys are wanted during training; this is **not produced by this designer** — it routes through developer follow-up if authorised.

### 3.2 Launch Commands (for `training-runner` reference)

```bash
# Run 1 — Cell C — node 106 cuda:0 — seed 44
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml \
  --agent_config configs/models/recurrent_ppo.yaml \
  --episodes 10000000 \
  --num-envs 128 \
  --seed 44 \
  --device cuda:0 \
  --log-interval 50 \
  --wandb-group hypervigilance \
  --wandb-job-type prod \
  --wandb-name "hypervigilance-round26-C-seed44_n106_gpu0" \
  --tag "hypervigilance-round26-C-seed44_n106_gpu0"

# Run 2 — Cell A1 — node 106 cuda:1 — seed 45
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml \
  --agent_config configs/models/recurrent_ppo.yaml \
  --episodes 10000000 \
  --num-envs 128 \
  --seed 45 \
  --device cuda:1 \
  --log-interval 50 \
  --wandb-group hypervigilance \
  --wandb-job-type prod \
  --wandb-name "hypervigilance-round26-A1-seed45_n106_gpu1" \
  --tag "hypervigilance-round26-A1-seed45_n106_gpu1"
```

**Wallclock estimate** (carried forward from R2.5): ~22 h (Cell C) and ~17 h (Cell A1) on node 106. Total wallclock ~22 h with both cells in parallel.

## 4. Pre-Registered Analysis Plan

### 4.1 Primary verdict — Cell C (event-level class-conditional defence)

Computed via `scripts/eval_rollout.py` + `scripts/motif_cluster.py` on the **final saved checkpoint** of the Cell C run, with the toolkit-design §3 protocol (200 deterministic episodes, seeds 1000–1199, R=3.0 cells, K=5 steps, K_motif=7 steps). The output directory will be `results/eval/<run_tag>/<final_step>/`.

| Outcome | Primary thresholds (toolkit measures) | Secondary cross-check | Implies |
|---|---|---|---|
| **H₁(C-event) confirmed** | `M2_BushDiveRate_predator ≥ 0.80` AND `Δ_M2_class ≡ M2_predator − M2_rabbit ≥ +30 pp` AND `M5_EatUnderThreatRatio_predator < 0.80` | Per-tag rabbit_TL vs rabbit_BR agree within ±5 pp on M2 and ±0.10 on M5 | The Round-2.5 §12 finding (Cell C is spatially class-blind but behaviourally class-discriminating) is **seed-stable**. Event-level class-conditional defence under matched olfactory smells is a robust feature of plain-RPPO under this configuration. |
| **H₀(C-event) confirmed (Round 2.5 was a one-off)** | `M2_predator < 0.50` OR `Δ_M2_class < +10 pp` | (any) | The §12 verdict reverts to provisional; the Round-2.5 single-seed result was seed-specific. Schedule a Round 2.6.5 with seeds 46 and 47 to disambiguate. |
| **Borderline** | `0.50 ≤ M2_predator < 0.80` OR `+10 pp ≤ Δ_M2_class < +30 pp` OR `0.80 ≤ M5_predator < 0.95` | (any) | Escalate per Appendix C — one more seed (seed 46) on a future GPU rotation. |
| **Inverted** | `Δ_M2_class ≤ −10 pp` (agent bush-dives MORE for rabbits than predators) | (any) | Surprising; would imply the GRU has class-mistaken the labels. Treat as a novel finding requiring its own follow-up. |

### 4.2 Primary verdict — Cell A1 (corner-camping seed-stability)

Computed from training-time WandB scalars (last-10% window, episodes 9.0 M – 10.0 M) AND from the offline toolkit run on the final saved checkpoint. The training-time half tests the headline "agent survives by camping"; the toolkit half tests the per-tag event-level signal.

| Outcome | Training-time thresholds | Toolkit cross-check | Implies |
|---|---|---|---|
| **H₁(A1-stable) — corner-camping is a generic basin** | `Episode/Steps ≥ 470/500` AND per-tag `\|Δ_TL ≡ MeanDistPredator_TL − MeanDistRabbit_TL\| ≤ 0.30` cells | `M2_BushDiveRate_predator < 0.10` AND `M2_BushDiveRate_rabbit < 0.10` (the camping signature: agent rarely uses bushes because it rarely encounters threats) | The Round-2.5 §11 corner-camping verdict for Cell A1 is **seed-stable**. The same-corner predator and same-corner rabbit policy collapses to location-conditional safety across seeds 43 and 45. |
| **H₀(A1-escape) — seed 45 finds a different policy** | `Steps < 400/500` AND the agent visits TL meaningfully (proxy: `MeanDistRabbit_TL` near random-policy baseline of ~3 instead of ~6.6 cells, OR `RabbitHits_TL > 1/ep`) | `M1_InterruptedFeedingRate_predator > 0.05` (agent eats with predator nearby in TL) | Corner-camping is seed-specific. The analyzer fills a new behaviour-class section: characterise the new policy, run the toolkit on it, and decide whether a Round 2.7 with seeds 46–48 is needed to map the basin distribution. |
| **Borderline** | `400 ≤ Steps < 470` OR `0.30 < \|Δ_TL\| ≤ 1.0` | (any) | Escalate per Appendix C — seed 46 on a future rotation. |
| **Class-conditional discrimination at TL** | `Δ_TL ≥ +1.0` cells AND `M1_predator_TL − M1_rabbit_TL ≥ +20 pp` AND `Steps ≥ 400` | (any) | Surprising — the agent escapes corner-camping AND learns class discrimination. Headline finding; route to Round 3 design (close the spatial loophole). |

### 4.3 Secondary verdicts (cross-round comparability with R2.5 + R1)

Retained from R2.5 §4.2 / §4.3 to preserve the apples-to-apples comparison across rounds. **All thresholds and metric definitions** for these secondary checks are inherited from [`sameprop_round25_design.md`](sameprop_round25_design.md) §§4.2–4.3; do not re-quote them here. The relevant cross-tabs the analyzer must produce:

- **Cell C secondary**: aggregated `Δ ≡ MeanDistPredator − MeanDistRabbit`, aggregated `ΔH ≡ RabbitHits − PredatorHits`, per-tag `MeanDistRabbit_TL` vs `MeanDistRabbit_BR` vs `MeanDistPredator_full`. The R2.5 verdict was "Inverted band" (Δ = −0.532, both rabbit tags above predator). R2.6 confirmation: Δ ≤ −0.30 (inverted band sign-stable) AND both per-tag rabbit distances above `MeanDistPredator_full` (bilateral-rabbit-avoidance sign-stable).
- **Cell A1 secondary**: aggregated Δ, per-tag Δ_TL, `Term_MaxSteps` (≥ 0.85 corroborates H₁(A1-stable) per the R2.5 §5 row-2 contamination signature).

### 4.4 Temporal evolution checks

Mandatory per project convention. For each cell, partition training into 10 equal episode-count windows (~1 M episodes each). The primary verdict must be **stable across windows 8, 9, and 10** — i.e., the last 30 % of training. Plot:

1. `Episode/Steps` and `Episode/Term_MaxSteps` per window (both cells; primary for A1).
2. Per-tag distances per window (both cells; secondary).
3. Aggregated Δ per window (both cells; secondary).

The toolkit measures (M1/M2/M5/M7) are not currently computed per-window during training (they require eval-rollouts); for R2.6 they are computed once on the final checkpoint. If a within-training trajectory of M2/M5 is wanted, it requires running eval-rollouts at multiple checkpoints — flagged as a future work item, not required for R2.6 verdicts.

### 4.5 Cross-cell + cross-round contrasts

| Comparison | What it isolates |
|---|---|
| C (R2.6, seed 44) vs C (R2.5, seed 42, `bdnfc0lu`) | **Primary seed-lock contrast.** Replicates §12 toolkit numbers across seeds. Both seeds should hit `M2_predator ≥ 0.80`. |
| A1 (R2.6, seed 45) vs A1 (R2.5, seed 43, `nm8gn7y2`) | **Primary seed-lock contrast.** Replicates §11 corner-camping verdict. Both seeds should hit `Steps ≥ 470` + `Δ_TL ≤ 0.30`. |
| C (R2.6) per-tag `MeanDistRabbit_TL` vs `MeanDistRabbit_BR` | Symmetric vs asymmetric rabbit avoidance — required for the "bilateral rabbit avoidance" reading to seed-lock. |
| C (R2.6) seed-44 vs C (R1, seed 42, `rg5nl1ov`) | The food-coupling effect under sameProp, with the new seed; cross-references the R2.5 §4.5 contrast. |
| A1 (R2.6) seed-45 vs A1 (R1, seed 43, `6ks4bjbq`) | The HUNT-ablation + patrol-area-match effect, with the new seed; same. |
| Combined Cell C + Cell A1 toolkit measures, R2.5 + R2.6 (n=2 per cell) | First two-seed reading of the toolkit's primary verdict criteria; provides initial seed-noise bounds on M2 and M5 for downstream studies (Round 3, NMN comparisons). |

## 5. Failure-Mode Catalog

Round 2.5 §5 still applies in full — re-use that catalog. Round 2.6 adds the following rows that are specific to its primary-verdict upgrade:

| Failure | Resolution |
|---|---|
| **(NEW R2.6 — M2 saturation)** Cell C M2_predator ≈ 1.0 (ceiling-limited; the bush-dive measure is bounded at 100%). | Confirms H₁(C-event) under §4.1 but note the ceiling in the verdict text. The +30 pp Δ_M2_class threshold is robust to ceiling on the predator branch (M2_rabbit was 50.8% in R2.5, with plenty of headroom to be exceeded); only flag if M2_rabbit also saturates near 1.0 (then both classes are "always bush-dive" and the measure has lost discriminative power — escalate to a tighter R or shorter K in v2). |
| **(NEW R2.6 — A1 escapes corner-camping)** Seed 45 produces the H₀(A1-escape) branch. | The analyzer fills a new section (not "H₁(A1) class-discrimination confirmed" or "H₀(A1) corner-camping confirmed" — neither applies). Characterise the new policy with: per-tag distances, training-time `Term_*` breakdown, toolkit M1/M2/M5/M7 on the final checkpoint, and motif-distribution comparison vs Cell C and vs Round 1. The verdict statement: "Corner-camping is one of at least two basins under this config; seed-43 and seed-45 sampled different basins." |
| **(NEW R2.6 — toolkit eval-rollout disk usage)** N=200 episodes × 2 checkpoints (final-checkpoint only per §3.5 of the toolkit design) × ~150 MB per checkpoint = **~300 MB total**. | Within the `results/` budget; negligible. No mitigation required. If the analyzer expands to multiple checkpoints during temporal-evolution analysis (§4.4 future-work item), each additional checkpoint adds ~150 MB. |
| **(NEW R2.6 — toolkit eval-rollout reproducibility)** If `scripts/eval_rollout.py` re-runs on the same checkpoint produce M1/M2/M5 values differing by > 1% across reruns. | Toolkit-design §5.5 fires. Halt analysis; the toolkit's deterministic-policy mode is leaking non-determinism. Surface to `senior-developer` immediately. |
| **(NEW R2.6 — per-cell verdict-sign agreement)** R2.6 and R2.5 produce opposite signs on the primary verdict (e.g., R2.5 H₁(C-event) but R2.6 H₀(C-event), or vice versa). | Headline reading is "the effect is seed-dependent." Treat as a null result for the headline claim; route to a 4–6 seed Cell-C study (Round 2.7) before claiming class-conditional defence as a general property of plain-RPPO under sameProp. |

## 6. Predicted Outcomes (designer's prior, pre-registered)

Recorded so the post-hoc reading does not adapt:

- **Cell C (seed 44):** R2.5 already showed `Δ_M2_class ≈ +37 pp` and `M2_predator = 88%` single-seed under sameProp. The toolkit-toolkit design priors (§9.2 of the toolkit doc) predicted no class-conditional response at all for Cell C — that prior was refuted by the observed +37 pp. Given the +37 pp was much wider than the toolkit's H₁(M2) threshold of +15 pp, the expected behaviour at seed 44 is **same direction, similar magnitude**. **Prior on H₁(C-event) confirmed: ~80%.** Prior on H₀(C-event) (the Round-2.5 result was seed-specific): ~15%. Prior on Borderline + Inverted combined: ~5%.

- **Cell A1 (seed 45):** R2.5 showed `Δ_TL = +0.004` cells at `Steps = 486/500`, `Term_MaxSteps = 0.92` — the corner-camping signature in full. The §12 motif-distribution analysis showed 60% of trajectory windows were stationary/freeze motifs confined to BR. Corner-camping is a strong attractor in this configuration — the geometry (TL predator + TL rabbit + BR rabbit + food in TL/BR) creates a low-effort survival policy that the agent finds easily. **Prior on H₁(A1-stable) confirmed: ~70%.** Prior on H₀(A1-escape) (seed 45 finds a different policy — most plausibly a "visit-TL-only-when-starving" mixed policy with lower survival but real class discrimination): ~25%. Prior on the class-conditional-at-TL branch (§4.2 row 4 — agent escapes camping AND learns class discrimination): ~3%. Prior on something unexpected: ~2%.

**Combined-cell joint prediction**: if both cells confirm their H₁ branch (which is the modal outcome at ~80% × ~70% ≈ 56% joint probability), the headline reading of the sameProp study becomes: "Under matched olfactory smells, plain-RPPO produces (a) seed-stable corner-camping when the predator's quadrant is a hard avoidance target (Cell A1), and (b) seed-stable event-level class-conditional defence when the spatial structure forces threat encounters (Cell C). Mean-distance metrics dissolve effect (b) into a misleading 'no class-conditional avoidance' verdict; the toolkit recovers it." This is the paper-grade story Round 2.6 either locks in or reopens.

## 7. Metrics Requested

**None beyond what the toolkit already emits.** The toolkit's M1/M2/M5/M7 keys cover the primary verdict criteria; the per-tag distance metrics shipped in commit `0a73613` (verified `6d3d382`) cover the secondary criteria. No new logger work is required for Round 2.6. The Round-2.5 §7 "second-tier disambiguator" items (`TimeFractionNearTL_*`, `QuadrantOccupancy_*`) remain deferred — only escalate if Round 2.6 lands in §4.1 or §4.2 Borderline bands.

## 8. Launchable Status

**Configs**: ready ✓ (env-level)
- `configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml` — verified rabbits TL/BR, predator full (re-checked 2026-05-12 against `tag:` fields at lines 128, 137, 156).
- `configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml` — verified rabbits TL/BR, predator TL co-located with TL rabbit (re-checked 2026-05-12 against `tag:` fields at lines 148, 157, 181).

**⚠ `behavior_measures:` config block — currently MISSING from both configs.**
- Verified by `grep -n "behavior_measures" 02-sameProp_R2_decoupleFood.yaml 02-sameProp_R2_passivePredator.yaml` returning empty (2026-05-12).
- **Impact on R2.6**: this designer's read is that **the primary verdict criteria can still be met without the online block**, because the toolkit's primary measures are computed *offline* via `scripts/eval_rollout.py` + `scripts/motif_cluster.py` on the final saved checkpoint. That is exactly what produced the R2.5 §12 numbers — the R2.5 configs also did not carry a `behavior_measures:` block during training. The offline scripts read the env config to instantiate the eval environment but use script-internal defaults (R=3.0, K=5, K_motif=7) for the protocol parameters.
- **However**, the toolkit-plan §10 "Gate 1 — schema loader" pre-registered the `behavior_measures:` block as a no-fallback-defaults requirement for *online* M1/M2/M5/M7 emission during training. If the user wants online toolkit keys logged to WandB during the R2.6 training runs (useful for temporal-evolution figures over the full 10 M episodes), the block must be backfilled to both configs BEFORE launch.
- **Remediation if online keys are wanted (NOT this designer's edit)**: route to `developer` to backfill the `behavior_measures:` block per [`behavior_measure_toolkit_v1_design.md`](../behavior_measures/behavior_measure_toolkit_v1_design.md) §6 (the locked schema with 14 mandatory keys) on both `02-sameProp_R2_decoupleFood.yaml` and `02-sameProp_R2_passivePredator.yaml`. After the backfill, both configs must pass through `env-config-auditor` per the toolkit-design §8.3 hand-off table (the auditor must verify the `obstacles` list contains a `hides_agent: true` entry — it does, in both configs at the bush rows).
- **Remediation if offline-only post-hoc verdict is sufficient (no config edit)**: configs are launchable as-is; the toolkit pipeline runs on the final saved checkpoint and produces the primary verdict numbers per the R2.5 §12 protocol.

The user owns this decision. The §4.1 / §4.2 primary verdict criteria do not depend on online keys — they are evaluated against the offline eval-rollout output, identical to the R2.5 §12 evaluation. Recommendation: **proceed offline-only** for R2.6 to keep the launch unblocked and seed-lock the R2.5 verdicts; backfill the online block in a separate developer ticket for Round 3 onward.

**Code state**: ready ✓
- Per-tag metrics shipped (commit `0a73613`), Site-1 RPPO bug fixed (commit `6d3d382`), full verification pass (per-tag plan §"Verification Report").
- Toolkit implementation plan `verification_status: pass`; `scripts/eval_rollout.py` and `scripts/motif_cluster.py` both exist and were exercised on R2.5 final checkpoints (`results/eval/models/10000003/` for Cell A1, `results/eval/models/10000022/` for Cell C).

**Pre-flight gates** (must pass before runner launches):
1. ⏸ User decision on the online-block question above (defer to offline-only OR backfill).
2. ⏸ `env-config-auditor` re-audit if (1) chooses backfill; otherwise quick re-audit of the unchanged configs.
3. ⏸ User authorization — explicit "go" after auditor sign-off.

**Runner handoff**: `training-runner` will fill the actual columns of §3 (Status, Launched at, WandB run ID, Log path) at launch time. Per the diary protocol, runner must call `diary training-start --tag … --node 106 --gpu cuda:0/cuda:1 --cell C/A1 --wandb <id> --doc docs/experiments/active/hypervigilance/sameprop_round26_design.md` for each run.

---

## Appendix

### A. Cell mapping table (channel ablations — inherited from R2.5)

Identical to R2.5 Appendix A — see [`sameprop_round25_design.md`](sameprop_round25_design.md) Appendix A. No re-quoting.

### B. Config diff vs Round 2.5

**None at the env level.** Round 2.6 reuses the Round 2.5 configs unchanged at the environment / sensor / body level. Only seed numbers and WandB tag prefixes differ.

### C. Escalation policy

Identical to R2.5 Appendix C. If Round 2.6 lands in any §4.1 / §4.2 Borderline band, or if a primary-criterion is borderline relative to its threshold (within 25 % of the threshold value, e.g., `+22 pp ≤ Δ_M2_class < +30 pp` or `0.475 ≤ Steps/max_steps < 0.94`), schedule Round 2.6.5:

| Round 2.6.5 run | Cell | Tag | Seed | Node/GPU |
|---|---|---|---|---|
| 1 | C — decoupleFood | `hypervigilance-round265-C-seed46_<n>_<gpu>` | 46 | next-available 2-GPU rotation |
| 2 | A1 — passivePredator | `hypervigilance-round265-A1-seed47_<n>_<gpu>` | 47 | same node, second GPU |

If R2.5 + R2.6 produce **opposing signs** on either primary verdict (per §5 NEW row 5), do NOT use the Appendix-C "one more seed" remediation — escalate directly to a 4-seed study (Round 2.7) with seeds 46, 47, 48, 49.

### D. Changelog

| Date | Change | Author |
|---|---|---|
| 2026-05-12 | Initial pre-registered design for Round 2.6; 2 cells × 1 seed each (C@44, A1@45) on node 106 cuda:0/cuda:1; primary verdict criteria upgraded to toolkit M2/M5 for Cell C and Steps+Δ_TL+M2-low for Cell A1; mean-distance measures retained as secondary cross-round comparators. Configs reused unchanged from Round 2.5; `behavior_measures:` block flagged as MISSING in §8 with remediation routes (offline-only OR developer backfill). | experiment-designer |
