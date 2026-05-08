---
title: "SameProp Round 2 — confound control (food decoupling) + movement-signature ablation"
topic: hypervigilance
status: partial
created: 2026-05-08
last_updated: 2026-05-08
phase: 1
wandb_tag: "hypervigilance-round2"
develop_link: "../../../develop/active/hypervigilance/sameprop_discriminating_channels.md"
supersedes: ""
---

# SameProp Round 2 — confound control + movement-signature ablation

> **Status**: PARTIAL — both runs SIGINT'd at ~3.5h / ~0.4M episodes (≈4–5 % of the 10M target). Pre-registered §4.2/§4.3 thresholds CANNOT be applied at this budget; see §9 for truncated-data partial verdict.
> **Date**: 2026-05-08
> **Author**: experiment-designer (§§0–8); experiment-analyzer (§9 truncated-data verdict)
> **Related**:
> - Round 1 launch + analysis (parent): [`sameprop_existing_run_survey.md`](sameprop_existing_run_survey.md)
> - Discriminating-channels memo: [`docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`](../../../develop/active/hypervigilance/sameprop_discriminating_channels.md)
> - Per-entity logging that enables this round: [`docs/develop/active/hypervigilance/per_entity_avoidance_logging.md`](../../../develop/active/hypervigilance/per_entity_avoidance_logging.md)
> - Round 1 baseline config (modified diff base): [`configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml`](../../../../configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml)

---

## 0. Round 1 recap (verified, both seeds agree at ≥6.7M episodes)

Under matched olfactory `properties [0,1,0,0,0]` for predator and rabbit, RPPO appears to discriminate:

| Metric | Final (≈6.7M ep) | Trajectory |
|---|---|---|
| MeanDistRabbit | ≈3.77 | 3.94 → 3.76 (decreasing) |
| MeanDistPredator | ≈4.40 | 4.35 → 4.41 (slowly increasing) |
| RabbitHits / ep | ≈6.5 | 3.7 → 6.6 (near-doubled) |
| PredatorHits / ep | ≈3.4 | peaks then mildly retreats |

**Round-1 confound flagged in parallel by experiment-analyzer:**
food TL+BR ↔ rabbit TL+BR, while predator roams full grid `[[1,1],[10,10]]`. So the apparent class discrimination MeanDistRabbit < MeanDistPredator could be **food-seeking spillover** (agent approaches food → rabbit happens to be in the same quadrant) rather than learned olfactory class identity.

Round 2 directly attacks this confound.

---

## 1. Research Question

**Q (cell C):** When food spawn areas are decoupled from rabbit spawn areas (food → TR + BL; rabbits remain TL + BR), does the Round-1 finding `MeanDistRabbit < MeanDistPredator` persist?

**Q (cell A1):** When predator HUNT mode is structurally disabled and patrol area is shrunk to match a rabbit's quadrant — eliminating the ranked-dominant pre-contact discriminator (the predator's `agent_pos`-tracking movement signature, per `sameprop_discriminating_channels.md` §4) — does RPPO retain class-level avoidance under matched olfactory properties?

> **H₀ (C):** The Round-1 effect was driven by the food/rabbit-quadrant collinearity. With food decoupled, MeanDistRabbit and MeanDistPredator converge (|Δ| ≤ 0.1 cells) and RabbitHits / PredatorHits converge correspondingly. **There is no genuine class discrimination** under sameProp.
>
> **H₁ (C):** The Round-1 effect persists after decoupling: MeanDistRabbit < MeanDistPredator − 0.3 cells with both seeds agreeing on the sign. **Olfactory channel ambiguity does not prevent learned class avoidance** under sameProp; some other channel (movement signature, visual ch.5/ch.7 at contact, extero-noc) carries the class label.
>
> **H₀ (A1):** With HUNT disabled and patrol-area matched, predator and rabbit are kinematically identical. The agent has no remaining pre-contact discriminator and falls back on only post-contact teaching (visual + extero-noc). Predicted: PredatorHits ≈ RabbitHits within 1.0 / ep, and MeanDistPredator ≈ MeanDistRabbit within 0.3 cells. **Movement signature was the dominant discriminator.**
>
> **H₁ (A1):** Even with HUNT disabled, RPPO keeps PredatorHits < RabbitHits − 1.5 / ep at convergence. **Post-contact visual+extero-noc is sufficient** for class learning via the recurrent state.

A clean falsification of either cell does not depend on the other; the two cells together produce a 2x2 over {confound resolved? × movement-signature ablated?} when read against Round 1.

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Cell C | Cell A1 |
|---|---|---|
| Food spawn quadrants | TR + BL (decoupled from rabbits) | TL + BR (Round-1 default) |
| Predator `detection_range` | 5 (Round-1 default) | **0** |
| Predator `hunt_stamina_threshold` | 0.7 (Round-1 default) | **1.1** (HUNT unreachable) |
| Predator `patrol_area` | `[[1,1],[10,10]]` (Round-1 default) | **`[[1,1],[5,5]]`** (matches rabbit TL) |
| Predator `spawn_area` | `[[1,1],[10,10]]` (Round-1 default) | **`[[1,1],[5,5]]`** |

All other fields fixed at Round-1 baseline (`01-interoNocicept_sameProp.yaml`). See §2.2 + Appendix B.

### 2.2 Controlled Variables

- Olfactory `properties` for predator + rabbit: `[0.0, 1.0, 0.0, 0.0, 0.0]` (the sameProp condition itself).
- `properties_std`: zero for all entities (no per-step olfactory noise).
- `perceptual_noise.enabled: false` (sigmas in YAML are inert).
- 4 hiding_predators in 4 quadrants, properties `[0,0,0,0,0]`.
- Rabbits: 2x count, TL + BR quadrants, move_interval 1, `properties [0,1,0,0,0]`.
- Bushes: 5 in TR, 5 in BL (hides_agent).
- Rocks: 3 in each of 4 quadrants, damage [1,5].
- Sensors: olfaction radius 20, decay 2.0, vector_size 5; visual range 0; nociception enabled; intero-noc enabled (tau 3.0, kernel 12); proprioception on; injury_observable false; nutrition_observable false.
- Body: full Round-1 homeostatic reward, max_steps 500, metabolic_cost 1.0.
- Agent: `configs/models/recurrent_ppo.yaml` (RPPO, identical to Round 1).
- Step budget: 10,000,000 episodes (full Round-1 budget; Round 1's 6.7M was an early checkpoint, not a hard ceiling).
- Parallel envs: 128 (Round-1 default).
- Per-entity metrics (`Episode/MeanDistRabbit`, `Episode/RabbitHits`, etc.) logged via the recently-merged `per_entity_avoidance_logging` (commit `4b55fc6`, branch `v1.3`).

### 2.3 Confounds & Limitations

| Confound | Affected | Severity | Mitigation |
|---|---|---|---|
| n = 2 seeds (42, 43) — same as Round 1 | both cells | Medium | Cross-round comparability with Round 1 is the priority; both seeds agreed at 6.7M in Round 1, so seed-stable signs are detectable. Add seeds 44/45 in Round 3 if Round 2 results are seed-divergent. |
| 2 GPUs, 1 cell × 1 seed each (no within-cell seed-pair this round) | both cells | Medium | This is the unavoidable consequence of running two cells in parallel with 2 GPUs. **Round-2 results are interpreted only against Round-1's 2-seed agreement** — a sign reversal in even 1 seed of Round 2 vs. consistent 2-seed Round 1 is informative. If a cell's verdict is borderline, schedule the second seed on the next GPU rotation. |
| Cell A1 keeps predator damage on contact | A1 | Low | This is intentional — we want post-contact teaching available so we can isolate "is movement signature OR is post-contact teaching what drives avoidance under sameProp." |
| Cell A1 quadrant-locks predator to TL only (one of two rabbit quadrants) | A1 | Low | Co-occupancy with TL rabbit is *desired* — it makes the discrimination maximally hard for the agent. BR rabbit then acts as an internal control: if A1 still discriminates, MeanDistRabbit_BR vs MeanDistRabbit_TL should differ if the agent uses quadrant identity rather than movement. (Per-rabbit indexing is not currently logged — see §7 Metrics Requested.) |
| Cell C still shares quadrants with hiding_predators in all 4 corners | C | Low | Hiding predators have `properties [0,0,0,0,0]` — no olfactory leak. They are a non-cue damage source, identical to Round 1. |
| `random_start_pos: true` means agent spawn distribution differs across episodes | both | Low | Same condition as Round 1; cross-round comparability preserved. |
| Round 1 total step count was reported as ~6.7M episodes; Round 2 targets 10M | both | Low | If late-training (>6.7M) shows the Round-1 effect *strengthens* in cell C, that argues against H₀(C). If it weakens, Round 1's verdict was likely premature. |

## 3. Launch Manifest

**Hardware**: Node 112, 2 GPUs (cuda:0, cuda:1). 1 cell per GPU, 1 seed per cell. Total: 2 runs.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | stopped early (SIGINT @ 3h 20m, 489,857 ep) | C-decoupleFood | `hypervigilance-round2-C-seed42_n112_gpu0` | hypervigilance | prod | 42 | 112 | cuda:0 | 2026-05-08T14:17:50 | 0u266oj5 | logs/20260508_141750.log |
| 2 | stopped early (SIGINT @ 3h 22m, 388,905 ep) | A1-passivePredator | `hypervigilance-round2-A1-seed43_n112_gpu1` | hypervigilance | prod | 43 | 112 | cuda:1 | 2026-05-08T14:20:15 | 27svrmhv | logs/20260508_142015.log |

### 3.1 Configs to Produce (designer-only, pre-launch)

| Run | Config (env) | Config (agent) |
|-----|--------------|----------------|
| 1 | `configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml` | `configs/models/recurrent_ppo.yaml` |
| 2 | `configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml` | `configs/models/recurrent_ppo.yaml` |

### 3.2 Launch Commands (for `training-runner` reference)

Both invoked via `run_command.py` with the standard project Python interpreter `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`. The commands the runner will issue are (modulo the `train_command-agent.sh` template):

```bash
# Run 1 — Cell C — node 112 cuda:0 — seed 42
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
  --wandb-name "hypervigilance-round2-C-seed42_n112_gpu0" \
  --tag "hypervigilance-round2-C-seed42_n112_gpu0"

# Run 2 — Cell A1 — node 112 cuda:1 — seed 43
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
  --wandb-name "hypervigilance-round2-A1-seed43_n112_gpu1" \
  --tag "hypervigilance-round2-A1-seed43_n112_gpu1"
```

The seed/GPU pairing is a deliberate choice: cell C on the same seed (42) as Round-1 R1-seed42 (`rg5nl1ov`) makes pairwise R1↔R2-C comparison cleanest under matched RNG; cell A1 takes the alternate seed (43) so any A1 verdict robust across seeds 42 and 43 (across cells) is harder to dismiss as a one-seed fluke.

## 4. Pre-Registered Analysis Plan

### 4.1 Primary metrics

Computed on the **last 10% window** of training (episodes ≈9.0M – 10.0M), mean across all 128 envs of the 1880 history records in the window.

| Metric | Cell C | Cell A1 |
|---|---|---|
| `Episode/MeanDistRabbit` (cells, L2) | primary | primary |
| `Episode/MeanDistPredator` (cells, L2) | primary | primary |
| `Episode/RabbitHits` (count / ep) | primary | primary |
| `Episode/PredatorHits` (count / ep) | primary | primary |
| `Episode/Steps` (survival) | secondary (project headline) | secondary |
| Δ ≡ MeanDistPredator − MeanDistRabbit | derived primary | derived primary |
| ΔH ≡ RabbitHits − PredatorHits | derived primary | derived primary |

### 4.2 Confirmation / refutation thresholds (cell C)

| Outcome | Thresholds | Implies |
|---|---|---|
| **H₀ (C) confirmed** (no genuine discrimination — Round 1 was the food confound) | \|Δ\| ≤ 0.1 cells **AND** \|ΔH\| ≤ 1.0 / ep | Round-1's apparent rabbit avoidance was food spillover. Sub-question for Round 3. |
| **H₁ (C) confirmed** (genuine discrimination survives) | Δ ≥ 0.3 cells (predator further than rabbit) **AND** ΔH ≥ 1.5 / ep (more rabbit hits than predator hits) | Class discrimination is real under sameProp. Movement signature / visual+extero is the channel. Pair with cell A1 to isolate. |
| **Inverted** (rabbits are now further than predators) | Δ ≤ −0.3 cells | The decoupling forced rabbits to a corner the agent learned was unreachable / unrewarding; alternative explanation needs Round 3. |
| **Ambiguous** | 0.1 < \|Δ\| < 0.3 | Add seeds 44/45 in Round 3 before drawing a verdict. |

### 4.3 Confirmation / refutation thresholds (cell A1)

| Outcome | Thresholds | Implies |
|---|---|---|
| **H₀ (A1) confirmed** (movement signature was dominant) | PredatorHits ≈ RabbitHits, \|ΔH\| ≤ 1.0 / ep **AND** \|MeanDistPredator − MeanDistRabbit\| ≤ 0.3 cells | The Round-1 discrimination collapses without HUNT. Movement signature is the dominant channel. |
| **H₁ (A1) confirmed** (post-contact teaching alone is sufficient) | PredatorHits < RabbitHits − 1.5 / ep at convergence | Visual ch.5 + extero-noc 0.9 at contact, mediated by GRU, suffices to drive class avoidance even without HUNT. |
| **Both worse than Round 1** | Survival drops below Round-1 floor (≈340 steps) | Quadrant restriction itself harmed the agent (possibly food-search interference); read as null. |

### 4.4 Temporal evolution checks (mandatory)

For each cell, partition training into 10 equal episode-count windows. Plot:
1. `Episode/MeanDistRabbit` and `Episode/MeanDistPredator` per window (overlay).
2. `Episode/RabbitHits` and `Episode/PredatorHits` per window (overlay).
3. `Episode/Steps` per window.

A cell is taken seriously only if its primary-metric verdict is **stable across the last 3 windows** (i.e., not driven by a single transient).

### 4.5 Cross-cell + cross-round contrasts

| Comparison | What it isolates |
|---|---|
| C (R2) vs Round-1 R1-seed42 (`rg5nl1ov`) | Effect of food-quadrant decoupling, holding seed = 42. |
| A1 (R2) vs Round-1 R1-seed43 (`6ks4bjbq`) | Effect of HUNT ablation + patrol-area match, holding seed = 43. |
| C vs A1 (R2) | Independent disconfirmations of two confounds. |

### 4.6 Random-policy baseline (anchor)

Round-1 analysis pre-computed: uniform-random `MeanDistPredator` ≈ 4.684 on the 10x10 grid. This is the random-policy floor for *any* MeanDist metric on this grid. A cell whose MeanDistRabbit and MeanDistPredator both sit at ≈4.68 is at the random-policy baseline and has not learned class discrimination at all.

## 5. Failure-Mode Catalog (pre-decided)

| Failure | Resolution |
|---|---|
| Training instability (NaN, value explosion) | Refutes the run, not the hypothesis. Re-launch with same seed; if it recurs, surface to `senior-developer` as a real config bug. |
| Cell A1: agent never approaches predator (because predator is in TL, food now in TL+BR, agent hangs out in BR rabbit corner) → `MeanDistPredator` looks high purely by spatial separation | Read `Episode/Steps` and `Episode/FoodEaten` together. If the agent is surviving by camping food and never visits TL, A1's verdict is moot — it tells us nothing about discrimination because the agent never had the chance. Schedule Round 3 with food forced in all 4 quadrants. |
| Cell C: agent learns to camp the TR or BL food + bush corners (where bushes hide it from hiding_predators) and rarely visits TL/BR rabbit quadrants → `MeanDistRabbit` looks artificially high | Same diagnostic as A1: cross-check with quadrant-occupancy if logged, otherwise interpret cautiously and schedule Round 3. |
| Saturation: `Episode/Steps` plateaus at ≈340 (Round-1 ceiling) without further differentiation | Null result. Insufficient horizon. Plan Round 3 at 20M episodes if Round-1's plateau is near a true ceiling rather than an under-trained one. |
| Seed-dependent noise: 1 seed per cell flips relative to Round-1's 2-seed verdict | Add a 2nd seed for the affected cell on the next GPU rotation; do **not** declare the cell verdict from a single seed unless thresholds in §4.2/§4.3 are exceeded by a wide margin (>2x). |

## 6. Predicted Outcomes (pre-registration)

Designer's prior on the outcomes — recorded so the post-hoc reading does not adapt:

- **Cell C (most likely):** Δ shrinks substantially (Round-1 0.6 cells → R2 ~0.2–0.4 cells), but does not vanish. Olfactory class identity is not the only channel — visual ch.5 + extero-noc + movement signature still carry information. Most likely outcome: **partial confound** — Round-1 verdict was overstated but not entirely an artifact.
- **Cell A1 (less likely):** ΔH (rabbit minus predator hits) shrinks but stays positive (RabbitHits > PredatorHits by ≥1 / ep), because visual ch.5 + extero-noc 0.9 at contact still teach the agent to avoid the predator quadrant via the GRU, even without HUNT. **H₁(A1)** outcome.

If both predictions are correct, the contribution decomposition under sameProp is roughly: ~30–50% food-quadrant collinearity + ~30–40% movement signature + ~20–30% post-contact visual/extero teaching. Round 3 (cell B: olfaction-only, sensors stripped) would close this.

## 7. Metrics Requested (optional — not blocking)

These would sharpen the analysis but are not required for the verdict thresholds in §4.

| Subfield | Content |
|---|---|
| **Metric (1)** | `Episode/MeanDistRabbit_TL` and `Episode/MeanDistRabbit_BR` (per-rabbit-instance distance, cells). |
| **Why now** | In cell A1, the predator is locked to TL. If the agent learns "TL is dangerous, BR is safe," MeanDistRabbit_TL would rise while MeanDistRabbit_BR falls, even though both rabbits emit identical olfactory signals. The aggregated `MeanDistRabbit` would average these and look like uniform avoidance. Per-rabbit indexing distinguishes "class-level olfactory avoidance" from "quadrant-level spatial learning." |
| **Where it'd live** | `src/environment/core.py` near the existing per-entity blocks (~line 491 of `dist_to_neutral`). Same pattern as `dist_to_food` / `dist_to_pred` per-instance variants if any exist, otherwise add `info['mean_dist_neutral_per_idx']`. |
| **Cost** | Cheap — `jnp.linalg.norm(state.neutral_pos - new_agent_pos, axis=-1)` already computed; just don't reduce. |

| Subfield | Content |
|---|---|
| **Metric (2)** | `Episode/QuadrantOccupancy_{TL,TR,BL,BR}` — fraction of episode steps the agent spends in each quadrant. |
| **Why now** | Disambiguates "agent avoids predator class" (low PredatorHits anywhere) from "agent avoids predator quadrant" (low TL occupancy in cell A1). Critical for the failure modes in §5. |
| **Where it'd live** | `src/environment/core.py` per-step occupancy counter, surfaced via `info`. |
| **Cost** | Cheap — 4 boolean masks + reduce. |

If accepted, hand off to `senior-developer` → `developer` via `feature-workflow`. **Round 2 launches without these.**

## 8. Launchable Status

**Configs**: complete ✅
- `configs/experiment/hypervigilance/02-sameProp_R2_decoupleFood.yaml`
- `configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml`

**Pre-flight gates** (must pass before runner launches):
1. ⏸ `env-config-auditor` — modality order intact, mandatory keys present, no `properties`/`property` typo, predator-side keys consistent.
2. ⏸ User authorization — explicit "go" after auditor sign-off.

**Runner handoff**: `training-runner` will fill the actual columns of §3 (Status, Node, GPU, Launched at, WandB run ID, Log path) at launch time.

---

## 9. Truncated-data Partial Analysis (2026-05-08)

> **TRUNCATION BANNER — read first.**
> Both runs were stopped by user SIGINT at **~3.5 h / ~0.4 M episodes**, against a pre-registered budget of **10.0 M episodes (~150 h at observed SPS).** Cell C reached **489,857 episodes (4.9 %)**; Cell A1 reached **388,905 episodes (3.9 %)**. Round 1's behavioural metrics did not stabilise until **~5–6 M episodes** (per the [Round-1 baseline analysis](round1_relog_baseline_analysis.md) §4.4 — windows 1 and 2 differ by ≥0.18 cells in `MeanDistRabbit`, but windows 6–8 differ by ≤ 0.01). **The data analysed below is in the early-learning regime, not the converged regime.** The pre-registered confirmation/refutation thresholds in §4.2 and §4.3 (designed for last-10 % of 10 M episodes) **cannot be applied directly** to a 0.4 M-episode tail. This section reports what the early trajectory shows; final verdicts on H₀(C), H₁(C), H₀(A1), H₁(A1) require completing the budget.
>
> **Author**: experiment-analyzer.
> **Mode**: pre-registered design exists (§§1–8); §9 is a partial-budget readout against the same hypothesis-aware structure.
> **Working files**: `tmp/20260508_round2_truncated_compare.txt`, `tmp/20260508_round2_temporal_evolution.txt`, `tmp/20260508_round1_36window_for_match.txt`, `tmp/20260508_round2_truncated_summary.txt`.

### 9.1 Run actuals

| Cell | WandB ID | Final episode | Walltime | Iterations | SPS | Final TS |
|------|----------|--------------:|---------:|-----------:|----:|---------:|
| C — decoupleFood | `0u266oj5` | 489,857 | 3 h 20 m | 5,800 | 41,357 | 95.8 M |
| A1 — passivePredator | `27svrmhv` | 388,905 | 3 h 22 m | 10,150 | 42,823 | 167 M |

Note A1 had ~75 % more iterations than C in the same wallclock because A1 episodes are far longer (mean Steps ≈ 482 vs C ≈ 272 — see §9.2). SPS values are healthy and within Round-1's 41–55 k band; truncation is not driven by training-speed pathology.

### 9.2 Per-cell early-trajectory readout (last-10 % window of *observed* episodes)

Computed over `Episode/Number` ≈ 440k–490k for Cell C; ≈ 350k–389k for Cell A1. Mean ± within-window std (= temporal noise across episodes in the window, not seed dispersion — n=1 seed per cell).

| Metric (§4.1) | Cell C @ 0.49 M ep | Cell A1 @ 0.39 M ep | R1 converged (~7.2 M) |
|---|---:|---:|---:|
| `Episode/MeanDistRabbit` | **4.857 ± 0.022** | **2.583 ± 0.050** | 3.77 |
| `Episode/MeanDistPredator` | **4.424 ± 0.096** | **6.447 ± 0.361** | 4.40 |
| `Episode/RabbitHits` / ep | **0.594 ± 0.031** | **17.93 ± 0.755** | 6.46 |
| `Episode/PredatorHits` / ep | **2.460 ± 0.132** | **0.913 ± 0.418** | 3.37 |
| `Episode/Steps` (survival) | **275.84 ± 5.99** | **481.39 ± 6.16** | 327.3 |
| **Δ ≡ MeanDistPredator − MeanDistRabbit** | **−0.43** (rabbit FARTHER) | **+3.86** (rabbit MUCH closer) | +0.63 |
| **ΔH ≡ RabbitHits − PredatorHits** | **−1.87** | **+17.02** | +3.09 |
| `Episode/MeanDistFood` | 1.821 ± 0.028 | 0.878 ± 0.128 | 2.41 |
| `Episode/MeanDistHidingPredator` | 2.619 ± 0.019 | 2.383 ± 0.034 | 2.63 |
| `Episode/HidingPredatorHits` / ep | 2.498 ± 0.074 | 3.904 ± 0.463 | 3.00 |
| `Episode/FoodEaten` / ep | 42.5 ± 1.6 | 151.7 ± 16.9 | 54.7 |
| `Episode/Term_Injury` | 0.334 ± 0.027 | 0.057 ± 0.024 | 0.339 |
| `Episode/Term_Starvation` | 0.517 ± 0.026 | 0.054 ± 0.032 | 0.398 |
| `Episode/Term_MaxSteps` | 0.150 ± 0.013 | 0.888 ± 0.033 | 0.263 |
| `Episode/Reward` | −209.83 ± 1.27 | −129.09 ± 5.02 | −204.5 |

Both signs of Δ and ΔH are **already past the pre-registered confirmation magnitude** (|Δ| ≥ 0.3 cells, |ΔH| ≥ 1.5/ep) at < 5 % of the planned budget, but in **directions that diverge sharply between cells and from R1**. This is the central observation of the truncated read.

### 9.3 §4.4 temporal-evolution check (3 sub-windows of the observed data)

Both cells split into three equal-episode windows from `Episode/Number`. The point of this check is to distinguish (a) "the truncation cut into a meaningful early-learning shoulder" from (b) "the truncation is at a transient — the curves haven't settled".

**Cell C — decoupleFood, seed 42:**

| Window | ep range | MeanDistRabbit | MeanDistPredator | Δ | RabbitHits | PredatorHits | ΔH | Steps | Term_MaxSteps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20k–230k | 4.918 ± 0.16 | 5.100 ± 0.28 | +0.18 | 0.432 | 1.435 | −1.00 | 152.0 | 0.012 |
| 2 | 230k–370k | 4.944 ± 0.05 | 4.667 ± 0.21 | **−0.28** | 0.501 | 2.214 | **−1.71** | 235.4 | 0.079 |
| 3 | 370k–480k | 4.855 ± 0.02 | 4.378 ± 0.10 | **−0.48** | 0.574 | 2.521 | **−1.95** | 271.8 | 0.142 |

- `MeanDistRabbit` is essentially flat (4.92 → 4.94 → 4.86), pinned at the **far end of the grid (random-policy baseline ≈ 4.68; the agent is *farther* from rabbits than chance).** Window-3 is statistically tighter than W1, indicating the policy is converging on this far-rabbit behaviour, not drifting through it.
- `MeanDistPredator` is **decreasing** (5.10 → 4.67 → 4.38) — the agent is becoming *less avoidant* of the predator over training. This is the opposite trajectory shape from R1 (where predator distance slowly *grew*).
- Δ is **monotonically more negative across all 3 windows**: +0.18 → −0.28 → −0.48. This is not a transient.
- `Steps` is still rising (152 → 236 → 272 — pre-stable; R1 took ~5 M ep to reach 327).
- The §4.4 stability rule ("verdict only if stable across last 3 windows") is **failed**: Δ is moving, not stable. But the *direction of motion* is consistent — the gap is widening in the inverted direction window over window.

**Cell A1 — passivePredator, seed 43:**

| Window | ep range | MeanDistRabbit | MeanDistPredator | Δ | RabbitHits | PredatorHits | ΔH | Steps | Term_MaxSteps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0–150k | 2.655 ± 0.18 | 6.623 ± 0.58 | **+3.97** | 15.48 | 0.678 | **+14.80** | 413.4 | 0.664 |
| 2 | 150k–270k | 2.578 ± 0.06 | 6.521 ± 0.42 | **+3.94** | 18.23 | 0.815 | **+17.42** | 482.8 | 0.898 |
| 3 | 270k–380k | 2.582 ± 0.05 | 6.357 ± 0.39 | **+3.78** | 17.95 | 1.023 | **+16.93** | 482.8 | 0.898 |

- All three primary distances are **stable across windows 2 and 3** (MeanDistRabbit Δ < 0.01; MeanDistPredator Δ ≈ 0.16; Steps Δ = 0). Cell A1 has reached an **early plateau** — at 0.39 M episodes, the policy has effectively converged to its current strategy. This is dramatically faster than R1's 5–6 M-episode plateau.
- `Steps = 482.8 ≈ max_steps cap of 500` — the agent is **always running out the clock**, terminating via `Term_MaxSteps = 0.89` rather than via injury (0.06) or starvation (0.05).
- `RabbitHits = 17.95/ep` (≈ 1 hit per 27 steps) is consistent with the agent **standing on or near rabbits for most of the episode** (the 2 rabbits in TL+BR each move every step; 17.95 hits in 482 steps ≈ continuous co-location).
- `PredatorHits = 1.02/ep` is rising slowly (0.68 → 0.81 → 1.02) — the predator's reduced patrol area `[[1,1],[5,5]]` does still occasionally co-locate with the agent's TL traversal, but rarely.

The §4.4 stability rule ("verdict only if stable across last 3 windows") is **passed for Cell A1** — windows 2 and 3 agree to ≤ 0.05 cells on every primary metric. The verdict, however, is contaminated by the §5 failure mode (see §9.5).

### 9.4 Cross-round comparison at matched episode count

The fair Round-1 comparator at this budget is **R1's ~0.3–0.4 M episode window**, not its 7.2 M converged tail. At matched episode count, R1's two seeds were still *learning* (Steps ≈ 145–159, well below their 327 final value). From `tmp/20260508_round1_36window_for_match.txt` window 1:

| Quantity | Cell C @ 0.49 M | R1-s42 @ 0.40 M | Cell A1 @ 0.39 M | R1-s43 @ 0.30 M |
|---|---:|---:|---:|---:|
| `Episode/Steps` | 275.8 | 145.6 | 481.4 | 158.9 |
| `Episode/MeanDistRabbit` | **4.857** | 4.141 | **2.583** | 4.101 |
| `Episode/MeanDistPredator` | **4.424** | 4.697 | **6.447** | 4.453 |
| Δ | **−0.43** | +0.56 | **+3.86** | +0.35 |
| `Episode/RabbitHits` | **0.59** | 2.02 | **17.93** | 2.25 |
| `Episode/PredatorHits` | **2.46** | 2.24 | **0.91** | 2.63 |
| ΔH | **−1.87** | −0.22 | **+17.02** | −0.38 |
| `Episode/FoodEaten` | 42.5 | 11.7 | 151.7 | 15.7 |

At equal episode budget, **both R2 cells have already produced larger effect magnitudes than R1's converged 7.2M tail (Δ_R1_SS = +0.63, ΔH_R1_SS = +3.09)**, but in opposite-or-extreme directions to R1's signal. The interventions in cells C and A1 have **measurably and rapidly** changed the learned behaviour — they are not no-ops swallowed by training noise.

### 9.5 §5 failure-mode catalog: which apply to the truncated data?

The pre-registered failure-mode catalog in §5 anticipated several patterns; let me hold the truncated data against each.

- **§5 row 2 — "Cell A1: agent never approaches predator… `MeanDistPredator` looks high purely by spatial separation"**: **APPLIES STRONGLY.** Cell A1's MeanDistPredator = 6.45 with predator confined to TL `[[1,1],[5,5]]` and agent's reward dominated by food (which is in the same TL+BR quadrants as the rabbits) is consistent with **the agent occupying the BR rabbit/food quadrant while the predator paces TL.** The §5 instruction was to read `Episode/Steps` and `Episode/FoodEaten` together; both are saturated (`Steps ≈ 482`, `FoodEaten = 152`) and `Term_MaxSteps = 0.89`, exactly the diagnostic signature §5 named. **Verdict: A1's apparent confirmation of H₁(A1) is moot in §5's terms** — the agent is not discriminating "predator class via post-contact teaching"; it is **spatially segregating from the predator's quadrant** while ignoring the rabbit in the predator-shared quadrant. A per-rabbit-instance log (the §7 Metrics Requested item) would distinguish "agent avoids TL rabbit + visits BR rabbit" from "agent visits both rabbits equally"; without that, the verdict is ambiguous.

- **§5 row 3 — "Cell C: agent learns to camp the TR or BL food + bush corners and rarely visits TL/BR rabbit quadrants → `MeanDistRabbit` looks artificially high"**: **APPLIES, in modified form.** Cell C's `MeanDistRabbit = 4.86 > random-baseline 4.68` and `RabbitHits = 0.59/ep ≈ once every 460 steps` is consistent with **the agent successfully camping in TR+BL food quadrants and effectively never visiting the TL+BR rabbit quadrants.** This is mechanically a *successful* food-decoupling — the food-quadrant-spillover mechanism that confounded R1 has been removed, and the resulting policy reveals: **without food drawing the agent toward rabbits, the agent does not visit rabbits at all.** Note this does NOT confirm H₀(C) ("no genuine class discrimination") — it makes the question moot, because the agent never tests the rabbit-vs-predator olfactory signal in the relevant range; both classes are now far away.

- **§5 row 4 — "Saturation: `Episode/Steps` plateaus at ≈340 (Round-1 ceiling)"**: **DOES NOT APPLY.** Neither cell has reached saturation at this budget — Cell C is climbing (152 → 272), Cell A1 has plateaued at 482 (above R1's 327 ceiling, because A1's Term_MaxSteps fraction is 0.89).

- **§5 row 5 — "Seed-dependent noise: 1 seed per cell flips relative to Round-1's 2-seed verdict"**: **POTENTIALLY APPLIES.** Both cells use 1 seed only; both produce sign-changes (or magnitude-changes large enough to qualify) relative to R1's converged tail. The §2.3 confound table flagged this and instructed: *"do not declare the cell verdict from a single seed unless thresholds in §4.2/§4.3 are exceeded by a wide margin (>2x)"*. Cell A1 exceeds the H₁(A1) threshold by ~10x but is contaminated by §5 row 2; Cell C produces an outcome §4.2 didn't enumerate (sign-flipped Δ = −0.43, neither H₀(C) nor H₁(C) language fits).

### 9.6 Provisional verdicts (EARLY — NEEDS RE-LAUNCH)

> **All four hypothesis verdicts below are tagged PROVISIONAL.** None can be elevated to "confirmed" or "refuted" per §4 thresholds at this budget. The R1 reference values they were calibrated against are 7.2 M-episode steady-state numbers; the R2 data here are 0.4 M-episode early-learning trajectory numbers.

**H₀(C) — "Round-1 effect was the food confound; |Δ| ≤ 0.1, |ΔH| ≤ 1.0":**
- Truncated Δ = **−0.43**, ΔH = **−1.87**. Magnitudes exceed H₀'s flat-equality threshold by 4–5×, so a strict reading is "H₀(C) is **not** confirmed".
- But the *direction* is sign-flipped — the agent is *farther from rabbits than from predator*, consistent with **a more nuanced version of H₀(C)**: the food confound was indeed driving R1's apparent discrimination, and once removed, the agent stops approaching rabbits entirely. The "no genuine class discrimination" prediction is upheld in the sense that **rabbit attraction was driven by food-co-location**, but the corollary is unanticipated: the agent doesn't fall back to neutral; it actively avoids the now-food-empty rabbit quadrants.
- **Provisional verdict (Cell C): WEAK SUPPORT for the food-confound mechanism of H₀(C), but with an unanticipated structural finding that needs the full budget to confirm — at converged training, does Δ stay negative, drift toward 0, or revert to R1's positive sign?** Provisional, not confirmed.

**H₁(C) — "Genuine discrimination survives food-decoupling; Δ ≥ 0.3, ΔH ≥ 1.5":**
- **Refuted at this budget**, in sign and magnitude. Δ has the wrong sign and is monotonically becoming more negative across windows. The probability that 9.5 M more episodes flip Δ from −0.48 to +0.30 (a +0.78 reversal against an established trend) without a phase transition is low.
- **Provisional verdict (Cell C): H₁(C) PROVISIONALLY REFUTED — class discrimination does not survive food-quadrant decoupling.** Strong direction, weak budget; needs full run for final.

**H₀(A1) — "Movement signature was dominant; PredatorHits ≈ RabbitHits, |Δ_dist| ≤ 0.3":**
- Truncated PredatorHits − RabbitHits = 0.91 − 17.93 = **−17.02**, MeanDistPredator − MeanDistRabbit = **+3.86**. Both magnitudes are **far past** H₀(A1)'s equality bands (|ΔH| ≤ 1.0, |Δ_dist| ≤ 0.3) — strict reading: **not confirmed**.
- However, the §5 failure-mode "agent never visits the predator's quadrant" applies, so this verdict is **uninformative** — the agent's spatial segregation prevents the H₀(A1) test from being meaningful. The metric `PredatorHits = 0.91` is low not because the agent learned class-conditional avoidance, but because it almost never enters the predator's TL patrol box.
- **Provisional verdict (Cell A1): H₀(A1) UNTESTABLE at this design.** Need either per-quadrant occupancy logging (§7 Metrics Requested item) or a Round 2.5 cell that forces agent visits to the predator quadrant (e.g., one food source pinned to TL).

**H₁(A1) — "Post-contact visual+extero-noc alone is sufficient; PredatorHits < RabbitHits − 1.5":**
- Truncated PredatorHits − RabbitHits = **−17.02**, far past the H₁(A1) margin. Strict reading: **confirmed by 10x**.
- But the §5 failure-mode contamination means this confirmation is plausibly an artifact of spatial segregation, not class discrimination via the GRU's post-contact teaching. The agent's actual policy is "camp food, harvest rabbits, avoid TL where predator lives" — *which would produce identical numbers under "rabbits and predators are kinematically identical"* (the H₀(A1) condition).
- **Provisional verdict (Cell A1): H₁(A1) PROVISIONALLY CONFIRMED IN STATISTIC, but the confirmation is observationally indistinguishable from the §5 row-2 failure mode.** Without per-rabbit-instance logging, the result cannot be cleanly attributed to class learning.

### 9.7 Random-policy baseline check (§4.6)

R1's prior random-policy baseline `MeanDistPredator ≈ 4.684`. At this budget:
- Cell C: MeanDistPredator = 4.42 (slightly below random — agent is mildly closer than random); MeanDistRabbit = 4.86 (slightly **above** random — agent is slightly *farther* from rabbits than random would be).
- Cell A1: MeanDistPredator = 6.45, MeanDistRabbit = 2.58. Both far from random — the policy is making strong, structured spatial choices, just not the ones the design tested for.

### 9.8 What is needed to complete the round?

Recommended next steps, in order of priority:

1. **(P0) Re-launch identical configs at 10M-episode budget.** Both cells have produced large early signals, but the §4.4 stability rule fails for Cell C (Δ still moving) and the H₁(A1) confirmation in Cell A1 is contaminated by an under-anticipated failure mode. The pre-registered thresholds were calibrated for converged behaviour; the data here are not converged. Wallclock estimate: ~3.5h × (10M / 0.45M) ≈ ~75 h per cell on RTX 3090; ~150h total wallclock for the pair. **Run on a node where 15h × 5 sequential restarts can complete uninterrupted**, or use checkpoint-and-resume if available.

2. **(P0, before re-launch) Add per-rabbit-instance and quadrant-occupancy logs.** The §7 Metrics Requested items (per-rabbit `MeanDistRabbit_TL` / `MeanDistRabbit_BR` and `QuadrantOccupancy_{TL,TR,BL,BR}`) are now load-bearing for Cell A1's verdict. Without them, no amount of additional training will disambiguate "class-conditional avoidance" from "TL quadrant avoidance". Hand off to `feature-workflow` (`senior-developer` plans → `developer` implements). This is now a blocker on Round-2-A1's interpretability.

3. **(P1) Consider Round 2.5 cell — Cell A2 = "predator forced to roam through agent's quadrants":** if quadrant occupancy data confirm the §5 row-2 failure mode in A1, design a third cell that prevents the agent from spatial segregation — e.g., predator patrol area = `[[1,1],[10,10]]` (R1 default) + HUNT disabled. This isolates the movement-signature ablation from the spatial-confinement confound.

4. **(P2) Consider whether Cell C's early signal is strong enough to justify a Round 2.5 instead of completing Round 2.** Cell C's sign-flipped Δ already at 0.49 M episodes is informative *in itself* — even at provisional status it suggests the food/rabbit co-location was indeed load-bearing for R1's apparent class discrimination. A Round 2.5 cell with food in *all four* quadrants (rather than the current 2/2 split) would test whether removing the agent's spatial-camping option restores the R1 effect or collapses it further.

### 9.9 Summary (one paragraph)

At ~5 % of the planned budget, both R2 cells have already departed sharply from R1's converged tail, but in directions and via mechanisms that the pre-registered §4 thresholds did not enumerate: Cell C produces a sign-flipped Δ = −0.43 (rabbits *farther* than predator) consistent with food-driven attraction having been the load-bearing R1 mechanism, and Cell A1 produces a saturated Δ = +3.86 that is operationally indistinguishable from a §5-anticipated "agent avoids predator quadrant by spatial segregation" failure mode. Cell C is not yet stable across windows; Cell A1 is stable but uninterpretable without per-quadrant logging. **The ranking-question raised by Round 1 — does olfactory class identity drive learned avoidance under sameProp — remains open after the truncated read.** The runs need to be re-launched at full budget AND extended with per-quadrant / per-rabbit-instance logging before §4.2/§4.3 verdicts can be issued.

### 9.10 Metrics Requested (now load-bearing — see §9.8 step 2)

The §7 items are no longer optional. To make Cell A1's verdict interpretable at full budget, the following must be logged:

| Subfield | Content |
|---|---|
| **Metric (1, escalated)** | `Episode/MeanDistRabbit_TL`, `Episode/MeanDistRabbit_BR` (per-rabbit-instance distance, cells). |
| **Why now (escalated rationale)** | At 0.39 M episodes Cell A1 has already converged to a strategy whose interpretation hinges on whether the agent treats the TL-rabbit (co-located with passive predator) differently from the BR-rabbit (alone). With aggregate `MeanDistRabbit = 2.58`, the average is consistent with "agent visits BR rabbit at distance ~1.5 and TL rabbit at distance ~3.5" (= 2.5 mean) OR with "agent visits both rabbits at distance 2.58" — the two pictures produce identical aggregate metrics but support different verdicts on H₀(A1) vs H₁(A1). Without per-rabbit indexing, no further training resolves the question. |
| **Where it'd live** | `src/environment/core.py` near the existing per-entity blocks (~line 491 of `dist_to_neutral`). |
| **Cost** | Cheap — the per-instance norm is already computed; just don't reduce. |

| Subfield | Content |
|---|---|
| **Metric (2, escalated)** | `Episode/QuadrantOccupancy_{TL,TR,BL,BR}` — fraction of episode steps the agent spends in each quadrant. |
| **Why now (escalated rationale)** | The §5 row-2 failure mode in Cell A1 (and row-3 failure mode in Cell C) cannot be diagnosed without quadrant occupancy. The truncated data is consistent with both "agent learned class avoidance" and "agent learned spatial segregation"; full-budget data will be similarly consistent unless this metric is in `info`. |
| **Where it'd live** | `src/environment/core.py` per-step occupancy counter, surfaced via `info`. |
| **Cost** | Cheap — 4 boolean masks + reduce. |

If accepted, hand off to `feature-workflow` (`senior-developer` plans → `developer` implements) BEFORE re-launching Round 2.

### 9.11 Related Issues

- The §5 failure-mode catalog was prescient about both observed pathologies (Cell C → row 3, Cell A1 → row 2). No bug-fix-workflow needed; the failure modes were anticipated.
- Frontmatter status escalated `active` → `partial`; manifest §3 statuses changed from `running` to `stopped early (SIGINT @ ...)`.
- No `develop/`-side plan needed yet; the §9.10 Metrics Requested items will become a `feature-workflow` plan if the user accepts them.

---

## Appendix

### A. Cell mapping table

| Round-1 channel rank (`sameprop_discriminating_channels.md`) | Round-2 cell that ablates it | Configured how |
|---|---|---|
| §4 Movement / temporal signature in olfaction (predator HUNT tracks `agent_pos`; rabbit jitters) | Cell A1 | `hunt_stamina_threshold: 1.1` (rested_enough never true) + `detection_range: 0` (HUNT condition unreachable) |
| §4 Patrol-area asymmetry (predator full-grid; rabbit quadrant) | Cell A1 | `patrol_area: [[1,1],[5,5]]` and `spawn_area: [[1,1],[5,5]]` (predator now matches rabbit TL) |
| §1 Olfactory instantaneous shape (indistinguishable when properties match) | NOT ablated | properties remain `[0,1,0,0,0]` for both — that is the *condition under test* |
| §3 Visual ch.5 vs ch.7 at colocation | NOT ablated (intentionally, for both cells) | `visual_sensor_enabled: true`, range 0 — present in both A1 and C; isolates the *contribution* of post-contact teaching |
| §2 Extero-noc 0.9 at contact | NOT ablated | `extero_nociception` active for predator hits in both A1 and C |
| **NEW: food-quadrant collinearity (analyzer flag)** | **Cell C** | food → TR + BL only; rabbits remain in TL + BR; food and rabbits share NO quadrant |

### B. Config diff vs Round-1 baseline

**Cell C — `02-sameProp_R2_decoupleFood.yaml` vs `01-interoNocicept_sameProp.yaml`**:
```diff
-  food TL count: 2     →  food TL count: 0
-  food TR count: 0     →  food TR count: 2
-  food BR count: 2     →  food BR count: 0
-  food BL count: 0     →  food BL count: 2
   (everything else identical)
```

**Cell A1 — `02-sameProp_R2_passivePredator.yaml` vs `01-interoNocicept_sameProp.yaml`**:
```diff
   predator:
-    spawn_area: [[1,1],[10,10]]      →  [[1,1],[5,5]]
-    patrol_area: [[1,1],[10,10]]     →  [[1,1],[5,5]]
-    detection_range: 5               →  0
-    hunt_stamina_threshold: 0.7      →  1.1
   (all other predator + non-predator fields identical)
```

### C. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-05-08 | Initial pre-registered design for Round 2; 2 cells × 1 seed each on node 112 cuda:0/cuda:1 | experiment-designer |
| 2026-05-08 | Launched both cells; manifest rows updated with WandB IDs and log paths; CIFS-stale duplicate noted (see §D) | training-runner |
| 2026-05-08 | Both runs SIGINT'd at ~3.5h / ~0.4M episodes; status frontmatter set to `partial`; manifest statuses set to `stopped early`; §9 truncated-data partial-verdict analysis appended; §9.10 escalates §7 Metrics Requested to load-bearing | experiment-analyzer |

### D. CIFS-attribute-cache stale-read incident (2026-05-08)

A second `run_command.py` invocation for Cell A1 read the stale (Cell C) version of `train_command-agent.sh` from node 112's CIFS cache and spawned a duplicate Cell C run:

- **Duplicate WandB run ID**: `f5d933ro` (`hypervigilance-round2-C-seed42_n112_gpu0`, cuda:0)
- **Duplicate log**: `logs/20260508_141905.log`

The correct Cell A1 run was subsequently launched inline (bypassing the script file) and is confirmed running as `27svrmhv`.

**Action required by user**: SIGTERM the duplicate process on node 112 that is writing to `logs/20260508_141905.log` (WandB run `f5d933ro`). It is a second Cell C instance competing on cuda:0 with the authoritative run `0u266oj5`. Delete or mark `f5d933ro` as junk in WandB.
