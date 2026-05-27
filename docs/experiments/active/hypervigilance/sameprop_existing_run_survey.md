---
title: SameProp existing-run survey — does RPPO avoid rabbits when olfactory channels collide?
topic: hypervigilance
status: draft
created: 2026-05-07
last_updated: 2026-05-08  # analyzer appended Round 1 verdict + partial-budget actuals
phase: post-hoc-survey
wandb_tag: interoNocicept_smaProp
---

# SameProp existing-run survey — does RPPO avoid rabbits when olfactory channels collide?

> **Status**: COMPLETE (post-hoc, single seed)
> **Date**: 2026-05-07
> **Author**: experiment-analyzer
> **Mode**: B — post-hoc, no pre-registered design doc
> **Related**:
> - Config under test: `configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml`
> - Senior-developer follow-up plan (parallel work): `docs/develop/active/hypervigilance/sameprop_discriminating_channels.md` (referenced by path even if not yet written)

> **POST-HOC NOTICE**: this analysis was framed retroactively. There is no
> pre-registered hypothesis, no power calculation, n = 1 seed. Conclusions are
> weaker than for a pre-registered experiment and should be treated as
> motivation for new training rather than a final answer.

---

## 1. Research Question

When the olfactory `properties` vector is identical for the patrolling predator
and the neutral rabbit (both `[0.0, 1.0, 0.0, 0.0, 0.0]`), does RPPO learn to
avoid the rabbit despite getting no informative olfactory cue, or does it leak
the predator's olfactory signature onto the rabbit?

> **H₀** (null): on the sameProp config, RPPO converges to a policy that
> avoids both the predator and the rabbit equivalently — i.e. it learns
> avoidance from a non-olfactory cue (visual, collision, nociception).
>
> **H₁** (alternative, user's claim): on the sameProp config, RPPO **fails**
> to avoid the rabbit (and, since rabbit and predator share the only
> long-range exteroceptive cue, fails to avoid the predator at distance as
> well). Performance is dominated by short-range collision/nociception
> reactivity rather than learned olfactory avoidance.

A direct test would compare `MeanDistRabbit` and `MeanDistPredator` against
each other and against a random-policy baseline. **No such metric is logged
in the existing run** — see §5.3.

## 2. Experimental Design

This is a survey of pre-existing training data, not a designed experiment.

### 2.1 Independent Variables

None — single config, single agent, single run.

### 2.2 Controlled Variables (i.e. config under test)

```yaml
# configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml
predators:
  - name: predator
    properties: [0.0, 1.0, 0.0, 0.0, 0.0]   # same as rabbit
neutral_animals:
  - name: rabbit
    properties: [0.0, 1.0, 0.0, 0.0, 0.0]   # same as predator
hiding_predator:
    properties: [0.0, 0.0, 0.0, 0.0, 0.0]   # no olfactory signature
food:
    properties: [1.0, 0.0, 0.0, 0.0, 0.0]   # discriminating channel
```

The agent therefore has:
- **Channel 0 (food)** — discriminating, working signal
- **Channel 1 (predator + rabbit)** — collides, ambiguous
- **Channels 2-4** — unused

### 2.3 Confounds & Limitations

| Confound | Severity | Note |
|---|---|---|
| n = 1 seed | High | Cannot estimate seed dispersion; final survival 342 steps may be ±50 across seeds. |
| No `MeanDistRabbit` / `hit_neutral` log | High | Agent–rabbit avoidance is **not directly observable** in this run. Only predator and food distances are logged. |
| Confounded with hiding-predator nociception | Medium | Hiding predators (4 corners) emit nociception 0.9 on contact; some learned avoidance may come from those rather than the patrolling predator. |
| Visual sensor radius = 0 | Low | Agent has no visual at distance, so olfaction + collision are the only long/short-range cues. |
| RPPO config not inspected for entropy schedule etc. | Low | Loss curves are healthy (see §4.2), so unlikely to invalidate. |

## 3. Launch Manifest

Pre-existing run, located by greppping `wandb/run-*/files/wandb-metadata.json`
for `01-interoNocicept_sameProp.yaml`. Exactly **one** RPPO match found.

| Run | Status | Cell | Tag | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|-----|------|------|-----|-------------|--------------|----------|
| 1 | completed | sameProp | `interoNocicept_smaProp` | unknown (default) | docker | NVIDIA RTX 6000 Ada | 2026-05-06T06:09:49Z | `rmw6m8zg` | `wandb/run-20260506_150949-rmw6m8zg/` |

Walltime 54,255 s ≈ 15.1 h; iteration 188,040; 3.08e9 timesteps;
9,999,705 episodes; 128 envs; 10 M episode budget; git commit `836f90eb`.

### 3.1 Configs

| Run | Config (env) | Config (agent) |
|-----|--------------|----------------|
| 1 | `configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml` | `configs/models/recurrent_ppo/recurrent_ppo.yaml` |

### 3.2 Sibling RPPO runs (NOT sameProp, reference only)

Found by grepping for any `interoNocicept` config:

| WandB ID | Config | Notes |
|----------|--------|-------|
| `hjhtmmlf` (20260501_050424) | `01-interoNocicept_noise.yaml` | predator `[0,0.7,0.5,0,0]`, rabbit `[0,0.5,0.7,0,0]` — channels 1+2 carry overlapping but discriminable signals |
| `haet89lw` (20260501_050337) | `01-interoNocicept_noise.yaml` | same config, different launch |

These are **not direct comparisons** to sameProp because the olfactory profiles
differ. They could anchor a "what does RPPO survival look like when the
olfactory cue actually discriminates?" baseline, but that comparison is left
for the next experiment.

## 4. Results

### 4.1 Primary Metrics (Hypothesis Test)

Final-window means (last 10 % of training, episodes ≈ 8.91 M – 9.90 M;
n_history_records ≈ 1880 in window):

| Metric | Final value | Interpretation |
|--------|-------------|----------------|
| Episode/Steps | **341.9 / 500** (68.4 %) | survival improved from ~200 → ~342 |
| Episode/MeanDistPredator | **4.456** | random-policy baseline ≈ 4.68; agent slightly *closer* than random |
| Episode/MeanDistFood | 2.366 | food-finding learned (random ≈ 4.68) |
| Episode/PredatorHits | **3.13 / ep** | ~1 hit per 110 steps — frequent contact |
| Episode/DangerHits (hiding_predator) | 3.23 / ep | hiding-predator contacts |
| Episode/DamagePredator | **94.0** | per episode; 41 % of total damage budget |
| Episode/DamageDanger (hiding_predator) | 96.8 | 42 % of damage budget |
| Episode/DamageObstacle (rocks) | 55.1 | 24 % of damage budget |
| Episode/Term_Injury | **36.3 %** | of episodes end in injury death |
| Episode/Term_Starvation | 32.9 % | starvation death |
| Episode/Term_MaxSteps | 30.8 % | timeout (the "win" condition) |

**Random baseline** for `MeanDistPredator`, computed by sampling 100 k
uniform-random agent and predator positions on the 1..10 × 1..10 grid:
**4.684 mean / 4.592 median**. The agent's 4.456 is below the uniform-random
baseline.

> **Verdict on H₁**: **partially supported, with a critical caveat**.
>
> - The agent's mean distance to the patrolling predator (4.46) is
>   indistinguishable from — slightly worse than — uniform random (4.68).
>   Across 10 training windows it stays in [4.35, 4.46]. The agent does NOT
>   learn to keep distance from the predator.
> - Since the rabbit and predator share the *only* long-range exteroceptive
>   cue (channel 1 of olfaction), and the agent has no useful predator-
>   avoidance policy, **by transitivity the agent has no avoidance signal
>   for the rabbit either**. The user's intuition holds.
> - **Caveat**: this conclusion is indirect. The run logs no per-rabbit
>   distance or per-rabbit collision count, so we cannot quote
>   `MeanDistRabbit` or `RabbitHits`. The transitive argument is sound but
>   the direct measurement is missing — see §5.3 / Metrics Requested.

### 4.2 Secondary Metrics

| Metric | Final | Trajectory | Note |
|--------|-------|-----------|------|
| Episode/FoodEaten | 58.4 / ep | 25 → 58 | clear food-finding learning |
| Episode/RestCount | 106.7 | 80 → 107 | majority of survival is rest |
| Episode/Reward | -201.4 | -213 → -201 | dominated by death penalty |
| loss/total | 0.139 | 0.11 → 0.14 | stable, no divergence |
| loss/value | 0.287 | 0.23 → 0.29 | stable |
| loss/entropy | -0.497 | -0.53 → -0.50 | policy entropy stayed near initial — agent did not collapse to deterministic |
| loss/grad_norm | 0.279 | 0.50 → 0.28 | well-behaved |

Training health is fine. The failure is policy-shape, not optimization.

### 4.3 Diagnostic Metrics

None — RPPO doesn't expose modulator gates or world-model internals.

### 4.4 Learning Dynamics

Windowed by `Episode/Number` into 10 equal windows.

```
Episode/Steps:           202.4 → 275.9 → 298.8 → 314.0 → 321.7 → 330.7 → 336.0 → 339.5 → 338.3 → 341.9
Episode/MeanDistPredator: 4.42 →  4.35 →  4.37 →  4.41 →  4.43 →  4.45 →  4.46 →  4.45 →  4.45 →  4.46
Episode/PredatorHits:     2.87 →  3.28 →  3.43 →  3.37 →  3.23 →  3.16 →  3.17 →  3.19 →  3.13 →  3.13
Episode/DamagePredator:   86.2 →  98.4 → 102.8 → 101.0 →  97.0 →  94.9 →  94.9 →  95.8 →  93.7 →  94.0
Episode/Term_Injury:      0.226→ 0.355 → 0.369 → 0.383 → 0.380 → 0.367 → 0.372 → 0.379 → 0.368 → 0.363
Episode/Term_Starvation:  0.724→ 0.497 → 0.437 → 0.387 → 0.369 → 0.357 → 0.337 → 0.319 → 0.334 → 0.329
Episode/FoodEaten:        25.0 →  42.7 →  47.3 →  51.4 →  53.5 →  55.6 →  56.9 →  58.0 →  57.5 →  58.4
```

Two qualitatively distinct learning trajectories:

1. **Food / starvation curve**: monotonic improvement, plateau by window ~7.
   The agent learned olfactory-driven food approach.
2. **Predator-distance curve**: essentially flat at the random-policy
   baseline from window 1 onwards. There is *no* learning of predator
   avoidance.

`Term_Injury` *rises* over training (0.23 → 0.36), driven by episodes
surviving long enough to die from injury rather than starve. The injury death
rate is not falling.

## 5. Analysis

### 5.1 Key Findings

**Finding 1 — Predator-distance curve is flat at random-policy baseline.**
- *What*: `MeanDistPredator` ∈ [4.35, 4.46] across all 10 training windows; uniform-random baseline ≈ 4.68.
- *Why*: with channel 1 (the only useful long-range olfactory channel) emitting an identical signal from a thing-that-kills-you and a thing-that-doesn't-matter, the gradient of the value function with respect to "approach the source of channel 1" is approximately zero. The agent cannot learn approach OR avoidance from this channel, because the same action sometimes earns food (when rabbit is nearby) and sometimes earns 30 damage (when predator is nearby).
- *Evidence*: `MeanDistPredator` window means; `loss/entropy` stays near initial (-0.50), consistent with the agent treating channel-1 olfactory states as low-information.
- *Confidence*: High — direct comparison with random baseline is unambiguous; trajectory across 10 windows shows no learning.

**Finding 2 — Food-finding works because food's olfactory channel (0) is unambiguous.**
- *What*: `MeanDistFood` 4.10 → 2.37, `FoodEaten` 25 → 58, `Term_Starvation` 72 % → 33 %.
- *Why*: channel 0 is emitted only by food, no confound. Standard policy gradient extracts the gradient and the agent learns approach.
- *Evidence*: trajectory above.
- *Confidence*: High.

**Finding 3 — Survival improvement is driven by food, not by predator avoidance.**
- *What*: episode length rises from 202 → 342 steps; the gain is entirely
  attributable to fewer starvation deaths (Term_Starvation 0.72 → 0.33).
  Term_Injury actually *rises* (0.23 → 0.36) over the same window —
  a 60 % relative increase in injury-death rate.
- *Why*: as agents survive longer, they accumulate more predator hits (constant
  hit-rate × longer episode = more total damage), and become more likely to
  cross the injury threshold before timeout.
- *Evidence*: termination breakdown trajectory.
- *Confidence*: High.

**Finding 4 — Indirect argument that the rabbit is not avoided.**
- *What*: rabbit avoidance cannot be observed directly (no log), but the agent
  has demonstrably failed to use channel 1 for predator avoidance, and the
  rabbit is the strictly weaker case (no damage to learn from).
- *Why*: a policy that does not avoid a thing-that-kills-you has no instrumental
  reason to avoid a thing-that-does-nothing emitting the same olfactory signal.
- *Evidence*: §4.1 verdict, transitivity.
- *Confidence*: Medium — the transitive argument is sound but the direct
  measurement is missing.

### 5.2 Cross-Run Comparisons

Not done — only one RPPO run on sameProp exists. The two RPPO runs on
`01-interoNocicept_noise.yaml` (different olfactory profiles) could provide a
"discriminating-channels" baseline, but their config is not isolating just the
sameProp variable, so the comparison is confounded.

### 5.3 Failure Modes & Pathologies

**P1 — Olfactory-channel collision pathology.**
This is what the experiment was set up to expose. The agent cannot extract
informative gradient from a channel that fires for both reward-positive and
reward-negative entities at random ratios. The pathology shows up in
`MeanDistPredator ≈ random`. Phase 3 (new training with discriminating
channels) is the proper next step.

**P2 — Behavioural-metric blind spot.**
There is no log of `dist_to_neutral`, `hit_neutral`, or rabbit-cell visit
count. Any future hypervigilance experiment will be forced into the same
transitive-only argument unless this is fixed first. See Metrics Requested
below.

**P3 — Hiding-predator damage as confound.**
`Episode/DamageDanger` (96.8) ≈ `Episode/DamagePredator` (94.0). Half the
damage the agent absorbs comes from non-olfactory hiding predators (corner
ambushers with `properties=[0,0,0,0,0]`). Any "the agent doesn't avoid the
predator" claim has to factor out the contribution from these cue-less
ambushers. The patrolling predator's contribution is ~94 / 247 ≈ 38 % of
total damage, *not* the dominant share.

## 6. Conclusions

### 6.1 Summary

- **One** RPPO run exists on `01-interoNocicept_sameProp.yaml`:
  `wandb/run-20260506_150949-rmw6m8zg/`, single seed, ~15 h walltime,
  10 M episodes.
- Final survival 341.9 / 500 steps; injury-death rate 36 %.
- **MeanDistPredator stays at the uniform-random baseline (4.46 vs 4.68)**
  across the entire 10 M episode run. Predator-avoidance never learns.
- Food-finding learns cleanly (channel 0 is unambiguous).
- The user's claim that the agent fails to avoid rabbits is **transitively
  supported but not directly observable** — `MeanDistRabbit` is not logged.
- **Phase 3 (new training with discriminating channels) is justified.**

### 6.2 Limitations & Open Questions

- n = 1 seed; cannot estimate variance across seeds.
- No direct rabbit-avoidance metric. The whole argument is mediated by
  predator avoidance + olfactory-channel transitivity.
- Did not verify the `recurrent_ppo.yaml` hyperparameters in detail.
- Sibling `noise` configs are not direct sameProp counterparts and were
  not analyzed in depth.

### 6.3 Recommended Next Experiments

| Priority | Experiment | Rationale | Effort |
|----------|-----------|-----------|--------|
| **P0** | Add `dist_to_neutral` and `hit_neutral` to `info` in `core.py` and `Episode/MeanDistRabbit`, `Episode/RabbitHits` to the trainers | Without this metric, every future hypervigilance analysis is blocked at the transitive argument. | small (logger only — see Metrics Requested below) |
| P1 | Re-run sameProp with the new metric + 5 seeds, then run a `discriminating` variant where predator + rabbit have separable olfactory signatures. Direct comparison resolves the question. | Pre-registered design, n ≥ 5, isolates the channel-discriminability variable. | medium (1 designer doc + 10 runs) |
| P2 | Compare RPPO sameProp vs the existing `01-interoNocicept_noise.yaml` runs to anchor "RPPO survival when olfaction discriminates" | Establishes performance ceiling for the architecture | small (re-extract noise runs) |

## Metrics Requested

| Subfield | Content |
|---|---|
| **Metric (1)** | `Episode/MeanDistRabbit` (cells, L2 to nearest neutral animal) — analogue of `MeanDistPredator`. |
| **Metric (2)** | `Episode/RabbitHits` (count per episode of agent stepping onto a rabbit cell) — analogue of `PredatorHits`. Rabbits don't damage, but the contact-count is the cleanest "did you avoid them?" metric. |
| **Why now** | The whole sameProp / hypervigilance question hinges on per-entity behaviour. Without these, every analysis is forced into a transitive predator-distance argument that confounds with hiding predators (50 % of damage). |
| **Where it'd live** | `src/environment/core.py` (~line 491-494, alongside `dist_to_food` and `dist_to_pred`); plumbed through `info` dict; surfaced in `src/models/recurrent_ppo_trainer.py` (~line 19/202) and `src/models/dreamer_v3_trainer.py` (~line 624) `EpisodeInfo` / step trackers; logged in `train.py` (~line 1241-1253). |
| **Cost** | Cheap — one `jnp.min(jnp.linalg.norm(state.neutral_pos - new_agent_pos, axis=-1))` per step + one `jnp.any(jnp.all(state.neutral_pos == new_agent_pos, axis=-1))` boolean. Same shape and cost as the existing predator equivalents. |

If accepted, hand off to `senior-developer` → `developer` via `feature-workflow`.

## Related Issues

- **Cross-link** to `docs/develop/active/hypervigilance/sameprop_discriminating_channels.md` (parallel work by `senior-developer`). When that doc lands, link it both ways.
- **Possible bug / design issue**: the user's interpretation (rabbit and predator should have non-overlapping properties for hypervigilance to be testable) is implied by the sameProp config name itself — sameProp is the *control*, not the production setting. Verify that elsewhere in the project (if any code or doc treats sameProp as a real evaluation condition rather than a sanity check, that's the bug).

---

## Appendix

### A. Raw Data Tables

Window summary JSON: `tmp/20260507_sameProp_windows.json`
Long-format CSV (per-step values, 18 MB): `tmp/20260507_sameProp_metrics.csv`
Working notes: `tmp/20260507_213500_sameprop_survey_workingfile.md`

### B. Config Diffs

The relevant difference between `01-interoNocicept_sameProp.yaml` (this run)
and `01-interoNocicept_noise.yaml` (sibling RPPO runs):

| Entity | sameProp | noise |
|--------|----------|-------|
| predator.properties | `[0.0, 1.0, 0.0, 0.0, 0.0]` | `[0.0, 0.7, 0.5, 0.0, 0.0]` |
| rabbit.properties | `[0.0, 1.0, 0.0, 0.0, 0.0]` | `[0.0, 0.5, 0.7, 0.0, 0.0]` |
| sensory.perceptual_noise.enabled | `false` | (likely true — not verified) |

### C. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-05-07 | Initial post-hoc analysis on the single existing RPPO run | experiment-analyzer |
| 2026-05-07 | Appended Round 1 re-logged baseline launch manifest (two new RPPO runs, node 112) | training-runner |
| 2026-05-08 | Filled Round 1 partial-budget actuals (final episodes, walltime, status: stopped early) and added Round 1 Verdict cross-link to `round1_relog_baseline_analysis.md` | experiment-analyzer |

---

## Round 1 — Re-logged Baseline Launch Manifest

**Purpose**: Re-run the existing sameProp baseline with the new per-entity avoidance logging (commit `4b55fc6`, branch `v1.3`). Two seeds on node 112 to capture seed variance and enable a direct comparison to the original single-seed run (`rmw6m8zg`, seed=42).

**Config**: `configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml` (identical to original; not edited).

**Agent config**: `configs/models/recurrent_ppo/recurrent_ppo.yaml`.

**Budget**: 10,000,000 episodes, 128 parallel envs, matching original run.

### Launch Table

| Cell | Role | Seed | Node | GPU | WandB tag | WandB run ID | WandB name | Launched at | Log path | Final episodes | Walltime | Status |
|------|------|------|------|-----|-----------|--------------|------------|-------------|----------|----------------|----------|--------|
| R1-seed42 | reproducibility (matches original) | 42 | 112 | cuda:0 | `hypervigilance-sameprop-relog-seed42_n112_gpu0` | `rg5nl1ov` | `hypervigilance-sameprop-relog-seed42_n112_gpu0` | 2026-05-07T22:31:07 | `logs/20260507_223107.log` | 7,284,221 / 10 M | 54,974 s ≈ 15h 16m | stopped early (user, asymmetry pattern stable) |
| R1-seed43 | second-seed cell | 43 | 112 | cuda:1 | `hypervigilance-sameprop-relog-seed43_n112_gpu1` | `6ks4bjbq` | `hypervigilance-sameprop-relog-seed43_n112_gpu1` | 2026-05-07T22:35:41 | `logs/20260507_223536.log` | 7,212,724 / 10 M | 54,712 s ≈ 15h 12m | stopped early (user, asymmetry pattern stable) |

### Round 1 Verdict (cross-link)

Full Round 1 analysis: [`round1_relog_baseline_analysis.md`](round1_relog_baseline_analysis.md). Headline: rabbit-vs-predator distance asymmetry (`MeanDistPredator − MeanDistRabbit ≈ 0.63` cells) reproduces across both seeds and grows with training, confirming the user's claim *behaviorally* — but a critical food/rabbit spawn-quadrant co-location confound (food TL+BR ≡ rabbit TL+BR; predator full-grid) means Round 1 cannot attribute the asymmetry to olfactory channel discriminability vs food-seeking spillover. Round 2 must decouple food and rabbit spawn areas (or match predator patrol area to rabbit quadrants) to isolate the mechanism.

### WandB URLs

- R1-seed42: https://wandb.ai/sungwoolee/grid_world_pain/runs/rg5nl1ov
- R1-seed43: https://wandb.ai/sungwoolee/grid_world_pain/runs/6ks4bjbq

### Tag → Cell Binding (for analyzer dereference)

| Tag | Cell | Seed | GPU | Role |
|-----|------|------|-----|------|
| `hypervigilance-sameprop-relog-seed42_n112_gpu0` | R1-seed42 | 42 | cuda:0 | reproducibility — matches original run `rmw6m8zg` |
| `hypervigilance-sameprop-relog-seed43_n112_gpu1` | R1-seed43 | 43 | cuda:1 | second-seed cell |

### Notes on This Launch

- **Node hardware**: node 112 has RTX 3090 (24 GB each) vs. original node's RTX 6000 Ada (51 GB). With `--num-envs 128`, peak VRAM on GPU 0 is ~5.5 GB at steady state — well within limits.
- **CIFS cache incident**: a duplicate run (`gh1cz01q`) briefly launched on cuda:0/seed=42 due to CIFS attribute-cache lag on node 112 reading the stale script. It was killed (SIGKILL) within ~2 min before any checkpoints were written. `gh1cz01q` should be treated as aborted / ignore in WandB.
- **Run 2 launched via inline command**: to bypass the CIFS cache issue, `run_command.py` for R1-seed43 was invoked with the full command string inline rather than relying on `bash train_command-agent.sh`. The two runs are functionally identical to what `train_command-agent.sh` (current state: seed=43/cuda:1) would have produced.
- **`train_command-agent.sh` final state**: reflects seed=43/cuda:1 (the last run). Analyzer / next runner should note this and update before a subsequent launch.
