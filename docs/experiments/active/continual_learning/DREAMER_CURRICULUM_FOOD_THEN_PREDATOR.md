---
title: "DreamerV3 hypervigilance recovery — 3-stage food→predator curriculum"
topic: continual_learning
status: active
created: 2026-05-07
last_updated: 2026-05-07T18:00:15
phase: 1
wandb_tag: "dreamer_curriculum_food_then_predator_s0"
---

# DreamerV3 hypervigilance recovery — 3-stage food→predator curriculum

> **Status**: COLLECTING
> **Date**: 2026-05-07
> **Author**: experiment-designer
> **Related**:
> - Diagnosis: [`docs/develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md`](../../../develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md)
> - Schema-add plan (cont_loss_weight knob): [`docs/develop/active/diagnosis/dreamer_cont_loss_weight_knob.md`](../../../develop/active/diagnosis/dreamer_cont_loss_weight_knob.md)
> - Reference demo (continual plumbing): [`docs/experiments/active/continual_learning/CONTINUAL_5X5_NOPRED_TO_PREDINT3_DEMO.md`](./CONTINUAL_5X5_NOPRED_TO_PREDINT3_DEMO.md)
> - Stage 1 config: [`configs/experiment/dreamer_curriculum/01_food_only.yaml`](../../../../configs/experiment/dreamer_curriculum/01_food_only.yaml)
> - Stage 2 config: [`configs/experiment/dreamer_curriculum/02_predator_slow.yaml`](../../../../configs/experiment/dreamer_curriculum/02_predator_slow.yaml)
> - Stage 3 config: [`configs/experiment/dreamer_curriculum/03_predator_full.yaml`](../../../../configs/experiment/dreamer_curriculum/03_predator_full.yaml)
> - Schedule: [`configs/continual/dreamer_curriculum_food_then_predator.yaml`](../../../../configs/continual/dreamer_curriculum_food_then_predator.yaml)
> - Agent config: [`configs/models/dreamer_v3_curriculum.yaml`](../../../../configs/models/dreamer_v3_curriculum.yaml)

> **Note**: Single-seed pilot. The result is observational (not statistical). The
> deliverable is a single WandB run whose Stage 3 segment shows survival
> meaningfully above the documented failing-run floor (~30 steps), which would
> be a positive signal that the two-lever + curriculum intervention can recover
> DreamerV3 from hypervigilance. A multi-seed follow-up is the next step if
> Stage 3 survival exceeds 100 (see §6).

---

## 1. Research Question

The diagnosis [`dreamer_hypervigilance_learning_failure.md`](../../../develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md)
established that, at the failing run's settings, DreamerV3 on the 5×5 task
collapsed two ways simultaneously:
- **Reward-head asymmetry** (`mae_pos: 4.56 → 0.37` vs `mae_neg: 5.17 → 4.06`)
  — the head learns the eating-tail but never resolves the heavy-negative
  death-penalty mass.
- **Continuation-head class imbalance** (98.4% acc ≈ 96.3% trivial baseline)
  — imagined rollouts almost never terminate, so the actor never sees the
  death-penalty gradient.

Both pathologies are downstream of the WM being trained on a replay buffer
that contains essentially only short death-trajectories with near-deterministic
reward (std 0.34 across ~1M episodes). **The death-loop is self-locking.**

This experiment asks: **can a 3-stage curriculum that (a) gives the actor
a death-event-free warm-up to learn food-finding, (b) gradually introduces
predator dynamics with weakened lethality, and (c) only then exposes the
agent to the full failing task — combined with the diagnosis's two priority
levers (`entropy_scale` 3e-4 → 1e-3 and `cont_loss_weight` 1.0 → 5.0) — break
the hypervigilance lock-in?**

> **H₀** (null): "DreamerV3 on Stage 3 with curriculum + levers performs
> indistinguishably from DreamerV3 on the same task without curriculum
> (failing-run baseline survival ≈ 30 steps)."
>
> **H₁** (alternative): "Curriculum + levers produce Stage 3 final survival
> above 100 mean steps and a non-flat learning curve, breaking the
> hypervigilance regime."

H₁ is a **soft, single-seed** prediction in this run. A 3-seed repeat would
elevate it to statistical; this pilot is a yes-feasible / no-not-yet check.

## 2. Experimental Design

### 2.1 Independent Variables

This is a **two-lever × curriculum** intervention vs the documented failing
baseline. The factors are confounded by design (we are not isolating which
lever does the work in this run; that is the next experiment); the question
here is whether the *combined* package recovers learning at all.

| Variable | Failing baseline (qont5dac) | This experiment | Rationale |
|----------|---|---|---|
| Curriculum | none (single config = 03 anchor verbatim) | 3 stages: 01 food-only → 02 predator_slow → 03 predator_full | Per diagnosis §6.1, the death-loop is self-locking once the actor is uniform. A pre-training stage with no death events lets the WM/actor leave the random regime before any death-penalty samples enter the buffer. |
| `agent.entropy_scale` | `3e-4` (Hafner-2023 default) | `1e-3` (~3.3× boost) | Per diagnosis §4.3.4, rPPO's effective entropy bonus is ~30× stronger than DreamerV3's. We pick the lower edge of the user's range (1e-3) so the actor maintains exploration without collapsing to noise. |
| `agent.cont_loss_weight` | `1.0` (pre-knob behavior; the knob just landed) | `5.0` (5× boost) | Per diagnosis §4.3.3 / §7.3 priority 2, the continuation head exploits class imbalance. 5× is the conservative end of the diagnosis's "10×" recommendation; we pair it with Stage 2 which has substantially higher death-event density per step than Stage 3. |

### 2.2 Controlled Variables

Held constant across the 3 stages (verified by the cross-stage modality
fingerprint match — see §4):

```text
# Sensor / observability flags (from configs/environment/default.yaml + stage merge)
sensory.olfactory_enabled              = True
sensory.nociception_enabled            = True
sensory.proprioception_enabled         = True
sensory.visual_sensor_enabled          = False
sensory.location_sensor                = False        # <- location_sensor_enabled in EnvParams
sensory.interoceptive_nociception_enabled = True
sensory.injury_observable              = False
sensory.nutrition_observable           = False
sensory.vector_size                    = 5
sensory.nociception_size               = 1
sensory.collision_sensor_range         = 1
sensory.visual_sensor_range            = 0
visualization.local_view_size          = 5

# Body / homeostatic regime
body.with_satiation                    = True
body.with_nutrition                    = True
body.with_injury                       = True
body.use_homeostatic_reward            = True
body.food_nutrition_gain               = 18
body.metabolic_cost                    = 1.0
body.death_penalty                     = 100
body.injury_smoothing_duration         = 3

# Environment / action enables
environment.height                     = 5
environment.width                      = 5
environment.max_steps                  = 500
environment.rest_action_enabled        = True
environment.eat_action_enabled         = True
environment.random_start_pos           = True

# Noise
perceptual_noise.enabled               = False
```

**The five observability flags listed first are inherited from
`configs/environment/default.yaml` and are NOT in the stage-overlay YAMLs.**
This is by design (matches the basic configs the curriculum stages were
copied from). The cross-stage live-load probe (§4) confirms they end up
identical across all three stages after the default merge.

Agent-side (controlled by `configs/models/dreamer_v3_curriculum.yaml` applied
to all three stages):

```text
agent.algorithm           = "DreamerV3"
agent.batch_size          = 16
agent.sequence_length     = 128
agent.replay_ratio        = 0.5
agent.collect_interval    = 128
agent.train_steps         = 64
agent.model_lr            = 1e-4
agent.actor_lr            = 3e-5
agent.value_lr            = 3e-5
agent.sampling_mode       = "mixture"  (5 positive + 5 recent + 6 random slots)
agent.encoding_mode       = "hierarchical"
agent.modulation.type     = null  (baseline DreamerV3, no neuromodulation)
agent.entropy_scale       = 1e-3   (** EXPERIMENTAL **)
agent.cont_loss_weight    = 5.0    (** EXPERIMENTAL **)
agent.unimix              = 0.01
agent.use_layer_norm      = True
```

Training side: `num_envs = 16`, `seed = 0`, `--log-interval = 50`,
`device = cuda:3`, schedule total = 766,000 episodes.

### 2.3 Per-stage env-step budget (back-calculation)

Anchor: failing DreamerV3 run (qont5dac in [diagnosis §3](../../../develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md#3-run-inventory)) ran 27.1 M
env-steps in 3 h 13 min = **2,330 env-steps/sec** at num_envs=16 on cuda:3.
Budget for this run: ~4 h ≈ 33.5 M env-steps.

| Stage | Episodes (delta) | Estimated mean ep length | Estimated env-steps | Cumulative |
|---|---:|---:|---:|---:|
| 01 food_only       | 16,000  | ~150 steps | ~2.4 M | 2.4 M |
| 02 predator_slow   | 50,000  | ~100 steps | ~5.0 M | 7.4 M |
| 03 predator_full   | 700,000 | ~30–40 steps (early) → longer if recovery | ~25 M | ~32 M |

**Total ≈ 32–33 M env-steps**, comfortably within the 4 h target. The episode-
length estimates are deliberately conservative on the floor side (assume
no recovery happens). If Stage 3 recovery does occur, episodes will be
longer and the actual env-step budget will run *higher* but the episode
count target will stay.

### 2.4 Confounds & Limitations

| Confound | Affected runs | Severity | Mitigation |
|----------|---------------|----------|------------|
| Single seed | the only run | High for any quantitative claim, OK for feasibility pilot | Pre-registered as observational. A 3-seed follow-up at seeds {0, 7, 13} is queued (see §6). |
| Two-lever + curriculum confound | the only run | High for attribution | This is intentional; the diagnosis recommends *both* levers + curriculum. If the package works, follow-up ablations can isolate (entropy-only / cont-only / curriculum-only). If it fails, the *package* hypothesis is refuted as a whole. |
| Replay buffer cleared at each stage transition (DreamerV3-specific) | Stage 2/3 starts | Medium | Documented behavior of the continual-feature for off-policy algos; without clearing, Stage 1's no-death replay would contaminate Stage 2/3 termination learning. The clearing trades sample-efficiency for clean WM signal — which is the entire point of the curriculum. |
| Mid-episode envs at boundary are silently dropped | Stage 1→2, Stage 2→3 boundaries | Low | At most num_envs−1 = 15 episodes' work is lost at each boundary (drift up to 15 from the nominal). |
| `--num-envs=16` matches the failing baseline but is small | the only run | Low | Matches the failing-baseline anchor (qont5dac), which is the comparison we care about. Larger num_envs would change the answer to "did sample efficiency mask the failure", which is a separate question. |
| One mid-run env rebuild forces JIT recompile (~30–60 s × 2 boundaries) | Stage transitions | Low | Wall-clock blip only, not a learning artifact. Curves are aligned to `Episode/Number`. |
| Stage-2 lever values not pre-validated by an in-env probe | Stage 2 | Medium | The slowed-predator parameters are designer choices; they *might* be too easy (agent never sees a death event) or too hard (still locked-in). The Stage-2 success criterion in §5 calls this out explicitly. |

## 3. Launch Manifest

System-of-record for every training run in this experiment.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | running | curriculum_food_then_predator | `dreamer_curriculum_food_then_predator_s0` | `dreamer_curriculum` | `curriculum` | 0 | 114 | cuda:3 | 2026-05-07T18:00:15 | 95mpudwp | logs/20260507_180004.log |

**WandB tags (extra, not the run name)**: `["dreamer", "continual", "curriculum", "5x5"]` — applied via the standard `--wandb-tags` channel if available, or post-hoc on the WandB UI side. The Tag (= wandb-name) above is the unique run identifier used by `train.py`'s naming logic.

### 3.1 Configs to Produce (designer-only, pre-launch)

| Run | Schedule | Configs-dir | Agent config |
|-----|----------|-------------|--------------|
| 1 | `configs/continual/dreamer_curriculum_food_then_predator.yaml` | `configs/experiment/dreamer_curriculum/` | `configs/models/dreamer_v3_curriculum.yaml` |

All five files are produced fresh in this experiment (no edits to pre-existing `configs/experiment/basic/` or `configs/models/dreamer_v3.yaml`).

### 3.2 Exact launch command

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --configs-dir configs/experiment/dreamer_curriculum/ \
  --continual-schedule configs/continual/dreamer_curriculum_food_then_predator.yaml \
  --agent_config configs/models/dreamer_v3_curriculum.yaml \
  --num-envs 16 \
  --seed 0 \
  --device cuda:3 \
  --log-interval 50 \
  --wandb-group dreamer_curriculum \
  --wandb-job-type curriculum \
  --wandb-name "dreamer_curriculum_food_then_predator_s0" \
  --tag "dreamer_curriculum_food_then_predator_s0"
```

Note:
- `--episodes` MUST NOT be passed when `--configs-dir` is set; the schedule's
  last boundary (766,000) is the budget.
- `--checkpoint-frequency` MUST NOT be passed; per-stage frequencies come
  from the schedule YAML.
- `--config` MUST NOT be passed (mutually exclusive with `--configs-dir`).
- `--log-interval 50` is more conservative than the demo's 5 because Dreamer's
  iter cost is higher and Stage 3's 700K episodes will produce many log points
  even at 1/50.

## 4. Cross-stage modality fingerprint — LIVE-VERIFIED

Mirrors `train.py`'s `_build_continual_schedule` + `_modality_fingerprint(p)`
flow. The probe at `tmp/20260507_dreamer_curriculum_probe.py` was run with the
project conda env on this commit:

```
$ /home/vncuser/miniconda3/envs/grid_world_pain/bin/python tmp/20260507_dreamer_curriculum_probe.py
```

**Result (verbatim from probe stdout):**

| Stage | obs_dim | action_dim | 13-tuple fingerprint |
|---|---:|---:|---|
| 01_food_only      | 19 | 6 | `(False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)` |
| 02_predator_slow  | 19 | 6 | `(False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)` |
| 03_predator_full  | 19 | 6 | `(False, 0, 5, True, 5, True, 1, True, False, True, False, False, 1)` |

**Cross-stage match: OK (all stages identical).**

Tuple decoding (in the order `_modality_fingerprint` returns):

```text
( visual_sensor_enabled = False,
  visual_sensor_range   = 0,
  local_view_size       = 5,
  olfactory_enabled     = True,
  olfactory_vector_size = 5,
  nociception_enabled   = True,
  nociception_size      = 1,
  interoceptive_nociception_enabled = True,
  location_sensor_enabled = False,
  proprioception_enabled = True,
  injury_observable     = False,
  nutrition_observable  = False,
  sensor_range          = 1 )
```

The five observability flags called out by the auditor in the previous demo
(`interoceptive_nociception_enabled`, `injury_observable`, `nutrition_observable`,
`visual_sensor_enabled`, `location_sensor_enabled`) all appear in this fingerprint
and are identical across all three stages.

`obs_dim = 19` matches the failing-baseline observation space exactly (per
diagnosis §2.2: Sat=1, IntNoci=1, ExtNoci=1, Olf=5, Coll=5, Prop=6 = 19).
`action_dim = 6` likewise (4 movement + rest + eat).

Schedule YAML validation (live-checked via `Config.get_mandatory`):
- `len(episode_boundaries) == 3 == num_stages` ✓
- `len(checkpoint_frequencies) == 3 == num_stages` ✓
- boundaries strictly increasing: `[16000, 66000, 766000]` ✓
- `boundaries[0] > 0` ✓
- all checkpoint frequencies > 0 ✓
- per-stage episode delta: `[16000, 50000, 700000]`

Agent-config keys (live-checked):
- `agent.entropy_scale = 0.001` (float) ✓
- `agent.cont_loss_weight = 5.0` (float) ✓ — the trainer reads this via
  `config.get_mandatory('agent.cont_loss_weight', float)` since the schema
  add at HEAD~1 (see [`docs/develop/active/diagnosis/dreamer_cont_loss_weight_knob.md`](../../../develop/active/diagnosis/dreamer_cont_loss_weight_knob.md))

## 5. Pre-registered confirmation / refutation criteria

The intervention is **the entire package** (curriculum + entropy_scale lever +
cont_loss_weight lever). Per-criterion:

### 5.1 Confirmation criteria (any single sufficient → H₁ supported, observational)

- **C1 (primary, Stage 3)**: Stage 3 final-1000-episode mean survival > **100 steps**.
  Threshold rationale: failing-baseline Stage 3 floor was ~30 steps. Doubling
  is the minimum signal that the curriculum changed something material;
  100+ rules out "noise" given the baseline std of 0.8.
- **C2 (Stage 1 plumbing)**: `WorldModel/model_reward_mae_neg` improves by
  **≥ 2× during Stage 1 alone** (Stage-1 final/Stage-1 initial). Threshold
  rationale: the failing run achieved only 1.27× over the entire 27 M
  env-steps. A 2× reduction in Stage 1's much smaller env-step budget would
  show that the food-only stage successfully exposes the reward head to
  enough of the heavy-negative tail to break the asymmetry.
- **C3 (actor unfreeze)**: `Behavior/mean_advantage` Stage 1 std-across-time
  is **> 0.05** (vs the failing-baseline ~0.005). A 10× spread relative to
  the failing constant-advantage signature would confirm the actor receives
  per-action-distinguishable gradient.
- **C4 (Stage 2 bridge effect)**: `Episode/Term_Injury` rate at Stage 2 end
  is **< 90%** AND `Episode/Term_MaxSteps` rate is **> 5%**. Failing baseline
  was 99% Term_Injury / 0% Term_MaxSteps for the entire run. Any non-trivial
  presence of MaxSteps terminations in Stage 2 means the agent has learned
  *some* avoidance under the easier predator regime.

### 5.2 Refutation criteria (any single sufficient → H₁ refuted)

- **R1 (Stage 1 actor stays uniform)**: `Behavior/mean_advantage` Stage 1
  end-window std **< 0.01** AND `Behavior/mean_entropy` Stage 1 end > **1.7**
  (≈ ln(6) = 1.79). If even the food-only easy stage cannot get the actor
  off uniform, the diagnosis's actor-frozen finding holds independent of
  death events, and the curriculum approach is not the right fix.
- **R2 (Stage 3 collapse despite Stage 2 success)**: Stage 2 satisfies C4
  (predator avoidance learned), but Stage 3 final-1000-episode mean survival
  collapses to **< 50 steps**. Would indicate the slowed-predator regime is
  too far from the full task to transfer (Stage 2 ≠ Stage 3 in the relevant
  feature).
- **R3 (training instability)**: NaN warnings in any stage, or
  `loss_model` rises monotonically over **any 1 M env-step window** in any
  stage. Refutes the run, not the hypothesis (re-run with same config; if
  reproducible, refutes the hypothesis).
- **R4 (cont_loss_weight saturation)**: `WorldModel/loss_cont` rises above
  **1.0** at any point in Stage 2 (5× a value that was 0.07 → 0.04 in the
  failing baseline). Would indicate the continuation head is unable to
  satisfy the higher-weighted loss given its capacity, and the lever was
  too aggressive.
- **R5 (entropy collapse)**: `Behavior/mean_entropy` drops **below 0.5**
  in any stage. With `entropy_scale = 1e-3` and a 6-action policy, a healthy
  trained DreamerV3 should sit between ~0.5 and ~1.5; below 0.5 means the
  policy has prematurely committed (would be unexpected here given the
  *upward* lever direction).

### 5.3 Failure-mode catalog (pre-decided interpretations)

| Observation | Interpretation |
|---|---|
| Stage 1 survival never breaks 100 (food-only, no predator) | Food-finding is harder than expected at ~150 mean steps; budget Stage 1 longer in a follow-up. NOT a refutation of the hypothesis (refutes the budget assumption). |
| Stage 3 survival ramp matches rPPO's 17 M-env-step inflection point | Ideal positive result — full hypervigilance recovery. |
| Stage 3 survival sits at 30 steps for 5 M+ env-steps after Stage 2 success | Curriculum successfully primed the agent for a sub-task that doesn't transfer to the anchor. Recommend ablation runs (entropy-only, cont-only, no-curriculum). |
| `WorldModel/loss_kl` collapses below 0.5 in any stage | RSSM posterior/prior collapse; refute the run (re-train). Would NOT have happened in the failing baseline (1.16-1.23 throughout) so this is a new failure if it appears. |
| `Behavior/mean_advantage` flat at -0.19 in Stage 1 (matches failing baseline) | Hypervigilance is structural, NOT death-loop driven. Refutes the curriculum hypothesis at the most basic level. |
| Stage 3 starts stronger than baseline but plateaus below max_steps=500 | Partial transfer; useful info for follow-up. Quantify transfer as `(Stage 3 mean survival) / 500`. |
| `stage/transition` fires more than twice or at unexpected episode counts | Plumbing bug, not a hypothesis test. Halt; route to senior-developer. |

## 6. Analysis Plan (post-run)

Single seed → no inferential statistics. Post-run fill-in for §7 / §8 will:

1. **Plumbing checks first** (mirroring the demo's §5 pattern):
   - Confirm exactly two `stage/transition` events at Episode/Number ∈
     `[16000, 16015]` and `[66000, 66015]` (drift up to num_envs-1=15).
   - Confirm `stage/index` flips 0→1 at the first boundary, 1→2 at the
     second.
   - Confirm a single WandB run-ID; both `models/stage_00_*.yaml`,
     `models/stage_01_*.yaml`, `models/stage_02_*.yaml`, and
     `models/schedule.yaml` present.
   - Confirm replay buffer is cleared at each transition (Dreamer-specific
     log: `stage/buffer_cleared_main` should fire at each boundary).

2. **Per-stage temporal evolution** (project rule):
   - For each stage, compute 20 windows of: `Episode/Steps` mean ± std,
     `Episode/Reward` mean ± std, `Behavior/mean_entropy`,
     `Behavior/mean_advantage`, `WorldModel/model_reward_mae_pos/neg`,
     `WorldModel/model_cont_acc`, `Episode/Term_Injury / _MaxSteps / _Starvation`.
   - Plot all on the `Episode/Number` axis with vertical lines at the two
     stage boundaries.

3. **Comparison to failing baseline** (qont5dac):
   - Side-by-side temporal trajectories on the *same x-axis* (env-steps,
     not episode count, since baseline ran 27 M env-steps and this run runs
     ~32 M).
   - Headline metric: Stage 3's first 1 M env-steps vs the failing run's
     first 1 M env-steps. Are they the same trajectory or has the agent
     entered the new task with non-trivial competence?

4. **Confirm/refute checklist**: Walk C1–C4 and R1–R5 from §5 with explicit
   metric values; record verdict.

5. **Wall-clock attribution**: Total time + JIT recompile blips at the two
   boundaries.

## 7. Plumbing-check table (§5.1 quick fill-in)

> _To be filled in after the run completes._

- [ ] Single WandB run-ID
- [ ] Schedule banner prints at startup (`Continual mode: 3 stages from configs/experiment/dreamer_curriculum/`)
- [ ] Cross-stage probe passes (no `ValueError` at startup re obs_dim/action_dim/fingerprint)
- [ ] `stage/transition` event #1 at Episode/Number ∈ `[16000, 16015]`
- [ ] `stage/transition` event #2 at Episode/Number ∈ `[66000, 66015]`
- [ ] `stage/index` = 0 → 1 → 2 at the two boundaries
- [ ] `stage/buffer_cleared_main` fires at both boundaries (DreamerV3-specific)
- [ ] Three stage YAMLs (`stage_00_*.yaml`, `stage_01_*.yaml`, `stage_02_*.yaml`) present in run's `models/`
- [ ] No NaN warnings in `output.log`
- [ ] Total wall-clock within ~4 h budget

## 8. Results

> _To be filled in after the run completes._

### 8.1 Plumbing checks
> _Fill from §7 above._

### 8.2 Curves
> _Insert WandB run link, screenshots of the three-stage temporal curves
> with stage boundaries marked, and 1–2 sentence narrative on each kink._

### 8.3 Verdict (C1–C4 / R1–R5 walk)
> _Per-criterion explicit metric values + pass/fail._

### 8.4 Comparison to failing baseline (qont5dac)
> _Headline survival, reward-MAE-neg, mean-entropy, and mean-advantage
> trajectories on the same env-step axis._

## 9. Conclusions

> _To be filled in after the run completes._

### 9.1 Summary
> _3-5 bullets: what worked, what didn't, what's next._

### 9.2 Recommended next experiments

> _Pre-populated possibilities (delete those that don't apply post-run):_
>
> | Priority | Experiment | Rationale | Effort |
> |---|---|---|---|
> | 1 | 3-seed repeat of this curriculum at seeds {7, 13, 21} | Confirm result is not seed-pathological. Required before any quantitative claim. | 3 runs × 4 h ≈ 12 GPU-h |
> | 2 | Lever-attribution ablation: 4 cells × {curriculum × {on, off}} × {levers × {on, off}} | If C1 holds, isolate which factor does the work. | 4 runs × 4 h ≈ 16 GPU-h |
> | 3 | Curriculum-only (no levers): same 3 stages but `dreamer_v3.yaml` | Pure curriculum effect. | 1 run × 4 h |
> | 4 | Reward-head class-balanced loss (diagnosis §7.3 priority 4) | If R3-style imagination drift persists, target the reward head directly. | new schema add + 1 run |
> | 5 | Slower predator at Stage 2 (move_interval=12, attack_delay=12) | If R2 fires (Stage 2 succeeds, Stage 3 fails), test whether a softer Stage 2 transfers better. | 1 run |
> | 6 | Stage-2-only direct training (skip curriculum, train directly on 02_predator_slow) | Tests whether the slow-predator config alone (without Stage 1 warm-up) is enough. | 1 run |

---

## Appendix

### A. Live-load probe full output

Saved at `tmp/20260507_dreamer_curriculum_probe.py`. Output above in §4 is verbatim.

### B. Lever values — copy from failing-run vs this experiment

| Lever | Failing baseline | This experiment | Ratio |
|---|---:|---:|---:|
| `agent.entropy_scale` | `3e-4` | `1e-3` | ~3.33× |
| `agent.cont_loss_weight` | `1.0` | `5.0` | 5× |
| Curriculum stages | 1 (anchor only) | 3 | — |
| Total env-step budget | 27.1 M (3 h 13 min) | ~32 M (4 h target) | ~1.18× |
| Total episodes | 909,858 | 766,000 (max) | ~0.84× (longer episodes expected if recovery) |

### C. Schedule sanity table

| Stage | name | until_ep (cum) | delta_ep | ckpt_freq | env-step est |
|---:|---|---:|---:|---:|---:|
| 0 | `01_food_only`     | 16,000  | 16,000  | 8,000   | ~2.4 M |
| 1 | `02_predator_slow` | 66,000  | 50,000  | 25,000  | ~5.0 M |
| 2 | `03_predator_full` | 766,000 | 700,000 | 100,000 | ~25 M |

### D. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-05-07 | Initial design + 3 stage configs + agent config + schedule + manifest. Cross-stage live-load probe verified. | experiment-designer |
