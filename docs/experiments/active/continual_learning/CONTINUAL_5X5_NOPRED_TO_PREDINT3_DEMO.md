---
title: "Continual learning demo — 5x5 NoPred -> PredInterval3 (RecurrentPPO)"
topic: continual_learning
status: active
created: 2026-05-07
last_updated: 2026-05-07T16:30:45
phase: 1
wandb_tag: "continual_5x5_NoPred-to-PredInt3_rppo_s0"
develop_link: "../../../develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md"
---

# Continual learning demo — 5x5 NoPred -> PredInterval3 (RecurrentPPO)

> **Status**: COLLECTING
> **Date**: 2026-05-07
> **Author**: experiment-designer
> **Related**:
> - Feature spec: [`docs/develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md`](../../../develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md)
> - Schedule YAML: [`configs/continual/basic_5x5_NoPred_to_PredInt3.yaml`](../../../../configs/continual/basic_5x5_NoPred_to_PredInt3.yaml)
> - Stage 0 config: [`configs/experiment/basic/00-5X5_NoPred.yaml`](../../../../configs/experiment/basic/00-5X5_NoPred.yaml)
> - Stage 1 config: [`configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml`](../../../../configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml)

> **Note**: This is a **plumbing/visualization demo** of `train.py --configs-dir`,
> not a publishable experiment. Single seed → no statistical claims; the
> deliverable is a single WandB run whose curve is **observably two-staged**
> with a clean `stage/transition` marker at the boundary.

---

## 1. Research Question

Does the existing `--configs-dir` / `--continual-schedule` feature in
`train.py` produce a single, end-to-end WandB run whose `Episode/Number`-axis
curves show:

1. a recognizable Stage 0 learning curve (NoPred, food-foraging only),
2. a single discrete `stage/transition = 1` marker at the boundary
   (somewhere in `[800, 800 + num_envs - 1]` after acceptable drift), and
3. a Stage 1 segment that visibly **changes** behavior (return / survival
   / entropy) compared to the end of Stage 0, when a predator is added?

Frame as a soft hypothesis (single seed → observational, not statistical):

> **H_demo** (qualitative): On RecurrentPPO at num_envs=16 with
> `episode_boundaries=[800, 1600]`, Stage 0 reaches a roughly-plateauing
> survival curve before episode 800; at the boundary the WandB run shows
> the `stage/transition` marker on the Episode/Number axis; Stage 1 then
> shows a **visible drop** in survival/return (the agent is briefly worse
> off because the predator is novel and recurrent state was reset) followed
> by partial recovery as RPPO adapts.

Refutation criteria (any one falsifies the demo for this run):

- **R1 (plumbing)** — `stage/transition` marker is missing, off-by-many-
  episodes (>= num_envs episodes past 800), or fires more than once.
- **R2 (continuity)** — the WandB run is split across two run IDs (the
  feature is supposed to keep a single process / single `wandb.init`).
- **R3 (boundary visibility)** — there is no perceptible kink in the
  primary curves (survival, return, policy entropy) at the boundary;
  Stage 1 is visually indistinguishable from a continuation of Stage 0.
- **R4 (probe failure)** — `train.py` raises at startup because the
  obs-dim or modality fingerprint differs across stages (would indicate
  a hidden mismatch the diff missed).

Note that R3 in particular is *expected to fail to refute* because the only
environment difference between stages is `predators[0].count: 0 -> 1`, which
materially changes the dynamics.

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| `predators[0].count` (across stages) | `0` (Stage 0) → `1` (Stage 1) | Sole environment delta between the two basic configs (verified by full diff). |
| Stage transition is at episode `800` | one boundary | Designer-locked; first phase long enough to plateau-ish on 5×5 RPPO. |

Everything else, including the agent architecture, optimizer, RNN type, and
all sensor enables, is held constant by construction (same `recurrent_ppo.yaml`
applied through the whole run; the feature reuses one model/optimizer across
the boundary).

### 2.2 Controlled Variables

Held constant across both stages (verified by `diff` of the two stage YAMLs
showing exactly one differing line):

```text
environment.height = 5
environment.width  = 5
environment.max_steps = 500
environment.rest_action_enabled = true
environment.eat_action_enabled  = true
sensory.olfactory_enabled      = true
sensory.nociception_enabled    = true
sensory.proprioception_enabled = true
sensory.visual_sensor_enabled  = false
sensory.location_sensor        = false
sensory.vector_size            = 5
sensory.nociception_size       = 1
body.with_satiation = true
body.with_nutrition = true
body.with_injury    = true
body.use_homeostatic_reward = true
body.food_nutrition_gain = 18
body.metabolic_cost      = 1.0
perceptual_noise.enabled = false
```

This list is the **modality-fingerprint surface** the train.py probe checks
(`_modality_fingerprint(p)` at train.py validation block — see Fix 3 in
the feature spec). All values are byte-identical between the two stage YAMLs.

Agent / algorithm side, controlled by `configs/models/recurrent_ppo/recurrent_ppo.yaml`
applied to both stages:

```text
agent.algorithm = "RecurrentPPO"
agent.gamma = 0.95
agent.K_epochs = 4
agent.eps_clip = 0.1
agent.sequence_length = 128
agent.hidden_size = 128
agent.entropy_coef = 0.01
agent.gae_lambda  = 0.95
agent.return_mode = "MC"
agent.encoding_mode = "hierarchical"
agent.modulation.type = null      # baseline RPPO, no neuromodulation
```

Training side: `num_envs = 16`, `seed = 0`, `log_interval = 5` (see §2.4),
`device = cuda:1`, schedule total = 1600 episodes.

### 2.3 Confounds & Limitations

| Confound | Affected Runs | Severity | Mitigation |
|----------|---------------|----------|------------|
| Single seed | the only run | High for any quantitative claim, fine for plumbing demo | Pre-registered as observational; no statistical verdict requested. |
| Mid-episode envs at boundary are silently dropped | Stage 1 start | Low | Documented in feature spec; `num_envs=16` so at most 15 episodes of work is lost — drift relative to nominal boundary `800`. |
| Recurrent state reset at boundary (Fix 4) confounds "agent transferred memory" with "agent re-discovered" | Stage 1 start | Low (this run does not test transfer) | Demo intentionally treats reset as the standard continual-RL choice; transfer-vs-no-transfer is a separate study. |
| One mid-run env rebuild forces JIT recompile (~30–60 s) | Stage 1 first iters | Low | Visible only as a wall-clock blip, not as a learning artifact. Curves are aligned to `Episode/Number`, not iterations. |
| Schedule-config drift if user later edits `00-5X5_NoPred.yaml` or `01-...yaml` | future re-runs | Medium | The two stage YAMLs are pinned inputs; per the user's instruction this designer is forbidden from editing them. |

## 3. Launch Manifest

System-of-record for every training run in this experiment. **Owned jointly:**
- `experiment-designer` writes the planned columns (this commit).
- `training-runner` fills the actual columns at launch time.
- `experiment-analyzer` reads the table to find WandB folders.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | running | NoPred→PredInt3 | `continual_5x5_NoPred-to-PredInt3_rppo_s0` | `continual_demo` | `demo` | 0 | 113 | cuda:1 | 2026-05-07T16:30:45 | wcrov9nj | logs/20260507_163045.log |

**WandB tags (extra, not the run name)**: `["continual", "rppo", "demo", "5x5"]` — set via `--wandb-tags` if the runner supports it; otherwise applied post-hoc on the WandB UI side. Tag (= wandb-name) above is the unique run identifier and is what `train.py:592`'s fallback uses.

### 3.1 Configs to Produce (designer-only, pre-launch)

| Run | Schedule | Configs-dir | Agent config |
|-----|----------|-------------|--------------|
| 1 | `configs/continual/basic_5x5_NoPred_to_PredInt3.yaml` | `configs/experiment/basic/` | `configs/models/recurrent_ppo/recurrent_ppo.yaml` |

Schedule and basic configs already exist on disk; this experiment authors only the schedule YAML (the basic stage YAMLs are pre-existing inputs and **must not be edited**).

### 3.2 Exact launch command

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --configs-dir configs/experiment/basic/ \
  --continual-schedule configs/continual/basic_5x5_NoPred_to_PredInt3.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --num-envs 16 \
  --seed 0 \
  --device cuda:1 \
  --log-interval 5 \
  --tag "continual_5x5_NoPred-to-PredInt3_rppo_s0"
```

Note:

- `--episodes` MUST NOT be passed when `--configs-dir` is set (train.py raises `ValueError` otherwise; the schedule's last boundary is the budget).
- `--checkpoint-frequency` MUST NOT be passed; per-stage frequencies come from the schedule YAML.
- `--config` MUST NOT be passed (mutually exclusive with `--configs-dir`).

### 3.3 What success looks like on WandB

- Single run, single run-ID, single x-axis sweep across `Episode/Number ∈ [0, 1600]`.
- One row in the run table, group `continual_demo`, job-type `demo`.
- A `stage/transition` panel showing exactly one spike (`y=1`) at some
  Episode/Number ∈ `[800, ~815]` (drift of up to `num_envs−1=15` is acceptable per spec).
- A `stage/index` panel that is `0` for `Episode/Number < 800` and
  `1` thereafter.
- Primary curves (`Episode/Reward_Mean`, `Episode/Length_Mean`, `Behavior/*`)
  show a visually identifiable kink at the boundary.

## 4. Predicted Outcomes

What this designer expects to see in the WandB curves; recorded here so a
reader checking the run later can compare prediction vs reality without
re-deriving the prediction.

### Stage 0 (episodes 0 → ~800, NoPred, food-foraging only)

- **Survival / episode length**: starts low (~30–80 steps as the agent
  learns to forage), climbs toward `max_steps=500` as the policy learns
  to eat → maintain nutrition → not starve. Should be roughly plateauing
  by ~600 episodes on 5×5 RPPO.
- **Episode reward (homeostatic)**: rises from negative (drive-distance
  shrinks as nutrition is maintained near setpoint) toward small positive
  steady-state values.
- **Policy entropy**: drops monotonically as the policy commits to a
  forage strategy.

### Boundary (Episode/Number ≈ 800)

- `stage/transition = 1` logged exactly once.
- `stage/index` flips `0 → 1`.
- Mid-episode envs are silently dropped (Stage 1 starts on a hard
  `env.reset`); the curve point that lands on the boundary may show a
  one-iteration discontinuity in episode-count accumulation.
- RPPO recurrent hidden state is reset (Fix 4 in feature spec), so the
  first few Stage 1 episodes use a fresh GRU memory.

### Stage 1 (episodes ~800 → 1600, predator added, move_interval=3)

- **Survival / episode length**: visible drop near the boundary as the
  agent walks into the predator's strike zone with a reset memory; partial
  recovery as RPPO updates the policy/value with predator-aware rollouts.
  Wall-clock is too short to expect full re-convergence in 800 episodes —
  a *trend toward recovery* is the prediction, not a return to Stage-0
  steady state.
- **Episode reward**: drops below Stage 0's plateau and recovers partially.
- **Policy entropy**: brief uptick as the policy is briefly less confident
  (hidden state was reset), then resumes decay.
- **Injury**: nonzero for the first time in the run (`Behavior/injury_mean`
  or analog).

### What would surprise me (still consistent with H_demo, just unexpected)

- Stage 1 immediately matching Stage 0's plateau (would suggest the agent
  generalises to "avoid moving things" purely from olfaction + nociception
  features even without predator-specific training; possible but not
  expected at only 800 episodes of additional training).
- Survival in Stage 1 *exceeding* Stage 0 (would suggest a confound with
  episode-length cutoff or something I haven't considered).

## 5. Analysis Plan (post-run)

Single seed → no inferential statistics. Post-run fill-in for §6 / §7 of
the doc will:

1. **Plumbing checks first** (R1, R2, R4 from §1):
   - Confirm one and only one `stage/transition` event in the WandB run.
   - Confirm `Episode/Number` of that event lies in `[800, 800+15]`.
   - Confirm a single WandB run-ID, single config dump, both
     `stage_00_*.yaml` and `stage_01_*.yaml` present in the run's
     `models/` folder.

2. **Boundary-aligned curve overlay**:
   - Plot `Episode/Reward_Mean`, `Episode/Length_Mean`, and
     `Behavior/policy_entropy_mean` (or whichever entropy panel is logged)
     on the `Episode/Number` axis with a vertical line at the
     `stage/transition` event.
   - Eyeball the kink. If absent, R3 fires.

3. **Wall-clock attribution**:
   - Note total wall-clock from the run's first → last log timestamp.
   - Note any two-iteration gap around the boundary that would indicate
     the env-rebuild JIT recompile.

No quantitative effect-size threshold (single seed). The verdict is
qualitative: did the feature deliver a visibly two-staged curve in one
WandB run? If yes, the demo passes; if any of R1–R4 fires, the failure
mode is documented in §7.

## 6. Failure-Mode Catalog (pre-decided)

Pre-deciding interpretations so a noisy single-seed result is not
litigated post-hoc.

| Observation | Interpretation |
|---|---|
| Stage 0 fails to learn (survival flat near random) within 800 ep | "Demo too short, not a feature bug." Rerun with `episode_boundaries: [1500, 3000]`. Not a refutation of the continual feature itself. |
| `stage/transition` event missing entirely | R1 fires → BUG in the feature. Halt; route to senior-developer. |
| `stage/transition` fires twice or fires before episode 800 | R1 fires → BUG. Halt; route to senior-developer. |
| `stage/transition` fires at episode > 800 + num_envs - 1 = 815 | Drift exceeds spec. Halt; route to senior-developer. |
| WandB shows two run IDs / two `wandb.init` calls | R2 fires → BUG. Halt; route to senior-developer. |
| Startup probe raises ValueError ("obs_dim" or "fingerprint" mismatch) | R4 fires. Re-diff the two basic configs; investigate. Should be impossible given the verified diff (see §2.2). |
| Stage 1 shows no visible kink | R3 fires (unlikely given the predator addition). Could indicate that 5×5 with 1 predator at move_interval=3 is too easy to detect, in which case the demo's *visualization* fails but the *plumbing* still works. Re-run with `01-5X5_PredInterval3_NutGain18.yaml`'s predator made more aggressive — but that is out of scope for this designer (cannot edit basic configs). |
| One iteration takes 30+ s right after the boundary | Expected: env rebuild forced a JIT recompile. Not a failure; document in the wall-clock note. |
| `stage/buffer_cleared_main` panel exists | Should NOT exist for RPPO (the buffer-clear path is Dreamer-only). If present, file a bug. |

## 7. Results

> _To be filled in after the run completes._

### 7.1 Plumbing checks

- [ ] Single WandB run-ID
- [ ] One `stage/transition` event at Episode/Number ∈ [800, 815]
- [ ] `stage/index` = 0 before, 1 after
- [ ] Both `models/stage_00_*.yaml` and `models/stage_01_*.yaml` present
- [ ] No startup ValueError
- [ ] Total wall-clock ≤ ~10 minutes

### 7.2 Curves

> _Insert link to WandB run, screenshot of the two-staged curve, and a
> 1–2 sentence narrative on the boundary kink._

### 7.3 Verdict on H_demo

> _Qualitative pass / fail._

## 8. Conclusions

> _To be filled in after the run completes._

---

## Appendix

### A. Wall-clock estimate (designer's back-of-envelope)

Anchored to recently-measured RecurrentPPO @ num_envs=16, num_steps=128
(2048 env-steps/iter) on this codebase: **~1.2 s/iter steady state** post-warmup
(see [`docs/develop/active/diagnosis/dreamer_replay_ratio_sweep.md`](../../../develop/active/diagnosis/dreamer_replay_ratio_sweep.md) §10, RPPO baseline ≈ 1437–1720 SPS depending on JIT amortization).

Episodes per iteration depend on episode length. On 5×5 basic:

- Stage 0 (no predator): episode length will climb from ~30 (random walk
  hits starvation quickly) to near `max_steps=500` once forage works.
  Conservative average ≈ ~150 → ~14 ep/iter mid-stage. 800 episodes ≈ 60–80 iters.
- Stage 1 (1 predator): episode length will dip then partially recover.
  Conservative average ≈ ~80 → ~25 ep/iter. 800 episodes ≈ 30–40 iters.

Total iters ≈ 90–120 of steady state at ~1.2 s/iter ≈ **~120–150 s of compute**.
Add:

- ~15–30 s RPPO cold-start JIT compile,
- ~30–60 s env-rebuild JIT recompile at the boundary (one-shot, full env
  re-trace because `EnvParams` static fields don't change but the entire
  ParallelEnv is reinstantiated per spec).

**Wall-clock estimate: ~3–5 minutes total**, padded to ~10 min as the
"do-not-be-surprised" upper bound. Well within "fast demo" intent.

### B. log_interval sanity check

`log_interval` is in **iterations** (train.py:443). The default is `10`.
At ~14 ep/iter (Stage 0) that's ~140 episodes between WandB log points —
about 5–6 points in the 800-episode Stage 0 window. That's borderline for a
recognizable curve.

**This experiment passes `--log-interval 5`** (every ~70 episodes early-Stage-0,
every ~125 episodes mid-Stage-1) → ~10–15 points per stage. Enough to see a
plateau/kink/recovery shape clearly.

The transition event itself is logged unconditionally inside the transition
block (train.py:1128–1133, no `iteration % log_interval == 0` gate), so the
`stage/transition` and `stage/index` markers land on the exact boundary
Episode/Number regardless of log_interval. Lowering log_interval only
improves the resolution of the *return / length / entropy* panels around the
boundary — exactly what the demo wants to make visible.

### C. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-05-07 | Initial design + schedule + manifest. | experiment-designer |
