---
name: wandb-analysis
description: "Analyze WandB training logs for this project. Use when user asks about training results, run comparison, speed benchmarks, or training health metrics. Trigger on mentions of WandB, training runs, s/it, SPS, or metric analysis."
---

# WandB Training Analysis Skill

Analyze and compare training runs logged to Weights & Biases for the grid_world_pain project.

## When to Use This Skill

Trigger when user:
- Asks about training results, speed, or metrics from WandB
- Wants to compare two or more training runs
- Mentions WandB, run names (YYYYMMDD-HHMMSS format), s/it, SPS, or metric names
- Asks "how is training going?" or "check the latest run"
- Provides run directory names for analysis

## Project Context

- **WandB Entity**: `sungwoolee`
- **WandB Project**: `grid_world_pain`
- **Run Naming**: `YYYYMMDD-HHMMSS_model_config_description` (the timestamp prefix is the primary identifier for matching)
- **Algorithms**: DreamerV3 (primary), RecurrentPPO (baseline)
- **Diagnostics Plan**: `docs/DREAMER_DIAGNOSTICS_PLAN_v2.md` (contains pass criteria and healthy ranges)

## Available Scripts

All scripts are in `scripts/` and run from the project root.

### 1. Speed Benchmark (`scripts/benchmark_wandb_speed.py`)

Extracts wall-clock speed metrics from WandB logs.

```bash
python scripts/benchmark_wandb_speed.py RUN_NAME1 RUN_NAME2 [...]
```

**Output**: Markdown table with `s/it`, `it/s`, `SPS`, total wall-clock time, iterations, timesteps.

**Options**:
- `--entity ENTITY` — WandB entity (default: sungwoolee)
- `--project PROJECT` — WandB project (default: grid_world_pain)
- `--csv` — Output CSV instead of markdown

### 2. Training Metrics Comparison (`scripts/compare_wandb_runs.py`)

Pulls and compares training health metrics across runs.

```bash
python scripts/compare_wandb_runs.py \
  --labels "label1,label2" \
  RUN_NAME1 RUN_NAME2
```

**Output**: Markdown tables organized by category:
- Episode Performance (Steps, Reward)
- World Model (loss_recon, loss_rew, KL, latent_entropy, reward_mae, cont_acc)
- Actor-Critic (mean_entropy, value_mae, advantage, actor/critic losses)
- System (effective_replay_ratio)
- Trajectory Summary (first → last values)

**Options**:
- `--labels "A,B,C"` — Comma-separated labels (must match run count)
- `--entity ENTITY` — WandB entity
- `--project PROJECT` — WandB project

If `--labels` is omitted, runs are labeled Run1, Run2, etc.

## Analysis Workflow

When the user asks to analyze runs, follow this order:

### Step 1: Identify Runs
Extract run names from user message. Run names follow the format:
`YYYYMMDD-HHMMSS_model_envs_config...`

### Step 2: Run Speed Benchmark
```bash
python scripts/benchmark_wandb_speed.py RUN_NAME1 RUN_NAME2
```

### Step 3: Run Training Metrics Comparison
```bash
python scripts/compare_wandb_runs.py --labels "descriptive1,descriptive2" RUN_NAME1 RUN_NAME2
```

### Step 4: Analyze Results Against Pass Criteria

Use these healthy ranges (from `docs/DREAMER_DIAGNOSTICS_PLAN_v2.md` Section 7):

| Metric | Healthy Range | Red Flag |
|:---|:---|:---|
| `mean_entropy` | 0.5 - 1.8 | < 0.3 (entropy collapse) |
| `loss_recon` | Decreasing, < 0.1 | Increasing or stuck |
| `loss_rew` | Decreasing | Stuck or increasing |
| `model_reward_mae_pos` | > 0 | = 0 (no food discovery) |
| `value_mae` | < 5, decreasing | > 20 or diverging |
| `loss_dyn_kl` | > 1.0, stable | = 1.0 (floor) or exploding |
| `latent_entropy` | 1.0 - 2.5 | < 0.5 (collapsed representation) |
| `eff_replay_ratio` | ≈ configured replay_ratio | Very different (gradient balance broken) |
| `Episode/Steps` | Increasing | Flat or decreasing |
| `cont_acc` | > 0.95 | Trivially predicting "always continue" |
| `loss_actor_entropy` | Meaningful fraction of loss_actor | ≈ 0 (entropy bonus negligible) |

### Step 5: Report Findings

Structure the analysis as:
1. **Speed comparison table** — s/it, SPS, total iterations
2. **Training health assessment** — flag any metrics outside healthy ranges
3. **Red flags** — critical issues requiring immediate attention
4. **Comparison** — which run performed better and why
5. **Recommendations** — actionable next steps

### Step 6: Update Diagnostics Plan (If Requested)

Add findings as a new Section 8.X in `docs/DREAMER_DIAGNOSTICS_PLAN_v2.md` following the template:
```
### 8.X [Title]
**Date**: YYYY-MM-DD
**Phase**: [1/2/3]
**Context**: [Config, run tag, WandB link]
**Observation**: [What was seen]
**Analysis**: [Root cause investigation]
**Resolution**: [Fix applied or next steps]
```

## WandB Metric Naming Conventions

Metrics are logged under these prefixes in WandB:
- `Episode/*` — Per-episode aggregates (Steps, Reward, Number)
- `WorldModel/*` — World model losses and reconstruction metrics
- `Behavior/*` — Actor-critic losses and policy statistics
- `Params/*` — Training system parameters (replay ratio)
- `Modulator/*` — Neuromodulation outputs (if enabled)
- `value_mae` — Critic MAE (logged at root level, not under Behavior/)

**Important**: Episode metrics and training metrics are logged at different intervals. The comparison script handles this by pulling them separately.

## Debugging Playbook

**Entropy Collapse** (`mean_entropy < 0.3`):
1. Check `entropy_scale` — try 3e-3 or 1e-2
2. Verify advantage computation uses same Moments normalization
3. Check `to_twohot` target receives raw `lambda_returns`

**Critic Divergence** (`value_mae > 20`):
1. Verify critic trains on `to_twohot(lambda_returns)` (raw space)
2. Check `from_twohot()` returns raw-space values

**Zero Food Discovery** (`model_reward_mae_pos = 0`):
1. Usually caused by entropy collapse — fix entropy first
2. Check environment config (food placement, eat_enabled)

**Replay Ratio Mismatch** (`eff_replay_ratio ≠ config`):
1. Check `Ratio` class normalization — `global_step` must be `global_step // num_steps`
2. Verify `collect_interval` interaction (v1 Section 14)

**Slow Training** (`s/it >> target`):
1. Confirm `buffer_device: "gpu"`
2. Check JIT retracing: `JAX_LOG_COMPILES=1`
3. Check batch dimensions: `batch_size × sequence_length × horizon` imagined transitions per step

## Arguments

When invoked as `/wandb-analysis RUN1 RUN2`, `$ARGUMENTS` contains the run names.
Parse them and run both scripts automatically.
