---
name: wandb-analysis
description: "Analyze WandB training logs. Use when user asks about training results, run comparison, speed benchmarks, or metric analysis. Trigger on mentions of WandB, training runs, s/it, SPS, metrics, or training health. Works with any algorithm."
---

# WandB Training Analysis Skill

Analyze and compare training runs logged to Weights & Biases. **Algorithm-agnostic** — automatically discovers what metrics exist in any run and extracts them for analysis.

## When to Use This Skill

Trigger when user:
- Asks about training results, speed, or metrics from WandB
- Wants to compare two or more training runs
- Mentions WandB, run names (YYYYMMDD-HHMMSS format), s/it, SPS, or metric names
- Asks "how is training going?", "is training working?", or "check the latest run"
- Provides run directory names for analysis
- Asks what metrics or config a run has
- Asks about how a metric **evolved, developed, or changed** over training (use `timeseries`)

## Project Context

- **WandB Entity**: `sungwoolee`
- **WandB Project**: `grid_world_pain`
- **Run Naming**: `YYYYMMDD-HHMMSS_model_config_description` (timestamp prefix is the primary identifier)

## Available Scripts

All scripts are in `scripts/wandb/` and require `PYTHONPATH=scripts/wandb` or adding that folder to the path.

### Primary Tool: `scripts/wandb/wandb_metrics.py`

Five subcommands: `config`, `discover`, `extract`, `compare`, `timeseries`.

#### `config` — Show run hyperparameters

```bash
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py config RUN_NAME
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py config RUN_NAME --json
```

Shows algorithm type, hyperparameters, environment settings, and all logged config. Use this to **identify what algorithm a run used** before analyzing metrics.

#### `discover` — List all metrics in a run

```bash
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py discover RUN_NAME
```

Lists every metric grouped by prefix (e.g., Episode/, Policy/, WorldModel/, etc.). **Always run this first** to see what metrics are available before extracting.

#### `extract` — Pull stats for a single run

```bash
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py extract RUN_NAME
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py extract RUN_NAME --metrics "Episode/*,*loss*"
```

Outputs per-metric stats: steady-state (last 20%), final value, min, max, count, plus trajectory (first -> last).

#### `compare` — Compare multiple runs

```bash
# Auto-discovery (works with any algorithm)
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py compare RUN1 RUN2 --labels "A,B"

# Filter to specific metrics
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py compare RUN1 RUN2 --metrics "Episode/*,*loss*"

# Use a named preset (curated metrics with pass/fail criteria)
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py compare RUN1 RUN2 --preset dreamer_v3
```

**Available presets**: `dreamer_v3`, `recurrent_ppo`. Presets add a "Criterion" column with expected ranges.

#### `timeseries` — Temporal analysis of metric evolution

Use this when the user asks **how** a metric changed over training, not just what it converged to. Supports two modes:

**Windowed mode** (default) — divides training into N equal windows and shows mean/std/min/max per window:
```bash
# Default 10 windows
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py timeseries RUN_NAME --metrics "Episode/Steps"

# More granular: 20 windows
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py timeseries RUN_NAME --metrics "Episode/Steps" --windows 20

# Multiple metrics
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py timeseries RUN_NAME --metrics "Episode/Steps,Episode/Reward,*loss*"
```

**Raw mode** — outputs every downsampled data point (step, value):
```bash
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py timeseries RUN_NAME --metrics "Episode/Steps" --raw
```

**Multi-run comparison** — side-by-side temporal comparison:
```bash
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py timeseries RUN1 RUN2 --metrics "Episode/Steps" --labels "A,B"
```

**Options**:
- `--metrics` (required) — comma-separated glob patterns
- `--windows N` — number of windows (default: 10)
- `--raw` — output raw data points instead of windowed summary
- `--samples N` — max data points to fetch from WandB (default: 10000)
- `--labels` — comma-separated labels for multi-run

**When to use `timeseries` vs `extract`**:
- Use `extract` for "what did the metric converge to?" (summary stats)
- Use `timeseries` for "how did the metric evolve?", "when did learning start?", "was there a collapse?", "show me the learning curve" (temporal shape)

**Common options** (all subcommands):
- `--entity ENTITY` — WandB entity (default: sungwoolee)
- `--project PROJECT` — WandB project (default: grid_world_pain)

### Speed Benchmark: `scripts/benchmark_wandb_speed.py`

```bash
PYTHONPATH=scripts/wandb python scripts/wandb/benchmark_wandb_speed.py RUN1 RUN2 [...]
PYTHONPATH=scripts/wandb python scripts/wandb/benchmark_wandb_speed.py RUN1 --csv
```

Outputs: `s/it`, `it/s`, `SPS` (env steps/sec), total wall-clock time, iterations, timesteps.

### Legacy: `scripts/wandb/compare_wandb_runs.py`

Backward-compatible wrapper. Defaults to `--preset dreamer_v3`.

## Analysis Workflow

### Step 1: Identify the run(s)
Extract run names from user message. Format: `YYYYMMDD-HHMMSS_model_config...`

### Step 2: Get run config
```bash
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py config RUN_NAME
```
This tells you the algorithm, hyperparameters, and environment. Essential for knowing how to interpret the metrics.

### Step 3: Discover available metrics
```bash
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py discover RUN_NAME
```
See what the algorithm actually logged. Different algorithms log different metrics.

### Step 4: Check training speed
```bash
PYTHONPATH=scripts/wandb python scripts/wandb/benchmark_wandb_speed.py RUN_NAME
```

### Step 5: Extract or compare metrics
```bash
# Single run — summary stats
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py extract RUN_NAME

# Multi-run comparison
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py compare RUN1 RUN2 --labels "A,B"
```

### Step 6: Temporal analysis (when needed)
```bash
# How did key metrics evolve over training?
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py timeseries RUN_NAME --metrics "Episode/Steps,Episode/Reward"

# Compare learning curves between runs
PYTHONPATH=scripts/wandb python scripts/wandb/wandb_metrics.py timeseries RUN1 RUN2 --metrics "Episode/Steps" --labels "A,B" --windows 15
```

### Step 7: Assess training health

Use the **General Training Health Checklist** below to evaluate whether training is working properly.

## General Training Health Checklist

These checks apply to **any RL algorithm**. Use the extracted metrics to verify each one:

### 1. Reward Signal
- **Episode reward** should trend **upward** (or toward the goal) over training
- Check trajectory: `first -> last` — is there meaningful improvement?
- Use `timeseries` to see if improvement is gradual, sudden, or oscillating
- Red flag: flat, decreasing, or oscillating wildly

### 2. Episode Length
- **Episode steps** should change as the agent learns
- In survival tasks: increasing steps = agent living longer = good
- In goal-reaching tasks: decreasing steps = agent solving faster = good
- Use `timeseries` to identify when learning started and if there were regressions
- Red flag: completely flat from start (agent not learning)

### 3. Loss Convergence
- All loss metrics (policy loss, value loss, reconstruction loss, etc.) should generally **decrease** over training
- Red flag: losses increasing, exploding (NaN/Inf), or stuck at initial values

### 4. Policy Entropy
- Look for metrics with "entropy" in the name
- Entropy should **decrease gradually** as the agent becomes more confident
- Red flag: entropy collapsed to near-zero early (premature convergence — agent stopped exploring)
- Red flag: entropy stuck at maximum (agent not learning a policy)

### 5. Value Prediction
- Look for metrics with "value", "mae", or "critic" in the name
- Value prediction error should **decrease** over training
- Red flag: value error increasing or diverging (critic not learning)

### 6. Training Speed
- `s/it` should be **stable** (not increasing over time)
- `SPS` (steps per second) should match expectations for the hardware
- Red flag: training slowing down significantly over time

### 7. NaN/Missing Data
- Check metric counts (`N` column) — are all metrics being logged?
- Red flag: metrics with very few data points or sudden gaps

### 8. Gradient Health (if logged)
- Look for grad_norm or similar metrics
- Should be stable, not exploding
- Red flag: gradient norm growing unboundedly

## Report Structure

When reporting analysis results, structure as:
1. **Run Info** — algorithm, key hyperparameters (from config)
2. **Speed** — s/it, SPS, total wall-clock time
3. **Training Health** — checklist results with specific metric values
4. **Red Flags** — any critical issues found
5. **Comparison** — if multiple runs, which performed better and why
6. **Recommendations** — actionable next steps

## Algorithm-Specific Presets

For curated analysis with known pass/fail criteria, use `--preset`:
- `dreamer_v3` — World model losses, latent entropy, continuation accuracy, etc.
- `recurrent_ppo` — Policy entropy, KL divergence, clip fraction, etc.

For detailed algorithm-specific diagnostics, see relevant docs in `docs/`.

## Arguments

When invoked as `/wandb-analysis RUN1 RUN2`, `$ARGUMENTS` contains the run names.
Parse them and run the appropriate scripts automatically.
