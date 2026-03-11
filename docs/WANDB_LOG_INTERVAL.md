# Add Configurable WandB Log Interval

> **Status**: PLANNED
> **Opened**: 2026-03-11
> **Related**: [WANDB_METRICS_REFERENCE.md](WANDB_METRICS_REFERENCE.md), [train.py](../train.py)

---

## Context

At 100M+ timesteps, the current logging frequency generates an excessive number of WandB data points. RecurrentPPO, DQN, DRQN, and PPO all log **every iteration**, while DreamerV3 logs training metrics every 10th iteration (hardcoded). This slows down WandB dashboard loading, metric export, and downstream analysis.

A configurable `log_interval` would let users reduce logging density for long runs (e.g., `log_interval: 50` → ~50x fewer data points).

## Analysis

### Current Logging Sites in `train.py`

There are **10 `wandb.log()` calls** across 5 algorithm branches:

| Algorithm | Episode Logging | Training Logging | Lines |
|-----------|----------------|-----------------|-------|
| **RecurrentPPO** | Every iter (if episodes) | Every iter | 881–910, 923–958 |
| **DreamerV3** | Every iter (if episodes) | Every 10th iter (hardcoded) | 1129–1158, 1188–1214 |
| **DQN** | Every iter (if episodes) | Every iter | 1332–1362, 1325–1364 |
| **DRQN** | Every iter (if episodes) | Every iter | 1496–1528, 1529–1537 |
| **PPO** | Every iter (if episodes) | Every iter | 1600–1630, 1595–1637 |

### Critical Detail: Episode Data Accumulation

Currently, `iteration_episodes` is reset to `[]` at the **top of each iteration** (lines 806, 1272, 1429, 1557 — or for DreamerV3, episodes accumulate within the iteration's done-tracking loop). The reset behavior must change depending on the accumulation mode (see Design).

The `ep_info_buffer` (deque, maxlen=100) already accumulates episodes across iterations for the tqdm progress bar — but it's capped and can't be used for accurate windowed aggregation.

## Implementation Plan

### Design

**Two config keys**:
- `training.log_interval` (integer, default `1`) — how often to log, in iterations.
- `training.log_accumulate` (boolean, default `true`) — whether to accumulate episode data across the interval window.

**CLI overrides**: `--log-interval <int>`, `--log-accumulate` / `--no-log-accumulate`.

#### Accumulation Modes

| Mode | `log_accumulate` | Episode Behavior | Training Metrics |
|------|-----------------|------------------|-----------------|
| **Accumulate** | `true` (default) | Episodes from ALL iterations in the window are collected. At log time, `np.mean()` is computed over the full window. **No data is lost.** | Point-in-time snapshot from the logging iteration only. |
| **Hard interval** | `false` | `iteration_episodes` resets every iteration as it does today. At log time, only episodes from that specific (logging) iteration are reported. **Episodes from skipped iterations are discarded.** | Same — point-in-time snapshot. |

**When to use each**:
- `log_accumulate: true` — Reduced WandB points but accurate aggregate statistics. Each logged point reflects all episodes in the window. Best for post-hoc analysis.
- `log_accumulate: false` — Lightweight snapshot, discards episodes between log points. No list growth between intervals. Best for live monitoring where you just want a rough signal.

**The approach**:
1. Read `log_interval` and `log_accumulate` from config/CLI at startup.
2. The `iteration_episodes` reset pattern depends on mode:
   - `log_accumulate=true`: reset `iteration_episodes = []` only on logging iterations (when `iteration % log_interval == 0`), **before** appending new episodes. Episodes accumulate across skipped iterations.
   - `log_accumulate=false`: reset `iteration_episodes = []` **every iteration** (current behavior). Only the logging iteration's episodes are reported.
3. Gate all `wandb.log()` calls behind `iteration % log_interval == 0`.
4. For DreamerV3, replace the hardcoded `iteration % 10` with `iteration % log_interval`.

**Unified reset expression**: `if not log_accumulate or iteration % log_interval == 0` — this captures both modes in one line:
- `log_accumulate=false` → `not False` = `True` → always reset.
- `log_accumulate=true` → `not True` = `False` → only reset on log iterations.

**Training metrics (loss, modulator, etc.)**: Always point-in-time snapshots regardless of accumulation mode. At `log_interval > 1`, we log the current iteration's values and skip the in-between ones. This is acceptable because:
- Loss values are already noisy and typically smoothed in WandB.
- Modulator stats are instantaneous snapshots, not cumulative.
- The user can always reduce `log_interval` if finer granularity is needed.

**Backward compatibility**: Defaults (`log_interval: 1`, `log_accumulate: true`) preserve current behavior exactly. Both keys are optional — use `config.get()` (not `get_mandatory`) since these are non-critical convenience parameters.

#### Reset Logic Truth Table

| `log_accumulate` | `iteration % log_interval == 0` | `iteration_episodes` reset? | Logged? |
|---|---|---|---|
| `true` | Yes (logging iteration) | **Yes** — clear, then accumulate fresh | **Yes** |
| `true` | No (skipped iteration) | **No** — episodes accumulate | No |
| `false` | Yes (logging iteration) | **Yes** — clear, log current iter's episodes | **Yes** |
| `false` | No (skipped iteration) | **Yes** — clear (episodes discarded) | No |

### File Changes

#### `train.py` (line 153) — Add CLI arguments

```python
# BEFORE (after line 153):
    parser.add_argument("--wandb-entity", type=str, help="WandB Entity Name")

# AFTER:
    parser.add_argument("--wandb-entity", type=str, help="WandB Entity Name")
    parser.add_argument("--log-interval", type=int, help="WandB logging interval in iterations (default: 1)")
    parser.add_argument("--log-accumulate", action=argparse.BooleanOptionalAction, default=None,
                        help="Accumulate episode metrics across log interval (default: true). Use --no-log-accumulate for hard interval.")
```

> **Note on `BooleanOptionalAction`**: Available in Python 3.9+. Generates both `--log-accumulate` and `--no-log-accumulate` flags. With `default=None`, we can distinguish "not specified" from explicit true/false, allowing config file fallback.

#### `train.py` (after line ~294) — Read config values

Insert near other training config reads (around line 294 where `num_envs` is read):

```python
# BEFORE:
    num_envs = args.num_envs or config.get_mandatory('training.num_envs')

# AFTER:
    num_envs = args.num_envs or config.get_mandatory('training.num_envs')
    log_interval = args.log_interval or config.get('training.log_interval', 1)
    log_accumulate = args.log_accumulate if args.log_accumulate is not None else config.get('training.log_accumulate', True)
```

#### `train.py` (line 806) — RecurrentPPO: Conditional `iteration_episodes` reset

```python
# BEFORE (line 806):
                iteration_episodes = []

# AFTER:
                if not log_accumulate or iteration % log_interval == 0:
                    iteration_episodes = []
```

#### `train.py` (lines 881–910) — RecurrentPPO: Gate episode logging

```python
# BEFORE (line 881):
                    if wandb_enabled and iteration_episodes:

# AFTER:
                    if wandb_enabled and iteration_episodes and iteration % log_interval == 0:
```

#### `train.py` (lines 923–958) — RecurrentPPO: Gate training metric logging

```python
# BEFORE (line 923):
                    if wandb_enabled:
                        wandb_logs = {

# AFTER:
                    if wandb_enabled and iteration % log_interval == 0:
                        wandb_logs = {
```

#### `train.py` (lines 1129–1158) — DreamerV3: Gate episode logging

DreamerV3 doesn't have an explicit `iteration_episodes = []` reset at the top of its branch — episodes are accumulated within the done-tracking loop (lines 1055–1126) and appended at line 1086. We need to:

1. Add conditional reset **before** the episode accumulation loop (around line 1055):
```python
# ADD before the done-tracking loop:
                    if not log_accumulate or iteration % log_interval == 0:
                        iteration_episodes = []
```

2. Gate the episode log:
```python
# BEFORE (line 1129):
                    if wandb_enabled and iteration_episodes:

# AFTER:
                    if wandb_enabled and iteration_episodes and iteration % log_interval == 0:
```

#### `train.py` (line 1188) — DreamerV3: Replace hardcoded `% 10` with `log_interval`

```python
# BEFORE (line 1188):
                    if wandb_enabled and iteration % 10 == 0:

# AFTER:
                    if wandb_enabled and iteration % log_interval == 0:
```

#### `train.py` (line 1272) — DQN: Conditional reset

```python
# BEFORE (line 1272):
                    iteration_episodes = []

# AFTER:
                    if not log_accumulate or iteration % log_interval == 0:
                        iteration_episodes = []
```

#### `train.py` (lines 1325–1364) — DQN: Gate both log calls

```python
# BEFORE (line 1325):
                    if wandb_enabled:
                        logs = {

# AFTER:
                    if wandb_enabled and iteration % log_interval == 0:
                        logs = {
```

(This gates both the episode log at 1362 and the training log at 1364 since they're within the same `if` block.)

#### `train.py` (line 1429) — DRQN: Conditional reset

```python
# BEFORE (line 1429):
                    iteration_episodes = []

# AFTER:
                    if not log_accumulate or iteration % log_interval == 0:
                        iteration_episodes = []
```

#### `train.py` (lines 1489–1537) — DRQN: Gate logging

Follow the same pattern:

```python
# BEFORE (DRQN wandb logging):
                    if wandb_enabled:

# AFTER:
                    if wandb_enabled and iteration % log_interval == 0:
```

#### `train.py` (line 1557) — PPO: Conditional reset

```python
# BEFORE (line 1557):
                    iteration_episodes = []

# AFTER:
                    if not log_accumulate or iteration % log_interval == 0:
                        iteration_episodes = []
```

#### `train.py` (lines 1595–1637) — PPO: Gate logging

```python
# BEFORE (line 1595):
                    if wandb_enabled:
                        logs = {

# AFTER:
                    if wandb_enabled and iteration % log_interval == 0:
                        logs = {
```

### Config Keys

**YAML path**: `training.log_interval`
**Type**: int
**Default**: `1` (log every iteration — current behavior)
**Example values**: `1` (default), `10` (every 10th), `50` (recommended for 100M+ runs)

**YAML path**: `training.log_accumulate`
**Type**: bool
**Default**: `true` (accumulate episodes across the interval window)
**Example values**: `true` (mean over window — no data lost), `false` (hard interval — discard between log points)

Both keys are **optional** and do not need to be added to existing config files. They fall back to defaults via `config.get()`.

## Checkpoints

- [ ] Checkpoint 1 — With `log_interval=1, log_accumulate=true` (defaults), verify behavior is identical to current (no regressions). Run a short training (~1000 steps) and compare WandB point count.
- [ ] Checkpoint 2 — With `log_interval=10, log_accumulate=true`, verify WandB data points are ~10x fewer. Confirm episode metrics represent the mean over the full 10-iteration window (more episodes per logged point than default).
- [ ] Checkpoint 3 — With `log_interval=10, log_accumulate=false`, verify WandB data points are ~10x fewer AND that episode counts per log point are smaller (only from the logging iteration, not accumulated). Episodes from skipped iterations should be discarded.
- [ ] Checkpoint 4 — Verify DreamerV3 branch: the hardcoded `% 10` is replaced by `% log_interval`, so `log_interval=1` now logs DreamerV3 training metrics every iteration (more than before). This is intentional and correct.
- [ ] Checkpoint 5 — Verify `--no-log-accumulate` CLI flag works and overrides a `log_accumulate: true` in the config YAML.

## Implementation Report

> **Implemented by**:
> **Date**:

## Verification Report

> **Verified by**:
> **Date**:

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**:
