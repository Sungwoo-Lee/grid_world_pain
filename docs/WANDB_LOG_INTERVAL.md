# Add Configurable WandB Log Interval

> **Status**: PLANNED
> **Opened**: 2026-03-11
> **Related**: [WANDB_METRICS_REFERENCE.md](WANDB_METRICS_REFERENCE.md), [train.py](../train.py)

---

## Context

At 100M+ timesteps, the current logging frequency generates an excessive number of WandB data points. RecurrentPPO, DQN, DRQN, and PPO all log **every iteration**, while DreamerV3 logs training metrics every 10th iteration (hardcoded). This slows down WandB dashboard loading, metric export, and downstream analysis.

A configurable `log_interval` would let users reduce logging density for long runs (e.g., `log_interval: 50` → ~50x fewer data points) without losing data accuracy — episode metrics accumulated during skipped iterations are aggregated at log time.

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

Currently, `iteration_episodes` is reset to `[]` at the **top of each iteration** (lines 806, 1272, 1429, 1557 — or for DreamerV3, episodes accumulate within the iteration's done-tracking loop). When `log_interval > 1`, we must **not** discard episodes from non-logging iterations. Instead, episodes must accumulate across the window and be aggregated at log time.

The `ep_info_buffer` (deque, maxlen=100) already accumulates episodes across iterations for the tqdm progress bar — but it's capped and can't be used for accurate windowed aggregation. We need a separate accumulator.

## Implementation Plan

### Design

**Single config key**: `training.log_interval` (integer, default `1`).
**One CLI override**: `--log-interval <int>`.

The approach:
1. Read `log_interval` from config/CLI at startup.
2. Replace `iteration_episodes = []` reset with conditional: only clear it after a logging iteration.
3. Gate all `wandb.log()` calls behind `iteration % log_interval == 0`.
4. For DreamerV3, replace the hardcoded `iteration % 10` with `iteration % log_interval`.

**Episode accumulation**: `iteration_episodes` currently resets every iteration. With this change, it resets only on logging iterations (after the data is logged). This means episodes from non-logging iterations naturally accumulate and get included in the next log point's `np.mean()`. No new data structures are needed.

**Training metrics (loss, modulator, etc.)**: These are point-in-time snapshots, not accumulated values. At `log_interval > 1`, we simply log the current iteration's values and skip the in-between ones. This is acceptable because:
- Loss values are already noisy and typically smoothed in WandB.
- Modulator stats are instantaneous snapshots, not cumulative.
- The user can always reduce `log_interval` if finer granularity is needed.

**Backward compatibility**: Default `log_interval: 1` preserves current behavior exactly. The key is optional — use `config.get('training.log_interval', 1)` (not `get_mandatory`) since this is a non-critical convenience parameter.

### File Changes

#### `train.py` (line 153) — Add CLI argument

```python
# BEFORE (after line 153):
    parser.add_argument("--wandb-entity", type=str, help="WandB Entity Name")

# AFTER:
    parser.add_argument("--wandb-entity", type=str, help="WandB Entity Name")
    parser.add_argument("--log-interval", type=int, help="WandB logging interval in iterations (default: 1)")
```

#### `train.py` (after line ~294) — Read config value

Insert near other training config reads (around line 294 where `num_envs` is read):

```python
# BEFORE:
    num_envs = args.num_envs or config.get_mandatory('training.num_envs')

# AFTER:
    num_envs = args.num_envs or config.get_mandatory('training.num_envs')
    log_interval = args.log_interval or config.get('training.log_interval', 1)
```

#### `train.py` (line 806) — RecurrentPPO: Conditional `iteration_episodes` reset

```python
# BEFORE (line 806):
                iteration_episodes = []

# AFTER:
                if iteration % log_interval == 0:
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

DreamerV3 doesn't have an explicit `iteration_episodes = []` reset — episodes are accumulated within the done-tracking loop (lines 1055–1126). The `iteration_episodes` list persists from one call to the next because it's defined outside the algorithm-specific block. However, looking at the code flow, episodes are appended at line 1086 and logged at 1129. We need to:

1. Gate the episode log:
```python
# BEFORE (line 1129):
                    if wandb_enabled and iteration_episodes:

# AFTER:
                    if wandb_enabled and iteration_episodes and iteration % log_interval == 0:
```

2. Clear `iteration_episodes` only after logging:
```python
# AFTER the wandb.log(ep_log) call at line 1158, add:
                        iteration_episodes = []
```

3. Also clear when it's a non-logging iteration but episodes exist (to prevent unbounded growth on non-logging iters — wait, we WANT to accumulate). Actually, do NOT clear on non-logging iterations. The list will grow until the next logging iteration, then get cleared. With `log_interval=50` and ~5 episodes/iter, that's ~250 episode dicts in memory — negligible.

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
                    if iteration % log_interval == 0:
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
                    if iteration % log_interval == 0:
                        iteration_episodes = []
```

#### `train.py` (lines 1489–1537) — DRQN: Gate logging

The DRQN logging block structure needs to be checked, but follow the same pattern:

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
                    if iteration % log_interval == 0:
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

### Config Key

**YAML path**: `training.log_interval`
**Type**: int
**Default**: `1` (log every iteration — current behavior)
**Example values**: `1` (default), `10` (every 10th), `50` (recommended for 100M+ runs)

This key is **optional** and does not need to be added to existing config files. It falls back to `1` via `config.get('training.log_interval', 1)`.

## Checkpoints

- [ ] Checkpoint 1 — With `log_interval=1`, verify behavior is identical to current (no regressions). Run a short training (~1000 steps) and compare WandB point count.
- [ ] Checkpoint 2 — With `log_interval=10`, verify that WandB data points are ~10x fewer. Confirm episode metrics still appear (just less frequently).
- [ ] Checkpoint 3 — Verify that `iteration_episodes` accumulates correctly across non-logging iterations: the episode count logged at each log point should reflect ALL episodes in the window, not just the last iteration's.
- [ ] Checkpoint 4 — Verify DreamerV3 branch: the hardcoded `% 10` is replaced by `% log_interval`, so `log_interval=1` now logs DreamerV3 training metrics every iteration (more than before). This is intentional and correct.

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
