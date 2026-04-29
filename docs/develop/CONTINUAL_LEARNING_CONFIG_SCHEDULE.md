# Continual Learning via Config Schedule

> **Status**: PLANNED
> **Opened**: 2026-04-20
> **Updated**: 2026-04-20 — folded in findings from [`docs/environment/13_checkpoint_scheduling.md`](../environment/13_checkpoint_scheduling.md)
> **Related**: [`docs/environment/ENVIRONMENT_SUMMARY.md`](../environment/ENVIRONMENT_SUMMARY.md), [`docs/environment/13_checkpoint_scheduling.md`](../environment/13_checkpoint_scheduling.md)

---

## Context

We want a **continual-learning (curriculum) training mode** where a single training run sweeps through an ordered list of environment configs. The user specifies a directory of stage configs (e.g. `01_*.yaml`, `02_*.yaml`, …), a list of cumulative episode boundaries saying when to switch to the next stage, and a parallel list of checkpoint frequencies. Ordering of stages is derived from alphabetic filename sort, so prefixing filenames with `01_`, `02_`, … suffices.

The current `train.py` hard-codes one config, resolves `episodes` once, computes `EnvParams` once, and instantiates `ParallelEnv(params)` once — no mechanism exists to rotate configs mid-run. This plan describes the additions needed to support the scheduled multi-stage training described above.

## Analysis

### Where config is loaded today

`train.py:184-236` merges five YAMLs in a fixed order: `configs/environment/default.yaml` → `configs/train/default.yaml` → `configs/evaluation/default.yaml` → `configs/logger/wandb.yaml` → `configs/visualization/default.yaml` → `--config` user/ablation → `--agent_config` (required). `load_env_params(config)` (at `train.py:328`) converts the merged dict into an `EnvParams` pytree.

### Where the training loop uses it

- `train.py:295` — `episodes = args.episodes if args.episodes is not None else config.get_mandatory('episodes')` (single scalar).
- `train.py:328` — `params = load_env_params(config)` (single call, used for `ParallelEnv(params)` at `train.py:445`).
- `train.py:808` — main `while total_episodes_completed < episodes` loop.
- `train.py:1657-1694` — checkpoint logic gated by a single `training.checkpoint_frequency`, ticking via `main.last_checkpoint_save`.

### JAX/XLA implications

From `ENVIRONMENT_SUMMARY.md`, `EnvParams` contains **static fields** (`struct.field(pytree_node=False)`): `height`, `width`, `placement_mode`, `use_homeostatic_reward`, `predator_enabled`, all sensor enables, etc. Changing a static field at runtime forces XLA recompilation of every JIT’d function that closes over `EnvParams`, including `ParallelEnv.step` and `ParallelEnv.reset`. Dynamic fields (counts, damage magnitudes, rewards) can be swapped without recompilation.

The user accepted the recompilation cost (stage transitions are rare, ≤ N_stages times per run), so the plan treats every stage transition as a full **env rebuild**: new `ParallelEnv(new_params)`, new JIT cache.

### Observation / action-dim constraint

Model architecture is constructed once from stage-0 obs/action dims (`train.py:451-454`). If a later stage enabled/disabled a sensor or the eat/rest action, dims would change and trained weights would no longer fit. Per the answered question, stage transitions that change `obs_dim` or `action_dim` are **forbidden and must fail fast at startup** — we validate this before training begins rather than discovering it at the first transition.

### WandB integration today

`train.py:391-402` calls `wandb.init(...)` once with `group`, `name`, `config` frozen, then defines metric schemas for `iteration`, `timesteps`, `Episode/Number`, `loss/*`, `modulator/*`, `behavior/*`. Metrics are logged every `log_interval` iterations. There is no concept of a “stage” dimension today.

### Checkpoint resume (out of scope, but flagged)

`train.py:719-800` supports `--load-checkpoint` to resume a run; it reads `episode`, `iteration`, `step` from the checkpoint. For continual runs, resuming must also restore the **current stage index**. Covered in File Changes below.

### Mid-episode envs at stage boundaries (accepted behavior)

With `num_envs > 1`, when the stage-transition gate fires, only a subset of envs (those that happened to complete an episode in the iteration that crossed `boundaries[i]`) are at natural episode boundaries. The rest are mid-episode in the OLD env's state.

Per user direction, the plan uses **hard reset + silently drop**:

- `env.reset(reset_key, num_envs)` replaces every env's state with a fresh start in the NEW env. Mid-episode envs' state (agent position, nutrition, partial return) is discarded.
- `episode_returns[:] = 0.0`, `episode_lengths[:] = 0`, and all behavior accumulators are wiped so partial OLD-stage data does not contaminate NEW-stage metrics.
- Those mid-episode episodes are **never** counted in `total_episodes_completed`, never logged to `ep_info_buffer`, never flushed to WandB. Their compute is lost.

Example: `num_envs=5`, boundary=100. Iteration K ends with `total_episodes_completed=101` after `e0, e1` finished. `e2, e3, e4` are mid-episode. Iteration K+1 triggers the transition; `e2, e3, e4`'s work is silently discarded.

This is accepted because: (a) it's simple and deterministic; (b) stage transitions happen at most `N_stages − 1` times per run, so total waste scales linearly in the schedule, not in training duration; (c) the alternative (drain until all envs naturally finish) adds variable-length drain periods and complicates the transition trigger.

### DreamerV3 replay-buffer clearing (new requirement)

The `ReplayBuffer` (`src/models/dreamer_v3_trainer.py:862-884`) stores up to `buffer_capacity` transitions collected under OLD-stage dynamics. If left intact across a transition, the world model continues sampling OLD-stage transitions while the agent acts in the NEW stage — cross-stage dynamics contamination.

Per user direction, the plan **clears both `buffer` and `positive_buffer`** at every stage transition (DreamerV3 only; RecurrentPPO is on-policy and has no persistent buffer). Cheap reset: set `buffer.idx = 0` and `buffer.size = 0` — `sample()` checks `self.size` before returning anything, so old array contents become unreachable.

This forces the world model to re-learn environment dynamics from scratch in the new stage. Training stability may dip immediately after each transition (warm-up period until the buffer refills past the minimum-size threshold at `train.py:1177`), which is acceptable given the deliberate dynamics shift.

### Checkpoint drift under parallel envs (accepted behavior)

Detailed in [`13_checkpoint_scheduling.md`](../environment/13_checkpoint_scheduling.md). Per user direction, the current checkpoint scheduler is **kept as-is** and the stage-transition trigger must use the **same drift semantics**:

- The checkpoint gate at `train.py:1664` is evaluated **once per training iteration** (not per episode). Saved Orbax step keys are `total_episodes_completed`, which drifts above the nominal multiple of `checkpoint_frequency` by up to `num_envs − 1` episodes.
- The stage-transition gate in this plan fires under the same rule: **the first iteration whose `total_episodes_completed >= boundaries[current_stage]`** triggers the transition. No forced save at the exact boundary, no nominal-milestone alignment, no changes to `main.last_checkpoint_save`, `max_to_keep`, or the step key passed to `checkpointer.save`.
- Consequence: a stage whose boundary is `3000` may actually transition at episode `3073`. This matches the behavior of the checkpointer today and is accepted.

The review doc (`13_checkpoint_scheduling.md`) describes the drift for reference but **does not prescribe any change** to the scheduler as part of this plan.

---

## Implementation Plan

### Design

**Core idea**: add a new CLI flag `--configs-dir <dir>` plus `--continual-schedule <schedule.yaml>`. When present, `train.py`:
1. Loads every `*.yaml` in `<dir>` (alphabetic sort) as an ordered list of stage configs.
2. Loads the schedule YAML which declares `episode_boundaries` (cumulative) and `checkpoint_frequencies` (parallel to stages).
3. Validates that `len(boundaries) == len(ckpt_freqs) == len(stage_configs)`, and that every stage produces the same `obs_dim` / `action_dim` (no static shape changes through the architecture-visible surface).
4. Sets `episodes = boundaries[-1]`.
5. Instantiates `ParallelEnv` from stage 0 and initializes the model normally.
6. In the main loop, when `total_episodes_completed ≥ boundaries[current_stage]` (checked once per iteration, same granularity as the existing checkpoint gate), advances `current_stage`, reloads params, rebuilds `ParallelEnv`, force-resets all envs, and looks up the next stage's `checkpoint_frequencies[i]`. Model, optimizer, RNG key, hidden state, and `main.last_checkpoint_save` all persist across the transition (no forced save at the boundary — drift is accepted per user direction).
7. For **DreamerV3 only**: at the same transition point, clear the replay buffer (`buffer.idx = 0; buffer.size = 0`) and the positive buffer if present. This prevents cross-stage dynamics contamination of the world model.
8. The existing checkpoint scheduler at `train.py:1656-1694` is **untouched**. The only change is that `checkpoint_freq` is looked up from `schedule.checkpoint_frequencies[current_stage]` instead of the single scalar config.
9. WandB: a single run; every iteration log includes `stage/index` and `stage/name`, and each transition logs a `stage/transition` marker. Stage metadata is added to `wandb.config` as a list.

**Why this shape**
- Alphabetic ordering + filename prefix is the simplest durable convention and matches what the user described.
- Cumulative boundaries (vs per-stage budgets) are what the user selected; they read more naturally as "switch at episode X".
- Single WandB run makes curriculum comparisons (reward/return across stage transitions) trivially plottable; stage tags stay queryable.
- Rebuilding `ParallelEnv` is the conservative choice — it sidesteps the fragile question of which fields are dynamic vs. static inside `EnvParams`. Cost is O(N_stages) JIT compiles, acceptable.
- **Stage-transition gate mirrors the checkpoint gate** (per-iteration check, accept drift up to `num_envs − 1` episodes). This keeps behavior consistent with the scheduler the user wants to preserve.

### File Changes

> All line numbers reference the current `train.py` (1783 lines).

#### `train.py` — CLI additions (near line 127)

```python
# BEFORE (line 127–156, excerpt):
parser.add_argument("--episodes", type=int, help="Number of episodes to train")
...
parser.add_argument("--config", type=str, help="Path to base config YAML (Environment/Ablation)")
...
parser.add_argument("--checkpoint-frequency", type=int, help="Save checkpoint every N episodes/evals")

# AFTER (add after the --config argument):
parser.add_argument("--configs-dir", type=str, default=None,
                    help="Directory of stage config YAMLs for continual learning. "
                         "Files are ordered alphabetically; prefix names with 01_, 02_, ... to control order. "
                         "Mutually exclusive with --config.")
parser.add_argument("--continual-schedule", type=str, default=None,
                    help="Path to schedule YAML (episode_boundaries, checkpoint_frequencies). "
                         "Required when --configs-dir is used.")
```

#### `train.py` — continual schedule loading (inserted after line 236, before line 239 "CLI Overrides")

Add a helper that returns either `None` (single-config mode, current behavior) or a `ContinualSchedule` dataclass.

```python
# NEW: tiny dataclass + loader. Place near the top of train.py (after imports).
from dataclasses import dataclass
from typing import List, Optional
import glob

@dataclass
class ContinualSchedule:
    stage_config_paths: List[str]          # absolute paths, alphabetic order
    stage_names: List[str]                 # file stem, e.g. "01_predator_intro"
    stage_configs: List[Config]            # pre-loaded Config per stage (base + stage overlay)
    episode_boundaries: List[int]          # cumulative, strictly increasing
    checkpoint_frequencies: List[int]      # parallel to stages

    @property
    def num_stages(self) -> int:
        return len(self.stage_config_paths)

    def stage_for_episode(self, episode: int) -> int:
        """Return stage index for the given (0-based) episode count."""
        for i, b in enumerate(self.episode_boundaries):
            if episode < b:
                return i
        return self.num_stages - 1   # past the end -> stay in final stage

def _build_continual_schedule(base_config: Config,
                              configs_dir: str,
                              schedule_path: str) -> ContinualSchedule:
    # 1. Discover stage files
    if not os.path.isdir(configs_dir):
        raise ValueError(f"--configs-dir '{configs_dir}' is not a directory.")
    paths = sorted(glob.glob(os.path.join(configs_dir, "*.yaml")))
    if not paths:
        raise ValueError(f"No *.yaml files found in {configs_dir}.")
    names = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    # 2. Load schedule YAML
    schedule = Config.load_yaml(schedule_path)
    boundaries = schedule.get_mandatory("continual.episode_boundaries")
    ckpt_freqs  = schedule.get_mandatory("continual.checkpoint_frequencies")

    if len(boundaries) != len(paths):
        raise ValueError(
            f"episode_boundaries length ({len(boundaries)}) != number of stage configs "
            f"({len(paths)}) in {configs_dir}.")
    if len(ckpt_freqs) != len(paths):
        raise ValueError(
            f"checkpoint_frequencies length ({len(ckpt_freqs)}) != number of stage configs "
            f"({len(paths)}).")
    if sorted(boundaries) != list(boundaries) or len(set(boundaries)) != len(boundaries):
        raise ValueError(f"episode_boundaries must be strictly increasing: {boundaries}")
    if any(f <= 0 for f in ckpt_freqs):
        raise ValueError(f"checkpoint_frequencies must be > 0: {ckpt_freqs}")

    # 3. Pre-build per-stage Config objects by cloning base and merging each stage YAML
    stage_configs = []
    for p in paths:
        stage_cfg = Config(yaml.safe_load(yaml.dump(base_config.to_dict())))  # deep copy
        stage_cfg.merge(Config.load_yaml(p))
        stage_configs.append(stage_cfg)

    return ContinualSchedule(
        stage_config_paths=paths,
        stage_names=names,
        stage_configs=stage_configs,
        episode_boundaries=list(boundaries),
        checkpoint_frequencies=list(ckpt_freqs),
    )
```

Call this after the existing base/train/eval/logger/vis merges, **instead of** the single `args.config` merge:

```python
# BEFORE (line 223–228):
# Merge Base/User/Ablation Config (--config)
if args.config:
    if not args.quiet:
        print(f"Loading override config from: {args.config}")
    user_config = Config.load_yaml(args.config)
    config.merge(user_config)

# AFTER:
schedule: Optional[ContinualSchedule] = None
if args.configs_dir is not None:
    if args.config:
        raise ValueError("--configs-dir and --config are mutually exclusive.")
    if args.continual_schedule is None:
        raise ValueError("--continual-schedule is required when --configs-dir is set.")
    schedule = _build_continual_schedule(config, args.configs_dir, args.continual_schedule)
    # Merge stage-0 into the live config so downstream code sees a fully-populated Config
    # for the starting stage.
    config = schedule.stage_configs[0]
    if not args.quiet:
        print(f"Continual mode: {schedule.num_stages} stages from {args.configs_dir}")
        for i, (n, b, f) in enumerate(zip(schedule.stage_names,
                                          schedule.episode_boundaries,
                                          schedule.checkpoint_frequencies)):
            print(f"  [{i:02d}] {n:30s}  until_ep={b:>6d}  ckpt_freq={f}")
elif args.config:
    if not args.quiet:
        print(f"Loading override config from: {args.config}")
    user_config = Config.load_yaml(args.config)
    config.merge(user_config)
```

#### `train.py` — `episodes` and `params` resolution (lines 295–328)

```python
# BEFORE (line 295):
episodes = args.episodes if args.episodes is not None else config.get_mandatory('episodes')

# AFTER:
if schedule is not None:
    if args.episodes is not None:
        raise ValueError("--episodes is incompatible with --configs-dir; "
                         "episode budget is set by the schedule's last boundary.")
    episodes = schedule.episode_boundaries[-1]
else:
    episodes = args.episodes if args.episodes is not None else config.get_mandatory('episodes')
```

```python
# BEFORE (line 328):
params = load_env_params(config)

# AFTER:
params = load_env_params(config)  # stage 0 (or single-config)

# Validate obs/action dim consistency across all stages BEFORE training starts.
if schedule is not None:
    env_probe = ParallelEnv(params)
    probe_key = jax.random.PRNGKey(0)
    _, probe_obs = env_probe.reset(probe_key, 1)
    stage0_obs_dim = int(probe_obs.shape[-1])
    stage0_action_dim = 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)
    for i in range(1, schedule.num_stages):
        p_i = load_env_params(schedule.stage_configs[i])
        env_i = ParallelEnv(p_i)
        _, obs_i = env_i.reset(probe_key, 1)
        a_i = 4 + int(p_i.rest_action_enabled) + int(p_i.eat_action_enabled)
        if int(obs_i.shape[-1]) != stage0_obs_dim or a_i != stage0_action_dim:
            raise ValueError(
                f"Stage {i} ({schedule.stage_names[i]}) changes "
                f"obs_dim ({stage0_obs_dim} -> {int(obs_i.shape[-1])}) or "
                f"action_dim ({stage0_action_dim} -> {a_i}). "
                "Continual learning forbids architecture-visible dimension changes.")
    del env_probe
```

#### `train.py` — results dir & config dump (lines 345–360)

When in continual mode, persist each stage config under `models/stage_NN_name.yaml` for full auditability.

```python
# AFTER the existing "config_save_path" block:
if schedule is not None:
    for i, (name, cfg) in enumerate(zip(schedule.stage_names, schedule.stage_configs)):
        out = os.path.join(models_dir, f"stage_{i:02d}_{name}.yaml")
        with open(out, "w") as f:
            yaml.dump(cfg.to_dict(), f, default_flow_style=False)
    # Also dump the schedule itself for reference
    sched_dump = {
        "continual": {
            "episode_boundaries": schedule.episode_boundaries,
            "checkpoint_frequencies": schedule.checkpoint_frequencies,
            "stage_names": schedule.stage_names,
        }
    }
    with open(os.path.join(models_dir, "schedule.yaml"), "w") as f:
        yaml.dump(sched_dump, f, default_flow_style=False)
```

#### `train.py` — WandB init (lines 368–402)

```python
# BEFORE (line 373-385, excerpt of wandb_kwargs["config"]):
"config": {
    "algorithm": algorithm,
    ...
    **config.to_dict()
},

# AFTER:
wandb_config_payload = {
    "algorithm": algorithm,
    "framework": "JAX/Flax NNX",
    "total_timesteps": total_timesteps,
    "num_envs": num_envs,
    "num_steps": num_steps,
    "lr": lr,
    "hidden_size": hidden_size,
    "seed": seed,
    **config.to_dict(),
}
if schedule is not None:
    wandb_config_payload["continual"] = {
        "num_stages": schedule.num_stages,
        "stage_names": schedule.stage_names,
        "episode_boundaries": schedule.episode_boundaries,
        "checkpoint_frequencies": schedule.checkpoint_frequencies,
    }
wandb_kwargs["config"] = wandb_config_payload
```

Add a new metric definition just after the existing `define_metric` block (around line 400):

```python
wandb.define_metric("stage/index",       step_metric="Episode/Number")
wandb.define_metric("stage/transition",  step_metric="Episode/Number")
```

#### `train.py` — training loop: stage-transition check (inside the `while` at line 808)

Add at the top of each iteration (just after `iteration += 1` at line 812). The gate uses the same per-iteration granularity as the existing checkpoint scheduler — drift relative to `boundaries[i]` is accepted.

```python
# NEW: stage transition check (only in continual mode)
if schedule is not None:
    new_stage = schedule.stage_for_episode(total_episodes_completed)
    if new_stage != current_stage:
        old_name = schedule.stage_names[current_stage]
        new_name = schedule.stage_names[new_stage]
        if not args.quiet:
            pbar.write(f"[STAGE] {current_stage}:{old_name} -> {new_stage}:{new_name} "
                       f"at ep={total_episodes_completed} "
                       f"(boundary was {schedule.episode_boundaries[current_stage]})")

        # Rebuild env with new params. Model, optimizer, h_state, key persist.
        # NO forced save here — the periodic scheduler at train.py:1656-1694 is
        # the single source of truth for saves, same drift as today.
        params = load_env_params(schedule.stage_configs[new_stage])
        env = ParallelEnv(params)
        key, reset_key = jax.random.split(key)
        env_state, obs = env.reset(reset_key, num_envs)
        # Wipe in-flight episode accumulators for the boundary envs (they are mid-episode).
        # Their partial episodes are silently dropped per user decision.
        episode_returns[:] = 0.0
        episode_lengths[:] = 0
        for k in BEHAVIOR_KEYS:
            episode_behavior[k][:] = 0.0
        for k in BEHAVIOR_DIST_KEYS:
            episode_dist_sums[k][:] = 0.0

        # --- DreamerV3 only: clear replay buffers to prevent cross-stage
        # dynamics contamination of the world model. ---
        if algorithm == "DreamerV3":
            # Cheap reset: mark as empty. sample() gates on self.size so the
            # (now stale) array contents become unreachable without a full wipe.
            pre_size = buffer.size
            buffer.idx = 0
            buffer.size = 0
            pos_pre_size = 0
            if positive_buffer is not None:
                pos_pre_size = positive_buffer.size
                positive_buffer.idx = 0
                positive_buffer.size = 0
            if not args.quiet:
                pbar.write(f"[STAGE] Cleared Dreamer replay buffer "
                           f"({pre_size} transitions) and positive buffer "
                           f"({pos_pre_size} transitions).")
            if wandb_enabled:
                wandb.log({
                    "stage/buffer_cleared_main":     pre_size,
                    "stage/buffer_cleared_positive": pos_pre_size,
                    "Episode/Number": total_episodes_completed,
                })

        current_stage = new_stage

        if wandb_enabled:
            wandb.log({
                "stage/index": current_stage,
                "stage/transition": 1,
                "Episode/Number": total_episodes_completed,
            })
```

`current_stage` must be initialized to 0 just above the main loop (near line 702, next to `total_episodes_completed = 0`):

```python
current_stage = 0
```

Note: `main.last_checkpoint_save` is **not** reset on transition. It keeps advancing monotonically across stages, which is consistent with "one global episode counter, stage-specific frequency lookup" — the next save will fire when `total_episodes_completed ≥ main.last_checkpoint_save + checkpoint_frequencies[current_stage]`.

#### `train.py` — checkpoint frequency lookup (line 1657)

Minimal change: per-stage lookup in continual mode, otherwise unchanged. **Everything else in the scheduler stays exactly as it is today** — same drift, same step key (`total_episodes_completed`), same `main.last_checkpoint_save` singleton, same YAML-configurable `max_to_keep`.

```python
# BEFORE (train.py:1657):
checkpoint_freq = args.checkpoint_frequency or config.get_mandatory('training.checkpoint_frequency')

# AFTER:
if schedule is not None:
    checkpoint_freq = schedule.checkpoint_frequencies[current_stage]
else:
    checkpoint_freq = args.checkpoint_frequency or config.get_mandatory('training.checkpoint_frequency')
```

No other changes to the scheduler block (`train.py:1658-1694`).

#### `train.py` — per-iteration WandB logs (around line 892 and line 934)

Every `wandb.log({...})` call inside the loop should include the stage tag. Simplest approach: define a helper near the top of the loop:

```python
def _stage_tag() -> dict:
    if schedule is None:
        return {}
    return {"stage/index": current_stage, "stage/name": schedule.stage_names[current_stage]}
```

And add `**_stage_tag()` into the dicts at lines 900 (Episode log), 935 (iteration log), and 1735 (eval log). `stage/name` is a string; WandB stores it as text — fine for filtering.

#### `train.py` — checkpoint payload (lines 1671–1689)

Add the current stage index so resume can reproduce curriculum state:

```python
ckpt_data["stage"] = current_stage   # add to both RecurrentPPO and DreamerV3 branches
```

#### `train.py` — resume block (lines 741 / 794)

Restore `current_stage` from the checkpoint:

```python
# Inside each restore branch, after the existing total_episodes_completed restore:
if schedule is not None:
    current_stage = restored.get('stage', 0)
```

No changes to `main.last_checkpoint_save` handling — after resume it is recreated by the existing `if not hasattr(main, 'last_checkpoint_save')` guard and updated on the next save.

#### `configs/continual/` — new directory with an example schedule YAML

```yaml
# configs/continual/example_schedule.yaml
continual:
  # Cumulative episode boundaries. Stage i runs while episode_count < episode_boundaries[i].
  # Total episodes = episode_boundaries[-1].
  episode_boundaries: [1000, 3000, 3500]

  # Parallel to boundaries. One int per stage.
  checkpoint_frequencies: [200, 500, 100]
```

(No changes to any `src/` files — all wiring lives in `train.py`.)

### Usage

```bash
python train.py \
    --configs-dir configs/experiment/curriculum_basic/ \
    --continual-schedule configs/continual/example_schedule.yaml \
    --agent_config configs/models/dreamerv3.yaml \
    --tag curriculum_basic_run01
```

Where `configs/experiment/curriculum_basic/` contains e.g.:

```
01_no_predator.yaml
02_predator_slow.yaml
03_predator_fast.yaml
```

## Checkpoints

- [ ] **Ckpt 1** — Single-config mode still works: running `train.py --config <file>` produces identical behaviour to `main` (no regression). Diff-check `args.configs_dir is None` path never touches new code.
- [ ] **Ckpt 2** — Schedule validation fails loudly: test with mismatched list lengths, non-increasing boundaries, and a stage that toggles a sensor — each raises a clear `ValueError` before training starts.
- [ ] **Ckpt 3** — Alphabetic ordering test: given files `03_c.yaml`, `01_a.yaml`, `02_b.yaml`, `glob.glob(...)` + `sorted(...)` yields `[01_a, 02_b, 03_c]`. Print `schedule.stage_names` at startup to confirm.
- [ ] **Ckpt 4** — Stage transition fires exactly once: set `boundaries=[5, 10, 15]` with `num_envs=1` and a dummy agent; log stage index each iteration and assert transitions occur only at the expected episode counts.
- [ ] **Ckpt 5** — Env rebuild actually takes effect: after transition, change a cheap-to-observe param like `food_nutrition_gain` between stages and confirm the agent observes the new value (via `behavior/ate_food_reward` WandB metric).
- [ ] **Ckpt 6** — Per-stage ckpt freq honoured: stage with freq=1000 followed by stage with freq=100 produces saves spaced ≈100 episodes apart in the new stage (with the same drift the user already accepts). No forced save at the boundary.
- [ ] **Ckpt 7** — Scheduler untouched: diff of `train.py:1658-1694` shows only the one-line `checkpoint_freq` lookup change (line 1657). Step key passed to `checkpointer.save` is still `total_episodes_completed`; `main.last_checkpoint_save` still snaps via `(total // freq) * freq`; `max_to_keep` logic unchanged.
- [ ] **Ckpt 8** — WandB schema: after one transition, the WandB run has `stage/index` increasing 0 → 1 with a `stage/transition=1` spike, and `wandb.config.continual.stage_names` populated.
- [ ] **Ckpt 9** — DreamerV3 buffer cleared at transition: launch with `algorithm=DreamerV3`, log `buffer.size` every iteration; immediately after a transition, `buffer.size == 0` and then regrows with new-stage transitions. Verify `stage/buffer_cleared_main` appears in WandB with the pre-clear size. For RecurrentPPO, the transition code path skips the clear block (no buffer to clear).
- [ ] **Ckpt 10** — Mid-episode drop is observable: with `num_envs=5` and a boundary=100 schedule, log per-env `episode_returns` immediately before and after the transition. Verify that envs with non-zero pre-transition returns show zero post-transition, and `ep_info_buffer` gains no entries for them.
- [ ] **Ckpt 11** — Resume restores stage: save a checkpoint mid-stage 1, then `--load-checkpoint <path>`; verify `current_stage==1` at resume and that the stage-1 env params are loaded (not stage-0).

## Implementation Report

> **Implemented by**: _pending_
> **Date**: _pending_

<!-- Filled by the implementing agent. -->

## Verification Report

> **Verified by**: _pending_
> **Date**: _pending_

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `train.py` | add `--configs-dir`, `--continual-schedule`, schedule loader, stage-transition logic (hard reset + drop mid-episode; DreamerV3 replay-buffer clear), per-stage `checkpoint_freq` lookup (one-line change at 1657), stage-tagged wandb logs, stage-aware resume | | |
| `configs/continual/example_schedule.yaml` | new example schedule | | |

**Conclusion**: _pending_
