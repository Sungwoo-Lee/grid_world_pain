---
title: "Sheeprl Training: How-To"
topic: diagnosis
status: active
created: 2026-05-12
last_updated: 2026-05-12
phase: usage-guide
---

# Sheeprl Training — How-To

> **Purpose**. This document tells a fresh Claude session (or human collaborator) exactly how to run another sheeprl DreamerV3 training on our environment. It assumes you do NOT have prior context from the original smoke-launch session.

> **What this is**. A practical usage guide for an already-built system. The system: a thin Python wrapper that lets sheeprl's stock DreamerV3 train on our 5×5 gridworld task. The pipeline was built and validated on 2026-05-11 by a smoke run (WandB run `jzgkcep4`) that reached survival ~500 within 25,000 env steps on the food-only no-predator task. The system runs unmodified upstream sheeprl with its own published `dreamer_v3_XS` model and hyperparameters — only the env is ours.

> **What this is NOT**. Not a re-implementation plan. Not an analysis of why sheeprl outperforms our DreamerV3. See:
> - The original plan + analysis: [`sheeprl_drop_in_test.md`](sheeprl_drop_in_test.md)
> - The line-by-line code walkthrough of sheeprl: [`../../../project/references/sheeprl_dreamer_v3/INDEX.md`](../../../project/references/sheeprl_dreamer_v3/INDEX.md)
> - The fix-cascade summary: [`../../../experiments/summaries/20260511_1559_dreamer_v3_fix_cascade.md`](../../../experiments/summaries/20260511_1559_dreamer_v3_fix_cascade.md)

---

## §1. TL;DR — launch a training in one paragraph

The lab cluster keeps the project mounted at `/media/nas01/projects/Interoceptive-AI/grid_world_pain/` on every node (NAS-shared). The `sheeprl_bridge` conda environment is **per-node** at `/home/vncuser/miniconda3/envs/sheeprl_bridge/` — it exists on **node 114** (where the smoke ran) and **not yet on any other node**. The recommended launch path is the project's `run_command.py` (port-1800 SSH multiplexed, backgrounds + logs automatically) calling the reusable workload `scripts/launch_sheeprl.sh`:

```bash
./run_command.py 114 "bash scripts/launch_sheeprl.sh \
    configs/experiment/dreamer_curriculum/01_food_only.yaml 0 gwp_food_only"
```

That fires a 200,000-env-step training (~11 hours on a single RTX 6000 Ada) with `dreamer_v3_XS`, WandB logging to project `grid_world_pain_sheeprl_test`, log at `logs/YYYYMMDD_HHMMSS.log`. Tail the log or open WandB to watch it.

Positional args to `launch_sheeprl.sh` are `<config-yaml> <gpu-index> <env-id-tag> [total-steps]`. The script handles `cd` to project root, sets `GWP_CONFIG_PATH` + `JAX_PLATFORMS=cpu` + `CUDA_VISIBLE_DEVICES=<gpu>`, and invokes the `sheeprl_bridge` Python explicitly — you don't have to think about any of that. For variations (different env config, different model size, different step budget) see §6 below.

---

## §2. Architecture — what's where

The system has four pieces. All four already exist; you do not create anything for a standard run.

### 2.1 The env bridge

**File**: `tmp/sheeprl/sheeprl/envs/grid_world_pain.py`

A 113-line Python module defining `GridWorldPainWrapper(gym.Env)`. It:

- Forces `JAX_PLATFORMS=cpu` at module import (before any JAX import) so torch can claim the GPU.
- Adds the grid_world_pain project root to `sys.path` so `from src.environment.* import ...` works regardless of Hydra's working-directory shuffling.
- In `__init__`: loads our YAML config via `Config.load_yaml(config_path)` merged onto `get_default_config()`, calls `load_env_params(cfg)` to build the JAX `EnvParams`, derives `gym.spaces.Discrete(n_actions)` from `EnvParams.action_dim`, derives `gym.spaces.Dict({"state": Box(shape=obs.shape)})` from a probe `get_observation` call.
- In `reset` / `step`: calls `jax_reset` / `jax_step`, converts JAX arrays to numpy, returns the gymnasium 5-tuple.
- **Drops the env's info dict** (`return ..., {}` from step) because `gym.vector.SyncVectorEnv` cannot stack JAX arrays from N parallel envs. If you need behavioral metrics inside info, extend the bridge — see §7.4.

### 2.2 The three sheeprl YAML configs

All in `tmp/sheeprl/sheeprl/configs/`:

**`env/grid_world_pain.yaml`** — registers our wrapper as the env target:
```yaml
defaults: [default, _self_]
id: grid_world_pain
wrapper:
  _target_: sheeprl.envs.grid_world_pain.GridWorldPainWrapper
  config_path: ${oc.env:GWP_CONFIG_PATH}   # read from env var at launch time
  seed: ${seed}
num_envs: 4
sync_env: True
capture_video: False
frame_stack: 1
action_repeat: 1
clip_rewards: False
grayscale: False
screen_size: 64
max_episode_steps: null
```

**`exp/dreamer_v3_grid_world_pain.yaml`** — the experiment recipe:
```yaml
# @package _global_
defaults:
  - dreamer_v3
  - override /algo: dreamer_v3_XS          # 256-unit MLP, 1 layer, 256 recurrent
  - override /env: grid_world_pain
  - override /logger@metric.logger: wandb
  - _self_

seed: 42
fabric:
  accelerator: cuda
algo:
  total_steps: 200_000
  per_rank_sequence_length: 64
  cnn_keys:
    encoder: []
    decoder: []
  mlp_keys:
    encoder: [state]
    decoder: [state]
```

**`logger/wandb.yaml`** — Lightning's PyTorch-side WandB logger (sheeprl ships only TensorBoard + MLflow by default):
```yaml
_target_: lightning.pytorch.loggers.WandbLogger
project: grid_world_pain_sheeprl_test
name: ${run_name}
save_dir: logs/runs/${root_dir}
log_model: False
```

All three are tracked in git (committed under `tmp/sheeprl/` because we vendor sheeprl into the repo).

### 2.3 The `sheeprl_bridge` conda environment

**Location**: `/home/vncuser/miniconda3/envs/sheeprl_bridge/` — **node-local, not shared**.

**Currently exists on**: node 114 only. Verify before launching elsewhere:
```bash
ssh vncuser@192.168.0.<NODE> '/home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python -c "import torch, jax, sheeprl, wandb; print(\"torch\", torch.__version__, \"jax\", jax.__version__)"'
```
On node 114 this prints `torch 2.5.0+cu121, jax 0.10.0, sheeprl ok, wandb 0.26.1`. To create the env on a new node, follow §5.

### 2.4 The grid_world_pain project root

**Location**: `/media/nas01/projects/Interoceptive-AI/grid_world_pain/` — **NAS-shared**, visible at the same path on every node.

The launch command relies on this — the `GWP_CONFIG_PATH` env var points at a YAML inside this tree, and the bridge imports `src.environment.*` from there.

---

## §3. Prerequisites checklist

Before launching anything, confirm all five:

1. **Target node accessible**: `ssh vncuser@192.168.0.<NODE>` works without password (key-based).
2. **Conda env present** (§2.3): `import torch, jax, sheeprl, wandb` succeeds on the target node.
3. **Project mounted**: `ls /media/nas01/projects/Interoceptive-AI/grid_world_pain/tmp/sheeprl/sheeprl.py` succeeds on the target node.
4. **WandB credentials**: `cat ~/.netrc | grep -A1 wandb` shows a valid `password` line. The shared key file is at `/media/nas01/projects/Interoceptive-AI/grid_world_pain/.wandb_api_key` (gitignored). If `wandb.login()` fails inside the run, copy the key into `~/.netrc` on the target node or run `wandb login` interactively once.
5. **GPU free**: `ssh vncuser@192.168.0.<NODE> nvidia-smi` shows the chosen GPU index has <1 GB used. Sheeprl_XS will consume ~2 GB.

If any fails, fix it before launching. The training will fail fast and obviously if any of these are off (no silent hangs).

---

## §4. The full launch flow, explained

The recommended launch is a single `run_command.py` call into `scripts/launch_sheeprl.sh`:

```bash
./run_command.py 114 "bash scripts/launch_sheeprl.sh <config> <gpu> <tag> [steps]"
```

Two pieces, each with a separate job:

### `run_command.py` — the thin SSH wrapper

After the 2026-05-12 refactor, `run_command.py` does only this:
- Open an SSH-multiplexed connection to the target node on port 1800 (`vncuser@192.168.0.<node>`, key auth).
- Run the passed command under `nohup bash -c "<cmd>" > logs/YYYYMMDD_HHMMSS.log 2>&1 &` on the remote side.
- Optionally auto-tail the log (default; pass `--no-tail` to skip; `--foreground` for synchronous output).

It does NOT `cd` anywhere, NOT activate any conda env, NOT validate the command. Those responsibilities are inside the bash script.

### `scripts/launch_sheeprl.sh` — the workload

The bash script you actually run takes 3 positional args + 1 optional:
- `$1` — config YAML (relative-to-project-root or absolute)
- `$2` — GPU index (0, 1, 2, …)
- `$3` — env-id tag (becomes part of the WandB run name)
- `$4` — total env steps (default `200_000`)

At launch it:
- `cd /media/nas01/projects/Interoceptive-AI/grid_world_pain` (so relative paths work)
- `export GWP_CONFIG_PATH=$(realpath "$1")` (absolute path for the bridge)
- `export JAX_PLATFORMS=cpu` (keep GPU for torch)
- `export CUDA_VISIBLE_DEVICES=$2` (the GPU index)
- `exec /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python tmp/sheeprl/sheeprl.py exp=dreamer_v3_grid_world_pain env.id="$3" algo.total_steps="$STEPS"`

Read the script: 50 lines, no magic.

### Concrete launches from the 2026-05-12 session

```bash
./run_command.py 114 "bash scripts/launch_sheeprl.sh configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml 0 gwp_5x5_pred"
./run_command.py 114 "bash scripts/launch_sheeprl.sh configs/experiment/hypervigilance/01-interoNocicept.yaml 1 gwp_10x10_intero"
```

Two trainings on the same node, different GPUs, different configs — each WandB run named uniquely by the `tag` argument.

---

## §5. Setting up a new node (only if not node 114)

If the conda env doesn't exist on your target node (`ls /home/vncuser/miniconda3/envs/sheeprl_bridge` fails), create it. **Takes ~5 minutes**.

```bash
ssh vncuser@192.168.0.<NODE> 'bash -lc "\
conda create -n sheeprl_bridge python=3.11 -y && \
# Check the node's CUDA driver:
nvidia-smi | grep -i 'cuda version'"'
```

The output shows the CUDA driver version (e.g., `CUDA Version: 12.2` → use cu121; `CUDA Version: 13.0` → use cu130). Then:

```bash
# cu121 path (CUDA driver 12.x):
ssh vncuser@192.168.0.<NODE> '/home/vncuser/miniconda3/envs/sheeprl_bridge/bin/pip install \
    torch==2.5.0 --index-url https://download.pytorch.org/whl/cu121'

# cu130 path (CUDA driver 13.x):
ssh vncuser@192.168.0.<NODE> '/home/vncuser/miniconda3/envs/sheeprl_bridge/bin/pip install \
    torch --index-url https://download.pytorch.org/whl/cu130'

# Then for both:
ssh vncuser@192.168.0.<NODE> '/home/vncuser/miniconda3/envs/sheeprl_bridge/bin/pip install \
    -e /media/nas01/projects/Interoceptive-AI/grid_world_pain/tmp/sheeprl && \
    /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/pip install "jax[cpu]" flax omegaconf pyyaml wandb && \
    /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/pip install \
    -e /media/nas01/projects/Interoceptive-AI/grid_world_pain'

# Verify:
ssh vncuser@192.168.0.<NODE> '/home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python \
    -c "import torch, jax, sheeprl, wandb; print(\"OK\", torch.__version__, jax.__version__)"'
```

WandB credentials are per-machine — copy `~/.netrc` from a working node, or run `wandb login` interactively once on the new node.

---

## §6. Common variations

### 6.1 Different env YAML

Change line 5 of the launch command (the `GWP_CONFIG_PATH`). Any YAML compatible with `load_env_params` works. Examples:

| YAML | Task |
|---|---|
| `configs/experiment/dreamer_curriculum/01_food_only.yaml` | 5×5 grid, food only, no predator. **Used by smoke.** |
| `configs/experiment/basic/00-5X5_NoPred.yaml` | Same task but with hiding_predator (count=1). |
| `configs/experiment/basic/00-10X10_Pred.yaml` | Full hypervigilance task with mobile predator. |

The bridge will pick up the new obs/action shapes automatically (no config changes needed inside sheeprl).

### 6.2 Different model size

Sheeprl ships five model variants. Override `algo` on the CLI:

```bash
... tmp/sheeprl/sheeprl.py exp=dreamer_v3_grid_world_pain algo=dreamer_v3_S
```

| Variant | dense_units | mlp_layers | recurrent_size | When |
|---|---|---|---|---|
| `dreamer_v3_XS` | 256 | 1 | 256 | Smoke / quick iteration. **Default.** |
| `dreamer_v3_S` | 512 | 2 | 512 | Mid-size; closer to paper recommendations for small envs. |
| `dreamer_v3_M` | 768 | 3 | 1024 | Real eval territory. |
| `dreamer_v3_L` | 1024 | 4 | 2048 | DMC-scale. |
| `dreamer_v3_XL` | 1024 | 5 | 4096 | Atari/Crafter-scale. The published default. |

Caveat: large models on our tiny env are dramatically over-parameterized. XS is the right starting point.

### 6.3 Different total steps

```bash
... tmp/sheeprl/sheeprl.py exp=dreamer_v3_grid_world_pain algo.total_steps=500_000
```

At sheeprl's default `replay_ratio=1`, throughput on RTX 6000 Ada with XS is ~5 env steps/sec, so wall-clock ≈ steps / 5 / 3600 hours. 200k = ~11h; 500k = ~28h; 1M = ~55h.

### 6.4 Different seed (multi-seed sweep)

```bash
... tmp/sheeprl/sheeprl.py exp=dreamer_v3_grid_world_pain seed=43
... tmp/sheeprl/sheeprl.py exp=dreamer_v3_grid_world_pain seed=44
```

Each goes to its own WandB run. WandB project `grid_world_pain_sheeprl_test` will accumulate runs across seeds.

### 6.5 Faster logging (default flushes every 5000 steps)

```bash
... tmp/sheeprl/sheeprl.py exp=dreamer_v3_grid_world_pain metric.log_every=500
```

Drops the WandB flush cadence from every 5000 → every 500 policy steps. First chart point appears ~8 minutes in instead of ~83 minutes. Algorithm unchanged; useful for short smokes.

### 6.6 Faster training (changes algorithm — flag clearly!)

The wall-clock bottleneck is sheeprl's default `replay_ratio=1` (one gradient update per env step). To trade learning fidelity for speed:

```bash
... tmp/sheeprl/sheeprl.py exp=dreamer_v3_grid_world_pain algo.replay_ratio=0.25
```

That's 4× faster but does 4× fewer gradient updates per env step. **Do not use this if you're comparing against the published sheeprl results** — those use `replay_ratio=1`.

### 6.7 Different WandB project

Edit `tmp/sheeprl/sheeprl/configs/logger/wandb.yaml` and change the `project:` field. Commit the change.

---

## §7. Watching a running training

### 7.1 Tail the log

```bash
ssh vncuser@192.168.0.114 'tail -f /media/nas01/projects/Interoceptive-AI/grid_world_pain/tmp/sheeprl_run_<TIMESTAMP>.log'
```

The log shows: Hydra config dump at start, WandB run URL, then periodic `Rank-0: policy_step=X, reward_env_i=Y` lines per env episode end.

### 7.2 WandB UI

Open `https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test` — every run lives in this project. The smoke run's URL was https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/jzgkcep4. Key panels:

- `Game/ep_len_avg` — **survival steps** (= our `Episode/Steps`). Compare against A1=106, Z2=~115.
- `Rewards/rew_avg` — cumulative episode return.
- `Loss/world_model_loss`, `Loss/reward_loss`, `Loss/value_loss`, `Loss/policy_loss`.
- `State/kl`, `State/post_entropy`, `State/prior_entropy`.

### 7.3 Verify exactly one PID is running

```bash
ssh vncuser@192.168.0.114 'pgrep -af sheeprl.py'
```

Expect exactly one line containing the launch command. Two = duplicate launch (kill the older one). Zero = it crashed (read the log).

### 7.4 Check GPU usage

```bash
ssh vncuser@192.168.0.114 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv'
```

XS should use ~1.9 GB. Utilization 20–30% is normal (the env-step is on CPU, so GPU sits idle between gradient updates).

### 7.5 Kill a run

```bash
ssh vncuser@192.168.0.114 'kill $(cat /media/nas01/projects/Interoceptive-AI/grid_world_pain/tmp/sheeprl_run.pid)'
ssh vncuser@192.168.0.114 'pgrep -af sheeprl.py'   # confirm gone
```

The WandB run will be marked "crashed" in the UI; you can manually mark it "killed" if you want.

---

## §8. Known gotchas

These were all surfaced during the 2026-05-11 smoke. Read them before debugging.

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA driver version is insufficient` at startup | Default sheeprl_bridge pip install grabs `torch+cu130` but the node's CUDA driver is older (e.g., node 114 = 12.2). | Reinstall torch with the matching cu-suffix (see §5). |
| `RuntimeError: ... fabric.accelerator` | Sheeprl's default fabric is CPU. Our exp config sets `fabric.accelerator: cuda` — confirm that line is still there. | Check `tmp/sheeprl/sheeprl/configs/exp/dreamer_v3_grid_world_pain.yaml`. |
| Mysterious `SyncVectorEnv` stacking error | Something inside our env's info dict is a JAX array; sync vector can't stack it across N envs. | The bridge already returns `{}` for info — if you extend it, make sure info values are numpy or Python scalars. |
| First WandB chart point takes ~1 hour to appear | Sheeprl's default `metric.log_every=5000` policy steps × `replay_ratio=1` × ~1s/grad-step = ~1.4 hours wall-clock. | Drop `log_every` (see §6.5) for short smokes. Or just wait. |
| `from src.utils.config import ...` fails | `sys.path` doesn't include the project root because Hydra changed cwd. | The bridge already injects `_PROJECT_ROOT` into `sys.path`. If you move the bridge file, update the `_PROJECT_ROOT = ...` calculation (four `..` levels up). |
| Training runs at ~5 env steps/sec; "shouldn't num_envs=4 give 4× speedup?" | No — with `replay_ratio=1`, sheeprl fires 4 grad steps per outer iteration when `num_envs=4`, and grad steps dominate wall time. `num_envs` doesn't help on tiny envs where the env-step is microseconds. | Accept it, or drop `replay_ratio` per §6.6 (not apples-to-apples then). |
| Node 114 CIFS staleness on `train_command-agent.sh` | Stale cached file on node 114's CIFS mount. | Not relevant here — we don't use `train_command-agent.sh` for sheeprl. Mentioned for completeness; project memory rule. |

---

## §9. Quick reference

### Files

| Thing | Path |
|---|---|
| Bridge | `tmp/sheeprl/sheeprl/envs/grid_world_pain.py` |
| Env YAML | `tmp/sheeprl/sheeprl/configs/env/grid_world_pain.yaml` |
| Exp YAML | `tmp/sheeprl/sheeprl/configs/exp/dreamer_v3_grid_world_pain.yaml` |
| Logger YAML | `tmp/sheeprl/sheeprl/configs/logger/wandb.yaml` |
| Sheeprl entry | `tmp/sheeprl/sheeprl.py` |
| Conda env (per node) | `/home/vncuser/miniconda3/envs/sheeprl_bridge/` |
| Project root (NAS) | `/media/nas01/projects/Interoceptive-AI/grid_world_pain/` |
| Default task config | `configs/experiment/dreamer_curriculum/01_food_only.yaml` |

### Critical env vars at launch

| Var | Value | Why |
|---|---|---|
| `GWP_CONFIG_PATH` | Absolute path to a YAML the bridge can load | Tells the bridge which task to run |
| `JAX_PLATFORMS` | `cpu` | Stops JAX from grabbing the GPU |
| `CUDA_VISIBLE_DEVICES` | `0` (or `1`, `2`, ...) | Which GPU torch uses |

### WandB

- Project: `grid_world_pain_sheeprl_test`
- Entity: `sungwoolee`
- Smoke reference run: `jzgkcep4` (https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/jzgkcep4)

### Last validated

2026-05-11 by smoke run `jzgkcep4` (200k env steps, food-only task, dreamer_v3_XS, node 114 GPU 0). Reached `Game/ep_len_avg=500` (env-cap) within 25k steps.

---

## §10. Related docs

- [`sheeprl_drop_in_test.md`](sheeprl_drop_in_test.md) — the original plan + Results/Analysis from the smoke. Read for "why this exists" + the original Implementation Report.
- [`../../../project/references/sheeprl_dreamer_v3/INDEX.md`](../../../project/references/sheeprl_dreamer_v3/INDEX.md) — line-by-line walkthrough of the sheeprl DreamerV3 source code (9 files, 226 functions, ~6,800 doc lines). For "what the sheeprl code actually does" questions.
- [`../../../experiments/summaries/20260511_1559_dreamer_v3_fix_cascade.md`](../../../experiments/summaries/20260511_1559_dreamer_v3_fix_cascade.md) — the fix-cascade story this sheeprl test exists to compare against.
- [`../../../project/concepts/dreamer_v3_implementation.md`](../../../project/concepts/dreamer_v3_implementation.md) — the paper-to-implementation reference doc; §6 is the 30-item deviation list against our codebase, §9 is the sheeprl-comparison section.
