#!/bin/bash
set -euo pipefail

# This script is launched via `run_command.py` which no longer cds to the
# project root or activates a conda env. Both responsibilities live here.
cd /media/nas01/projects/Interoceptive-AI/grid_world_pain

#
# train_command-agent.sh
# ----------------------
# Launch script edited ONLY by the `training-runner` agent.
# The user's manual launch script is `train_command-new.sh` — the agent never touches that.
#
# ---------------------------------------------------------------------------
# train.py CLI arguments (full list — agent fills the relevant ones below)
# ---------------------------------------------------------------------------
# Required:
#   --agent_config <path>     Path to agent/model config YAML (e.g., configs/models/dreamer_v3/dreamer_v3.yaml)
#
# Common (env / training):
#   --config <path>           Base env config YAML (single-stage)
#   --configs-dir <dir>       Directory of stage YAMLs for continual learning (mutually exclusive with --config)
#   --continual-schedule <p>  Schedule YAML (required when --configs-dir is used)
#   --episodes <int>          Number of episodes
#   --total-timesteps <int>   Total timesteps (overrides --episodes if set)
#   --seed <int>              Random seed
#   --num-envs <int>          Parallel envs
#   --num-steps <int>         Steps per iteration (rollout length)
#   --hidden-size <int>       Hidden layer size
#   --lr <float>              Learning rate (overrides agent config)
#   --device <str>            'cuda:N' / 'gpu' / 'cpu'
#   --no-satiation            Disable satiation
#   --no-overeating-death     Disable death by overeating
#   --checkpoint-frequency N  Save checkpoint every N evals
#   --load-checkpoint <path>  Resume from checkpoint
#   --results-dir <path>      Custom results directory
#
# WandB filtering (see convention block below):
#   --wandb-project <str>     Default: 'grid_world_pain' (leave unset; default applies)
#   --wandb-entity <str>      Default: 'sungwoolee'      (leave unset; default applies)
#   --wandb-group <str>       Experiment family
#   --wandb-job-type <str>    Operational category
#   --wandb-name <str>        Run display name in WandB web
#   --wandb-resume-id <str>   Resume an existing WandB run by id
#   --no-wandb                Disable WandB logging entirely
#
# Logging / dev:
#   --tag <str>               Project-internal tag (drives results/JAX_<algo>/<ts>_<tag>/ and logs/<ts>_<tag>.log)
#   --log-interval <int>      WandB logging interval (iterations)
#   --log-accumulate / --no-log-accumulate
#                             Accumulate episode metrics across the log interval (default: accumulate)
#   --quiet                   Suppress stdout / progress bar
#   --debug                   Verbose step-by-step progress
#   --profile                 jax.profiler trace; forces --no-wandb + --quiet
#
# ---------------------------------------------------------------------------
# WandB-field convention (the agent fills all four below)
# ---------------------------------------------------------------------------
#   --wandb-group     experiment family — top dir under configs/environment/experiment/
#                       e.g. 'basic', 'hypervigilance', 'noise'
#
#   --wandb-job-type  operational category. Default 'prod'.
#                     Set to 'debug' / 'test' / 'pilot' / 'ablation' only when the user says so.
#                     (Algorithm is already filterable via Config.agent.algorithm —
#                      job-type is reserved for ops metadata.)
#
#   --wandb-name      run display name in WandB web. Format:
#                       <algo>_<config_stem>_n<node>            (single seed)
#                       <algo>_<config_stem>_s<seed>_n<node>    (seed override)
#                     e.g. 'dreamer_v3_00-5X5_NoPred_n113'
#
#   --tag             identical to --wandb-name. Drives local paths:
#                       results/JAX_<algo>/<ts>_<tag>/
#                       logs/<ts>_<tag>.log
#                     and shows up as Config.tag in the WandB run config.
#
# Defaults from configs/logger/wandb.yaml: project=grid_world_pain, entity=sungwoolee.
# The agent leaves --wandb-project and --wandb-entity unset so those defaults apply.
# ---------------------------------------------------------------------------

# hunger_gated_lindecay sweep — 2026-06-20 through 2026-06-22
# 10 recurrent_ppo runs across nodes 101/102/103/110/106/108, seed 42, ~10M episodes, fresh-init.
# wandb-group: hunger_gated_lindecay, job-type: prod
# Launched via CIFS-bypass /tmp scripts; this file is the audit record.
#
# Run 01: rppo_hg01_s0_sig0_dp1_s42      — node 108, cuda:0  (held anchor; launched 2026-06-22)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/01-s0_sig0.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:0 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg01_s0_sig0_dp1_s42 \
#   --tag rppo_hg01_s0_sig0_dp1_s42
#
# Run 02: rppo_hg02_s0.05_sig0_dp1_s42   — node 101, cuda:0
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/hunger_gated_lindecay/02-s0.05_sig0.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --device cuda:0 \
  --num-envs 16 \
  --episodes 10000000 \
  --seed 42 \
  --wandb-group hunger_gated_lindecay \
  --wandb-job-type prod \
  --wandb-name rppo_hg02_s0.05_sig0_dp1_s42 \
  --tag rppo_hg02_s0.05_sig0_dp1_s42
#
# Run 03: rppo_hg03_s0.1_sig0_dp1_s42    — node 101, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/03-s0.1_sig0.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:1 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg03_s0.1_sig0_dp1_s42 \
#   --tag rppo_hg03_s0.1_sig0_dp1_s42
#
# Run 04: rppo_hg04_s0.25_sig0_dp1_s42   — node 102, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/04-s0.25_sig0.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:0 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg04_s0.25_sig0_dp1_s42 \
#   --tag rppo_hg04_s0.25_sig0_dp1_s42
#
# Run 05: rppo_hg05_s0.5_sig0_dp1_s42    — node 102, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/05-s0.5_sig0.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:1 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg05_s0.5_sig0_dp1_s42 \
#   --tag rppo_hg05_s0.5_sig0_dp1_s42
#
# Run 06: rppo_hg06_s0.1_sig0.2_dp1_s42  — node 103, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/06-s0.1_sig0.2.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:0 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg06_s0.1_sig0.2_dp1_s42 \
#   --tag rppo_hg06_s0.1_sig0.2_dp1_s42
#
# Run 07: rppo_hg07_s0.25_sig0.2_dp1_s42 — node 103, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/07-s0.25_sig0.2.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:1 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg07_s0.25_sig0.2_dp1_s42 \
#   --tag rppo_hg07_s0.25_sig0.2_dp1_s42
#
# Run 08: rppo_hg08_s0.5_sig0.2_dp1_s42  — node 110, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/08-s0.5_sig0.2.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:0 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg08_s0.5_sig0.2_dp1_s42 \
#   --tag rppo_hg08_s0.5_sig0.2_dp1_s42
#
# Run 09: rppo_hg09_s0.1_sig0.4_dp1_s42  — node 110, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/09-s0.1_sig0.4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:1 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg09_s0.1_sig0.4_dp1_s42 \
#   --tag rppo_hg09_s0.1_sig0.4_dp1_s42
#
# Run 10: rppo_hg10_s0.5_sig0.4_dp1_s42  — node 106, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/10-s0.5_sig0.4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:1 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg10_s0.5_sig0.4_dp1_s42 \
#   --tag rppo_hg10_s0.5_sig0.4_dp1_s42
