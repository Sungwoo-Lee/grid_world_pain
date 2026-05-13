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
#   --agent_config <path>     Path to agent/model config YAML (e.g., configs/models/dreamer_v3.yaml)
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
#   --wandb-group     experiment family — top dir under configs/experiment/
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

# JAX matched-config SPS measurement Run B — num_envs=16 (JAX-architectural parallelism)
# Node 113, cuda:0. Design doc: docs/experiments/active/sheeprl_bridge/JAX_SHEEPRL_MATCHED_SPS_DESIGN.md
# Measures JAX DreamerV3 env-SPS under sheeprl-XS matched recipe (replay_ratio=1.0) with JAX vmap parallelism.
# --episodes 0 required: env YAML pins episodes:100 which overrides --total-timesteps without this flag.
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/dreamer_curriculum/01_food_only.yaml \
  --agent_config configs/models/dreamer_v3_sheeprl_matched.yaml \
  --num-envs 16 \
  --episodes 0 \
  --total-timesteps 3000000 \
  --seed 0 \
  --device cuda:0 \
  --log-interval 50 \
  --wandb-group sheeprl_bridge \
  --wandb-job-type sps_measurement \
  --wandb-name jax_sheeprl_matched_n16_s0 \
  --tag jax_sheeprl_matched_n16_s0
