#!/bin/bash
set -euo pipefail
#
# launch_sheeprl.sh
# -----------------
# Reusable workload for sheeprl DreamerV3 training launches. Launched via
# `run_command.py` (which no longer cds or activates conda — both done here).
#
# Usage:
#   bash scripts/launch_sheeprl.sh <config-yaml> <gpu-index> <env-id-tag> [total-steps] [num-envs] [size] [hydra-overrides...]
#
# Args:
#   <config-yaml>        Path (relative to project root or absolute) to the
#                        env YAML for our gridworld task (e.g.
#                        configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml).
#   <gpu-index>          CUDA index for torch (0..N-1).
#   <env-id-tag>         Unique tag for the run (becomes part of WandB run name
#                        via sheeprl's exp_name template). E.g. "gwp_5x5_pred".
#   [total-steps]        Optional. Default 200_000.
#   [num-envs]           Optional. Default 1.
#   [size]               Optional. DreamerV3 size preset: XS, S, M, L, XL.
#                        Default XS. Passed as Hydra override `algo=dreamer_v3_<size>`.
#                        XS: 256 units / 1 MLP layer (baseline, low VRAM).
#                        S:  512 units / 2 MLP layers (2× capacity, ~4× VRAM).
#   [hydra-overrides...] Optional. Any extra Hydra key=value overrides passed
#                        through verbatim to pytorch_agents.run_dreamer_v3.
#                        E.g. env.use_jax_vector_env=true
#
# Examples:
#   bash scripts/launch_sheeprl.sh configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml 0 gwp_5x5_pred
#   bash scripts/launch_sheeprl.sh configs/experiment/hypervigilance/01-interoNocicept.yaml 1 gwp_10x10_intero 500000
#   bash scripts/launch_sheeprl.sh configs/experiment/hypervigilance/01-interoNocicept.yaml 3 sps_n4 5000 4
#   bash scripts/launch_sheeprl.sh configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml 0 gwp_5x5_pred_S 200_000 1 S
#   bash scripts/launch_sheeprl.sh configs/experiment/dreamer_curriculum/01_food_only.yaml 3 jaxvec_smoke 1000 4 XS env.use_jax_vector_env=true

cd /media/nas01/projects/Interoceptive-AI/grid_world_pain

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <config-yaml> <gpu-index> <env-id-tag> [total-steps] [num-envs] [size] [hydra-overrides...]" >&2
    exit 1
fi

CONFIG="$(realpath "$1")"
GPU="$2"
TAG="$3"
STEPS="${4:-200_000}"
NUM_ENVS="${5:-1}"
SIZE="${6:-XS}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: config not found: $CONFIG" >&2
    exit 1
fi

export GWP_CONFIG_PATH="$CONFIG"
# v2 spike: JAX shares the same GPU as torch (bound to 20% of GPU memory).
# Preallocate=false is mandatory — otherwise JAX would grab 90% at startup
# and OOM-fight torch's parameter slabs.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2
export CUDA_VISIBLE_DEVICES="$GPU"

echo "Launch:  sheeprl DreamerV3 $SIZE"
echo "  config:   $CONFIG"
echo "  gpu:      cuda:$GPU"
echo "  tag:      $TAG"
echo "  steps:    $STEPS"
echo "  num_envs: $NUM_ENVS"
echo "  size:     $SIZE  (algo=dreamer_v3_${SIZE})"
echo

# Tell sheeprl's Hydra search-path plugin where to find our env/exp/logger
# configs (Hydra resolves `pkg://pytorch_agents.configs` via importlib).
export SHEEPRL_SEARCH_PATH="pkg://pytorch_agents.configs"

exec /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python \
    -m pytorch_agents.run_dreamer_v3 \
    exp=dreamer_v3_grid_world_pain \
    "algo=dreamer_v3_${SIZE}" \
    env.id="$TAG" \
    algo.total_steps="$STEPS" \
    env.num_envs="$NUM_ENVS" \
    "${@:7}"
