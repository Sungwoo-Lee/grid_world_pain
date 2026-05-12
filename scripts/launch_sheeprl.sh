#!/bin/bash
set -euo pipefail
#
# launch_sheeprl.sh
# -----------------
# Reusable workload for sheeprl DreamerV3 training launches. Launched via
# `run_command.py` (which no longer cds or activates conda — both done here).
#
# Usage:
#   bash scripts/launch_sheeprl.sh <config-yaml> <gpu-index> <env-id-tag> [total-steps]
#
# Args:
#   <config-yaml>     Path (relative to project root or absolute) to the
#                     env YAML for our gridworld task (e.g.
#                     configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml).
#   <gpu-index>       CUDA index for torch (0..N-1).
#   <env-id-tag>      Unique tag for the run (becomes part of WandB run name
#                     via sheeprl's exp_name template). E.g. "gwp_5x5_pred".
#   [total-steps]     Optional. Default 200_000.
#
# Examples:
#   bash scripts/launch_sheeprl.sh configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml 0 gwp_5x5_pred
#   bash scripts/launch_sheeprl.sh configs/experiment/hypervigilance/01-interoNocicept.yaml 1 gwp_10x10_intero 500000

cd /media/nas01/projects/Interoceptive-AI/grid_world_pain

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <config-yaml> <gpu-index> <env-id-tag> [total-steps]" >&2
    exit 1
fi

CONFIG="$(realpath "$1")"
GPU="$2"
TAG="$3"
STEPS="${4:-200_000}"

if [ ! -f "$CONFIG" ]; then
    echo "Error: config not found: $CONFIG" >&2
    exit 1
fi

export GWP_CONFIG_PATH="$CONFIG"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES="$GPU"

echo "Launch:  sheeprl DreamerV3 XS"
echo "  config: $CONFIG"
echo "  gpu:    cuda:$GPU"
echo "  tag:    $TAG"
echo "  steps:  $STEPS"
echo

exec /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python \
    tmp/sheeprl/sheeprl.py \
    exp=dreamer_v3_grid_world_pain \
    env.id="$TAG" \
    algo.total_steps="$STEPS"
