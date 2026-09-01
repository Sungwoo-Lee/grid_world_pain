#!/bin/bash
# Launch one arm of the VISION-DEGRADATION LADDER sweep on a lab node.
#
#   bash scripts/lab/launch_ladder_arm.sh <arm> <cuda_index> <node_label>
#   e.g. bash scripts/lab/launch_ladder_arm.sh V3_blur10 2 n114
#
# Invoked remotely via run_command.py, which does NOT cd or activate an env --
# both are this script's job (same contract as launch_sensory_arm.sh).
#
# Config-owns-values: num_envs, seed (42) and checkpoint_frequency come from
# configs/train/default.yaml and are deliberately NOT passed here. This sweep is
# SINGLE SEED by design; comparisons are across arms, not across seeds.
set -euo pipefail
cd /media/nas01/projects/Interoceptive-AI/grid_world_pain

ARM="${1:?usage: launch_ladder_arm.sh <arm> <cuda_index> <node_label>}"
DEV="${2:?missing cuda index}"
NODE="${3:?missing node label}"

CONFIG="configs/environment/experiment/sensory_ladder/${ARM}.yaml"
[[ -f "$CONFIG" ]] || { echo "no such arm config: $CONFIG" >&2; exit 2; }

/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config "$CONFIG" \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --episodes 10000000 \
  --device "cuda:${DEV}" \
  --log-interval 50 \
  --wandb-group "sensory_ladder" \
  --tag "lad_${ARM}_${NODE}g${DEV}"
