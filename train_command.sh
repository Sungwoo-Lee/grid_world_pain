#!/bin/bash
set -e

python train.py \
  --agent_config configs/models/neuromodulated_ppo.yaml \
  --num-envs 128 \
  --episodes 10000000 \
  --checkpoint-frequency 1000000  \
  --tag "rppoNMN_128env_gsize1_preAct_NoLoc_resource"