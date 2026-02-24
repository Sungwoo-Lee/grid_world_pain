#!/bin/bash

# recurrent_ppo
# neuromodulated_ppo
# dreamer_v3
# neuromodulated_dreamer_v3

/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --agent_config configs/models/recurrent_ppo.yaml \
  --num-envs 128 \
  --episodes 10000000 \
  --checkpoint-frequency 1000000  \
  --device cuda:0 \
  --tag "recurrent_ppo_128env_hierarchical_groupEnc"