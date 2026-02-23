#!/bin/bash

# recurrent_ppo
# neuromodulated_ppo
# dreamer_v3
# neuromodulated_dreamer_v3

/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --agent_config configs/models/dreamer_v3.yaml \
  --num-envs 64 \
  --episodes 10000000 \
  --checkpoint-frequency 1000000  \
  --device cuda:1 \
  --tag "dreamer_v3_64env_bush_origparm_smallNet"