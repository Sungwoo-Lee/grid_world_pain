#!/bin/bash

# recurrent_ppo
# neuromodulated_ppo
# dreamer_v3
# neuromodulated_dreamer_v3

# configs/environment/default.yaml
# configs/experiment/ablation/homeostatic/08_location.yaml

/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/default.yaml \
  --agent_config configs/models/dreamer_v3.yaml \
  --num-envs 1 \
  --episodes 10000000 \
  --checkpoint-frequency 100000 \
  --device cuda:1 \
  --tag "dreamer_v3_1env_replayRatio1_collectInterval128"