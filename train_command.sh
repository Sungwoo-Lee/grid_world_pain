#!/bin/bash

# recurrent_ppo
# neuromodulated_ppo
# dreamer_v3
# neuromodulated_dreamer_v3

# configs/environment/default.yaml
# configs/experiment/ablation/homeostatic/08_location.yaml

# rppo
# dreamer_v3_1env_replayRatio1_collectInterval128

/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/ablation/homeostatic/08_location.yaml \
  --agent_config configs/models/recurrent_ppo.yaml \
  --num-envs 1 \
  --episodes 10000 \
  --checkpoint-frequency 1000 \
  --device cuda:1 \
  --tag "08_location_rppo_tanh_mc"