#!/bin/bash

# recurrent_ppo
# neuromodulated_ppo
# dreamer_v3
# neuromodulated_dreamer_v3

# configs/environment/default.yaml
# configs/experiment/ablation/homeostatic/08_location.yaml

/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/ablation/homeostatic/08_location.yaml \
  --agent_config configs/models/dreamer_v3.yaml \
  --num-envs 64 \
  --episodes 1000000 \
  --checkpoint-frequency 10000 \
  --device cuda:1 \
  --tag "08_location_dreamer_v3_64env_replayRatio1_collectInterval1"