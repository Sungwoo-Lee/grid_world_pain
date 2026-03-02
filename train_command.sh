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
  --config configs/environment/default.yamls \
  --agent_config configs/models/dreamer_v3.yaml \
  --num-envs 64 \
  --episodes 100000000 \
  --checkpoint-frequency 1000000 \
  --device cuda:0 \
  --tag "dreamer_v3_128envs_64batch_128collect_replay1_hierarchical"