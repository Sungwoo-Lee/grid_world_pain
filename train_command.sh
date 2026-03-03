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
  --config configs/environment/default.yaml \
  --agent_config configs/models/dreamer_v3.yaml \
  --num-envs 1 \
  --episodes 10000000 \
  --checkpoint-frequency 1000000 \
  --device cuda:0 \
  --tag "dreamer_v3_1envs_16batch_128collect_replay1_hierarchical_1e6buffer_noBodyEncoding"