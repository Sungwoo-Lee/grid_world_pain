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
  --agent_config configs/models/recurrent_ppo.yaml \
  --num-envs 128 \
  --episodes 10000000 \
  --checkpoint-frequency 1000000 \
  --device cuda:0 \
  --tag "rppo_10X10_100injury_3predators_metabolicCost1"