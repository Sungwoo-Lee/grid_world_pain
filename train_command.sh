#!/bin/bash
set -e

# recurrent_ppo
# neuromodulated_ppo
# dreamer_v3
# neuromodulated_dreamer_v3

python train.py \
  --agent_config configs/models/neuromodulated_ppo.yaml \
  --num-envs 128 \
  --episodes 10000000 \
  --checkpoint-frequency 1000000  \
  --device cuda:1 \
  --tag "rppoNMN_128env"