#!/bin/bash
set -e

# recurrent_ppo
# neuromodulated_ppo
# dreamer_v3
# neuromodulated_dreamer_v3
  # --episodes 10000000 \
  # --checkpoint-frequency 1000000  \


python train.py \
  --agent_config configs/models/dreamer_v3.yaml \
  --num-envs 128 \
  --episodes 10000000 \
  --checkpoint-frequency 100000  \
  --device cuda:0 \
  --tag "dreamer_v3_128env_128batch"