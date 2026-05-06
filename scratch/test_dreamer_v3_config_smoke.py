"""Smoke test: every mandatory agent.* key consumed by DreamerTrainer.__init__
is present in the DreamerV3 YAML configs. No GPU, no env, no model build.

Run:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python scratch/test_dreamer_v3_config_smoke.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import Config

# Hard-coded list of mandatory agent.* keys read in DreamerTrainer.__init__
# (src/models/dreamer_v3_trainer.py:64–80, plus sampling_mode at line 742).
# Update when that constructor changes.
MANDATORY_KEYS = [
    'agent.encoder_dim',
    'agent.encoder_fc_layers',
    'agent.rssm_deter_dim',
    'agent.rssm_stoch_dim',
    'agent.rssm_classes',
    'agent.decoder_fc_layers',
    'agent.reward_fc_layers',
    'agent.continue_fc_layers',
    'agent.actor_fc_layers',
    'agent.critic_fc_layers',
    'agent.use_layer_norm',
    'agent.encoding_mode',
    'agent.model_lr',
    'agent.actor_lr',
    'agent.value_lr',
    'agent.sampling_mode',
]

CONFIGS = [
    'configs/models/dreamer_v3.yaml',
    'configs/models/neuromodulated_dreamer_v3.yaml',
]

any_failed = False
for rel_path in CONFIGS:
    config_path = os.path.join(PROJECT_ROOT, rel_path)
    cfg = Config.load_yaml(config_path)
    missing = []
    for key in MANDATORY_KEYS:
        try:
            cfg.get_mandatory(key)
        except ValueError:
            missing.append(key)

    if missing:
        any_failed = True
        print(f'FAIL: {rel_path} is missing mandatory keys:')
        for k in missing:
            print(f'  - {k}')
    else:
        print(f'OK: all {len(MANDATORY_KEYS)} mandatory keys present in {rel_path}')

sys.exit(1 if any_failed else 0)
