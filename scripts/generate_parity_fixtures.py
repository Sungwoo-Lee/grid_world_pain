"""Generate parity fixtures for CP1 unified animal entity refactor.

Run BEFORE the refactor (on the current codebase at commit 27f7a0a) to capture
reference obs/state for all 86 migrated configs.

Saves one .npz per config under tests/env/fixtures/parity/<slug>.npz.

Usage:
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/generate_parity_fixtures.py
"""
import sys
import os
import re
import glob
import traceback

import numpy as np
import jax
import jax.numpy as jnp

# Ensure project root on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset, jax_step

ACTIONS = [0, 1, 2, 3, 4] * 20  # 100 steps
SEED = 0

FIXTURE_DIR = os.path.join(_ROOT, "tests", "env", "fixtures", "parity")
os.makedirs(FIXTURE_DIR, exist_ok=True)

def config_slug(config_path: str) -> str:
    """Convert a config path to a filesystem-safe slug."""
    rel = os.path.relpath(config_path, _ROOT)
    slug = re.sub(r'[/\\]', '__', rel)
    slug = re.sub(r'\.yaml$', '', slug)
    slug = re.sub(r'[^A-Za-z0-9_.-]', '_', slug)
    return slug


def collect_configs():
    configs = []
    configs += sorted(glob.glob(os.path.join(_ROOT, "configs", "experiment", "**", "*.yaml"), recursive=True))
    configs += sorted(glob.glob(os.path.join(_ROOT, "configs", "continual", "**", "*.yaml"), recursive=True))
    configs += sorted(glob.glob(os.path.join(_ROOT, "configs", "verification", "**", "*.yaml"), recursive=True))
    env_default = os.path.join(_ROOT, "configs", "environment", "default.yaml")
    if env_default not in configs:
        configs.append(env_default)
    return configs


def run_episode(params, key):
    """Reset + 100 steps. Returns (obs_list, state_list, info_list)."""
    state = jax_reset(params, key)
    obs_list = []
    state_list = [state]
    info_list = []
    for action in ACTIONS:
        state, reward, done, info = jax_step(state, action, params)
        obs_list.append(jnp.zeros(1))  # placeholder — we record state fields
        state_list.append(state)
        info_list.append(info)
    return state_list, info_list


def extract_state_fields(state) -> dict:
    """Pull all relevant state fields as numpy arrays."""
    fields = {}
    # Agent
    fields["agent_pos"] = np.array(state.agent_pos)
    fields["current_step"] = np.array(state.current_step)
    # Resources
    fields["res_pos"] = np.array(state.res_pos)
    fields["res_active"] = np.array(state.res_active)
    # Animals (unified — CP6 updated from legacy pred_*/neutral_* fields)
    fields["animal_pos"] = np.array(state.animal_pos)
    fields["animal_state"] = np.array(state.animal_state)
    fields["animal_stamina"] = np.array(state.animal_stamina)
    fields["animal_move_timer"] = np.array(state.animal_move_timer)
    fields["animal_attack_timer"] = np.array(state.animal_attack_timer)
    fields["animal_property_sampled"] = np.array(state.animal_property_sampled)
    # Obstacles
    fields["obs_pos"] = np.array(state.obs_pos)
    fields["obs_property_sampled"] = np.array(state.obs_property_sampled)
    # Body
    fields["satiation"] = np.array(state.satiation)
    fields["nutrition"] = np.array(state.nutrition)
    fields["injury_level"] = np.array(state.injury_level)
    fields["terminated"] = np.array(state.terminated)
    return fields


def extract_info_fields(info: dict) -> dict:
    """Pull relevant info dict entries as numpy arrays."""
    out = {}
    for key in ("hit_predator", "hit_neutral", "damage_predator", "damage_obstacle",
                "damage_hiding_predator", "dist_to_pred", "dist_to_neutral",
                "dist_per_predator", "dist_per_neutral", "agent_in_bush"):
        if key in info:
            out[key] = np.array(info[key])
    return out


def generate_fixture(config_path: str):
    slug = config_slug(config_path)
    out_path = os.path.join(FIXTURE_DIR, slug + ".npz")

    try:
        with open(config_path) as f:
            import yaml
            config_dict = yaml.safe_load(f)
        config = Config(config_dict)
        params = load_env_params(config)
    except Exception as e:
        print(f"  SKIP (load error): {config_path}\n    {e}")
        return False, "load_error"

    try:
        key = jax.random.PRNGKey(SEED)
        state_list, info_list = run_episode(params, key)
    except Exception as e:
        print(f"  SKIP (run error): {config_path}\n    {traceback.format_exc()}")
        return False, "run_error"

    # Flatten all steps into one npz
    npz_data = {}
    # Store reset state (step 0) and each post-step state (steps 1..100)
    for step_i, state in enumerate(state_list):
        step_fields = extract_state_fields(state)
        for fname, farr in step_fields.items():
            npz_data[f"step{step_i:03d}_{fname}"] = farr

    # Store info for each of the 100 steps (info from step 1..100)
    for step_i, info in enumerate(info_list):
        info_fields = extract_info_fields(info)
        for fname, farr in info_fields.items():
            npz_data[f"info{step_i:03d}_{fname}"] = farr

    # Also store the number of predators and neutrals for reconstruction
    npz_data["num_pred"] = np.array(len(params.predator_indices))
    npz_data["num_neutral"] = np.array(len(params.neutral_indices))

    np.savez_compressed(out_path, **npz_data)
    return True, "ok"


def main():
    configs = collect_configs()
    print(f"Found {len(configs)} configs to process")

    ok_count = 0
    skip_count = 0
    for i, cfg in enumerate(configs):
        rel = os.path.relpath(cfg, _ROOT)
        print(f"[{i+1}/{len(configs)}] {rel}", end=" ... ", flush=True)
        success, reason = generate_fixture(cfg)
        if success:
            print("OK")
            ok_count += 1
        else:
            print(f"SKIPPED ({reason})")
            skip_count += 1

    print(f"\nDone: {ok_count} fixtures saved, {skip_count} skipped")
    print(f"Fixture dir: {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
