"""Verifies that neutral-animal and predator olfactory observations are equal
when their (position, property) inputs are equal. Black-box; reads two YAMLs."""
import json
import os
import sys
import datetime as dt
import jax
import jax.numpy as jnp
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.core import jax_reset
from src.environment.sensor import get_observation, get_observation_breakdown


def load(path):
    with open(path) as f:
        return Config(yaml.safe_load(f))


def olf_slice(breakdown):
    idx = 0
    for name, dim in breakdown.items():
        if name == "Olfaction":
            return slice(idx, idx + dim)
        idx += dim
    raise KeyError("Olfaction not in breakdown")


def run_one(cfg_path, seed, agent_pos_override=None, property_override=None, is_predator=False):
    cfg = load(cfg_path)
    data = cfg.to_dict()
    if property_override is not None:
        if is_predator:
            data['environment']['predators'][0]['properties'] = property_override
        else:
            data['environment']['neutral_animals'][0]['properties'] = property_override
    
    params = load_env_params(cfg)
    if agent_pos_override is not None:
        params = params.replace(start_pos=jnp.array(agent_pos_override))
    
    state = jax_reset(params, jax.random.PRNGKey(seed))
    obs = get_observation(state, params, apply_noise=False)
    bk = get_observation_breakdown(params)
    return params, state, obs, bk


def main():
    seed = 0
    neutral_cfg = "configs/verification/olfaction_parity_neutral.yaml"
    predator_cfg = "configs/verification/olfaction_parity_predator.yaml"
    
    properties = [
        ("P1", [0.0, 0.5, 0.7, 0.0, 0.0]),
        ("P2", [1.0, 0.0, 0.0, 0.0, 0.0]),
        ("P3", [0.3, 0.3, 0.3, 0.3, 0.3])
    ]
    
    # Entity is at [0, 0]. Agent is at [2, 2] by default.
    # Distances (Manhattan):
    # d1: agent at [0, 1] -> dist = 1
    # d2: agent at [0, 2] -> dist = 2
    # d4: agent at [2, 2] -> dist = 4
    distances = [
        ("d1", [0, 1]),
        ("d2", [0, 2]),
        ("d4", [2, 2])
    ]

    rows = []
    for prop_name, prop in properties:
        for dist_name, agent_pos in distances:
            # Run neutral
            params_a, state_a, obs_a, bk_a = run_one(neutral_cfg, seed, agent_pos, prop, is_predator=False)
            # Run predator
            params_b, state_b, obs_b, bk_b = run_one(predator_cfg, seed, agent_pos, prop, is_predator=True)
            
            # Sanity checks
            if not (state_a.neutral_pos[0] == state_b.pred_pos[0]).all():
                print(f"Position mismatch: neutral={state_a.neutral_pos[0]} pred={state_b.pred_pos[0]}")
            if not (state_a.neutral_pos[0] == jnp.array([0, 0])).all():
                print(f"Entity not at [0,0]: {state_a.neutral_pos[0]}")
                print(f"Neutral spawn area: {params_a.neutral_spawn_area}")
                assert False, f"Entity not at [0,0]: {state_a.neutral_pos[0]}"
            
            s_a = olf_slice(bk_a)
            s_b = olf_slice(bk_b)
            assert s_a == s_b, f"Slice mismatch: {s_a} vs {s_b}"
            
            olf_a = obs_a[s_a]
            olf_b = obs_b[s_b]
            delta = jnp.max(jnp.abs(olf_a - olf_b))
            
            rows.append({
                "prop": prop_name,
                "dist": dist_name,
                "agent_pos": agent_pos,
                "obs_neutral": olf_a.tolist(),
                "obs_predator": olf_b.tolist(),
                "max_abs_delta": float(delta)
            })
            
            if delta >= 1e-6:
                print(f"FAIL: {prop_name} {dist_name} delta={delta:.2e}")
                print(f"  Neutral:  {olf_a}")
                print(f"  Predator: {olf_b}")
                # Do not assert yet, collect all failures
            
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = f"tmp/{ts}_olfaction_parity.json"
    os.makedirs("tmp", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    
    max_delta = max(r['max_abs_delta'] for r in rows)
    if max_delta < 1e-6:
        print(f"PASS — {len(rows)} cases, max delta {max_delta:.2e}")
    else:
        print(f"FAIL — {len(rows)} cases, max delta {max_delta:.2e}")
        sys.exit(1)
        
    print(f"Log: {out}")


if __name__ == "__main__":
    main()
