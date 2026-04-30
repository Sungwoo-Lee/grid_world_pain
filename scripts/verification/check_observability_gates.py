"""Verifies injury_observable / nutrition_observable correctly include / exclude
slots in get_observation and get_observation_breakdown."""
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


def main():
    expectations = {
        "S1": {"injury": False, "nutrition": False, "must_have": [],
               "must_not_have": ["Injury", "Nutrition"]},
        "S2": {"injury": True,  "nutrition": False, "must_have": ["Injury"],
               "must_not_have": ["Nutrition"]},
        "S3": {"injury": False, "nutrition": True,  "must_have": ["Nutrition"],
               "must_not_have": ["Injury"]},
        "S4": {"injury": True,  "nutrition": True,  "must_have": ["Injury", "Nutrition"],
               "must_not_have": []},
    }

    rows = []
    for tag, exp in expectations.items():
        cfg_path = f"configs/verification/observability_gates_{tag}.yaml"
        with open(cfg_path) as f:
            cfg = Config(yaml.safe_load(f))
        params = load_env_params(cfg)
        state = jax_reset(params, jax.random.PRNGKey(0))
        
        # Force known interoceptive values to read them back via the slot.
        state = state.replace(injury_level=jnp.float32(40.0),
                              nutrition=jnp.float32(70.0))
        
        # We also need to disable noise to get deterministic reads
        obs = get_observation(state, params, apply_noise=False)
        bk = get_observation_breakdown(params)

        print(f"Checking {tag}...")
        for k in exp["must_have"]:     
            assert k in bk, f"{tag}: {k} missing from breakdown {list(bk.keys())}"
        for k in exp["must_not_have"]: 
            assert k not in bk, f"{tag}: {k} should NOT be in breakdown {list(bk.keys())}"
        
        assert obs.shape[0] == sum(bk.values()), f"{tag}: shape mismatch {obs.shape[0]} vs sum(bk)={sum(bk.values())}"

        # Numerical read for present slots.
        idx = 0
        slot_values = {}
        for name, dim in bk.items():
            if name in ("Injury", "Nutrition"):
                slot_values[name] = float(obs[idx])
            idx += dim
            
        if "Injury" in bk:
            val = slot_values["Injury"]
            expected = 40.0 / params.max_injury
            assert abs(val - expected) < 1e-6, f"{tag}: Injury value mismatch. Got {val}, expected {expected}"
        if "Nutrition" in bk:
            val = slot_values["Nutrition"]
            expected = 70.0 / params.max_nutrition
            assert abs(val - expected) < 1e-6, f"{tag}: Nutrition value mismatch. Got {val}, expected {expected}"

        rows.append({"scenario": tag, "obs_dim": int(obs.shape[0]),
                     "breakdown": {k: int(v) for k, v in bk.items()},
                     "slot_values": slot_values})

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("tmp", exist_ok=True)
    out = f"tmp/{ts}_observability_gates.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print("PASS — all 4 scenarios match expectations")
    for r in rows:
        print(f"  {r['scenario']}: dim={r['obs_dim']}  keys={list(r['breakdown'])}")
    print(f"Log: {out}")


if __name__ == "__main__":
    main()
