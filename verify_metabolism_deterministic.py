import jax
import jax.numpy as jnp
from src.environment.config_loader import load_env_params
from src.environment.core import jax_step, jax_reset, calculate_drive
from src.utils.config import Config
import os
import pandas as pd
from tqdm import tqdm

def run_deterministic_metabolism():
    # 1. Load config
    config_path = "configs/test/metabolism_verification.yaml"
    config = Config.load_yaml(config_path)
    params = load_env_params(config)
    
    # 2. Define Action Sequence
    # 0: Up, 1: Right, 2: Down, 3: Left, 4: Rest, 5: Eat
    actions = []
    
    # Phase 1: Move to food
    actions += [1]
    
    # Phase 2: Wait until starving (stomach empty, body dropping)
    actions += [4] * 50
    
    # Phase 3: Eat once
    actions += [5]
    
    # Phase 4: Wait more
    actions += [4] * 50
    
    # Phase 5: Eat twice back-to-back
    actions += [5, 5]
    
    # Phase 6: Long autonomous decay
    actions += [4] * 100
    
    # 3. Setup simulation
    key = jax.random.PRNGKey(42)
    state = jax_reset(params, key)
    
    stats = []
    
    # Initial state
    stats.append({
        'step': 0,
        'satiation': float(state.satiation),
        'nutrition': float(state.nutrition),
        'injury': float(state.injury_level),
        'rest_streak': int(state.rest_streak),
        'pos_r': int(state.agent_pos[0]),
        'pos_c': int(state.agent_pos[1]),
        'drive': float(calculate_drive(state.satiation, state.injury_level, params)),
        'action': "Initial",
        'reward': 0.0
    })
    
    action_names = ["Up", "Right", "Down", "Left", "Rest", "Eat"]
    
    # 4. Step through sequence
    print(f"Executing metabolism plan ({len(actions)} steps)...")
    for i, act_idx in enumerate(tqdm(actions)):
        state, reward, done, info = jax_step(state, act_idx, params)
        
        stats.append({
            'step': i + 1,
            'satiation': float(state.satiation),
            'nutrition': float(state.nutrition),
            'injury': float(state.injury_level),
            'rest_streak': int(state.rest_streak),
            'pos_r': int(state.agent_pos[0]),
            'pos_c': int(state.agent_pos[1]),
            'drive': float(calculate_drive(state.satiation, state.injury_level, params)),
            'drive_hunger': float(info.get('drive_hunger', 0.0)),
            'drive_injury': float(info.get('drive_injury', 0.0)),
            'event_ate': bool(info.get('ate_food', False)),
            'event_damage': float(info.get('damage', 0.0)),
            'event_collided': bool(info.get('event_collided', False)),
            'event_rested': bool(info.get('rested', False)),
            'dist_to_food': float(jnp.min(jnp.where(jnp.logical_and(state.res_active, params.res_type == 0), jnp.linalg.norm(state.res_pos - state.agent_pos, axis=-1), 99.0))),
            'dist_to_pred': 99.0,
            'reward_homeostatic': float(info.get('reward_homeostatic', 0.0)),
            'reward_extrinsic': float(info.get('reward_extrinsic', 0.0)),
            'max_satiation': float(params.max_satiation),
            'max_injury': float(params.max_injury),
            'action': action_names[act_idx],
            'reward': float(reward),
            'metabolic_drain': float(info.get('metabolic_drain', 0.0))
        })
        if done: break

    # 5. Save and Plot
    os.makedirs("results/verification", exist_ok=True)
    df = pd.DataFrame(stats)
    csv_path = "results/verification/metabolism_deterministic_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"Stats saved to {csv_path}")
    
    # Run plotting script
    python_path = "/home/vncuser/miniconda3/envs/grid_world_pain/bin/python"
    os.system(f"{python_path} plot_physiology.py --csv {csv_path} --out results/verification/metabolism_deterministic_plot.png")

if __name__ == "__main__":
    run_deterministic_metabolism()
