import jax
import jax.numpy as jnp
from src.environment.config_loader import load_env_params
from src.environment.core import jax_step, jax_reset, calculate_drive
from src.environment.sensor import get_observation
from src.utils.evaluation_core import evaluate_jax_checkpoint
from src.utils.config import get_default_config
import os
import pandas as pd
from tqdm import tqdm

def run_deterministic_recovery():
    # 1. Load config
    from src.utils.config import Config
    config_path = "configs/test/recovery_verification.yaml"
    config = Config.load_yaml(config_path)
    params = load_env_params(config)
    
    # 2. Define Action Sequence
    # 0: Up, 1: Right, 2: Down, 3: Left, 4: Rest, 5: Eat
    actions = [
        1,       # Move onto danger at (1,2)
        4, 4, 4, 4, 4, # Stay on danger and REST (takes massive damage)
        3,       # Move back to (1,1)
    ]
    # Add 150 steps of Rest to see full spool-up
    actions += [4] * 150
    
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
    print(f"Executing deterministic plan ({len(actions)} steps)...")
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
            'action': action_names[act_idx],
            'reward': float(reward),
            'max_satiation': float(params.max_satiation),
            'max_injury': float(params.max_injury),
            'reward_homeostatic': float(info.get('reward_homeostatic', 0.0)),
            'reward_extrinsic': float(info.get('reward_extrinsic', 0.0)),
            'sense_nociception': 0.0 # Placeholder
        })
        if done: break

    # 5. Save and Plot
    os.makedirs("results/verification", exist_ok=True)
    df = pd.DataFrame(stats)
    csv_path = "results/verification/recovery_deterministic_stats.csv"
    df.to_csv(csv_path, index=False)
    print(f"Stats saved to {csv_path}")
    
    python_path = "/home/vncuser/miniconda3/envs/grid_world_pain/bin/python"
    os.system(f"{python_path} plot_physiology.py --csv {csv_path} --out results/verification/recovery_deterministic_plot.png")

if __name__ == "__main__":
    run_deterministic_recovery()
