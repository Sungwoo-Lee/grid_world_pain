
import numpy as np
import yaml
from src.environment.grid_world import GridWorld
from src.environment.body import InteroceptiveBody
from src.utils.config import Config

def test_l3_passing_through():
    # Load defaults
    with open('configs/environment/environment.yaml', 'r') as f:
        env_defaults = yaml.safe_load(f)
    with open('configs/ablation/survival/03_predator_intro.yaml', 'r') as f:
        l3_config = yaml.safe_load(f)
        
    config = Config(env_defaults)
    config.merge(l3_config)
    
    # Initialize Env
    # Force predator to move EVERY step for this test
    env = GridWorld(
        height=config.get('environment.height'),
        width=config.get('environment.width'),
        start=(0,0),
        resource_pos=(-1,-1),
        with_satiation=config.get('body.with_satiation'),
        max_steps=100,
        prob_switch_to_danger=0,
        min_danger_duration=0,
        damage_amount=0,
        prob_switch_to_food=0,
        min_food_duration=0,
        relocate_resource=False,
        relocation_steps=100,
        vector_size=5,
        food_property=[0]*5,
        danger_property=[0]*5,
        eat_action_enabled=False,
        rest_action_enabled=False,
        predator_enabled=True,
        predator_damage=100.0,
        predator_start_pos=(1,0), 
        predator_move_interval=1, # Move every step
        predator_random_start_pos=False,
        injury_smoothing_duration=3
    )
    
    # CASE: Predator moves onto agent, but agent moves away.
    # Predator at (1,0), Agent at (0,0).
    # Predator moves UP to (0,0).
    # Agent moves RIGHT to (0,1).
    
    env.agent_pos = (0,0)
    env.predator_pos = (1,0)
    
    print(f"Step Start: Agent at {env.agent_pos}, Predator at {env.predator_pos}")
    
    # Take step: Move Right (action 1)
    state, reward, done, info = env.step(1)
    
    print(f"Post Step: Agent at {env.agent_pos}, Predator at {env.predator_pos}")
    print(f"Info: {info}")

    if info['damage'] > 0:
        print("PASS: Damage detected even when moving away.")
    else:
        print("FAIL: No damage detected when moving away (Passing Through issue).")

if __name__ == "__main__":
    test_l3_passing_through()
