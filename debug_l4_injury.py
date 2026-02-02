
import numpy as np
import yaml
from src.environment.grid_world import GridWorld
from src.environment.body import InteroceptiveBody
from src.utils.config import Config

def test_l4_homeostatic_injury():
    # Load defaults
    with open('configs/environment/environment.yaml', 'r') as f:
        env_defaults = yaml.safe_load(f)
    # Load L4 Homeostatic config
    with open('configs/ablation/homeostatic/04_nociception.yaml', 'r') as f:
        l4_config = yaml.safe_load(f)
        
    config = Config(env_defaults)
    config.merge(l4_config)
    
    print("Testing Level 04 Homeostatic Predator Injury...")
    print(f"with_injury: {config.get('body.with_injury')}")
    print(f"Max Injury: {config.get('body.max_injury')}")
    print(f"Predator Damage: {config.get('predator.damage')}")
    print(f"Injury Smoothing: {config.get('body.injury_smoothing_duration')}")

    # Initialize Env
    env = GridWorld(
        height=config.get('environment.height'),
        width=config.get('environment.width'),
        start=(0,0),
        resource_pos=(-1,-1),
        with_satiation=config.get('body.with_satiation'),
        max_steps=200,
        prob_switch_to_danger=0,
        min_danger_duration=0,
        damage_amount=0,
        prob_switch_to_food=0,
        min_food_duration=0,
        relocate_resource=False,
        relocation_steps=200,
        vector_size=5,
        food_property=[0]*5,
        danger_property=[0]*5,
        eat_action_enabled=False,
        rest_action_enabled=True,
        predator_enabled=True,
        predator_damage=config.get('predator.damage'),
        predator_start_pos=(1,1), 
        predator_move_interval=1000,
        injury_smoothing_duration=config.get('body.injury_smoothing_duration')
    )
    
    # Initialize Body
    body = InteroceptiveBody(
        max_satiation=config.get('body.max_satiation'),
        start_satiation=config.get('body.start_satiation'),
        overeating_death=config.get('body.overeating_death'),
        random_start_satiation=config.get('body.random_start_satiation'),
        food_satiation_gain=config.get('body.food_satiation_gain'),
        use_homeostatic_reward=config.get('body.use_homeostatic_reward'),
        satiation_setpoint=config.get('body.satiation_setpoint'),
        death_penalty=config.get('body.death_penalty'),
        with_injury=config.get('body.with_injury'),
        max_injury=config.get('body.max_injury'),
        start_injury=config.get('body.start_injury'),
        injury_recovery=config.get('body.injury_recovery'),
        random_start_injury=config.get('body.random_start_injury'),
        injury_smoothing_duration=config.get('body.injury_smoothing_duration'),
        with_satiation=config.get('body.with_satiation')
    )
    
    # Force agent onto predator
    env.agent_pos = (1,1)
    env.predator_pos = (1,1)
    
    print(f"Agent Pos: {env.agent_pos}, Predator Pos: {env.predator_pos}")
    
    # Step 1: Stay (Action 0 for this test)
    # If rest enabled, Stay is not explicitly defined in action_spec but 0-3 are movement.
    # Level 4 has rest_action_enabled: True. 
    # Actions: 0=Up, 1=Right, 2=Down, 3=Left, 4=Rest.
    # We move the agent back to (1,1) if it moves, but let's just use Rest (4) to see if it cancels.
    
    print("\n--- Step 1: Agent on Predator, Action: Rest (4) ---")
    state, reward, done, info = env.step(4)
    print(f"Env Info: {info}")
    
    body_state, body_reward, body_done = body.step(info)
    print(f"Body Injury Level: {body.injury_level:.2f}, Done: {body_done}, Reward: {body_reward:.2f}")
    
    print("\n--- Step 2: Agent stays on Predator (manually), Action: Move Up (0) ---")
    env.agent_pos = (1,1)
    state, reward, done, info = env.step(0)
    print(f"Env Info: {info}")
    body_state, body_reward, body_done = body.step(info)
    print(f"Body Injury Level: {body.injury_level:.2f}, Done: {body_done}, Reward: {body_reward:.2f}")

    print("\n--- Step 3: Agent stays on Predator (manually), Action: Move Up (0) ---")
    env.agent_pos = (1,1)
    state, reward, done, info = env.step(0)
    print(f"Env Info: {info}")
    body_state, body_reward, body_done = body.step(info)
    print(f"Body Injury Level: {body.injury_level:.2f}, Done: {body_done}, Reward: {body_reward:.2f}")

    print("\n--- Step 4: Agent moves away, Action: Move Up (0) ---")
    # Agent is at (0,1) now. Predator at (1,1).
    state, reward, done, info = env.step(0)
    print(f"Env Info: {info}")
    body_state, body_reward, body_done = body.step(info)
    print(f"Body Injury Level: {body.injury_level:.2f}, Done: {body_done}, Reward: {body_reward:.2f}")

if __name__ == "__main__":
    test_l4_homeostatic_injury()
