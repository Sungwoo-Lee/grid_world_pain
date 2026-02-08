import jax
import jax.numpy as jnp
import numpy as np
import yaml
import os
from legacy.src.environment.grid_world import GridWorld, ResourceEntity
from legacy.src.environment.body import InteroceptiveBody
from src.environment.state import EnvState, EnvParams
from src.environment.core import jax_step, jax_reset
from src.environment.sensor import get_observation
from src.environment.sensor import ResourceSensor, CollisionSensor, LocationSensor, InteroceptiveNociceptor

def compare_states(np_env, np_body, jax_state, jax_params, step_num):
    """Compares the state of the NumPy and JAX environments."""
    print(f"\n--- Comparing Step {step_num} ---")
    
    # 1. Agent Position
    np_pos = np_env.agent_pos
    jax_pos = jax_state.agent_pos
    match_pos = (np_pos[0] == jax_pos[0]) and (np_pos[1] == jax_pos[1])
    print(f"Agent Pos: NP={np_pos}, JAX={jax_pos} | Match: {match_pos}")
    
    # 2. Body State
    np_sat = np_body.satiation
    jax_sat = float(jax_state.satiation)
    match_sat = abs(np_sat - jax_sat) < 1e-5
    print(f"Satiation: NP={np_sat:.1f}, JAX={jax_sat:.1f} | Match: {match_sat}")
    
    np_inj = np_body.injury_level
    jax_inj = float(jax_state.injury_level)
    match_inj = abs(np_inj - jax_inj) < 1e-5
    print(f"Injury:    NP={np_inj:.2f}, JAX={jax_inj:.2f} | Match: {match_inj}")
    
    return match_pos and match_sat and match_inj

def run_sensor_parity_test():
    """Test parity of sensors."""
    print("Running Sensor Parity Test...")
    
    # NP Env Setup
    np_env = GridWorld(
        height=4, width=4, start=(0,0), resources=[], predators=[],
        with_satiation=True, max_steps=100, eat_action_enabled=True, rest_action_enabled=True,
        predator_enabled=False, injury_smoothing_duration=3, vector_size=5
    )
    # Add resources
    f_res = ResourceEntity(0, {'name': 'f', 'type': 'food', 'properties': [1,0,0,0,0], 'spawn_area': [[1,1], [1,1]], 'max_consumption': -1, 'regeneration_delay': 0, 'damage': 0})
    f_res.pos = (1, 1)
    d_res = ResourceEntity(1, {'name': 'd', 'type': 'danger', 'properties': [0,1,0,0,0], 'spawn_area': [[2,2], [2,2]], 'max_consumption': -1, 'regeneration_delay': 0, 'damage': 5.0})
    d_res.pos = (2, 2)
    np_env.resources = [f_res, d_res]
    
    np_body = InteroceptiveBody(
        max_satiation=100, start_satiation=50, overeating_death=False, random_start_satiation=False,
        food_satiation_gain=10, use_homeostatic_reward=True, satiation_setpoint=50, death_penalty=10.0,
        with_injury=True, max_injury=20, start_injury=0, injury_recovery=1, random_start_injury=False,
        injury_smoothing_duration=3, with_satiation=True
    )
    
    # JAX Env Setup
    jax_params = EnvParams(
        height=4, width=4, max_steps=100,
        res_type=jnp.array([0, 1]), res_property=jnp.array([[1,0,0,0,0], [0,1,0,0,0]], dtype=jnp.float32),
        res_spawn_area=jnp.array([[1,1,1,1], [2,2,2,2]]),
        res_max_cons=jnp.array([-1, -1]), res_reg_delay=jnp.array([0, 0]), res_damage=jnp.array([0.0, 5.0]),
        pred_property=jnp.zeros((0, 5)), pred_move_int=jnp.zeros(0, dtype=jnp.int32),
        pred_damage=jnp.zeros(0), pred_patrol=jnp.zeros((0,4)), pred_detect=jnp.zeros(0),
        pred_max_stamina=jnp.zeros(0), pred_recovery=jnp.zeros(0), pred_hunt_thresh=jnp.zeros(0),
        max_satiation=100.0, max_injury=20.0, food_gain=10.0, setpoint=50.0,
        injury_recovery=1.0, smoothing_duration=3, death_penalty=10.0,
        overeating_death=False, use_homeostatic_reward=True, with_satiation=True, with_injury=True,
        sensor_radius=10.0, sensor_decay=2.0, sensor_range=3
    )
    
    jax_state = EnvState(
        agent_pos=jnp.array([0, 0]), current_step=jnp.array(0),
        res_pos=jnp.array([[1, 1], [2, 2]]), res_active=jnp.ones(2, dtype=jnp.bool_),
        res_cons_count=jnp.zeros(2, dtype=jnp.int32), res_reg_timer=jnp.zeros(2, dtype=jnp.int32),
        pred_pos=jnp.zeros((0, 2)), pred_state=jnp.zeros(0, dtype=jnp.int32),
        pred_stamina=jnp.zeros(0), pred_move_timer=jnp.zeros(0, dtype=jnp.int32),
        satiation=jnp.array(50.0), injury_level=jnp.array(0.0),
        injury_buffer=jnp.zeros(3), terminated=jnp.array(False),
        key=jax.random.PRNGKey(0)
    )
    
    # NP Sensors
    res_sensor = ResourceSensor(radius=10.0, vector_size=5, decay_power=2.0)
    coll_sensor = CollisionSensor(sensor_range=3)
    loc_sensor = LocationSensor(output_size=2)
    inj_sensor = InteroceptiveNociceptor(output_size=1)
    
    # Path
    actions = [2, 1, 2, 1, 0, 3]
    
    for i, action in enumerate(actions):
        # NP Step
        np_pos, _, _, np_info = np_env.step(action)
        np_body_state, _, _ = np_body.step(np_info)
        
        # NP Observations
        obs_res = res_sensor.sense(np_pos, np_env.get_active_resources())
        obs_coll = coll_sensor.sense(np_pos, np_env.height, np_env.width)
        obs_loc = loc_sensor.sense(np_pos, np_env.height, np_env.width)
        obs_inj = np.array([np_body.satiation / np_body.max_satiation, np_body.injury_level / np_body.max_injury])
        np_obs = np.concatenate([obs_res, obs_coll, obs_loc, obs_inj])
        
        # JAX Step
        jax_state, _, _, _ = jax_step(jax_state, action, jax_params)
        jax_obs = get_observation(jax_state, jax_params)
        
        # Compare
        obs_match = np.allclose(np_obs, np.array(jax_obs), atol=1e-5)
        print(f"Step {i+1} Observation Match: {obs_match}")
        if not obs_match:
            print(f"  NP Obs:  {np_obs}")
            print(f"  JAX Obs: {np.array(jax_obs)}")
            # break

    print("\nSensor Verification Complete.")

if __name__ == "__main__":
    run_sensor_parity_test()
