import jax
import jax.numpy as jnp
from src.environment.core import jax_reset, jax_step
from src.environment.config_loader import load_env_params
from src.environment.state import EnvParams

def test_nociception():
    # 1. Setup params manually
    params = EnvParams(
        height=10,
        width=10,
        max_steps=100,
        grid_location_type=jnp.zeros((10, 10), dtype=jnp.int32),
        res_type=jnp.array([1]), # One Danger
        res_property=jnp.zeros((1, 5)),
        res_nociception=jnp.array([0.9]),
        res_spawn_area=jnp.array([[0, 0, 9, 9]]),
        res_max_cons=0,
        res_reg_delay=0,
        res_damage=jnp.array([1.0]),
        pred_property=jnp.zeros((0, 5)),
        pred_nociception=jnp.zeros(0),
        pred_move_int=jnp.zeros(0, dtype=jnp.int32),
        pred_damage=jnp.zeros(0),
        pred_patrol=jnp.zeros((0, 4)),
        pred_detect=jnp.zeros(0),
        pred_max_stamina=jnp.zeros(0),
        pred_recovery=jnp.zeros(0),
        pred_hunt_thresh=jnp.zeros(0),
        obs_blocking=jnp.array([True]),
        obs_damage=jnp.array([0.5]),
        obs_property=jnp.array([[0, 0, 0, 0, 0.7]]), # Rock with scent!
        obs_nociception=jnp.array([0.3]),
        obs_spawn_area=jnp.array([[5, 5, 5, 5]]),
        max_satiation=100.0,
        max_injury=100.0,
        food_gain=20.0,
        setpoint=100.0,
        start_satiation=100.0,
        injury_recovery=1.0,
        smoothing_duration=1,
        death_penalty=0.0,
        overeating_death=False,
        use_homeostatic_reward=False,
        with_satiation=True,
        with_injury=True,
        random_start_satiation=False,
        random_start_injury=False,
        rest_action_enabled=True,
        eat_action_enabled=True,
        sensor_radius=5.0,
        sensor_decay=2.0,
        sensor_range=1,
        visual_sensor_enabled=False,
        visual_sensor_range=0,
        local_view_size=5
    )
    key = jax.random.PRNGKey(0)
    state = jax_reset(params, key)
    
    # Force agent next to the rock at (5, 5)
    state = state._replace(agent_pos=jnp.array([5, 4]))
    
    print("Initial State: Agent at (5, 4), Rock at (5, 5)")
    
    # 2. Test Collision Damage & Nociception (Move into blocking rock)
    action = 1 # Right
    state, reward, done, info = jax_step(state, action, params)
    
    print(f"Step 1 (Bunt Rock): Agent Pos: {state.agent_pos}, last_collision_noc: {state.last_collision_noc}")
    print(f"  Damage applied: {info['damage']}")
    
    from src.environment.sensor import get_observation
    obs = get_observation(state, params)
    
    # Calculate nociception index
    chem_dim = int(params.res_property.shape[-1])
    noc_idx = chem_dim # It's the second component
    noc_val = obs[noc_idx]
    
    print(f"  Nociception Value: {noc_val:.2f} (Expected 0.3)")
    
    # Check chemical sensor (rock should be at distance 1)
    # 1.0 / (1**2 + 1e-10) approx 1.0. Multiplied by 0.7 = 0.7
    chem_val = obs[4] # Last channel of chemical
    print(f"  Chemical Value (Rock scent): {chem_val:.2f} (Expected 0.7)")
    
    assert state.last_collision_noc == 0.3
    assert info['damage'] == 0.5
    assert jnp.isclose(noc_val, 0.3)
    assert jnp.isclose(chem_val, 0.7, atol=0.01)
    
    # 3. Test Predator Nociception
    # Force a predator at agent's position
    # Update params to have 1 predator with intensity 0.9
    params = params._replace(
        pred_property=jnp.zeros((1, 5)),
        pred_nociception=jnp.array([0.9]),
        pred_damage=jnp.array([3.0])
    )
    state = state._replace(pred_pos=jnp.array([[5, 4]]))
    obs = get_observation(state, params)
    noc_val = obs[noc_idx]
    print(f"Step 2 (Predator Contact): Nociception: {noc_val:.2f} (Expected 0.9)")
    assert jnp.isclose(noc_val, 0.9)
    
    print("\nVerification Successful!")

if __name__ == "__main__":
    test_nociception()
