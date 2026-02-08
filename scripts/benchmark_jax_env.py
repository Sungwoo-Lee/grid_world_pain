import jax
import jax.numpy as jnp
import time
import yaml
from src.environment.state import EnvParams, EnvState
from src.environment.wrapper import ParallelEnv, auto_reset_step

def run_benchmark(num_envs=1024, num_steps=100):
    # Setup dummy params (L04 Homeostatic)
    params = EnvParams(
        height=4, width=4, max_steps=200,
        res_type=jnp.zeros(0, dtype=jnp.int32),
        res_property=jnp.zeros((0, 5)),
        res_spawn_area=jnp.zeros((0, 4)),
        res_max_cons=jnp.zeros(0, dtype=jnp.int32),
        res_reg_delay=jnp.zeros(0, dtype=jnp.int32),
        res_damage=jnp.zeros(0),
        pred_property=jnp.array([[0,1,0,0,0]], dtype=jnp.float32),
        pred_move_int=jnp.array([3], dtype=jnp.int32),
        pred_damage=jnp.array([3.0]),
        pred_patrol=jnp.array([[0,0,4,4]]),
        pred_detect=jnp.array([10.0]),
        pred_max_stamina=jnp.array([500.0]),
        pred_recovery=jnp.array([1.0]),
        pred_hunt_thresh=jnp.array([0.0]),
        max_satiation=100.0, max_injury=20.0, food_gain=10.0, setpoint=50.0,
        injury_recovery=1.0, smoothing_duration=3, death_penalty=10.0,
        overeating_death=False, use_homeostatic_reward=True,
        with_satiation=False, with_injury=True,
        sensor_radius=10.0, sensor_decay=2.0, sensor_range=3
    )
    
    env = ParallelEnv(params)
    key = jax.random.PRNGKey(42)
    
    print(f"Benchmarking JAX ParallelEnv with {num_envs} environments and {num_steps} steps...")
    
    # Reset
    states, obs = env.reset(key, num_envs)
    
    # Jit the step function
    @jax.jit
    def step_batch(s, a):
        return env.step(s, a)
    
    # Warmup
    actions = jax.random.randint(key, (num_envs,), 0, 5)
    _ = step_batch(states, actions)
    jax.block_until_ready(_)
    
    start_time = time.time()
    
    curr_states = states
    total_steps = 0
    for _ in range(num_steps):
        actions = jax.random.randint(key, (num_envs,), 0, 5)
        curr_states, obs, rews, dones, infos = step_batch(curr_states, actions)
        total_steps += num_envs
        
    jax.block_until_ready(curr_states)
    end_time = time.time()
    
    duration = end_time - start_time
    sps = total_steps / duration
    
    print(f"Total Steps: {total_steps}")
    print(f"Duration:    {duration:.4f}s")
    print(f"Steps/sec:   {sps:.2f}")
    
    return sps

if __name__ == "__main__":
    print("--- 1 Env (Sequential Baseline) ---")
    run_benchmark(num_envs=1, num_steps=1000)
    
    print("\n--- 128 Envs ---")
    run_benchmark(num_envs=128, num_steps=100)
    
    print("\n--- 1024 Envs ---")
    run_benchmark(num_envs=1024, num_steps=100)
    
    print("\n--- 4096 Envs ---")
    run_benchmark(num_envs=4096, num_steps=100)
