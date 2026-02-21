#!/usr/bin/env python3
import jax
import jax.numpy as jnp
from flax import nnx
import time
import numpy as np

from src.models.dreamer_v3_trainer import DreamerTrainer
from src.environment.core import jax_reset, EnvParams

def test_collection_performance():
    print("=" * 80)
    print("PHASE 3 PERFORMANCE TEST: JITTED Collection Loop")
    print("=" * 80)

    # Setup
    obs_dim = 33
    act_dim = 6 # 0-5 actions
    num_envs = 32
    num_steps = 64
    
    config = {
        'encoder_dim': 128,
        'encoder_fc_layers': [128, 128],
        'rssm_deter_dim': 512,
        'rssm_stoch_dim': 32,
        'rssm_classes': 32,
        'decoder_fc_layers': [128, 128],
        'reward_fc_layers': [128, 128],
        'continue_fc_layers': [128, 128],
        'actor_fc_layers': [256, 256],
        'critic_fc_layers': [256, 256],
        'model_lr': 1e-4,
        'actor_lr': 3e-5,
        'value_lr': 8e-5,
    }

    rngs = nnx.Rngs(0)
    # Environment params from config
    import os
    from src.utils.config import get_default_config, Config
    from src.environment.config_loader import load_env_params
    
    config_obj = get_default_config()
    # Load default environment config
    env_config_path = "configs/environment/environment.yaml"
    if os.path.exists(env_config_path):
        config_obj.merge(Config.load_yaml(env_config_path))
    
    params = load_env_params(config_obj)
    
    # Initialize env state (vmapped)
    key = jax.random.PRNGKey(42)
    reset_keys = jax.random.split(key, num_envs)
    env_state = jax.vmap(jax_reset, in_axes=(None, 0))(params, reset_keys)
    
    # Infer dimensions
    from src.environment.sensor import get_observation
    dummy_obs = get_observation(jax.tree_util.tree_map(lambda x: x[0], env_state), params)
    obs_dim = dummy_obs.shape[0]
    act_dim = params.action_dim
    
    print(f"Detected dimensions - Obs: {obs_dim}, Act: {act_dim}")
    
    trainer = DreamerTrainer(obs_dim, act_dim, config, rngs, modulation_config=None)
    
    print(f"\nBenchmarking {num_steps} steps across {num_envs} environments...")

    # 1. Baseline: Python loop (simplified simulation of train.py)
    print("\n[Baseline] Running Python loop...")
    from src.environment. sensor import get_observation
    from src.environment.core import jax_step
    
    start_time = time.time()
    current_env_state = env_state
    current_dreamer_state = trainer.agent.wm.rssm.initial(num_envs)
    current_dreamer_state['prev_action'] = jnp.zeros((num_envs, act_dim))
    current_dreamer_state['is_first'] = jnp.ones((num_envs, 1))
    current_key = key
    
    for t in range(num_steps):
        obs = jax.vmap(get_observation, in_axes=(0, None))(current_env_state, params)
        current_key, act_key = jax.random.split(current_key)
        action_idx, current_dreamer_state = trainer.get_action(obs, current_dreamer_state, eval_mode=False, rng=act_key)
        
        # In actual train.py, it calls jax_step vmapped
        current_env_state, reward, done, info = jax.vmap(jax_step, in_axes=(0, 0, None))(current_env_state, action_idx.astype(jnp.int32), params)
        # Note: Baseline is MISSING auto-reset, but we let it run for timing
        
    baseline_time = time.time() - start_time
    print(f"  Baseline time: {baseline_time*1000:.2f}ms ({num_envs*num_steps/baseline_time:.1f} steps/sec)")

    # 2. Optimized: JITTED collect_sequence
    print("\n[Optimized] Running JITTED collect_sequence...")
    
    # Warmup
    _ = trainer.collect_sequence(env_state, params, num_steps, key)
    jax.block_until_ready(_)
    print("  ✓ JIT compilation complete")
    
    start_time = time.time()
    num_iterations = 10
    for _ in range(num_iterations):
        result = trainer.collect_sequence(env_state, params, num_steps, key)
        jax.block_until_ready(result)
    
    opt_time = (time.time() - start_time) / num_iterations
    print(f"  Optimized time: {opt_time*1000:.2f}ms ({num_envs*num_steps/opt_time:.1f} steps/sec)")
    
    speedup = baseline_time / opt_time
    print(f"\n{'─'*80}")
    print(f"SPEEDUP: {speedup:.1f}x")
    print(f"{'─'*80}")
    
    return speedup

if __name__ == "__main__":
    test_collection_performance()
