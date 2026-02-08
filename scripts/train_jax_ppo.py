"""
JAX RecurrentPPO Training Script with ParallelEnv Integration.

This script demonstrates full training using the JAX-native RecurrentPPO
with the parallel environment wrapper for massive throughput.
"""
import jax
import jax.numpy as jnp
import optax
import yaml
from flax import nnx
from typing import NamedTuple

from src.environment.state import EnvParams
from src.environment.wrapper import ParallelEnv
from src.environment.config_loader import load_env_params
from src.models.recurrent_ppo_network import ActorCriticRNN
from src.models.recurrent_ppo_trainer import train_iteration

class PPOConfig(NamedTuple):
    num_steps: int
    num_epochs: int
    gamma: float
    gae_lambda: float
    clip_eps: float
    ent_coef: float
    vf_coef: float
    lr: float

def load_env_params_from_config(config_path: str) -> EnvParams:
    """Loads environment parameters from a YAML config file."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    env_cfg = cfg.get('environment', {})
    body_cfg = cfg.get('body', {})
    
    # Build minimal EnvParams - this needs to be expanded for full configs
    return EnvParams(
        height=env_cfg.get('height', 10),
        width=env_cfg.get('width', 10),
        max_steps=env_cfg.get('max_steps', 500),
        res_type=jnp.zeros(0, dtype=jnp.int32),
        res_property=jnp.zeros((0, 5)),
        res_spawn_area=jnp.zeros((0, 4)),
        res_max_cons=jnp.zeros(0, dtype=jnp.int32),
        res_reg_delay=jnp.zeros(0, dtype=jnp.int32),
        res_damage=jnp.zeros(0),
        pred_property=jnp.zeros((0, 5)),
        pred_move_int=jnp.zeros(0, dtype=jnp.int32),
        pred_damage=jnp.zeros(0),
        pred_patrol=jnp.zeros((0, 4)),
        pred_detect=jnp.zeros(0),
        pred_max_stamina=jnp.zeros(0),
        pred_recovery=jnp.zeros(0),
        pred_hunt_thresh=jnp.zeros(0),
        max_satiation=body_cfg.get('max_satiation', 100.0),
        max_injury=body_cfg.get('max_injury', 20.0),
        food_gain=body_cfg.get('food_gain', 10.0),
        setpoint=body_cfg.get('setpoint', 50.0),
        injury_recovery=body_cfg.get('injury_recovery', 1.0),
        smoothing_duration=body_cfg.get('smoothing_duration', 3),
        death_penalty=body_cfg.get('death_penalty', 10.0),
        overeating_death=body_cfg.get('overeating_death', False),
        use_homeostatic_reward=body_cfg.get('use_homeostatic_reward', True),
        with_satiation=body_cfg.get('with_satiation', False),
        with_injury=body_cfg.get('with_injury', True),
        sensor_radius=env_cfg.get('sensor_radius', 10.0),
        sensor_decay=env_cfg.get('sensor_decay', 2.0),
        sensor_range=env_cfg.get('sensor_range', 3)
    )

def train_jax_ppo(
    config_path: str = None,
    num_envs: int = 256,
    total_timesteps: int = 100_000,
    num_steps: int = 128,
    num_epochs: int = 4,
    lr: float = 2.5e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    hidden_size: int = 64,
    seed: int = 0,
):
    """Main training function for JAX RecurrentPPO."""
    print(f"Training JAX RecurrentPPO with {num_envs} parallel environments")
    print(f"Total timesteps: {total_timesteps}, Steps per iteration: {num_steps * num_envs}")
    
    # Compute number of iterations
    timesteps_per_iter = num_steps * num_envs
    num_iterations = total_timesteps // timesteps_per_iter
    
    # Load EnvParams from config or use defaults
    if config_path:
        print(f"Loading config from: {config_path}")
        params = load_env_params(config_path)
    else:
        print("Using default minimal environment")
        params = EnvParams(
            height=10, width=10, max_steps=500,
            res_type=jnp.zeros(0, dtype=jnp.int32),
            res_property=jnp.zeros((0, 5)),
            res_spawn_area=jnp.zeros((0, 4)),
            res_max_cons=jnp.zeros(0, dtype=jnp.int32),
            res_reg_delay=jnp.zeros(0, dtype=jnp.int32),
            res_damage=jnp.zeros(0),
            pred_property=jnp.zeros((0, 5)),
            pred_move_int=jnp.zeros(0, dtype=jnp.int32),
            pred_damage=jnp.zeros(0),
            pred_patrol=jnp.zeros((0, 4)),
            pred_detect=jnp.zeros(0),
            pred_max_stamina=jnp.zeros(0),
            pred_recovery=jnp.zeros(0),
            pred_hunt_thresh=jnp.zeros(0),
            max_satiation=100.0, max_injury=20.0, food_gain=10.0, setpoint=50.0,
            injury_recovery=1.0, smoothing_duration=3, death_penalty=10.0,
            overeating_death=False, use_homeostatic_reward=True,
            with_satiation=False, with_injury=True,
            sensor_radius=10.0, sensor_decay=2.0, sensor_range=3
        )
    
    # Initialize ParallelEnv
    env = ParallelEnv(params)
    
    # Initialize random keys
    key = jax.random.PRNGKey(seed)
    key, model_key, env_key = jax.random.split(key, 3)
    
    # Initialize environment states
    env_state, obs = env.reset(env_key, num_envs)
    input_dim = obs.shape[-1]
    action_dim = 4  # UP, DOWN, LEFT, RIGHT
    
    print(f"Observation dim: {input_dim}, Action dim: {action_dim}")
    
    # Initialize NNX Model and Optimizer
    rngs = nnx.Rngs(model_key)
    model = ActorCriticRNN(input_dim=input_dim, action_dim=action_dim, hidden_size=hidden_size, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
    h_state = model.initial_state(batch_size=num_envs)
    
    # PPO Config
    config = PPOConfig(
        num_steps=num_steps,
        num_epochs=num_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        lr=lr
    )
    
    # JIT compile train_iteration
    print("JIT compiling train_iteration...")
    jit_train = nnx.jit(train_iteration, static_argnums=(6,))
    
    # Training loop
    print(f"Starting training for {num_iterations} iterations...")
    for i in range(num_iterations):
        env_state, h_state, key, epoch_logs = jit_train(
            model, optimizer, params, env_state, h_state, key, config
        )
        
        # Log progress every 10 iterations
        if (i + 1) % 10 == 0 or i == 0:
            final_loss, (p_loss, v_loss, e_loss) = epoch_logs[-1]
            timesteps_done = (i + 1) * timesteps_per_iter
            print(f"Iter {i+1}/{num_iterations} | Timesteps: {timesteps_done:,} | "
                  f"Loss: {final_loss:.4f} | Policy: {p_loss:.4f} | Value: {v_loss:.4f}")
    
    print("Training complete!")
    return model, optimizer

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train JAX RecurrentPPO")
    parser.add_argument("--config", type=str, default=None, help="Path to environment config YAML")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    train_jax_ppo(
        config_path=args.config,
        num_envs=args.num_envs,
        total_timesteps=args.total_timesteps,
        num_steps=args.num_steps,
        num_epochs=args.num_epochs,
        lr=args.lr,
        hidden_size=args.hidden_size,
        seed=args.seed
    )
