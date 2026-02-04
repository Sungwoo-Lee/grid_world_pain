import jax
import jax.numpy as jnp
import optax
from flax import nnx
from src.environment.jax_env.state import EnvParams
from src.environment.jax_env.core import jax_reset
from src.models.jax_models.recurrent_ppo_network import ActorCriticRNN
from src.models.jax_models.recurrent_ppo_trainer import train_iteration
from typing import NamedTuple

class PPOConfig(NamedTuple):
    num_steps: int
    num_epochs: int
    gamma: float
    gae_lambda: float
    clip_eps: float
    ent_coef: float
    vf_coef: float
    lr: float

def smoke_test():
    print("Starting RecurrentPPO JAX (Flax NNX) Smoke Test...")
    
    num_envs = 32
    num_steps = 10
    num_epochs = 2
    
    params = EnvParams(
        height=4, width=4, max_steps=100,
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
    
    # NNX Rngs
    rngs = nnx.Rngs(0)
    
    # Input dim: 33
    input_dim = 33
    action_dim = 4
    hidden_size = 64
    
    # Init NNX Model and Optimizer
    model = ActorCriticRNN(input_dim=input_dim, action_dim=action_dim, hidden_size=hidden_size, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    
    # Init Envs
    key = jax.random.PRNGKey(0)
    key, env_key = jax.random.split(key)
    env_keys = jax.random.split(env_key, num_envs)
    env_state = jax.vmap(jax_reset, in_axes=(None, 0))(params, env_keys)
    h_state = model.initial_state(batch_size=num_envs)
    
    config = PPOConfig(
        num_steps=num_steps,
        num_epochs=num_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        lr=1e-3
    )
    
    print("Compiling train_iteration with nnx.jit...")
    # config is index 6. env_params is index 2 (but it has static fields internally).
    jit_train = nnx.jit(train_iteration, static_argnums=(6,))
    
    print("Running first iteration...")
    # New NNX train_iteration takes (model, optimizer, ...)
    next_env_state, next_h_state, key, logs = jit_train(
        model, optimizer, params, env_state, h_state, key, config
    )
    
    jax.block_until_ready(next_h_state)
    print("Iteration complete.")
    
    # logs is now a Python list of (loss, aux) tuples
    final_loss, (p_loss, v_loss, e_loss) = logs[-1]
    print(f"Final epoch loss: {final_loss:.4f}")
    print(f"Policy Loss:  {p_loss:.4f}")
    print(f"Value Loss:   {v_loss:.4f}")
    print(f"Entropy Loss: {e_loss:.4f}")
    
    print("\nSmoke Test Passed!")

if __name__ == "__main__":
    smoke_test()
