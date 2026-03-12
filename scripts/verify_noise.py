import jax
import jax.numpy as jnp
import yaml
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.sensor import get_observation
from src.environment.state import EnvState

def verify_noise():
    # 1. Load default config
    config_path = "configs/environment/default.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # We need to wrap it in our Config class which handle nested keys
    # However, load_env_params expects a Config object.
    # Let's just use the Config class properly.
    # We need a dummy train/model config for it to work? 
    # No, Config just needs the dict.
    config = Config(config_dict)
    
    # 2. Load EnvParams
    params = load_env_params(config)
    
    # Check current status
    print(f"Perceptual noise enabled in config: {params.perceptual_noise_enabled}")
    
    # 3. Create dummy state
    key = jax.random.PRNGKey(0)
    state = EnvState(
        agent_pos=jnp.array([5, 5]),
        current_step=jnp.array(0),
        res_pos=jnp.zeros((params.res_type.shape[0], 2)),
        res_active=jnp.ones(params.res_type.shape[0], dtype=jnp.bool_),
        res_cons_count=jnp.zeros(params.res_type.shape[0], dtype=jnp.int32),
        res_reg_timer=jnp.zeros(params.res_type.shape[0], dtype=jnp.int32),
        pred_pos=jnp.zeros((params.pred_property.shape[0], 2)),
        pred_state=jnp.zeros(params.pred_property.shape[0], dtype=jnp.int32),
        pred_stamina=jnp.zeros(params.pred_property.shape[0], dtype=jnp.float32),
        pred_move_timer=jnp.zeros(params.pred_property.shape[0], dtype=jnp.int32),
        pred_attack_timer=jnp.zeros(params.pred_property.shape[0], dtype=jnp.int32),
        neutral_pos=jnp.zeros((params.neutral_property.shape[0], 2)),
        neutral_move_timer=jnp.zeros(params.neutral_property.shape[0], dtype=jnp.int32),
        obs_pos=jnp.zeros((params.obs_blocking.shape[0], 2)),
        satiation=jnp.array(100.0),
        nutrition=jnp.array(100.0),
        injury_level=jnp.array(50.0), # Set some injury to check state-dependent noise
        injury_buffer=jnp.zeros(params.smoothing_duration),
        last_collision_noc=jnp.array(0.0),
        rest_streak=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False),
        key=key,
        last_action=jnp.array(0, dtype=jnp.int32)
    )

    # 4. Test with noise DISABLED
    # Force disabled if it wasn't already
    disabled_params = params._replace(perceptual_noise_enabled=False)
    
    obs1 = get_observation(state, disabled_params, apply_noise=True)
    obs2 = get_observation(state, disabled_params, apply_noise=True)
    
    # Use different keys for get_observation? 
    # Actually get_observation salts the key: obs_key = jax.random.fold_in(state.key, 999)
    # So for identical state.key, it will have identical obs_key.
    # To test if noise is added, we should change state.key.
    
    state1 = state._replace(key=jax.random.PRNGKey(1))
    state2 = state._replace(key=jax.random.PRNGKey(2))
    
    obs1 = get_observation(state1, disabled_params, apply_noise=True)
    obs2 = get_observation(state2, disabled_params, apply_noise=True)
    
    if jnp.allclose(obs1, obs2):
        print("✅ SUCCESS: Observations are identical when noise is DISABLED even with different PRNG keys.")
    else:
        diff = jnp.max(jnp.abs(obs1 - obs2))
        print(f"❌ FAILURE: Observations differ when noise is DISABLED! Max diff: {diff}")
        # Print where they differ
        # print(f"Diff mask: {obs1 != obs2}")

    # 5. Test with noise ENABLED
    # We need to make sure some noise parameters are set to non-zero.
    # In default.yaml, many sigmas are non-zero.
    enabled_params = params._replace(perceptual_noise_enabled=True)
    
    obs_noise1 = get_observation(state1, enabled_params, apply_noise=True)
    obs_noise2 = get_observation(state2, enabled_params, apply_noise=True)
    
    if not jnp.allclose(obs_noise1, obs_noise2):
        diff = jnp.max(jnp.abs(obs_noise1 - obs_noise2))
        print(f"✅ SUCCESS: Observations differ when noise is ENABLED. Max diff: {diff}")
    else:
        print("❌ FAILURE: Observations are identical when noise is ENABLED!")

if __name__ == "__main__":
    verify_noise()
