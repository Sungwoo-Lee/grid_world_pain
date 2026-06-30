"""Verify perceptual noise system correctness.

B4 fix: EnvState constructor updated to use unified animal_* fields
(replaces old pred_pos / pred_state / neutral_pos / neutral_move_timer etc.).
"""
import jax
import jax.numpy as jnp
from src.utils.config import Config
from src.environment.config_loader import load_env_params, load_env_config
from src.environment.sensor import get_observation
from src.environment.state import EnvState

def verify_noise():
    # 1. Load default config
    config_path = "configs/environment/default.yaml"
    config = load_env_config(config_path)  # honours `extends:` if present; standalone otherwise

    # 2. Load EnvParams
    params = load_env_params(config)

    # Check current status
    print(f"Perceptual noise enabled in config: {params.perceptual_noise_enabled}")

    N = params.animal_property.shape[0]
    chem_dim = params.animal_property.shape[-1] if N > 0 else 5

    # 3. Create dummy state (B4 fix: use unified animal_* fields)
    key = jax.random.PRNGKey(0)
    state = EnvState(
        agent_pos=jnp.array([5, 5]),
        current_step=jnp.array(0),
        res_pos=jnp.zeros((params.res_type.shape[0], 2), dtype=jnp.int32),
        res_active=jnp.ones(params.res_type.shape[0], dtype=jnp.bool_),
        res_cons_count=jnp.zeros(params.res_type.shape[0], dtype=jnp.int32),
        res_reg_timer=jnp.zeros(params.res_type.shape[0], dtype=jnp.int32),
        res_property_sampled=jnp.zeros((params.res_type.shape[0], chem_dim), dtype=jnp.float32),
        # Unified animal fields (B4 fix)
        animal_pos=jnp.zeros((N, 2), dtype=jnp.int32),
        animal_state=jnp.zeros(N, dtype=jnp.int32),
        animal_stamina=jnp.zeros(N, dtype=jnp.float32),
        animal_move_timer=jnp.zeros(N, dtype=jnp.int32),
        animal_attack_timer=jnp.zeros(N, dtype=jnp.int32),
        animal_property_sampled=jnp.zeros((N, chem_dim), dtype=jnp.float32),
        animal_detect_sampled=jnp.zeros(N, dtype=jnp.float32),
        animal_max_stamina_sampled=jnp.zeros(N, dtype=jnp.float32),
        animal_recovery_sampled=jnp.zeros(N, dtype=jnp.float32),
        animal_hunt_thresh_sampled=jnp.zeros(N, dtype=jnp.float32),
        animal_lose_interest_sampled=jnp.zeros(N, dtype=jnp.float32),
        obs_pos=jnp.zeros((params.obs_blocking.shape[0], 2), dtype=jnp.int32),
        obs_property_sampled=jnp.zeros((params.obs_blocking.shape[0], chem_dim), dtype=jnp.float32),
        satiation=jnp.array(100.0),
        nutrition=jnp.array(100.0),
        injury_level=jnp.array(50.0),  # some injury to exercise state-dependent noise
        injury_buffer=jnp.zeros(params.smoothing_duration),
        nociception_history_buffer=jnp.zeros(params.interoceptive_kernel_length),
        last_collision_noc=jnp.array(0.0),
        rest_streak=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False),
        key=key,
        last_action=jnp.array(0, dtype=jnp.int32),
    )

    # 4. Test with noise DISABLED
    disabled_params = params._replace(perceptual_noise_enabled=False)

    state1 = state._replace(key=jax.random.PRNGKey(1))
    state2 = state._replace(key=jax.random.PRNGKey(2))

    obs1 = get_observation(state1, disabled_params, apply_noise=True)
    obs2 = get_observation(state2, disabled_params, apply_noise=True)

    if jnp.allclose(obs1, obs2):
        print("SUCCESS: Observations are identical when noise is DISABLED even with different PRNG keys.")
    else:
        diff = jnp.max(jnp.abs(obs1 - obs2))
        print(f"FAILURE: Observations differ when noise is DISABLED! Max diff: {diff}")

    # 5. Test with noise ENABLED
    enabled_params = params._replace(perceptual_noise_enabled=True)

    obs_noise1 = get_observation(state1, enabled_params, apply_noise=True)
    obs_noise2 = get_observation(state2, enabled_params, apply_noise=True)

    if not jnp.allclose(obs_noise1, obs_noise2):
        diff = jnp.max(jnp.abs(obs_noise1 - obs_noise2))
        print(f"SUCCESS: Observations differ when noise is ENABLED. Max diff: {diff}")
    else:
        print("FAILURE: Observations are identical when noise is ENABLED!")

if __name__ == "__main__":
    verify_noise()
