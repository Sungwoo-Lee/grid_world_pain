import jax
import jax.numpy as jnp
from .state import EnvState, EnvParams

def sense_resource(agent_pos, res_pos, res_active, res_property, radius, decay_power):
    """Vectorized Resource Sensor (Chemical signature gradient)."""
    # [num_res, 2]
    diff = res_pos - agent_pos
    dist = jnp.linalg.norm(diff, axis=-1)
    
    # Handle singularity (agent ON resource)
    # Original: dist < 0.001 -> decay = 2.0
    # Else: 1.0 / (dist ** decay_power)
    decay = jnp.where(dist < 0.001, 2.0, 1.0 / (jnp.power(dist, decay_power) + 1e-10))
    
    # Mask by radius and activity
    mask = jnp.logical_and(res_active, dist <= radius)
    
    # Apply mask and sum: [num_res, vector_size] -> [vector_size]
    weighted_props = res_property * decay[:, None] * mask[:, None]
    obs = jnp.sum(weighted_props, axis=0)
    return obs

def sense_collision(agent_pos, state: EnvState, params: EnvParams):
    """Manhattan Collision Sensor (checks OOB and blocking obstacles)."""
    sensor_range = params.sensor_range
    offsets = get_visual_offsets(sensor_range) # [num_cells, 2]
    num_cells = offsets.shape[0]
    cell_coords = agent_pos + offsets # [num_cells, 2]
    
    # 1. Bounds check
    is_out_of_bounds = jnp.any(jnp.logical_or(
        cell_coords < 0,
        cell_coords >= jnp.array([params.height, params.width])
    ), axis=-1)
    
    # 2. Blocking Obstacles (Rocks)
    def check_blocking_rock(coord):
        # coord: [2]
        is_here = jnp.all(state.obs_pos == coord, axis=-1)
        # Check if any rock at this position is blocking
        is_blocking = jnp.logical_and(is_here, params.obs_blocking)
        return jnp.any(is_blocking)
        
    is_blocked_by_rock = jax.vmap(check_blocking_rock)(cell_coords)
    
    # Total collision: OOB or Blocking Rock
    collision = jnp.logical_or(is_out_of_bounds, is_blocked_by_rock).astype(jnp.float32)
    
    return collision

def sense_location(agent_pos, height, width):
    """Normalized Agent Location Sensor."""
    # Center coordinates to [-1, 1]
    norm_r = (agent_pos[0] / (height - 1)) * 2 - 1
    norm_c = (agent_pos[1] / (width - 1)) * 2 - 1
    return jnp.array([norm_r, norm_c])

def sense_extero_nociception(agent_pos, state: EnvState, params: EnvParams):
    """
    Continuous Phasic Nociceptor: Detects contact with danger, predators, and rocks.
    Returns the maximum intensity among all current painful contacts.
    """
    # 1. Danger Resource Contact
    dist_res = jnp.linalg.norm(state.res_pos - agent_pos, axis=-1)
    # Intensity = intensity from params if at position and active
    res_intensities = jnp.where(jnp.logical_and(state.res_active, dist_res < 0.1), params.res_nociception, 0.0)
    max_res = jnp.max(res_intensities, initial=0.0)

    # 2. Predator Contact
    dist_pred = jnp.linalg.norm(state.pred_pos - agent_pos, axis=-1)
    # All predators are active
    pred_intensities = jnp.where(dist_pred < 0.1, params.pred_nociception, 0.0)
    max_pred = jnp.max(pred_intensities, initial=0.0)

    # 3. Rock Overlap Contact (Non-blocking)
    dist_obs = jnp.linalg.norm(state.obs_pos - agent_pos, axis=-1)
    obs_intensities = jnp.where(dist_obs < 0.1, params.obs_nociception, 0.0)
    max_obs_overlap = jnp.max(obs_intensities, initial=0.0)
    
    # 4. Rock Collision Contact (Bumping)
    # Stored in state from jax_step
    max_collision = state.last_collision_noc

    # Result is the maximum intensity
    final_noc = jnp.max(jnp.array([max_res, max_pred, max_obs_overlap, max_collision]), initial=0.0)
    return jnp.array([final_noc])

def get_visual_offsets(sensor_range):
    """Generates Manhattan diamond offsets in a consistent order."""
    offsets = []
    # Ordering: Manhattan distance 0, then 1, then 2...
    # Within each distance, we can use a fixed direction order (e.g. Up, Right, Down, Left)
    for d in range(sensor_range + 1):
        if d == 0:
            offsets.append([0, 0])
        else:
            # Manhattan distance d: |dr| + |dc| = d
            # We iterate to find all pairs
            for dr in range(-d, d + 1):
                dc_abs = d - abs(dr)
                if dc_abs == 0:
                    offsets.append([dr, 0])
                else:
                    # Both + and - for dc
                    offsets.append([dr, dc_abs])
                    offsets.append([dr, -dc_abs])
    
    # Sort for consistency: primary by distance, secondary by row, tertiary by col
    offsets_arr = jnp.array(offsets)
    dist = jnp.sum(jnp.abs(offsets_arr), axis=1)
    # Use jnp.lexsort or similar if needed, but for small ranges a simple nested loop is fine
    # Let's just use the manual order for range 0 and 1 as they are common
    if sensor_range == 1:
        return jnp.array([[0,0], [-1,0], [0,1], [1,0], [0,-1]])
    
    return offsets_arr

def sense_visual(agent_pos, state: EnvState, params: EnvParams):
    """Matmul-optimized Visual Sensor (simplified object recognition)."""
    # Channel mapping:
    # 0: Grass (loc 1), 1: Sand (loc 2), 2: Plain (loc 0)
    # 3: Food (res_type 0), 4: Danger (res_type 1)
    # 5: Predator, 6: Rock, 7: Neutral Animal
    
    vis_range = params.visual_sensor_range
    offsets = get_visual_offsets(vis_range) # [num_cells, 2]
    num_cells = offsets.shape[0]
    cell_coords = agent_pos + offsets # [num_cells, 2]
    
    # 1. Bounds check
    is_in_bounds = jnp.all(jnp.logical_and(
        cell_coords >= 0,
        cell_coords < jnp.array([params.height, params.width])
    ), axis=-1)
    
    # Safe coordinates for indexing background
    safe_coords = jnp.where(is_in_bounds[:, None], cell_coords, 0)
    
    # 2. Background (Grid Properties) - Vectorized Indexing
    loc_types = params.grid_location_type[safe_coords[:, 0], safe_coords[:, 1]]
    # Mapping: loc 1 -> channel 0, loc 2 -> channel 1, loc 0 -> channel 2
    vis_background = jax.nn.one_hot(jnp.where(loc_types == 1, 0, jnp.where(loc_types == 2, 1, 2)), 8)
    # Mask OOB background
    vis_background = vis_background * is_in_bounds[:, None]
    
    # 3. Dynamic Entities (Resources, Predators, Rocks, Neutrals) - Matmul Optimized
    
    # Combine all dynamic entity positions
    all_pos = jnp.concatenate([
        state.res_pos,
        state.pred_pos,
        state.obs_pos,
        state.neutral_pos
    ], axis=0) # [Total_E, 2]
    
    # Combine activity status (Predators/Rocks/Neutrals always active)
    all_active = jnp.concatenate([
        state.res_active,
        jnp.ones(state.pred_pos.shape[0], dtype=jnp.bool_),
        jnp.ones(state.obs_pos.shape[0], dtype=jnp.bool_),
        jnp.ones(state.neutral_pos.shape[0], dtype=jnp.bool_)
    ], axis=0) # [Total_E]
    
    # Create Visual Property Matrix [Total_E, 8]
    # Channels 3: Food, 4: Danger, 5: Predator, 6: Rock, 7: Neutral
    num_res = state.res_pos.shape[0]
    num_pred = state.pred_pos.shape[0]
    num_obs = state.obs_pos.shape[0]
    num_neutral = state.neutral_pos.shape[0]
    
    res_props = jax.nn.one_hot(jnp.where(params.res_type == 0, 3, 4), 8)
    pred_props = jax.nn.one_hot(jnp.full((num_pred,), 5), 8)
    obs_props = jax.nn.one_hot(jnp.full((num_obs,), 6), 8)
    neutral_props = jax.nn.one_hot(jnp.full((num_neutral,), 7), 8)
    
    all_props = jnp.concatenate([res_props, pred_props, obs_props, neutral_props], axis=0)
    
    # Apply activity mask
    all_props = all_props * all_active[:, None]
    
    # Compute Matches [num_cells, Total_E]
    # (num_cells, 1, 2) == (1, Total_E, 2)
    matches = jnp.all(cell_coords[:, None, :] == all_pos[None, :, :], axis=-1)
    
    # Sum properties using Matmul: [num_cells, Total_E] @ [Total_E, 8] -> [num_cells, 8]
    vis_entities = jnp.matmul(matches.astype(jnp.float32), all_props)
    
    # Final assembly
    total_vis = vis_background + vis_entities
    # Mask out-of-bounds cells (entities at [0,0] might match safe_coords if OOB)
    total_vis = total_vis * is_in_bounds[:, None]
    
    return total_vis.flatten()

@jax.jit
def apply_perceptual_noise(obs: jnp.ndarray, state: EnvState, params: EnvParams, key: jax.random.PRNGKey):
    """Applies vectorized, state-dependent Gaussian noise based on modality-specific modes."""
    if not params.perceptual_noise_enabled:
        return obs
        
    breakdown = get_observation_breakdown(params)
    
    # Mapping Sensor names to indices in params.noise_modes/sigmas/scales
    # Sync with config_loader.py ordering
    modality_map = {
        "Injury": 0,
        "Nutrition": 1,
        "Satiation": 2,
        "Extero Nociception": 3,
        "Olfaction": 4,
        "Collision": 5,
        "Proprioception": 6,
        "Visual": 7,
        "Location": 8
    }
    
    sigma_base_list = []
    alpha_list = []
    mode_list = []
    
    # Static iteration over breakdown (which depends on EnvParams/struct)
    for sensor_name, dim in breakdown.items():
        idx = modality_map[sensor_name]
        sigma_base_list.append(jnp.full((dim,), params.noise_sigmas[idx]))
        alpha_list.append(jnp.full((dim,), params.noise_injury_scales[idx]))
        mode_list.append(jnp.full((dim,), params.noise_modes[idx]))
        
    sigma_base = jnp.concatenate(sigma_base_list)
    alpha = jnp.concatenate(alpha_list)
    mode = jnp.concatenate(mode_list)
    
    # Normalized injury (0.0 to 1.0)
    norm_injury = state.injury_level / jnp.maximum(params.max_injury, 1e-6)
    
    # Effective Sigma calculation:
    # Mode 0: None (0.0)
    # Mode 1: Constant (sigma_base)
    # Mode 2: State-Dependent (sigma_base * (1 + alpha * injury))
    sigma_eff = jnp.where(
        mode == 2,
        sigma_base * (1.0 + alpha * norm_injury),
        jnp.where(mode == 1, sigma_base, 0.0)
    )
    
    # Clip ranges (Vectorized)
    clip_min = jnp.concatenate([jnp.full((dim,), params.noise_clip_min[modality_map[name]]) for name, dim in breakdown.items()])
    clip_max = jnp.concatenate([jnp.full((dim,), params.noise_clip_max[modality_map[name]]) for name, dim in breakdown.items()])
    
    noise = jax.random.normal(key, obs.shape) * sigma_eff
    return jnp.clip(obs + noise, clip_min, clip_max)

@jax.jit(static_argnames=['apply_noise'])
def get_observation(state: EnvState, params: EnvParams, apply_noise=True):
    """Assembles the full observation vector, including noise if enabled."""
    # Salt the state key for observation noise
    obs_key = jax.random.fold_in(state.key, 999)
    
    obs_parts = []
    
    # 1. Injury
    obs_parts.append(jnp.array([state.injury_level / params.max_injury]))
    
    # 2. Nutrition
    obs_parts.append(jnp.array([state.nutrition / params.max_nutrition]))
    
    # 3. Satiation
    obs_parts.append(jnp.array([state.satiation / params.max_satiation]))
    
    # 4. Extero Nociception (Phasic - Multi-source)
    if params.nociception_enabled:
        obs_parts.append(sense_extero_nociception(state.agent_pos, state, params))

    # 5. Olfaction Sensor (Resources + Predators + Obstacles + Neutral)
    if params.olfactory_enabled:
        res_chem = sense_resource(state.agent_pos, state.res_pos, state.res_active, params.res_property, params.sensor_radius, params.sensor_decay)
        pred_chem = sense_resource(state.agent_pos, state.pred_pos, jnp.ones(state.pred_pos.shape[0], dtype=jnp.bool_), params.pred_property, params.sensor_radius, params.sensor_decay)
        obs_chem = sense_resource(state.agent_pos, state.obs_pos, jnp.ones(state.obs_pos.shape[0], dtype=jnp.bool_), params.obs_property, params.sensor_radius, params.sensor_decay)
        neutral_chem = sense_resource(state.agent_pos, state.neutral_pos, jnp.ones(state.neutral_pos.shape[0], dtype=jnp.bool_), params.neutral_property, params.sensor_radius, params.sensor_decay)
        obs_parts.append(res_chem + pred_chem + obs_chem + neutral_chem)
    
    # 6. Collision
    obs_parts.append(sense_collision(state.agent_pos, state, params))
    
    # 7. Proprioception (Previous Action)
    if params.proprioception_enabled:
        obs_parts.append(jax.nn.one_hot(state.last_action, params.action_dim))
    
    # 8. Visual Sensor
    if params.visual_sensor_enabled:
        obs_parts.append(sense_visual(state.agent_pos, state, params))
    
    # 9. Location
    if params.location_sensor_enabled:
        obs_parts.append(sense_location(state.agent_pos, params.height, params.width))
    
    # Assemble final vector
    obs = jnp.concatenate(obs_parts)
    
    # Apply Perceptual Precision Modulation
    if apply_noise:
        return apply_perceptual_noise(obs, state, params, obs_key)
    return obs

def get_observation_breakdown(params: EnvParams):
    """Returns a dict of {sensor_name: dimension} for observation components."""
    breakdown = {}
    
    # 1. Injury
    breakdown["Injury"] = 1

    # 2. Nutrition
    breakdown["Nutrition"] = 1

    # 3. Satiation
    breakdown["Satiation"] = 1

    # 4. Extero Nociception
    if params.nociception_enabled:
        breakdown["Extero Nociception"] = 1
    
    # 5. Olfaction
    if params.olfactory_enabled:
        breakdown["Olfaction"] = int(params.res_property.shape[-1])
    
    # 6. Collision
    num_coll_cells = 2 * (params.sensor_range**2) + 2 * params.sensor_range + 1
    breakdown["Collision"] = int(num_coll_cells)
    
    # 7. Proprioception
    if params.proprioception_enabled:
        breakdown["Proprioception"] = int(params.action_dim)
        
    # 8. Visual
    if params.visual_sensor_enabled:
        num_vis_cells = 2 * (params.visual_sensor_range**2) + 2 * params.visual_sensor_range + 1
        breakdown["Visual"] = int(num_vis_cells * 8)
    
    # 9. Location
    if params.location_sensor_enabled:
        breakdown["Location"] = 2
        
    return breakdown

