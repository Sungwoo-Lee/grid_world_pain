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
    """One-hot Visual Sensor (simplified object recognition)."""
    # [Grass, Sand, Plain, Food, Danger, Predator, Rock]
    # Location mapping: 0: Plain, 1: Grass, 2: Sand
    # Resource mapping: 0: Food, 1: Danger
    
    vis_range = params.visual_sensor_range
    offsets = get_visual_offsets(vis_range) # [num_cells, 2]
    num_cells = offsets.shape[0]
    cell_coords = agent_pos + offsets # [num_cells, 2]
    
    # Bounds check
    is_in_bounds = jnp.logical_and(
        jnp.logical_and(cell_coords[:, 0] >= 0, cell_coords[:, 0] < params.height),
        jnp.logical_and(cell_coords[:, 1] >= 0, cell_coords[:, 1] < params.width)
    )
    
    # Safe coordinates for indexing
    safe_coords = jnp.where(is_in_bounds[:, None], cell_coords, 0)
    
    # 1. Location properties
    # grid_location_type is [H, W]
    loc_types = params.grid_location_type[safe_coords[:, 0], safe_coords[:, 1]]
    # Index 0: Grass (loc 1), 1: Sand (loc 2), 2: Plain (loc 0)
    loc_vis = jax.nn.one_hot(jnp.where(loc_types == 1, 0, jnp.where(loc_types == 2, 1, 2)), 7)
    
    # 2. Resources
    # state.res_pos: [num_res, 2], state.res_active: [num_res], params.res_type: [num_res]
    # We need to map across all num_cells and all num_res... 
    # Or just iterate over resources and add to their cells.
    
    def get_res_contrib(coord):
        # coord: [2]
        is_here = jnp.all(state.res_pos == coord, axis=-1)
        active_here = jnp.logical_and(is_here, state.res_active)
        # res_type 0: Food (vis 3), 1: Danger (vis 4)
        is_food = jnp.logical_and(active_here, params.res_type == 0)
        is_danger = jnp.logical_and(active_here, params.res_type == 1)
        
        contrib = jnp.zeros(7)
        contrib = contrib.at[3].add(jnp.sum(is_food.astype(jnp.float32)))
        contrib = contrib.at[4].add(jnp.sum(is_danger.astype(jnp.float32)))
        return contrib

    res_vis = jax.vmap(get_res_contrib)(cell_coords)

    # 3. Predators
    def get_pred_contrib(coord):
        # coord: [2]
        is_here = jnp.all(state.pred_pos == coord, axis=-1)
        # Predator (vis 5)
        contrib = jnp.zeros(7)
        contrib = contrib.at[5].add(jnp.sum(is_here.astype(jnp.float32)))
        return contrib
    
    pred_vis = jax.vmap(get_pred_contrib)(cell_coords)

    # 4. Rocks (Obstacles)
    def get_obs_contrib(coord):
        # coord: [2]
        is_here = jnp.all(state.obs_pos == coord, axis=-1)
        # Rock (vis 6)
        contrib = jnp.zeros(8)
        contrib = contrib.at[6].add(jnp.sum(is_here.astype(jnp.float32)))
        return contrib
        
    obs_vis = jax.vmap(get_obs_contrib)(cell_coords)

    # 5. Neutral Animals
    def get_neutral_contrib(coord):
        is_here = jnp.all(state.neutral_pos == coord, axis=-1)
        # Neutral Animal (vis 7)
        contrib = jnp.zeros(8)
        contrib = contrib.at[7].add(jnp.sum(is_here.astype(jnp.float32)))
        return contrib
    
    neutral_vis = jax.vmap(get_neutral_contrib)(cell_coords)
    
    # Update loc_vis, res_vis, pred_vis to use 8 slots
    # 1. Location properties
    loc_vis = jax.nn.one_hot(jnp.where(loc_types == 1, 0, jnp.where(loc_types == 2, 1, 2)), 8)
    
    # 2. Resources (update to 8)
    def get_res_contrib_8(coord):
        is_here = jnp.all(state.res_pos == coord, axis=-1)
        active_here = jnp.logical_and(is_here, state.res_active)
        is_food = jnp.logical_and(active_here, params.res_type == 0)
        is_danger = jnp.logical_and(active_here, params.res_type == 1)
        contrib = jnp.zeros(8)
        contrib = contrib.at[3].add(jnp.sum(is_food.astype(jnp.float32)))
        contrib = contrib.at[4].add(jnp.sum(is_danger.astype(jnp.float32)))
        return contrib
    res_vis = jax.vmap(get_res_contrib_8)(cell_coords)

    # 3. Predators (update to 8)
    def get_pred_contrib_8(coord):
        is_here = jnp.all(state.pred_pos == coord, axis=-1)
        contrib = jnp.zeros(8)
        contrib = contrib.at[5].add(jnp.sum(is_here.astype(jnp.float32)))
        return contrib
    pred_vis = jax.vmap(get_pred_contrib_8)(cell_coords)
    
    # Sum all contributions
    total_vis = loc_vis + res_vis + pred_vis + obs_vis + neutral_vis
    
    # Mask out-of-bounds cells
    total_vis = total_vis * is_in_bounds[:, None]
    
    return total_vis.flatten()

def apply_perceptual_noise(obs: jnp.ndarray, state: EnvState, params: EnvParams, key: jax.random.PRNGKey):
    """Applies vectorized, state-dependent Gaussian noise based on modality-specific modes."""
    if not params.perceptual_noise_enabled:
        return obs
        
    breakdown = get_observation_breakdown(params)
    
    # Mapping Sensor names to indices in params.noise_modes/sigmas/scales
    # Sync with config_loader.py ordering
    modality_map = {
        "Olfaction": 0,
        "Extero Nociception": 1,
        "Collision": 2,
        "Location": 3,
        "Satiation": 4,
        "Nutrition": 5,
        "Injury": 6,
        "Visual": 7,
        "Proprioception": 8
    }
    
    sigma_base_list = []
    alpha_list = []
    mode_list = []
    
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
    
    noise = jax.random.normal(key, obs.shape) * sigma_eff
    return obs + noise

def get_observation(state: EnvState, params: EnvParams):
    """Assembles the full observation vector, including noise if enabled."""
    # Salt the state key to get a deterministic but unique key for observation noise
    obs_key = jax.random.fold_in(state.key, 999)
    
    # 1. Olfaction Sensor (Resources + Predators + Obstacles + Neutral)
    # Renamed from chemical for consistency with documentation
    res_chem = sense_resource(state.agent_pos, state.res_pos, state.res_active, params.res_property, params.sensor_radius, params.sensor_decay)
    pred_chem = sense_resource(state.agent_pos, state.pred_pos, jnp.ones(state.pred_pos.shape[0], dtype=jnp.bool_), params.pred_property, params.sensor_radius, params.sensor_decay)
    obs_chem = sense_resource(state.agent_pos, state.obs_pos, jnp.ones(state.obs_pos.shape[0], dtype=jnp.bool_), params.obs_property, params.sensor_radius, params.sensor_decay)
    neutral_chem = sense_resource(state.agent_pos, state.neutral_pos, jnp.ones(state.neutral_pos.shape[0], dtype=jnp.bool_), params.neutral_property, params.sensor_radius, params.sensor_decay)
    olf_obs = res_chem + pred_chem + obs_chem + neutral_chem
    
    # 2. Extero Nociception (Phasic - Multi-source)
    noc_obs = sense_extero_nociception(state.agent_pos, state, params)
    
    # 3. Collision
    coll_obs = sense_collision(state.agent_pos, state, params)
    
    # 4. Location
    loc_obs = sense_location(state.agent_pos, params.height, params.width)
    
    # 5. Interoception
    intero_obs = jnp.array([
        state.satiation / params.max_satiation,
        state.nutrition / params.max_nutrition,
        state.injury_level / params.max_injury
    ])
    
    # 6. Visual Sensor
    if params.visual_sensor_enabled:
        vis_obs = sense_visual(state.agent_pos, state, params)
        obs = jnp.concatenate([olf_obs, noc_obs, coll_obs, loc_obs, intero_obs, vis_obs])
    else:
        obs = jnp.concatenate([olf_obs, noc_obs, coll_obs, loc_obs, intero_obs])
    
    # 7. Proprioception (Previous Action)
    if params.proprioception_enabled:
        proprio_obs = jax.nn.one_hot(state.last_action, params.action_dim)
        obs = jnp.concatenate([obs, proprio_obs])
    
    # Apply Perceptual Precision Modulation
    return apply_perceptual_noise(obs, state, params, obs_key)

def get_observation_breakdown(params: EnvParams):
    """Returns a dict of {sensor_name: dimension} for observation components."""
    # Component dimensions based on sensor.py logic:
    # 1. Olfaction: vector_size from resource properties
    chem_dim = int(params.res_property.shape[-1])
    
    # 2. Extero Nociception: 1 (contact)
    noc_dim = 1
    
    # 3. Collision: Manhattan range
    num_coll_cells = 2 * (params.sensor_range**2) + 2 * params.sensor_range + 1
    coll_dim = int(num_coll_cells)
    
    # 4. Location: 2 (normalized row, col)
    loc_dim = 2
    
    # 5. Interoception: 3 (satiation, nutrition, injury)
    intero_dim = 3
    
    breakdown = {
        "Olfaction": chem_dim,
        "Extero Nociception": noc_dim,
        "Collision": coll_dim,
        "Location": loc_dim,
        "Satiation": 1,
        "Nutrition": 1,
        "Injury": 1
    }
    
    if params.visual_sensor_enabled:
        # Range r -> 2r^2 + 2r + 1 cells
        num_cells = 2 * (params.visual_sensor_range**2) + 2 * params.visual_sensor_range + 1
        breakdown["Visual"] = int(num_cells * 8)
    
    # 7. Proprioception
    if params.proprioception_enabled:
        breakdown["Proprioception"] = int(params.action_dim)
        
    return breakdown

