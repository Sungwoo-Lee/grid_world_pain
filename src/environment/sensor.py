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

def sense_collision(agent_pos, height, width, sensor_range):
    """Radial Collision Sensor with high resolution (8 rays per unit range)."""
    num_sectors = sensor_range * 8
    angles = 2 * jnp.pi * jnp.arange(num_sectors) / num_sectors
    dr = -jnp.cos(angles)
    dc = jnp.sin(angles)
    directions = jnp.stack([dr, dc], axis=-1) # [num_sectors, 2]
    
    steps = jnp.arange(1, sensor_range + 1)
    
    # Check positions for all rays and steps: [num_sectors, sensor_range, 2]
    check_pos = agent_pos[None, None, :] + directions[:, None, :] * steps[None, :, None]
    check_pos = jnp.round(check_pos).astype(jnp.int32)
    
    # Bounds check: [num_sectors, sensor_range]
    is_out = jnp.any(jnp.logical_or(
        check_pos < 0,
        check_pos >= jnp.array([height, width])
    ), axis=-1)
    
    # Find first OOB index for each ray
    # Use indices where True, else a safe high value
    indices = jnp.where(is_out, jnp.arange(sensor_range), sensor_range + 1)
    first_hit_idx = jnp.min(indices, axis=1)
    
    hit_mask = first_hit_idx < sensor_range
    # Proximity: 1.0 at distance 1 (idx 0), 0.0 at range
    # Formula from original: 1.0 - (step - 1) / sensor_range
    proximity = 1.0 - (first_hit_idx) / sensor_range
    proximity = jnp.where(hit_mask, proximity, 0.0)
    
    return proximity

def sense_location(agent_pos, height, width):
    """Normalized Agent Location Sensor."""
    # Center coordinates to [-1, 1]
    norm_r = (agent_pos[0] / (height - 1)) * 2 - 1
    norm_c = (agent_pos[1] / (width - 1)) * 2 - 1
    return jnp.array([norm_r, norm_c])

def sense_extero_nociception(agent_pos, res_pos, res_active, res_type):
    """Phasic Nociceptor: Detects immediate contact with danger resources."""
    # res_type 0=food, 1=danger
    is_danger = (res_type == 1)
    # Contact check: dist == 0 (expressed as dist < 0.1 for safety on grid)
    diff = res_pos - agent_pos
    dist = jnp.linalg.norm(diff, axis=-1)
    
    # Active danger on current cell
    contact = jnp.logical_and(jnp.logical_and(res_active, is_danger), dist < 0.1)
    # Result is 1.0 if any danger contact
    activated = jnp.any(contact).astype(jnp.float32)
    return jnp.array([activated])

def get_observation(state: EnvState, params: EnvParams):
    """Assembles the full observation vector."""
    # 1. Chemical Sensor (Resources + Predators)
    res_chem = sense_resource(
        state.agent_pos, state.res_pos, state.res_active, params.res_property,
        radius=params.sensor_radius, decay_power=params.sensor_decay
    )
    pred_chem = sense_resource(
        state.agent_pos, state.pred_pos, jnp.ones(state.pred_pos.shape[0], dtype=jnp.bool_), params.pred_property,
        radius=params.sensor_radius, decay_power=params.sensor_decay
    )
    chem_obs = res_chem + pred_chem
    
    # 2. Extero Nociception (Phasic)
    noc_obs = sense_extero_nociception(
        state.agent_pos, state.res_pos, state.res_active, params.res_type
    )
    
    # 3. Collision
    coll_obs = sense_collision(
        state.agent_pos, params.height, params.width, params.sensor_range
    )
    
    # 4. Location
    loc_obs = sense_location(state.agent_pos, params.height, params.width)
    
    # 5. Interoception
    intero_obs = jnp.array([
        state.satiation / params.max_satiation,
        state.injury_level / params.max_injury
    ])
    
    # Concatenate all
    return jnp.concatenate([chem_obs, noc_obs, coll_obs, loc_obs, intero_obs])

def get_observation_breakdown(params: EnvParams):
    """Returns a dict of {sensor_name: dimension} for observation components."""
    # Component dimensions based on sensor.py logic:
    # 1. Chemical: vector_size from resource properties
    chem_dim = int(params.res_property.shape[-1])
    
    # 2. Extero Nociception: 1 (contact)
    noc_dim = 1
    
    # 3. Collision: sensor_range * 8 rays
    coll_dim = int(params.sensor_range) * 8
    
    # 4. Location: 2 (normalized row, col)
    loc_dim = 2
    
    # 5. Interoception: 2 (satiation, injury)
    intero_dim = 2
    
    return {
        "Chemical": chem_dim,
        "Extero Nociception": noc_dim,
        "Collision": coll_dim,
        "Location": loc_dim,
        "Interoception": intero_dim
    }

