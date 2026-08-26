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
    # On-source value: "standing on it" == "half a cell away", the grid's sampling
    # limit. At decay_power=1.0 this is bit-identical to the literal 2.0 this
    # replaced (1/0.5 == 2.0 exactly in float32), so byte-parity holds for every
    # config at the shipped decay power. At other gammas it stays on the curve,
    # where the constant did not.  [ONSOURCE_RULE_STUDY]
    on_source = 1.0 / jnp.power(0.5, decay_power)
    decay = jnp.where(dist < 0.001, on_source, 1.0 / (jnp.power(dist, decay_power) + 1e-10))
    
    # Mask by radius and activity
    mask = jnp.logical_and(res_active, dist <= radius)
    
    # Apply mask and sum: [num_res, vector_size] -> [vector_size]
    weighted_props = res_property * decay[:, None] * mask[:, None]
    obs = jnp.sum(weighted_props, axis=0)
    return obs

def _sense_olfaction_at(point, state: EnvState, params: EnvParams):
    """The three-pool olfactory sum evaluated at ONE sampling point.

    Kept as three separate sense_resource calls summed in the original order
    (res + animal + obs) so float accumulation is bit-identical to pre-DIRECTIONAL_SENSORS.
    """
    return (sense_resource(point, state.res_pos, state.res_active,
                           state.res_property_sampled, params.sensor_radius, params.sensor_decay)
            + sense_resource(point, state.animal_pos, state.animal_active,
                             state.animal_property_sampled, params.sensor_radius, params.sensor_decay)
            + sense_resource(point, state.obs_pos, state.obs_active,
                             state.obs_property_sampled, params.sensor_radius, params.sensor_decay))


def sense_olfaction_cells(state: EnvState, params: EnvParams):
    """Olfactory field sampled at every cell of a Manhattan diamond (DIRECTIONAL_SENSORS).

    olfactory_grid_range == 0 takes a static fallback to the original
    single-point expression, so parity does not depend on vmap-of-one compiling
    identically. Out-of-bounds cells read zero across all channels, matching the
    visual sensor. Flattened cell-major: cell 0's channels, then cell 1's, ...
    """
    if params.olfactory_grid_range == 0:          # static branch, trace time
        return _sense_olfaction_at(state.agent_pos, state, params)

    offsets = get_visual_offsets(params.olfactory_grid_range)   # [C,2]
    cells = state.agent_pos + offsets
    # Built here rather than reused from sense_visual: the two sensors have
    # INDEPENDENT ranges, so their cell sets and in-bounds masks differ.
    in_bounds = jnp.all(jnp.logical_and(
        cells >= 0, cells < jnp.array([params.height, params.width])), axis=-1)   # [C]
    per_cell = jax.vmap(lambda pt: _sense_olfaction_at(pt, state, params))(cells)  # [C,V]
    return (per_cell * in_bounds[:, None]).flatten()


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
    
    # 2. Blocking Obstacles (Rocks) — inactive obstacles are transparent (PER_EPISODE_ENV_VARIANCE)
    def check_blocking_rock(coord):
        # coord: [2]
        is_here = jnp.all(state.obs_pos == coord, axis=-1)
        # Check if any rock at this position is blocking AND active
        is_blocking = jnp.logical_and(is_here, params.obs_blocking & state.obs_active)
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
    Continuous Phasic Nociceptor: Detects contact with hiding predators, animals, and rocks.
    Returns the maximum intensity among all current painful contacts.

    B2 fix: predator contact now uses unified state.animal_pos / params.animal_nociception
    masked by params.animal_is_damaging (replaces old state.pred_pos / params.pred_nociception).
    """
    # 1. Hiding Predator Contact (resources with type==1)
    dist_res = jnp.linalg.norm(state.res_pos - agent_pos, axis=-1)
    # Intensity = intensity from params if at position and active
    res_intensities = jnp.where(jnp.logical_and(state.res_active, dist_res < 0.1), params.res_nociception, 0.0)
    max_res = jnp.max(res_intensities, initial=0.0)

    # 2. Animal Contact (B2 fix — unified; only damaging animals emit nociception)
    # AND with animal_active so inactive animals cannot trigger nociception (PER_EPISODE_ENV_VARIANCE)
    if state.animal_pos.shape[0] > 0:
        dist_animal = jnp.linalg.norm(state.animal_pos - agent_pos, axis=-1)
        animal_intensities = jnp.where(
            jnp.logical_and(dist_animal < 0.1, params.animal_is_damaging & state.animal_active),
            params.animal_nociception, 0.0
        )
        max_animal = jnp.max(animal_intensities, initial=0.0)
    else:
        max_animal = 0.0

    # 3. Rock Overlap Contact (Non-blocking) — AND with obs_active (PER_EPISODE_ENV_VARIANCE)
    dist_obs = jnp.linalg.norm(state.obs_pos - agent_pos, axis=-1)
    obs_intensities = jnp.where(dist_obs < 0.1, params.obs_nociception * state.obs_active, 0.0)
    max_obs_overlap = jnp.max(obs_intensities, initial=0.0)

    # 4. Rock Collision Contact (Bumping)
    # Stored in state from jax_step
    max_collision = state.last_collision_noc

    # Result is the maximum intensity
    final_noc = jnp.max(jnp.array([max_res, max_animal, max_obs_overlap, max_collision]), initial=0.0)
    return jnp.array([final_noc])

def sense_interoceptive_nociception(state: EnvState, params: EnvParams):
    """
    Tonic interoceptive pain. Two modes (selected statically at JIT time):

    - Convolution mode (default): convolves recent injury history with a
      normalized alpha kernel (peak at τ steps). Buffer slot 0 = most recent
      injury; kernel[0]=0 so current-step injury does not leak instantaneously.
    - Passthrough mode (`interoceptive_convolution_enabled=False`): bypasses
      the kernel and returns the current normalized injury directly. Use this
      for ablations where the agent should perceive injury without delay.

    Returns scalar in [0, 1].
    """
    if params.interoceptive_convolution_enabled:
        convolved = jnp.sum(state.nociception_history_buffer * params.interoceptive_kernel)
        return jnp.array([convolved / jnp.maximum(params.max_injury, 1e-6)])
    return jnp.array([state.injury_level / jnp.maximum(params.max_injury, 1e-6)])

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

def _psf_weights(cell_coords, all_pos, agent_pos, params: EnvParams):
    """Anisotropic point-spread weights, [num_cells, Total_E].

    A gaussian elongated ALONG the agent->entity ray: sigma_par grows with the
    entity's distance so WHERE it is becomes vague, while sigma_perp stays tight
    so WHICH WAY it is stays sharp. Normalised by the kernel's full analytic mass
    (2*pi*sp*st) so only the fraction landing inside the diamond is reported --
    that is what makes distant entities fade.  [VISUAL_PSF_MECHANISM_STUDY]

    v = c - e is computed BEFORE projecting, rather than as (c.u - e.u): the
    latter routes geometry through a matmul, which runs in reduced precision on
    Ampere-class GPUs and loses ~3 decimal digits.
    """
    ap = agent_pos.astype(jnp.float32)
    pos = all_pos.astype(jnp.float32)                                  # [E,2]
    rel = pos - ap
    d = jnp.sqrt(jnp.maximum(jnp.sum(rel * rel, axis=-1), 1e-12))      # [E]
    u = jnp.where((d > 1e-6)[:, None], rel / d[:, None],
                  jnp.array([1.0, 0.0], dtype=jnp.float32))            # [E,2] radial
    t = jnp.stack([-u[:, 1], u[:, 0]], axis=-1)                        # [E,2] tangential

    floor = params.visual_blur_sigma_floor
    sp = jnp.maximum(params.visual_blur_radial_scale * d, floor)       # [E]
    st = jnp.maximum(sp / params.visual_blur_anisotropy, floor)        # [E]

    v = cell_coords.astype(jnp.float32)[:, None, :] - pos[None, :, :]  # [C,E,2]
    vpar = jnp.einsum('ced,ed->ce', v, u)
    vperp = jnp.einsum('ced,ed->ce', v, t)
    W = jnp.exp(-(vpar * vpar) * (0.5 / (sp * sp))[None, :]
                - (vperp * vperp) * (0.5 / (st * st))[None, :])
    return W / (2.0 * jnp.pi * sp * st)[None, :]


def _visual_mask_gate(agent_pos, all_pos, all_mask):
    """Per-entity visibility gate, broadcast over cells -> [1, Total_E].

    Gates on the ENTITY's Manhattan distance from the agent, not the cell's.
    Under exact matching the two are equivalent; under blur they are not, and
    gating per cell would leave a 'far'-masked entity leaking its blur tail into
    the agent's own cell.  0 = none, 1 = far, 2 = all.
    """
    d_e = jnp.sum(jnp.abs(all_pos - agent_pos), axis=-1)               # [E]
    keep = jnp.where(all_mask == 2, 0.0,
                     jnp.where(all_mask == 1,
                               (d_e < 1).astype(jnp.float32), 1.0))
    return keep[None, :]


def _occlusion_gate(agent_pos, all_pos, all_active, all_blocks, params: EnvParams):
    """Line-of-sight gate, [1, Total_E]. DIRECTIONAL_SENSORS.

    An entity is hidden when a NEARER sight-blocking entity lies inside the shadow
    cone of the ray from the agent to it. A cone rather than a strict grid line:
    on an integer grid, exact collinearity fires almost only along the axes and
    perfect diagonals, which would make the sensor blind north-south and
    clear-sighted obliquely -- a worse artefact than the thing it models.

    Returns a per-entity factor, so it composes with the static visual_mask gate
    and the activity mask by plain multiplication. strength=1 hides fully;
    lower values attenuate.
    """
    rel = (all_pos - agent_pos).astype(jnp.float32)                    # [E,2]
    d = jnp.sqrt(jnp.maximum(jnp.sum(rel * rel, axis=-1), 1e-12))      # [E]
    u = rel / d[:, None]
    cos = u @ u.T                                                      # [E,E] pairwise bearing
    nearer = d[None, :] < d[:, None]                                   # j nearer than i
    blocker = (all_active & all_blocks)[None, :] & (d[None, :] > 1e-6)
    occluded = jnp.any(nearer & (cos > params.visual_occlusion_cos) & blocker, axis=1)
    return (1.0 - params.visual_occlusion_strength * occluded.astype(jnp.float32))[None, :]


def sense_visual(agent_pos, state: EnvState, params: EnvParams):
    """Matmul-optimized Visual Sensor (configurable per-entity appearance vectors).

    Each entity type carries a per-entity visual_property vector of length V
    (params.visual_vector_size, default 8) set from the config. The sensor
    aggregates these vectors via exact-cell-match matmul, identical in structure
    to the olfactory sensor.

    Default channel mapping at V=8 (matches pre-v3.0 one-hot encoding):
      0: Grass (loc 1), 1: Sand (loc 2), 2: Plain (loc 0)
      3: Food (res_type 0), 4: Hiding Predator (res_type 1)
      5: Predator (animal_visual_channel=5), 6: Rock (obstacle)
      7: Neutral Animal (animal_visual_channel=7)

    Custom visual_properties on any entity overrides the class default.
    Custom V (visual_vector_size≠8) requires explicit visual_properties on all
    entities and visual_background_properties in the config (enforced at load time).

    B2 fix: dynamic entities are now [res, animal, obs] — unified animals replace the
    old separate pred/neutral lists. Concat order is preserved for byte-parity.

    STATIC ONLY (v1): visual_property arrays are consumed directly from EnvParams
    (static per-episode). If per-episode visual jitter is added in a future version,
    it MUST reuse the per-subset property_key draw-order pattern established by
    UNIFIED_ANIMAL_ENTITY_AND_PER_EPISODE_SAMPLING to avoid PRNG parity breaks.
    """
    V = params.visual_vector_size  # width of visual property vectors

    vis_range = params.visual_sensor_range
    offsets = get_visual_offsets(vis_range)  # [num_cells, 2]
    num_cells = offsets.shape[0]
    cell_coords = agent_pos + offsets  # [num_cells, 2]

    # 1. Bounds check
    is_in_bounds = jnp.all(jnp.logical_and(
        cell_coords >= 0,
        cell_coords < jnp.array([params.height, params.width])
    ), axis=-1)

    # Safe coordinates for indexing background
    safe_coords = jnp.where(is_in_bounds[:, None], cell_coords, 0)

    # 2. Background (Grid Properties) - lookup into [3, V] table
    # Table rows: grass (index 0), sand (index 1), plain (index 2)
    # Selector: loc 1 -> 0, loc 2 -> 1, loc 0 -> 2
    loc_types = params.grid_location_type[safe_coords[:, 0], safe_coords[:, 1]]
    bg_selector = jnp.where(loc_types == 1, 0, jnp.where(loc_types == 2, 1, 2))
    vis_background = params.visual_background_property[bg_selector]  # [num_cells, V]
    vis_background = vis_background * is_in_bounds[:, None]

    # 3. Dynamic Entities (Resources, Animals, Obstacles) — B2 fix: unified animal list
    num_res    = state.res_pos.shape[0]
    num_animal = state.animal_pos.shape[0]
    num_obs    = state.obs_pos.shape[0]

    # Combine all dynamic entity positions — 3-way concat
    parts_pos = [state.res_pos]
    if num_animal > 0:
        parts_pos.append(state.animal_pos)
    parts_pos.append(state.obs_pos)
    all_pos = jnp.concatenate(parts_pos, axis=0)  # [Total_E, 2]

    # Combine activity status — use per-episode masks (PER_EPISODE_ENV_VARIANCE)
    parts_active = [state.res_active]
    if num_animal > 0:
        parts_active.append(state.animal_active)
    parts_active.append(state.obs_active)
    all_active = jnp.concatenate(parts_active, axis=0)  # [Total_E]

    # Visual Property Matrix [Total_E, V] — per-episode sampled vectors from EnvState.
    # With std=0 (default), sampled == mean == params.*_visual_property (byte-identical).
    # With std>0, each episode draws a fresh stochastic appearance vector per entity.
    res_props = state.res_visual_property_sampled        # [num_res, V]
    obs_props = state.obs_visual_property_sampled        # [num_obs, V]
    parts_props = [res_props]
    parts_mask = [params.res_visual_mask]
    parts_blocks = [params.res_blocks_sight]
    if num_animal > 0:
        # Each animal uses its per-episode sampled visual property vector
        animal_props = state.animal_visual_property_sampled  # [N, V]
        parts_props.append(animal_props)
        parts_mask.append(params.animal_visual_mask)
        parts_blocks.append(params.animal_blocks_sight)
    parts_props.append(obs_props)
    parts_mask.append(params.obs_visual_mask)
    parts_blocks.append(params.obs_blocks_sight)
    all_props = jnp.concatenate(parts_props, axis=0)  # [Total_E, V]
    all_mask = jnp.concatenate(parts_mask, axis=0)    # [Total_E] int
    all_blocks = jnp.concatenate(parts_blocks, axis=0)  # [Total_E] bool

    # Weight matrix [num_cells, Total_E]: gaussian point-spread when blur is on,
    # exact cell match otherwise. The OFF branch is bit-identical to pre-DIRECTIONAL_SENSORS --
    # the activity mask moved from all_props onto W, which is exact because both
    # matches and all_active are exactly 0.0 or 1.0.
    if params.visual_blur_enabled:                     # static branch, trace time
        W = _psf_weights(cell_coords, all_pos, agent_pos, params)
    else:
        W = jnp.all(cell_coords[:, None, :] == all_pos[None, :, :],
                    axis=-1).astype(jnp.float32)
    W = W * all_active[None, :].astype(jnp.float32)
    W = W * _visual_mask_gate(agent_pos, all_pos, all_mask)
    if params.visual_occlusion_enabled:                # static branch, trace time
        W = W * _occlusion_gate(agent_pos, all_pos, all_active, all_blocks, params)

    # Sum properties: [num_cells, Total_E] @ [Total_E, V] -> [num_cells, V]
    vis_entities = jnp.matmul(W, all_props)
    if params.visual_value_mode == 'clamp':            # static branch, trace time
        # Per-channel presence rather than a count: two rocks in a cell read 1.0,
        # not 2.0. Applied to the ENTITY contribution only -- terrain is ground,
        # not an object, and keeps its own value.
        vis_entities = jnp.minimum(vis_entities, 1.0)

    # Final assembly
    total_vis = vis_background + vis_entities
    total_vis = total_vis * is_in_bounds[:, None]

    return total_vis.flatten()

@jax.jit
def apply_perceptual_noise(obs: jnp.ndarray, state: EnvState, params: EnvParams, key: jax.random.PRNGKey):
    """Applies vectorized, state-dependent Gaussian noise based on modality-specific modes."""
    if not params.perceptual_noise_enabled:
        return obs
        
    breakdown = get_observation_breakdown(params)
    
    # Mapping Sensor names to indices in params.noise_modes/sigmas/scales
    # Derived from YAML order stored in params.
    modality_map = {name: i for i, name in enumerate(params.noise_modality_order)}
    
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
    clip_min = jnp.concatenate([jnp.full((dim,), params.noise_clip_min[modality_map[name]]) for name, dim in breakdown.items() if name in modality_map])
    clip_max = jnp.concatenate([jnp.full((dim,), params.noise_clip_max[modality_map[name]]) for name, dim in breakdown.items() if name in modality_map])
    
    noise = jax.random.normal(key, obs.shape) * sigma_eff
    return jnp.clip(obs + noise, clip_min, clip_max)

@jax.jit(static_argnames=['apply_noise'])
def get_observation(state: EnvState, params: EnvParams, apply_noise=True):
    """Assembles the full observation vector, including noise if enabled."""
    # Salt the state key for observation noise
    obs_key = jax.random.fold_in(state.key, 999)
    
    obs_parts = []
    
    # 1. Injury (hidden when injury_observable=False) — interoceptive
    if params.injury_observable:
        obs_parts.append(jnp.array([state.injury_level / params.max_injury]))

    # 2. Nutrition (hidden when nutrition_observable=False) — interoceptive
    if params.nutrition_observable:
        obs_parts.append(jnp.array([state.nutrition / params.max_nutrition]))

    # 3. Satiation — interoceptive
    obs_parts.append(jnp.array([state.satiation / params.max_satiation]))

    # 4. Interoceptive Nociception — interoceptive
    #    (Tonic — delayed function of hidden injury, or passthrough if convolution disabled)
    if params.interoceptive_nociception_enabled:
        obs_parts.append(sense_interoceptive_nociception(state, params))

    # 5. Extero Nociception — exteroceptive (phasic, multi-source contact)
    if params.nociception_enabled:
        obs_parts.append(sense_extero_nociception(state.agent_pos, state, params))

    # 5. Olfaction Sensor (Resources + Animals + Obstacles)
    # B2 fix: unified animal_chem replaces separate pred_chem + neutral_chem calls.
    if params.olfactory_enabled:
        # DIRECTIONAL_SENSORS: sampled at every cell of a Manhattan diamond of radius
        # olfactory_grid_range. At range 0 this is the pre-DIRECTIONAL_SENSORS single sample,
        # bit-identical. Per-episode active masks keep inactive entities silent.
        obs_parts.append(sense_olfaction_cells(state, params))
    
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
    
    # 1. Injury (hidden when injury_observable=False) — interoceptive
    if params.injury_observable:
        breakdown["Injury"] = 1
    # 2. Nutrition (hidden when nutrition_observable=False) — interoceptive
    if params.nutrition_observable:
        breakdown["Nutrition"] = 1
    # 3. Satiation — interoceptive
    breakdown["Satiation"] = 1
    # 4. Interoceptive Nociception — interoceptive (delayed/passthrough injury)
    if params.interoceptive_nociception_enabled:
        breakdown["Interoceptive Nociception"] = 1
    # 5. Extero Nociception — exteroceptive
    if params.nociception_enabled:
        breakdown["Extero Nociception"] = 1
    
    # 5. Olfaction
    if params.olfactory_enabled:
        n_olf_cells = (2 * (params.olfactory_grid_range ** 2)
                       + 2 * params.olfactory_grid_range + 1)
        breakdown["Olfaction"] = int(n_olf_cells * params.res_property.shape[-1])
    
    # 6. Collision
    num_coll_cells = 2 * (params.sensor_range**2) + 2 * params.sensor_range + 1
    breakdown["Collision"] = int(num_coll_cells)
    
    # 7. Proprioception
    if params.proprioception_enabled:
        breakdown["Proprioception"] = int(params.action_dim)
        
    # 8. Visual
    if params.visual_sensor_enabled:
        num_vis_cells = 2 * (params.visual_sensor_range**2) + 2 * params.visual_sensor_range + 1
        breakdown["Visual"] = int(num_vis_cells * params.visual_vector_size)
    
    # 9. Location
    if params.location_sensor_enabled:
        breakdown["Location"] = 2
        
    return breakdown

def build_sensory_viz(obs, state, params, true_obs=None):
    """Build the sensory_data list consumed by renderer.render_jax_state.

    If true_obs is None, computes it from state+params with apply_noise=False
    (unless perceptual_noise_enabled is False, in which case true_obs = obs).
    """
    import numpy as np  # renderer is host-side; np is fine here
    breakdown = get_observation_breakdown(params)
    if true_obs is None:
        true_obs = np.asarray(obs)
    obs = np.asarray(obs)
    
    ptr = 0
    t_ptr = 0
    viz = []
    
    for sensor_name, dim in breakdown.items():
        if sensor_name == "Olfaction":
            olf_obs = obs[ptr:ptr+dim]
            olf_true = true_obs[t_ptr:t_ptr+dim] if true_obs is not None else olf_obs
            ptr += dim; t_ptr += dim
            _V = int(params.res_property.shape[-1])
            # Chemical channels, NOT the visual ones. The previous label list was
            # the eight VISUAL channel names applied to a 5-wide chemical vector;
            # it was inert because draw_spectrum_pod ignores labels, but the
            # diamond renderer below does not.
            _olf_labels = (['FOOD', 'AN-A', 'AN-B', 'BUSH', 'TREE'] if _V == 5
                           else [f'C{i}' for i in range(_V)])
            if params.olfactory_grid_range > 0:
                # DIRECTIONAL_SENSORS: one reading per diamond cell -> render as a spatial grid,
                # the same pod the visual sensor uses.
                viz.append({'name': 'Olfactory', 'vector': olf_obs, 'true_vector': olf_true,
                            'type': 'visual_grid', 'num_features': _V,
                            'range': params.olfactory_grid_range, 'labels': _olf_labels})
            else:
                viz.append({'name': 'Olfactory', 'vector': olf_obs, 'true_vector': olf_true,
                            'type': 'spectrum', 'labels': _olf_labels})
        
        elif sensor_name == "Extero Nociception":
            noc_obs = float(obs[ptr])
            noc_true = float(true_obs[t_ptr]) if true_obs is not None else noc_obs
            ptr += dim; t_ptr += dim
            viz.append({'name': 'Extero Nociception', 'intensity': noc_obs, 'true_intensity': noc_true, 'color': '#c0392b', 'type': 'intensity'})
        
        elif sensor_name == "Collision":
            coll_obs = obs[ptr:ptr+dim]
            coll_true = true_obs[t_ptr:t_ptr+dim] if true_obs is not None else coll_obs
            ptr += dim; t_ptr += dim
            viz.append({'name': 'Collision', 'vector': coll_obs, 'true_vector': coll_true, 'type': 'diamond', 'range': params.sensor_range, 'num_features': 1})
        
        elif sensor_name == "Location":
            loc_vec = obs[ptr:ptr+dim]
            ptr += dim; t_ptr += dim
            viz.append({'name': 'LOC', 'value_text': f"({loc_vec[0]:.2f}, {loc_vec[1]:.2f})", 'color': '#ADB5BD', 'type': 'text'})
        
        elif sensor_name in ("Satiation", "Nutrition", "Injury", "Interoceptive Nociception"):
            s_obs = float(obs[ptr])
            s_true = float(true_obs[t_ptr]) if true_obs is not None else s_obs
            ptr += dim; t_ptr += dim
            display_name = "Intero Nociception" if sensor_name == "Interoceptive Nociception" else sensor_name
            color = "#8e44ad" if sensor_name == "Interoceptive Nociception" else None  # purple distinguishes pain
            tile = {'name': display_name, 'intensity': s_obs, 'true_intensity': s_true, 'type': 'intensity'}
            if color is not None:
                tile['color'] = color
            viz.append(tile)
        
        elif sensor_name == "Visual":
            vis_obs = obs[ptr:ptr+dim]
            vis_true = true_obs[t_ptr:t_ptr+dim] if true_obs is not None else vis_obs
            ptr += dim; t_ptr += dim
            _V = params.visual_vector_size
            _vis_labels = (
                ['GRS', 'SND', 'PLN', 'FOD', 'DNG', 'PRD', 'RCK', 'NEU']
                if _V == 8 else [str(i) for i in range(_V)]
            )
            viz.append({'name': 'Visual', 'vector': vis_obs, 'true_vector': vis_true, 'type': 'visual_grid', 'num_features': _V, 'range': params.visual_sensor_range, 'labels': _vis_labels})
        
        elif sensor_name == "Proprioception":
            proprio_vec = obs[ptr:ptr+dim]
            ptr += dim; t_ptr += dim
            viz.append({'name': 'Proprioception', 'vector': proprio_vec, 'type': 'radial', 'color': '#be4bdb'})

    return viz
