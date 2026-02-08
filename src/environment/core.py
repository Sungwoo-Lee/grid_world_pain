import jax
import jax.numpy as jnp
from .state import EnvState, EnvParams

def move_agent(pos: jnp.ndarray, action: int, obs_pos: jnp.ndarray, obs_blocking: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
    """Calculates New Agent position based on action, considering obstacles."""
    # 0: Up, 1: Right, 2: Down, 3: Left, 4+: Stay
    moves = jnp.array([
        [-1, 0], # Up
        [0, 1],  # Right
        [1, 0],  # Down
        [0, -1], # Left
        [0, 0],  # Rest/Stay
        [0, 0],  # Eat/Stay
    ], dtype=jnp.int32)
    
    # Clip action to valid range [0, 5]
    action = jnp.clip(action, 0, 5).astype(jnp.int32)
    move = moves[action]
    
    new_pos = pos + move
    # Clamp to grid boundaries
    new_pos = jnp.array([
        jnp.clip(new_pos[0], 0, params.height - 1),
        jnp.clip(new_pos[1], 0, params.width - 1)
    ])
    
    # Obstacle collision check
    is_collision = jnp.any(jnp.logical_and(
        jnp.all(obs_pos == new_pos, axis=-1),
        obs_blocking
    ))
    
    # If collision, stay at current position
    final_pos = jnp.where(is_collision, pos, new_pos)
    return final_pos

def calculate_drive(satiation, injury, params):
    """Calculates homeostatic drive (Euclidean distance to setpoint)."""
    target = jnp.array([params.setpoint, 0.0])
    current = jnp.stack([satiation, injury], axis=-1)
    return jnp.linalg.norm(current - target, axis=-1)

def update_body(state: EnvState, info: dict, params: EnvParams) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, bool]:
    """Updates satiation and injury levels."""
    prev_satiation = state.satiation
    prev_injury = state.injury_level
    
    # --- Satiation Dynamics ---
    if params.with_satiation:
        new_satiation = prev_satiation - 1.0
        new_satiation = jnp.where(info['ate_food'], new_satiation + params.food_gain, new_satiation)
        
        if not params.overeating_death:
            new_satiation = jnp.clip(new_satiation, 0.0, params.max_satiation)
        else:
            new_satiation = jnp.clip(new_satiation, 0.0, params.max_satiation + 1.0)
    else:
        new_satiation = prev_satiation
                
    # --- Injury Dynamics ---
    if params.with_injury:
        damage = info['damage']
        inc = damage / params.smoothing_duration
        new_buffer = jnp.roll(state.injury_buffer, -1).at[-1].set(0.0)
        new_buffer = new_buffer + inc
        
        applied_inc = new_buffer[0]
        new_injury = prev_injury + applied_inc
        
        # Recovery
        new_injury = jnp.where(jnp.logical_and(info['rested'], applied_inc <= 0), 
                               jnp.maximum(new_injury - params.injury_recovery, 0.0), 
                               new_injury)
        
        new_injury = jnp.clip(new_injury, 0.0, params.max_injury)
    else:
        new_injury = prev_injury
        new_buffer = state.injury_buffer
        damage = info['damage']
        
    # Termination check
    done = False
    if params.with_satiation:
        done = jnp.where(new_satiation <= 0.0, True, done)
        if params.overeating_death:
            done = jnp.where(new_satiation >= params.max_satiation, True, done)
            
    if params.with_injury:
        done = jnp.where(new_injury >= params.max_injury, True, done)
    else:
        # Instant death logic for levels without health system
        done = jnp.where(damage > 0, True, done)
    
    return new_satiation, new_injury, new_buffer, done 

def update_resources(res_active, res_reg_timer, res_cons_count, params):
    """Updates resource timers and regeneration."""
    # Regeneration
    needs_reg_update = jnp.logical_and(jnp.logical_not(res_active), res_reg_timer > 0)
    new_reg_timer = jnp.where(needs_reg_update, res_reg_timer - 1, res_reg_timer)
    
    # Respawn where timer hits 0
    respawn_mask = jnp.logical_and(jnp.logical_not(res_active), new_reg_timer <= 0)
    new_active = jnp.where(respawn_mask, True, res_active)
    new_cons_count = jnp.where(respawn_mask, 0, res_cons_count)
    
    return new_active, new_reg_timer, new_cons_count, respawn_mask

def update_predators(pred_pos, pred_state, pred_stamina, pred_move_timer, agent_pos, obs_pos, obs_blocking, params, key):
    """Updates predator states and positions, considering obstacles."""
    # 1. Timers
    new_move_timer = pred_move_timer - 1
    
    # Manhattan distance
    dist = jnp.sum(jnp.abs(pred_pos - agent_pos), axis=-1)
    
    # Check if back in patrol area
    # params.pred_patrol is [num_pred, 4] -> [min_r, min_c, max_r, max_c]
    in_zone = jnp.logical_and(
        jnp.logical_and(pred_pos[:, 0] >= params.pred_patrol[:, 0], pred_pos[:, 0] <= params.pred_patrol[:, 2]),
        jnp.logical_and(pred_pos[:, 1] >= params.pred_patrol[:, 1], pred_pos[:, 1] <= params.pred_patrol[:, 3])
    )
    
    # 2. State Transitions (Only when move_timer <= 0)
    # HUNT transitions
    rested_enough = pred_stamina >= (params.pred_max_stamina * params.pred_hunt_thresh)
    become_hunt = jnp.logical_and(dist <= params.pred_detect, rested_enough)
    
    # Lose interest
    lose_interest = jnp.logical_or(dist > params.pred_detect * 2, pred_stamina <= 0)
    
    # New State logic
    next_state = pred_state
    # Transition to HUNT (1)
    next_state = jnp.where(jnp.logical_and(pred_state != 1, become_hunt), 1, next_state)
    # Transition to RETURN (2) (if patrol area exists) or PATROL (0)
    next_state = jnp.where(jnp.logical_and(pred_state == 1, lose_interest), 2, next_state)
    # Transition to PATROL (0) when in zone and in RETURN state (or if no patrol area)
    next_state = jnp.where(jnp.logical_and(next_state == 2, in_zone), 0, next_state)
    
    # 3. Movement (Only when move_timer <= 0)
    should_move = new_move_timer <= 0
    
    # Target calculations for different states
    # RETURN: Target center of patrol zone
    tr_return = (params.pred_patrol[:, 0] + params.pred_patrol[:, 2]) // 2
    tc_return = (params.pred_patrol[:, 1] + params.pred_patrol[:, 3]) // 2
    
    # PATROL: Random jitter (-1, 0, 1)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    jitter_r = jax.random.randint(subkey1, (pred_pos.shape[0],), -1, 2)
    jitter_c = jax.random.randint(subkey2, (pred_pos.shape[0],), -1, 2)
    
    # Determine directional diff based on state
    # dr, dc = goal - current
    dr = jnp.zeros_like(pred_pos[:, 0])
    dc = jnp.zeros_like(pred_pos[:, 1])
    
    # HUNT (1) vectors
    dr = jnp.where(next_state == 1, agent_pos[0] - pred_pos[:, 0], dr)
    dc = jnp.where(next_state == 1, agent_pos[1] - pred_pos[:, 1], dc)
    
    # RETURN (2) vectors
    dr = jnp.where(next_state == 2, tr_return - pred_pos[:, 0], dr)
    dc = jnp.where(next_state == 2, tc_return - pred_pos[:, 1], dc)
    
    # PATROL (0) vectors (jitter)
    dr = jnp.where(next_state == 0, jitter_r, dr)
    dc = jnp.where(next_state == 0, jitter_c, dc)
    
    # Resolve step
    step_r = jnp.sign(dr)
    step_c = jnp.sign(dc)
    
    # JAX stochasticity for diagonal moves
    key, subkey3 = jax.random.split(key)
    rand_choice = jax.random.uniform(subkey3, (pred_pos.shape[0],)) < 0.5
    
    # Resolve diagonal (pick one axis to move along)
    final_move_r = jnp.where(jnp.logical_and(dr != 0, dc != 0), jnp.where(rand_choice, step_r, 0), step_r)
    final_move_c = jnp.where(jnp.logical_and(dr != 0, dc != 0), jnp.where(jnp.logical_not(rand_choice), step_c, 0), step_c)
    
    move_vec = jnp.stack([final_move_r, final_move_c], axis=-1)
    new_pos = jnp.where(should_move[:, None], pred_pos + move_vec, pred_pos)
    
    # 4. Spatial Bounds Clipping (Strictly enforce pred_patrol)
    new_pos = jnp.stack([
        jnp.clip(new_pos[:, 0], params.pred_patrol[:, 0], params.pred_patrol[:, 2]),
        jnp.clip(new_pos[:, 1], params.pred_patrol[:, 1], params.pred_patrol[:, 3])
    ], axis=-1)
    
    # Hard Grid Boundaries
    new_pos = jnp.clip(new_pos, 0, jnp.stack([params.height - 1, params.width - 1]))
    
    # 4.5 Obstacle Collision for Predators
    def check_collision(p_pos, old_p_pos):
        # p_pos: [2], old_p_pos: [2]
        is_coll = jnp.any(jnp.logical_and(jnp.all(obs_pos == p_pos, axis=-1), obs_blocking))
        return jnp.where(is_coll, old_p_pos, p_pos)
    
    # Check collision for each predator
    new_pos = jax.vmap(check_collision)(new_pos, pred_pos)
    
    # Reset timer
    new_move_timer = jnp.where(should_move, params.pred_move_int, new_move_timer)
    
    # Stamina
    new_stamina = jnp.where(next_state == 1, pred_stamina - 1.0, pred_stamina + params.pred_recovery)
    new_stamina = jnp.clip(new_stamina, 0.0, params.pred_max_stamina)
    
    return new_pos, next_state, new_stamina, new_move_timer, key

def jax_step(state: EnvState, action: int, params: EnvParams) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, dict]:
    """Orchestrates a full environment step in JAX."""
    
    # 0. Split key for random events (regeneration, predators)
    key, respawn_key, predator_key = jax.random.split(state.key, 3)

    # 1. Resource Regeneration (before agent moves)
    new_active, new_reg_timer, new_cons_count, respawn_mask = update_resources(
        state.res_active, state.res_reg_timer, state.res_cons_count, params
    )
    
    # Displace resources that just respawned
    num_res = params.res_type.shape[0]
    res_keys = jax.random.split(respawn_key, num_res)
    
    def sample_res_pos(rk, area):
        return jax.random.randint(rk, (2,), area[:2], area[2:])
        
    new_potential_pos = jax.vmap(sample_res_pos)(res_keys, params.res_spawn_area)
    # Only update position IF respawn_mask is true for that resource
    res_pos_after_reg = jnp.where(respawn_mask[:, None], new_potential_pos, state.res_pos)
    
    # 2. Predator Update
    new_pred_pos, new_pred_state, new_pred_stamina, new_pred_move_timer, _ = update_predators(
        state.pred_pos, state.pred_state, state.pred_stamina, state.pred_move_timer, 
        state.agent_pos, state.obs_pos, params.obs_blocking, params, predator_key
    )
    
    # 3. Agent Movement
    new_agent_pos = move_agent(state.agent_pos, action, state.obs_pos, params.obs_blocking, params)
    
    # 4. Interaction Logic
    # Check overlaps with resources (using positions AFTER regeneration)
    at_resource = jnp.all(res_pos_after_reg == new_agent_pos, axis=-1)
    interact_resource = jnp.logical_and(at_resource, new_active)
    
    # Danger interaction (Auto)
    is_danger = params.res_type == 1
    damage_res = jnp.sum(jnp.where(jnp.logical_and(interact_resource, is_danger), params.res_damage, 0.0))
    
    # Food interaction (Action-based or Auto)
    is_food = params.res_type == 0
    
    # Determine if eat action was triggered based on config
    eat_action_idx = jnp.where(params.rest_action_enabled, 5, 4)
    eat_action_triggered = jnp.logical_and(params.eat_action_enabled, action == eat_action_idx)
    
    # Auto-eat occurs if eat action is disabled and agent is on food
    ate_food_auto = jnp.logical_and(jnp.logical_not(params.eat_action_enabled), 
                                   jnp.logical_and(interact_resource, is_food))
    
    # Final 'ate_food' flag (used for satiation and lifecycle)
    ate_food = jnp.any(jnp.logical_or(
        ate_food_auto,
        jnp.logical_and(jnp.logical_and(interact_resource, is_food), eat_action_triggered)
    ))
    
    # Fix eat_triggered for lifecycle update (both auto and action)
    eat_lifecycle_triggered = jnp.logical_and(interact_resource, is_food)
    eat_lifecycle_triggered = jnp.logical_and(eat_lifecycle_triggered, 
                                             jnp.logical_or(jnp.logical_not(params.eat_action_enabled), eat_action_triggered))
    
    # Final interact mask for lifecycle update
    interacted_this_step = jnp.logical_or(
        jnp.logical_and(interact_resource, is_danger),
        eat_lifecycle_triggered
    )
    
    # Update Resource Lifecycle (Consumption)
    next_cons_count = new_cons_count + jnp.where(interacted_this_step, 1, 0)
    # Deactivate if exceeded max_cons (if max_cons > 0)
    # CRITICAL FIX: Only deactivate if it was active to avoid resetting timer during deactivation phase
    should_deactivate = jnp.logical_and(new_active, 
                                        jnp.logical_and(params.res_max_cons > 0, next_cons_count >= params.res_max_cons))
    final_active = jnp.where(should_deactivate, False, new_active)
    # Set reg timer
    next_reg_timer = jnp.where(should_deactivate, params.res_reg_delay, new_reg_timer)

    
    # ate_food is already calculated above
    
    # Predator Damage
    at_predator = jnp.all(new_pred_pos == new_agent_pos, axis=-1)
    damage_pred = jnp.sum(jnp.where(at_predator, params.pred_damage, 0.0))
    
    total_damage = damage_res + damage_pred
    
    # 5. Body Update
    # rested is always action 4 if enabled
    rested = jnp.logical_and(params.rest_action_enabled, action == 4)
    
    info = {
        'ate_food': ate_food,
        'damage': total_damage,
        'rested': rested,
        'hit_danger': jnp.any(jnp.logical_and(interact_resource, is_danger)),
        'hit_predator': jnp.any(at_predator),
    }
    
    new_satiation, new_injury, next_injury_buffer, done = update_body(state, info, params)
    
    # Max Steps Truncation
    next_step = state.current_step + 1
    truncated = next_step >= params.max_steps
    
    # Termination Reason (Integer codes for JIT compatibility)
    # 0: active, 1: max_steps, 2: starvation, 3: overeating, 4: injury
    reason = jnp.array(0, dtype=jnp.int32)
    reason = jnp.where(truncated, 1, reason)
    reason = jnp.where(new_satiation <= 0.0, 2, reason)
    if params.overeating_death:
        reason = jnp.where(new_satiation >= params.max_satiation, 3, reason)
    reason = jnp.where(new_injury >= params.max_injury, 4, reason)
    
    info['termination_reason'] = reason
    done = jnp.logical_or(done, truncated)
    
    # 6. Reward (Homeostatic)
    reward = 0.0
    if params.use_homeostatic_reward:
        prev_drive = calculate_drive(state.satiation, state.injury_level, params)
        curr_drive = calculate_drive(new_satiation, new_injury, params)
        reward = prev_drive - curr_drive
        reward = jnp.where(done, reward - params.death_penalty, reward)
    else:
        reward = jnp.where(ate_food, 1.0, 0.0)
        reward = jnp.where(done, -params.death_penalty, reward)

    # 7. Final State
    new_state = state._replace(
        agent_pos=new_agent_pos,
        current_step=next_step,
        res_pos=res_pos_after_reg,
        res_active=final_active,
        res_reg_timer=next_reg_timer,
        res_cons_count=next_cons_count,
        pred_pos=new_pred_pos,
        pred_state=new_pred_state,
        pred_stamina=new_pred_stamina,
        pred_move_timer=new_pred_move_timer,
        satiation=new_satiation,
        injury_level=new_injury,
        injury_buffer=next_injury_buffer,
        terminated=done,
        key=key
    )
    
    return new_state, reward, done, info


def jax_reset(params: EnvParams, key: jax.random.PRNGKey) -> EnvState:
    """Functional reset for the JAX environment."""
    key, agent_key, res_key, pred_key, body_key = jax.random.split(key, 5)
    
    # 1. Agent Position (Random)
    agent_pos = jax.random.randint(agent_key, (2,), 0, jnp.array([params.height, params.width]))
    
    # 2. Resources (Simplified: Random placement within spawn_area)
    num_res = params.res_type.shape[0]
    res_keys = jax.random.split(res_key, num_res)
    
    def sample_res_pos(rk, area):
        return jax.random.randint(rk, (2,), area[:2], area[2:])
        
    res_pos = jax.vmap(sample_res_pos)(res_keys, params.res_spawn_area)
    
    # 3. Predators (Simplified: Random placement)
    num_pred = params.pred_damage.shape[0]
    pred_spawn_keys = jax.random.split(pred_key, num_pred)
    
    def sample_pred_pos(pk, area):
        return jax.random.randint(pk, (2,), area[:2], area[2:])
        
    pred_pos = jax.vmap(sample_pred_pos)(pred_spawn_keys, params.pred_patrol)
    
    # 4. Obstacles
    num_obs = params.obs_blocking.shape[0]
    obs_keys = jax.random.split(key, num_obs)
    
    def sample_obs_pos(ok, area):
        return jax.random.randint(ok, (2,), area[:2], area[2:])
        
    obs_pos = jax.vmap(sample_obs_pos)(obs_keys, params.obs_spawn_area)
    
    # 5. Body (Random start support)
    body_key1, body_key2 = jax.random.split(body_key)
    
    if params.random_start_satiation:
        min_start = params.max_satiation / 2.0
        satiation = jax.random.uniform(body_key1, (), minval=min_start, maxval=params.max_satiation)
    else:
        satiation = params.start_satiation
    
    if params.random_start_injury:
        max_start_injury = params.max_injury / 2.0
        injury = jax.random.uniform(body_key2, (), minval=0.0, maxval=max_start_injury)
    else:
        injury = 0.0
        
    injury_buffer = jnp.zeros(params.smoothing_duration)
    
    state = EnvState(
        agent_pos=agent_pos,
        current_step=jnp.array(0, dtype=jnp.int32),
        res_pos=res_pos,
        res_active=jnp.ones(num_res, dtype=jnp.bool_),
        res_cons_count=jnp.zeros(num_res, dtype=jnp.int32),
        res_reg_timer=jnp.zeros(num_res, dtype=jnp.int32),
        pred_pos=pred_pos,
        pred_state=jnp.zeros(num_pred, dtype=jnp.int32), # PATROL
        pred_stamina=jnp.full(num_pred, params.pred_max_stamina, dtype=jnp.float32),
        pred_move_timer=jnp.zeros(num_pred, dtype=jnp.int32),
        obs_pos=obs_pos,
        satiation=jnp.array(satiation, dtype=jnp.float32),
        injury_level=jnp.array(injury, dtype=jnp.float32),
        injury_buffer=injury_buffer,
        terminated=jnp.array(False, dtype=jnp.bool_),
        key=key
    )
    
    return state

