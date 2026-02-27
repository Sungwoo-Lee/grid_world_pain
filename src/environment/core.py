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
    return final_pos, is_collision

def calculate_drive(satiation, injury, params):
    """Calculates homeostatic drive (Euclidean distance to setpoint)."""
    target = jnp.array([params.setpoint, 0.0])
    current = jnp.stack([satiation, injury], axis=-1)
    return jnp.linalg.norm(current - target, axis=-1)

def update_body(state: EnvState, info: dict, params: EnvParams) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, bool]:
    """Updates satiation, nutrition, and injury levels with streak-based recovery."""
    prev_nutrition = state.nutrition
    prev_injury = state.injury_level
    prev_rest_streak = state.rest_streak
    # --- Nutrition Dynamics (Linear Decay) ---
    if params.with_nutrition:
        # Nutrition decays linearly
        new_nutrition = prev_nutrition - params.metabolic_cost
        # Refill from food (immediate) - with consumption cost
        ate_food_gain = params.food_nutrition_gain - params.eating_nutrition_cost
        new_nutrition = jnp.where(info['ate_food'], new_nutrition + ate_food_gain, new_nutrition)
        new_nutrition = jnp.clip(new_nutrition, 0.0, params.max_nutrition)
    else:
        new_nutrition = prev_nutrition

    # --- Satiation Dynamics (Derived Non-linearly from Nutrition) ---
    if params.with_satiation:
        # Subjective fullness S = Max * (N/MaxN)^k
        fullness_ratio = jnp.clip(new_nutrition / params.max_nutrition, 0.0, 1.0)
        new_satiation = params.max_satiation * jnp.power(fullness_ratio, params.nutrition_to_satiation_scaling_factor)
    else:
        new_satiation = state.satiation

    # --- Injury Dynamics (Instant-start smoothing) ---
    damage = info['damage']
    if params.with_injury:
        # 1. Spread new damage across the buffer
        inc = damage / params.smoothing_duration
        temp_buffer = state.injury_buffer + inc
        
        # 2. Apply the first slice immediately
        applied_inc = temp_buffer[0]
        new_injury = prev_injury + applied_inc
        
        # 3. Shift the rest of the buffer for future steps
        new_buffer = jnp.roll(temp_buffer, -1).at[-1].set(0.0)
        
        # --- Recovery Dynamics (Exponential recovery based on rest streak) ---
        # Update rest streak
        new_rest_streak = jnp.where(info['rested'], prev_rest_streak + 1, 0)
        
        # Calculate exponential recovery: base * (1 + accel)^(streak-1)
        # streak 1 -> mult 1.0 (base)
        # streak 2 -> mult 1.5 (base * 1.5)
        recovery_mult = jnp.power(1.0 + params.recovery_accel_rate, (jnp.maximum(new_rest_streak, 1) - 1).astype(jnp.float32))
        recovery_amount = params.recovery_base_rate * recovery_mult
        
        # Recovery only applies if resting and not currently taking net damage
        can_recover = jnp.logical_and(info['rested'], applied_inc <= 0)
        new_injury = jnp.where(can_recover, new_injury - recovery_amount, new_injury)
        
        new_injury = jnp.clip(new_injury, 0.0, params.max_injury)
    else:
        new_injury = prev_injury
        new_buffer = state.injury_buffer
        new_rest_streak = prev_rest_streak
        
    # Termination check (Based on Nutrition and Injury)
    done = False
    if params.with_nutrition:
        done = jnp.where(new_nutrition <= 0.0, True, done)
        
    if params.with_injury:
        done = jnp.where(new_injury >= params.max_injury, True, done)
    else:
        # Instant death logic for levels without health system
        done = jnp.where(damage > 0, True, done)
    
    return new_satiation, new_nutrition, new_injury, new_buffer, new_rest_streak, done 

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

def update_predators(pred_pos, pred_state, pred_stamina, pred_move_timer, pred_attack_timer, agent_pos, obs_pos, obs_blocking, params, key):
    """Updates predator states and positions, considering obstacles."""
    # 1. Timers
    new_move_timer = pred_move_timer - 1
    new_attack_timer = jnp.maximum(pred_attack_timer - 1, 0)
    
    # Manhattan distance
    dist = jnp.sum(jnp.abs(pred_pos - agent_pos), axis=-1)
    
    # Check if back in patrol area
    # params.pred_patrol is [num_pred, 4] -> [min_r, min_c, max_r, max_c]
    in_zone = jnp.logical_and(
        jnp.logical_and(pred_pos[:, 0] >= params.pred_patrol[:, 0], pred_pos[:, 0] <= params.pred_patrol[:, 2]),
        jnp.logical_and(pred_pos[:, 1] >= params.pred_patrol[:, 1], pred_pos[:, 1] <= params.pred_patrol[:, 3])
    )
    
    # 2. State Transitions (Only when move_timer <= 0)
    # Bush concealment: agent is hidden if standing on an obstacle with hides_agent=True
    agent_hidden = jnp.any(jnp.logical_and(
        jnp.all(obs_pos == agent_pos, axis=-1),
        params.obs_hides_agent
    ))
    
    # HUNT transitions (suppressed when agent is hidden)
    rested_enough = pred_stamina >= (params.pred_max_stamina * params.pred_hunt_thresh)
    become_hunt = jnp.logical_and(
        jnp.logical_and(dist <= params.pred_detect, rested_enough),
        jnp.logical_not(agent_hidden)
    )
    
    # Lose interest (also triggered when agent hides)
    lose_interest = jnp.logical_or(
        jnp.logical_or(dist > params.pred_detect * params.pred_lose_interest_mult, pred_stamina <= 0),
        agent_hidden
    )
    
    # New State logic
    next_state = pred_state
    # Transition to HUNT (1)
    next_state = jnp.where(jnp.logical_and(pred_state != 1, become_hunt), 1, next_state)
    # Transition to RETURN (2) (if patrol area exists) or PATROL (0)
    next_state = jnp.where(jnp.logical_and(pred_state == 1, lose_interest), 2, next_state)
    
    # Target center for RETURN state logic
    tr_return = (params.pred_patrol[:, 0] + params.pred_patrol[:, 2]) // 2
    tc_return = (params.pred_patrol[:, 1] + params.pred_patrol[:, 3]) // 2
    
    # Transition to PATROL (0) when close to center and in RETURN state
    dist_to_center = jnp.sum(jnp.abs(pred_pos - jnp.stack([tr_return, tc_return], axis=-1)), axis=-1)
    reentered_home = jnp.logical_and(next_state == 2, dist_to_center <= 2)
    next_state = jnp.where(reentered_home, 0, next_state)
    
    # 3. Movement (Only when move_timer <= 0 and not attacking/delayed)
    should_move = jnp.logical_and(new_move_timer <= 0, new_attack_timer <= 0)
    
    # Target calculations for different states
    # tr_return and tc_return moved up
    
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
    
    # 4. Spatial Bounds Clipping (Strict enforcement for all states)
    new_pos = jnp.stack([
        jnp.clip(new_pos[:, 0], params.pred_patrol[:, 0], params.pred_patrol[:, 2]),
        jnp.clip(new_pos[:, 1], params.pred_patrol[:, 1], params.pred_patrol[:, 3])
    ], axis=-1)
    
    # Hard Grid Boundaries (Always enforced)
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
    
    new_stamina = jnp.where(next_state == 1, pred_stamina - 1.0, pred_stamina + params.pred_recovery)
    new_stamina = jnp.clip(new_stamina, 0.0, params.pred_max_stamina)
    
    return new_pos, next_state, new_stamina, new_move_timer, new_attack_timer, key

def update_neutral_animals(neutral_pos, neutral_move_timer, obs_pos, obs_blocking, params, key):
    """Updates neutral animal positions (random patrol)."""
    # 1. Timers
    new_move_timer = neutral_move_timer - 1
    should_move = new_move_timer <= 0
    
    # 2. Random Movement (Jitter)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    jitter_r = jax.random.randint(subkey1, (neutral_pos.shape[0],), -1, 2)
    jitter_c = jax.random.randint(subkey2, (neutral_pos.shape[0],), -1, 2)
    
    move_vec = jnp.stack([jitter_r, jitter_c], axis=-1)
    new_pos = jnp.where(should_move[:, None], neutral_pos + move_vec, neutral_pos)
    
    # 3. Spatial Bounds Clipping (Patrol Area)
    new_pos = jnp.stack([
        jnp.clip(new_pos[:, 0], params.neutral_patrol[:, 0], params.neutral_patrol[:, 2]),
        jnp.clip(new_pos[:, 1], params.neutral_patrol[:, 1], params.neutral_patrol[:, 3])
    ], axis=-1)
    
    # Hard Grid Boundaries
    new_pos = jnp.clip(new_pos, 0, jnp.stack([params.height - 1, params.width - 1]))
    
    # Obstacle Collision
    def check_collision(p_pos, old_p_pos):
        is_coll = jnp.any(jnp.logical_and(jnp.all(obs_pos == p_pos, axis=-1), obs_blocking))
        return jnp.where(is_coll, old_p_pos, p_pos)
    
    new_pos = jax.vmap(check_collision)(new_pos, neutral_pos)
    
    # Reset timer
    new_move_timer = jnp.where(should_move, params.neutral_move_int, new_move_timer)
    
    return new_pos, new_move_timer, key

@jax.jit
def jax_step(state: EnvState, action: int, params: EnvParams) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, dict]:
    """Orchestrates a full environment step in JAX."""
    
    # 0. Split key for random events
    key, respawn_key, predator_key, neutral_key, damage_key = jax.random.split(state.key, 5)

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
    
    # 2. Agent Movement
    new_agent_pos, just_collided = move_agent(state.agent_pos, action, state.obs_pos, params.obs_blocking, params)

    # 3. Predator Update (Using the NEW agent position)
    new_pred_pos, new_pred_state, new_pred_stamina, new_pred_move_timer, new_pred_attack_timer, _ = update_predators(
        state.pred_pos, state.pred_state, state.pred_stamina, state.pred_move_timer, state.pred_attack_timer,
        new_agent_pos, state.obs_pos, params.obs_blocking, params, predator_key
    )
    
    # 3.5 Neutral Animal Update
    new_neutral_pos, new_neutral_move_timer, _ = update_neutral_animals(
        state.neutral_pos, state.neutral_move_timer, state.obs_pos, params.obs_blocking, params, neutral_key
    )
    
    # 4. Interaction Logic
    # Check overlaps with resources (using positions AFTER regeneration)
    at_resource = jnp.all(res_pos_after_reg == new_agent_pos, axis=-1)
    interact_resource = jnp.logical_and(at_resource, new_active)
    
    # Calculate attempted position for collision damage targeting
    moves_map = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1], [0, 0], [0, 0]], dtype=jnp.int32)
    attempted_pos = state.agent_pos + moves_map[jnp.clip(action, 0, 5).astype(jnp.int32)]
    # Clamp to grid boundaries
    attempted_pos = jnp.array([
        jnp.clip(attempted_pos[0], 0, params.height - 1),
        jnp.clip(attempted_pos[1], 0, params.width - 1)
    ])
    
    # Danger interaction (Auto)
    is_danger = params.res_type == 1
    # Sample damage for each resource interaction
    sampled_res_damage = jax.random.uniform(damage_key, (params.res_type.shape[0],), 
                                           minval=params.res_damage[:, 0], 
                                           maxval=params.res_damage[:, 1])
    damage_res = jnp.sum(jnp.where(jnp.logical_and(interact_resource, is_danger), sampled_res_damage, 0.0))
    
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
    # Sample predator damage
    sampled_pred_damage = jax.random.uniform(damage_key, (params.pred_damage.shape[0],),
                                            minval=params.pred_damage[:, 0],
                                            maxval=params.pred_damage[:, 1])
    damage_pred = jnp.sum(jnp.where(at_predator, sampled_pred_damage, 0.0))
    
    # Trigger Attack Delay for predators that hit the agent
    new_pred_attack_timer = jnp.where(at_predator, params.pred_attack_delay, new_pred_attack_timer)
    
    # Rock/Obstacle Damage
    # 1. Overlap damage (non-blocking rocks at current pos)
    at_obs = jnp.all(state.obs_pos == new_agent_pos, axis=-1)
    # Sample obstacle damage
    sampled_obs_damage = jax.random.uniform(damage_key, (params.obs_damage.shape[0],),
                                           minval=params.obs_damage[:, 0],
                                           maxval=params.obs_damage[:, 1])
    damage_obs_overlap = jnp.sum(jnp.where(jnp.logical_and(at_obs, jnp.logical_not(params.obs_blocking)), sampled_obs_damage, 0.0))
    
    # 2. Collision damage (blocking rocks)
    # Target the specific obstacle we hit
    at_attempted_obs = jnp.all(state.obs_pos == attempted_pos, axis=-1)
    damage_obs_collision = jnp.where(just_collided, jnp.max(jnp.where(at_attempted_obs, sampled_obs_damage, 0.0), initial=0.0), 0.0)
    
    # Calculate collision NOC intensity for sensing
    collision_noc = jnp.where(just_collided, jnp.max(jnp.where(at_attempted_obs, params.obs_nociception, 0.0), initial=0.0), 0.0)
    
    total_damage = damage_res + damage_pred + damage_obs_overlap + damage_obs_collision
    
    # 5. Body Update
    # rested is always action 4 if enabled
    rested = jnp.logical_and(params.rest_action_enabled, action == 4)
    
    damage_danger = damage_res
    
    info = {
        'ate_food': ate_food,
        'damage': total_damage,
        'damage_danger': damage_danger,
        'damage_predator': damage_pred,
        'damage_obstacle': damage_obs_overlap + damage_obs_collision,
        'rested': rested,
        'hit_danger': jnp.any(jnp.logical_and(interact_resource, is_danger)),
        'hit_predator': jnp.any(at_predator),
    }
    
    new_satiation, new_nutrition, new_injury, next_injury_buffer, new_rest_streak, done = update_body(state, info, params)
    
    # Max Steps Truncation
    next_step = state.current_step + 1
    truncated = next_step >= params.max_steps
    
    # Termination Reason (Integer codes for JIT compatibility)
    # 0: active, 1: max_steps, 2: starvation, 3: overeating, 4: injury
    reason = jnp.array(0, dtype=jnp.int32)
    reason = jnp.where(truncated, 1, reason)
    reason = jnp.where(new_nutrition <= 0.0, 2, reason)
    if params.overeating_death:
        reason = jnp.where(new_satiation >= params.max_satiation, 3, reason)
    reason = jnp.where(new_injury >= params.max_injury, 4, reason)
    
    info['termination_reason'] = reason
    done = jnp.logical_or(done, truncated)
    
    # 6. Reward (Homeostatic driven by Satiation)
    reward_homeostatic = 0.0
    reward_extrinsic = 0.0
    
    # Calculate components for analysis
    # drive = (1 - satiation/100)^2 + (injury/100)^2
    drive_hunger = jnp.power(1.0 - (new_satiation / params.max_satiation), 2)
    drive_injury = jnp.power(new_injury / params.max_injury, 2)
    
    if params.use_homeostatic_reward:
        prev_drive = calculate_drive(state.satiation, state.injury_level, params)
        curr_drive = calculate_drive(new_satiation, new_injury, params)
        reward_homeostatic = prev_drive - curr_drive
        # Death penalty based on Nutrition starvation
        reward_homeostatic = jnp.where(done, reward_homeostatic - params.death_penalty, reward_homeostatic)
    else:
        reward_extrinsic = jnp.where(ate_food, 1.0, 0.0)
        reward_extrinsic = jnp.where(done, -params.death_penalty, reward_extrinsic)
    
    reward = reward_homeostatic + reward_extrinsic
    # Apply eating penalty if ate food
    reward = jnp.where(ate_food, reward - params.eating_reward_penalty, reward)
    
    info['reward_homeostatic'] = reward_homeostatic
    info['reward_extrinsic'] = reward_extrinsic
    info['drive_hunger'] = drive_hunger
    info['drive_injury'] = drive_injury
    info['metabolic_drain'] = params.metabolic_cost
    info['event_collided'] = just_collided
    
    # GPU-side distance calculations for stats
    dist_to_food = jnp.min(jnp.where(jnp.logical_and(state.res_active, params.res_type == 0), jnp.linalg.norm(state.res_pos - new_agent_pos, axis=-1), 99.0)) if state.res_pos.shape[0] > 0 else 99.0
    dist_to_pred = jnp.min(jnp.linalg.norm(state.pred_pos - new_agent_pos, axis=-1)) if state.pred_pos.shape[0] > 0 else 99.0
    info['dist_to_food'] = dist_to_food
    info['dist_to_pred'] = dist_to_pred

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
        pred_attack_timer=new_pred_attack_timer,
        satiation=new_satiation,
        nutrition=new_nutrition,
        injury_level=new_injury,
        injury_buffer=next_injury_buffer,
        last_collision_noc=collision_noc,
        rest_streak=new_rest_streak,
        terminated=done,
        key=key,
        last_action=jnp.array(action, dtype=jnp.int32),
        neutral_pos=new_neutral_pos,
        neutral_move_timer=new_neutral_move_timer
    )
    
    return new_state, reward, done, info


def resolve_overlaps_global(
    all_positions: jnp.ndarray,
    all_spawn_areas: jnp.ndarray,
    grid_height: int,
    grid_width: int,
    key: jax.random.PRNGKey
) -> jnp.ndarray:
    """Resolve entity position overlaps via single-pass sequential scan.
    
    Uses a flat boolean occupancy mask. For each entity, if its cell is
    already taken, picks a random free cell within its spawn area using
    a pre-shuffled global permutation. All ops are JIT-compatible.
    """
    num_entities = all_positions.shape[0]
    total_cells = grid_height * grid_width
    occupancy = jnp.zeros(total_cells, dtype=jnp.bool_)
    global_perm = jax.random.permutation(key, total_cells)
    cell_rows = jnp.arange(total_cells) // grid_width
    cell_cols = jnp.arange(total_cells) % grid_width
    
    def resolve_one(carry, _unused):
        occ, positions, i = carry
        flat_idx = positions[i, 0] * grid_width + positions[i, 1]
        is_taken = occ[flat_idx]
        
        min_r, min_c, max_r, max_c = all_spawn_areas[i]
        in_area = (cell_rows >= min_r) & (cell_rows < max_r) & \
                  (cell_cols >= min_c) & (cell_cols < max_c)
        valid = in_area & (~occ)
        
        valid_in_perm = valid[global_perm]
        first_valid_mask = valid_in_perm & (jnp.cumsum(valid_in_perm) == 1)
        replacement_flat = jnp.where(first_valid_mask, global_perm, 0).sum()
        
        new_flat = jnp.where(is_taken, replacement_flat, flat_idx)
        new_r = new_flat // grid_width
        new_c = new_flat % grid_width
        
        positions = positions.at[i].set(jnp.array([new_r, new_c]))
        occ = occ.at[new_flat].set(True)
        return (occ, positions, i + 1), None
    
    init_carry = (occupancy, all_positions, jnp.array(0))
    (_, all_positions, _), _ = jax.lax.scan(
        resolve_one, init_carry, None, length=num_entities
    )
    return all_positions


def place_in_area(
    subkey: jax.random.PRNGKey,
    area: jnp.ndarray,
    occupancy: jnp.ndarray,
    num_entities: int,
    max_entities: int,
    grid_height: int,
    grid_width: int,
) -> tuple:
    """Place entities in a rectangular area, avoiding occupied cells.
    
    Uses a single global permutation filtered to valid candidates.
    Returns (positions [max_entities, 2], flat_indices [max_entities]).
    
    Positions beyond num_entities are padded with zeros.
    """
    total_cells = grid_height * grid_width
    min_r, min_c, max_r, max_c = area[0], area[1], area[2], area[3]
    
    cell_rows = jnp.arange(total_cells) // grid_width
    cell_cols = jnp.arange(total_cells) % grid_width
    
    # Valid = in area AND not occupied
    in_area = (cell_rows >= min_r) & (cell_rows < max_r) & \
              (cell_cols >= min_c) & (cell_cols < max_c)
    valid = in_area & (~occupancy)
    
    # Permute all cells, filter to valid
    perm = jax.random.permutation(subkey, total_cells)
    valid_in_perm = valid[perm]
    
    # Take first num_entities valid cells
    cumsum = jnp.cumsum(valid_in_perm)
    selected = valid_in_perm & (cumsum <= num_entities)
    
    # Extract flat indices, padded to max_entities
    selected_flat = jnp.where(selected, perm, total_cells)  # sentinel
    selected_flat = jnp.sort(selected_flat)[:max_entities]
    selected_flat = jnp.where(
        jnp.arange(max_entities) < num_entities,
        selected_flat,
        0  # unused padding
    )
    
    pos_r = selected_flat // grid_width
    pos_c = selected_flat % grid_width
    positions = jnp.stack([pos_r, pos_c], axis=-1)
    return positions, selected_flat


@jax.jit
def jax_reset(params: EnvParams, key: jax.random.PRNGKey) -> EnvState:
    """Functional reset for the JAX environment.
    
    Uses 4-phase overlap-free entity placement:
    Supports two placement modes (selected via config):
      per_entity: vmap sampling + sequential overlap scan (fast on small grids)
      per_type:   lax.scan over type groups (better for large grids)
    """
    key, agent_key, placement_key, body_key = jax.random.split(key, 4)
    
    # 1. Agent Position
    random_pos = jax.random.randint(agent_key, (2,), 0, jnp.array([params.height, params.width]))
    agent_pos = jnp.where(params.random_start_pos, random_pos, params.start_pos)
    
    # 2. Entity Placement
    num_res = params.res_type.shape[0]
    num_pred = params.pred_damage.shape[0]
    num_obs = params.obs_blocking.shape[0]
    num_neutral = params.neutral_property.shape[0]
    
    if params.placement_mode == 'per_entity':
        # ── Per-Entity Scan: vmap sample + resolve_overlaps_global ──
        placement_key, res_key, pred_key, obs_key, neutral_key, resolve_key = \
            jax.random.split(placement_key, 6)
        
        # Sample initial positions (may have overlaps)
        res_keys = jax.random.split(res_key, num_res)
        res_pos = jax.vmap(lambda k, a: jax.random.randint(k, (2,), a[:2], a[2:]))(
            res_keys, params.res_spawn_area)
        
        pred_keys = jax.random.split(pred_key, num_pred)
        pred_pos = jax.vmap(lambda k, a: jax.random.randint(k, (2,), a[:2], a[2:]))(
            pred_keys, params.pred_spawn_area)
        
        obs_keys = jax.random.split(obs_key, num_obs)
        obs_pos = jax.vmap(lambda k, a: jax.random.randint(k, (2,), a[:2], a[2:]))(
            obs_keys, params.obs_spawn_area)
        
        neutral_keys = jax.random.split(neutral_key, num_neutral)
        neutral_pos = jax.vmap(lambda k, a: jax.random.randint(k, (2,), a[:2], a[2:]))(
            neutral_keys, params.neutral_spawn_area)
        
        # Resolve all overlaps in one sequential scan
        all_positions = jnp.concatenate([res_pos, pred_pos, obs_pos, neutral_pos], axis=0)
        all_spawn_areas = jnp.concatenate([
            params.res_spawn_area, params.pred_spawn_area,
            params.obs_spawn_area, params.neutral_spawn_area
        ], axis=0)
        all_positions = resolve_overlaps_global(
            all_positions, all_spawn_areas, params.height, params.width, resolve_key
        )
        
    else:  # per_type
        # ── Type-Level: lax.scan over spawn-area groups ──
        all_positions = jnp.zeros((params.num_entities, 2), dtype=jnp.int32)
        total_cells = params.height * params.width
        occupancy = jnp.zeros(total_cells, dtype=jnp.bool_)
        
        def place_type_group(carry, type_idx):
            occ, positions, rng = carry
            rng, subkey = jax.random.split(rng)
            area = params.type_areas[type_idx]
            count = params.type_counts[type_idx]
            type_pos, type_flat = place_in_area(
                subkey, area, occ, count, params.max_per_type,
                params.height, params.width
            )
            # Per-cell occupancy update (safe, avoids batched .at padding bug)
            valid = jnp.arange(params.max_per_type) < count
            for j in range(params.max_per_type):
                occ = jnp.where(valid[j], occ.at[type_flat[j]].set(True), occ)
            # Per-entity position scatter
            entity_indices = params.type_entity_map[type_idx]
            for j in range(params.max_per_type):
                eidx = entity_indices[j]
                positions = jnp.where(
                    valid[j],
                    positions.at[eidx].set(type_pos[j]),
                    positions
                )
            return (occ, positions, rng), None
        
        placement_key, scan_key = jax.random.split(placement_key)
        (_, all_positions, _), _ = jax.lax.scan(
            place_type_group,
            (occupancy, all_positions, scan_key),
            jnp.arange(params.num_types)
        )
    
    # 3. Split positions back into per-type arrays
    res_pos = all_positions[:num_res]
    pred_pos = all_positions[num_res:num_res + num_pred]
    obs_pos = all_positions[num_res + num_pred:num_res + num_pred + num_obs]
    neutral_pos = all_positions[num_res + num_pred + num_obs:]
    
    # 6. Body (Random start support)
    body_key1, body_key2, body_key3 = jax.random.split(body_key, 3)
    
    if params.random_start_nutrition:
        min_start_nutr = params.max_nutrition / 2.0
        nutrition = jax.random.uniform(body_key2, (), minval=min_start_nutr, maxval=params.max_nutrition)
    else:
        nutrition = params.start_nutrition

    # Satiation is derived from nutrition
    fullness_ratio = jnp.clip(nutrition / params.max_nutrition, 0.0, 1.0)
    satiation = params.max_satiation * jnp.power(fullness_ratio, params.nutrition_to_satiation_scaling_factor)

    if params.random_start_injury:
        max_start_injury = params.max_injury / 2.0
        injury = jax.random.uniform(body_key3, (), minval=0.0, maxval=max_start_injury)
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
        pred_attack_timer=jnp.zeros(num_pred, dtype=jnp.int32),
        obs_pos=obs_pos,
        satiation=jnp.array(satiation, dtype=jnp.float32),
        nutrition=jnp.array(nutrition, dtype=jnp.float32),
        injury_level=jnp.array(injury, dtype=jnp.float32),
        injury_buffer=injury_buffer,
        last_collision_noc=jnp.array(0.0, dtype=jnp.float32),
        rest_streak=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False, dtype=jnp.bool_),
        key=key,
        last_action=jnp.array(4 if params.rest_action_enabled else 5, dtype=jnp.int32), # Default to Rest/Stay
        neutral_pos=neutral_pos,
        neutral_move_timer=jnp.zeros(num_neutral, dtype=jnp.int32)
    )
    
    return state



