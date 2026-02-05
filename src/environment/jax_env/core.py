import jax
import jax.numpy as jnp
from .state import EnvState, EnvParams

def move_agent(pos: jnp.ndarray, action: int, params: EnvParams) -> jnp.ndarray:
    """Calculates New Agent position based on action."""
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
    return new_pos

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

def update_predators(pred_pos, pred_state, pred_stamina, pred_move_timer, agent_pos, params, key):
    """Updates predator states and positions."""
    # 1. Timers
    new_move_timer = pred_move_timer - 1
    
    # Manhattan distance
    dist = jnp.sum(jnp.abs(pred_pos - agent_pos), axis=-1)
    
    # 2. State Transitions (Only when move_timer <= 0)
    # HUNT transitions
    rested_enough = pred_stamina >= (params.pred_max_stamina * params.pred_hunt_thresh)
    become_hunt = jnp.logical_and(dist <= params.pred_detect, rested_enough)
    
    # Lose interest
    lose_interest = jnp.logical_or(dist > params.pred_detect * 2, pred_stamina <= 0)
    
    # New State logic
    next_state = pred_state
    next_state = jnp.where(jnp.logical_and(pred_state != 1, become_hunt), 1, next_state) # 1 = HUNT
    next_state = jnp.where(jnp.logical_and(pred_state == 1, lose_interest), 2, next_state) # 2 = RETURN (or 0 PATROL)
    
    # 3. Movement (Only when move_timer <= 0)
    # Simple deterministic pursuit for HUNT, return to center for RETURN
    should_move = new_move_timer <= 0
    
    # Pursuit vector
    dr = agent_pos[0] - pred_pos[:, 0]
    dc = agent_pos[1] - pred_pos[:, 1]
    
    # JAX stochasticity for diagonal moves
    key, subkey = jax.random.split(key)
    rand_choice = jax.random.uniform(subkey, (pred_pos.shape[0],)) < 0.5
    
    step_r = jnp.sign(dr)
    step_c = jnp.sign(dc)
    
    # Move priority
    move_r = jnp.where(dr != 0, step_r, 0)
    move_c = jnp.where(dc != 0, step_c, 0)
    
    # Resolve diagonal (simplified: use rand_choice)
    final_move_r = jnp.where(jnp.logical_and(dr != 0, dc != 0), jnp.where(rand_choice, move_r, 0), move_r)
    final_move_c = jnp.where(jnp.logical_and(dr != 0, dc != 0), jnp.where(jnp.logical_not(rand_choice), move_c, 0), move_c)
    
    # Apply move only if in HUNT state (simplified logic for now)
    new_pos = pred_pos
    move_vec = jnp.stack([final_move_r, final_move_c], axis=-1)
    new_pos = jnp.where(jnp.logical_and(should_move, next_state == 1)[:, None], pred_pos + move_vec, new_pos)
    
    # Clamp to grid
    new_pos = jnp.clip(new_pos, 0, jnp.array([params.height - 1, params.width - 1]))
    
    # Reset timer
    new_move_timer = jnp.where(should_move, params.pred_move_int, new_move_timer)
    
    # Stamina
    new_stamina = jnp.where(next_state == 1, pred_stamina - 1.0, pred_stamina + params.pred_recovery)
    new_stamina = jnp.clip(new_stamina, 0.0, params.pred_max_stamina)
    
    return new_pos, next_state, new_stamina, new_move_timer, key

def jax_step(state: EnvState, action: int, params: EnvParams) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, dict]:
    """Orchestrates a full environment step in JAX."""
    
    # 1. Resource Regeneration (before agent moves)
    new_active, new_reg_timer, new_cons_count, _ = update_resources(
        state.res_active, state.res_reg_timer, state.res_cons_count, params
    )
    
    # 2. Predator Update
    key, predator_key = jax.random.split(state.key)
    new_pred_pos, new_pred_state, new_pred_stamina, new_pred_move_timer, _ = update_predators(
        state.pred_pos, state.pred_state, state.pred_stamina, state.pred_move_timer, 
        state.agent_pos, params, predator_key
    )
    
    # 3. Agent Movement
    new_agent_pos = move_agent(state.agent_pos, action, params)
    
    # 4. Interaction Logic
    # Check overlaps with resources
    at_resource = jnp.all(state.res_pos == new_agent_pos, axis=-1)
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
    should_deactivate = jnp.logical_and(params.res_max_cons > 0, next_cons_count >= params.res_max_cons)
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
        'rested': rested
    }
    
    new_satiation, new_injury, next_injury_buffer, done = update_body(state, info, params)
    
    # Max Steps Truncation
    next_step = state.current_step + 1
    truncated = next_step >= params.max_steps
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
    
    # 4. Body (Random start support)
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
        satiation=jnp.array(satiation, dtype=jnp.float32),
        injury_level=jnp.array(injury, dtype=jnp.float32),
        injury_buffer=injury_buffer,
        terminated=jnp.array(False, dtype=jnp.bool_),
        key=key
    )
    
    return state

