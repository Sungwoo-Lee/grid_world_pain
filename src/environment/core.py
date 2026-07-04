import jax
import jax.numpy as jnp
from .state import EnvState, EnvParams

def move_agent(pos: jnp.ndarray, action: int, obs_pos: jnp.ndarray, obs_blocking: jnp.ndarray, params: EnvParams,
               obs_active: jnp.ndarray = None) -> jnp.ndarray:
    """Calculates New Agent position based on action, considering obstacles.

    obs_active: per-episode obstacle activation mask (PER_EPISODE_ENV_VARIANCE).
    When provided, inactive obstacles (obs_active=False) are transparent to
    collision. When None (caller did not pass it), falls back to all-True
    (byte-identical to pre-feature behaviour for degenerate-range configs).
    """
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

    # Obstacle collision check — AND with obs_active so inactive obstacles are transparent
    if obs_active is None:
        _eff_blocking = obs_blocking
    else:
        _eff_blocking = obs_blocking & obs_active
    is_collision = jnp.any(jnp.logical_and(
        jnp.all(obs_pos == new_pos, axis=-1),
        _eff_blocking
    ))

    # If collision, stay at current position
    final_pos = jnp.where(is_collision, pos, new_pos)
    return final_pos, is_collision

def calculate_drive(satiation, injury, params):
    """Calculates homeostatic drive (Euclidean distance to setpoint)."""
    target = jnp.array([params.setpoint, 0.0])
    current = jnp.stack([satiation, injury], axis=-1)
    return jnp.linalg.norm(current - target, axis=-1)

def update_body(state: EnvState, info: dict, params: EnvParams) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, bool]:
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

    # Roll the perceptual history buffer and write the new injury at slot 0.
    # Buffer is non-conditional on `with_injury`: if injury never updates, slot 0 stays at prev_injury (0 from reset).
    new_nociception_history = jnp.roll(state.nociception_history_buffer, 1).at[0].set(new_injury)
        
    # Termination check (Based on Nutrition and Injury)
    done = False
    if params.with_nutrition:
        done = jnp.where(new_nutrition <= 0.0, True, done)
        
    if params.with_injury:
        done = jnp.where(new_injury >= params.max_injury, True, done)
    else:
        # Instant death logic for levels without health system
        done = jnp.where(damage > 0, True, done)
    
    return new_satiation, new_nutrition, new_injury, new_buffer, new_nociception_history, new_rest_streak, done

def update_resources(res_active, res_reg_timer, res_cons_count, params,
                     res_allocated=None):
    """Updates resource timers and regeneration.

    res_allocated: per-episode allocation mask set once at jax_reset, never mutated
    (PER_EPISODE_ENV_VARIANCE fix).  Only slots that were allocated (K-mask=True at
    reset) are eligible for respawn.  Slots that were never activated (count_high − K
    inactive slots) have res_allocated=False and must stay inert all episode —
    otherwise update_resources would treat their timer=0 / res_active=False state as
    "eaten, please regrow" and revive them on step 1.

    When res_allocated is None (legacy callers) falls back to all-True so that
    degenerate-range configs (all slots allocated) are byte-identical to pre-fix.
    """
    # Regeneration: only tick the timer for slots that are allocated and inactive.
    if res_allocated is None:
        _allocated = jnp.ones_like(res_active, dtype=jnp.bool_)
    else:
        _allocated = res_allocated
    needs_reg_update = jnp.logical_and(jnp.logical_not(res_active), res_reg_timer > 0)
    new_reg_timer = jnp.where(needs_reg_update, res_reg_timer - 1, res_reg_timer)

    # Respawn where timer hits 0 AND the slot was actually allocated (fix: & _allocated).
    # Without the _allocated gate, slots that start with timer=0 and res_active=False
    # (the inactive K-mask slots) satisfy (~active & timer<=0) → True and revive on
    # step 1, silently collapsing per-episode count variance to count_high every episode.
    respawn_mask = jnp.logical_and(
        jnp.logical_and(jnp.logical_not(res_active), new_reg_timer <= 0),
        _allocated
    )
    new_active = jnp.where(respawn_mask, True, res_active)
    new_cons_count = jnp.where(respawn_mask, 0, res_cons_count)

    return new_active, new_reg_timer, new_cons_count, respawn_mask

def _hunt_step(hunt_pos, hunt_state, hunt_stamina, hunt_mt, hunt_at,
               hunt_detect, hunt_max_stamina, hunt_recovery, hunt_thresh,
               hunt_lose_interest, hunt_patrol, hunt_move_int,
               agent_pos, obs_pos, obs_blocking_for_collision, obs_hides_agent, key,
               grid_height: int = 10, grid_width: int = 10,
               obs_active: jnp.ndarray = None,
               attack_range_s: jnp.ndarray = None,
               attack_success_rate: jnp.ndarray = None,
               hunt_attack_delay: jnp.ndarray = None,
               hunt_active: jnp.ndarray = None,
               has_attack_feature: bool = False):
    """Hunt behaviour update (verbatim body of the old update_predators).

    Receives sliced arrays of shape (N_pred, ...) so PRNG draw shapes are
    byte-identical to the pre-refactor update_predators call. (B1 fix)

    B2 parity fix: the old update_predators used two different arrays:
      - obs_blocking (params.obs_blocking) for check_collision inside the function
      - params.obs_hides_agent for the agent_hidden computation inside the function
    Both must be passed separately to preserve byte-parity.

    obs_active: per-episode obstacle activation mask (PER_EPISODE_ENV_VARIANCE).
    Inactive obstacles are transparent to collision and do not conceal the agent.
    None → all-True (byte-identical to pre-feature code).

    Jump/pounce feature (attack_range_s / attack_success_rate / hunt_attack_delay /
    hunt_active / has_attack_feature): see
    docs/develop/active/env_entities/PREDATOR_JUMP_MECHANISM.md. `has_attack_feature`
    is a Python-static bool — when False (default), the jump block below is skipped
    entirely at TRACE time (no extra jax.random.split, no extra draws, no extra
    `where`s), so the disabled path is byte-identical to pre-feature `_hunt_step`.
    """
    # 1. Timers
    new_move_timer = hunt_mt - 1
    new_attack_timer = jnp.maximum(hunt_at - 1, 0)

    # Manhattan distance
    dist = jnp.sum(jnp.abs(hunt_pos - agent_pos), axis=-1)

    # Check if back in patrol area
    in_zone = jnp.logical_and(
        jnp.logical_and(hunt_pos[:, 0] >= hunt_patrol[:, 0], hunt_pos[:, 0] <= hunt_patrol[:, 2]),
        jnp.logical_and(hunt_pos[:, 1] >= hunt_patrol[:, 1], hunt_pos[:, 1] <= hunt_patrol[:, 3])
    )

    # 2. State Transitions
    # agent_hidden: uses obs_hides_agent (bush concealment) AND obs_active (inactive bushes don't hide).
    _eff_hides = obs_hides_agent if obs_active is None else (obs_hides_agent & obs_active)
    agent_hidden = jnp.any(jnp.logical_and(
        jnp.all(obs_pos == agent_pos, axis=-1),
        _eff_hides
    ))

    rested_enough = hunt_stamina >= (hunt_max_stamina * hunt_thresh)
    become_hunt = jnp.logical_and(
        jnp.logical_and(dist <= hunt_detect, rested_enough),
        jnp.logical_not(agent_hidden)
    )

    lose_interest = jnp.logical_or(
        jnp.logical_or(dist > hunt_detect * hunt_lose_interest, hunt_stamina <= 0),
        agent_hidden
    )

    next_state = hunt_state
    next_state = jnp.where(jnp.logical_and(hunt_state != 1, become_hunt), 1, next_state)
    next_state = jnp.where(jnp.logical_and(hunt_state == 1, lose_interest), 2, next_state)

    tr_return = (hunt_patrol[:, 0] + hunt_patrol[:, 2]) // 2
    tc_return = (hunt_patrol[:, 1] + hunt_patrol[:, 3]) // 2

    dist_to_center = jnp.sum(jnp.abs(hunt_pos - jnp.stack([tr_return, tc_return], axis=-1)), axis=-1)
    reentered_home = jnp.logical_and(next_state == 2, dist_to_center <= 2)
    next_state = jnp.where(reentered_home, 0, next_state)

    # 3. Movement
    should_move = jnp.logical_and(new_move_timer <= 0, new_attack_timer <= 0)

    key, subkey1, subkey2 = jax.random.split(key, 3)
    jitter_r = jax.random.randint(subkey1, (hunt_pos.shape[0],), -1, 2)
    jitter_c = jax.random.randint(subkey2, (hunt_pos.shape[0],), -1, 2)

    dr = jnp.zeros_like(hunt_pos[:, 0])
    dc = jnp.zeros_like(hunt_pos[:, 1])
    dr = jnp.where(next_state == 1, agent_pos[0] - hunt_pos[:, 0], dr)
    dc = jnp.where(next_state == 1, agent_pos[1] - hunt_pos[:, 1], dc)
    dr = jnp.where(next_state == 2, tr_return - hunt_pos[:, 0], dr)
    dc = jnp.where(next_state == 2, tc_return - hunt_pos[:, 1], dc)
    dr = jnp.where(next_state == 0, jitter_r, dr)
    dc = jnp.where(next_state == 0, jitter_c, dc)

    step_r = jnp.sign(dr)
    step_c = jnp.sign(dc)

    key, subkey3 = jax.random.split(key)
    rand_choice = jax.random.uniform(subkey3, (hunt_pos.shape[0],)) < 0.5

    final_move_r = jnp.where(jnp.logical_and(dr != 0, dc != 0), jnp.where(rand_choice, step_r, 0), step_r)
    final_move_c = jnp.where(jnp.logical_and(dr != 0, dc != 0), jnp.where(jnp.logical_not(rand_choice), step_c, 0), step_c)

    move_vec = jnp.stack([final_move_r, final_move_c], axis=-1)
    new_pos = jnp.where(should_move[:, None], hunt_pos + move_vec, hunt_pos)

    # 4. Spatial Bounds Clipping (Strict enforcement for all states)
    new_pos = jnp.stack([
        jnp.clip(new_pos[:, 0], hunt_patrol[:, 0], hunt_patrol[:, 2]),
        jnp.clip(new_pos[:, 1], hunt_patrol[:, 1], hunt_patrol[:, 3])
    ], axis=-1)

    # Hard Grid Boundaries (Always enforced)
    new_pos = jnp.clip(new_pos, 0, jnp.stack([grid_height - 1, grid_width - 1]))

    # Obstacle Collision — uses obs_blocking_for_collision AND obs_active (inactive obstacles transparent).
    _eff_blocking_hunt = obs_blocking_for_collision if obs_active is None else (obs_blocking_for_collision & obs_active)
    def check_collision(p_pos, old_p_pos):
        is_coll = jnp.any(jnp.logical_and(jnp.all(obs_pos == p_pos, axis=-1), _eff_blocking_hunt))
        return jnp.where(is_coll, old_p_pos, p_pos)

    new_pos = jax.vmap(check_collision)(new_pos, hunt_pos)

    # ── Jump / pounce override (predator lunge attack) ──────────────────────
    # Guarded by the STATIC has_attack_feature bool: when False (default / every
    # existing config), this entire block is skipped at TRACE time — no extra
    # jax.random.split, no extra draws, no extra `where`s — so the disabled path
    # is byte-identical to pre-feature `_hunt_step`. See
    # docs/develop/active/env_entities/PREDATOR_JUMP_MECHANISM.md.
    if has_attack_feature:
        jump_attempted = (
            (next_state == 1) & (dist <= attack_range_s) & jnp.logical_not(agent_hidden)
            & (new_attack_timer <= 0) & (attack_range_s > 0) & hunt_active
        )

        # Tail key — the `key` left over after `subkey3` above is otherwise
        # discarded, so these draws cannot perturb jitter_r/jitter_c/rand_choice
        # (parity-safe: the disabled path never reaches this branch at all).
        key, jkey_succ, jkey_nbr = jax.random.split(key, 3)
        success = jax.random.uniform(jkey_succ, (hunt_pos.shape[0],)) < attack_success_rate
        jump_success = jump_attempted & success

        # Miss target: uniformly-random VALID Chebyshev-1 neighbour of the agent
        # (same 8-cell candidate set for every predator). Bush (non-blocking)
        # cells ARE valid miss targets — only out-of-bounds / blocking obstacles
        # are excluded (Fork F3). Fallback: stay put if no neighbour is valid.
        offsets = jnp.array([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],           [0, 1],
            [1, -1],  [1, 0],  [1, 1],
        ], dtype=jnp.int32)
        cand = agent_pos[None, :] + offsets                                    # (8, 2)
        in_bounds = (
            (cand[:, 0] >= 0) & (cand[:, 0] < grid_height)
            & (cand[:, 1] >= 0) & (cand[:, 1] < grid_width)
        )
        blocked = jax.vmap(
            lambda c: jnp.any(jnp.all(obs_pos == c, axis=-1) & _eff_blocking_hunt)
        )(cand)
        valid = in_bounds & jnp.logical_not(blocked)                           # (8,)
        score = jnp.where(
            valid[None, :], jax.random.uniform(jkey_nbr, (hunt_pos.shape[0], 8)), -1.0
        )
        pick = jnp.argmax(score, axis=-1)                                      # (N_pred,)
        miss_pos = jnp.where(jnp.any(valid), cand[pick], hunt_pos)             # fallback: stay put

        jump_pos = jnp.where(
            jump_success[:, None], jnp.broadcast_to(agent_pos, hunt_pos.shape), miss_pos
        )
        new_pos = jnp.where(jump_attempted[:, None], jump_pos, new_pos)        # override AFTER clips
        # Cooldown on ANY attempt (hit OR miss).
        new_attack_timer = jnp.where(jump_attempted, hunt_attack_delay, new_attack_timer)

    # Reset timer
    new_move_timer = jnp.where(should_move, hunt_move_int, new_move_timer)

    new_stamina = jnp.where(next_state == 1, hunt_stamina - 1.0, hunt_stamina + hunt_recovery)
    new_stamina = jnp.clip(new_stamina, 0.0, hunt_max_stamina)

    return new_pos, next_state, new_stamina, new_move_timer, new_attack_timer


def _wander_step(wand_pos, wand_mt, wand_patrol, wand_move_int, obs_pos, obs_blocking, key,
                 grid_height: int = 10, grid_width: int = 10,
                 obs_active: jnp.ndarray = None):
    """Wander behaviour update (verbatim body of the old update_neutral_animals).

    Receives sliced arrays of shape (N_neutral, ...) so PRNG draw shapes are
    byte-identical to the pre-refactor update_neutral_animals call. (B1 fix)

    obs_active: per-episode obstacle activation mask (PER_EPISODE_ENV_VARIANCE).
    Inactive obstacles are transparent to collision. None → all-True (byte-identical).
    """
    # 1. Timers
    new_move_timer = wand_mt - 1
    should_move = new_move_timer <= 0

    # 2. Random Movement (Jitter)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    jitter_r = jax.random.randint(subkey1, (wand_pos.shape[0],), -1, 2)
    jitter_c = jax.random.randint(subkey2, (wand_pos.shape[0],), -1, 2)

    move_vec = jnp.stack([jitter_r, jitter_c], axis=-1)
    new_pos = jnp.where(should_move[:, None], wand_pos + move_vec, wand_pos)

    # 3. Spatial Bounds Clipping (Patrol Area)
    new_pos = jnp.stack([
        jnp.clip(new_pos[:, 0], wand_patrol[:, 0], wand_patrol[:, 2]),
        jnp.clip(new_pos[:, 1], wand_patrol[:, 1], wand_patrol[:, 3])
    ], axis=-1)

    # Hard Grid Boundaries
    new_pos = jnp.clip(new_pos, 0, jnp.stack([grid_height - 1, grid_width - 1]))

    # Obstacle Collision — inactive obstacles are transparent
    _eff_blocking_wand = obs_blocking if obs_active is None else (obs_blocking & obs_active)
    def check_collision(p_pos, old_p_pos):
        is_coll = jnp.any(jnp.logical_and(jnp.all(obs_pos == p_pos, axis=-1), _eff_blocking_wand))
        return jnp.where(is_coll, old_p_pos, p_pos)

    new_pos = jax.vmap(check_collision)(new_pos, wand_pos)

    # Reset timer
    new_move_timer = jnp.where(should_move, wand_move_int, new_move_timer)

    return new_pos, new_move_timer


def update_animals(state: 'EnvState', agent_pos, params: 'EnvParams', hunt_key, wander_key):
    """Unified animal update with per-subset call pattern (B1 fix).

    Slices hunt and wander subsets statically from the unified animal arrays,
    calls _hunt_step / _wander_step with their original draw shapes
    (N_pred,) / (N_neutral,), and scatters results back. PRNG bytes are
    byte-identical to the old update_predators / update_neutral_animals calls.

    `hunt_key` feeds _hunt_step (was `predator_key`).
    `wander_key` feeds _wander_step (was `neutral_key`).
    """
    animal_pos         = state.animal_pos
    animal_state       = state.animal_state
    animal_stamina     = state.animal_stamina
    animal_mt          = state.animal_move_timer
    animal_at          = state.animal_attack_timer
    detect_s           = state.animal_detect_sampled
    max_stam_s         = state.animal_max_stamina_sampled
    recovery_s         = state.animal_recovery_sampled
    hunt_thresh_s      = state.animal_hunt_thresh_sampled
    lose_int_s         = state.animal_lose_interest_sampled

    new_pos     = animal_pos
    new_state   = animal_state
    new_stamina = animal_stamina
    new_mt      = animal_mt
    new_at      = animal_at

    # Effective blocking for animals: union of physical blocking and the
    # optional blocks_animals flag.  Default false → obs_blocking unchanged
    # (byte-identical to pre-feature behaviour when blocks_animals is all-False).
    obs_block_for_animals = params.obs_blocking | params.obs_blocks_animals

    # ── Branch A: HUNT subset ────────────────────────────────────────────────
    if len(params.hunt_idx) > 0:
        h_idx = jnp.array(params.hunt_idx, dtype=jnp.int32)

        new_hunt_pos, new_hunt_state, new_hunt_stamina, new_hunt_mt, new_hunt_at = _hunt_step(
            animal_pos[h_idx],
            animal_state[h_idx],
            animal_stamina[h_idx],
            animal_mt[h_idx],
            animal_at[h_idx],
            detect_s[h_idx],
            max_stam_s[h_idx],
            recovery_s[h_idx],
            hunt_thresh_s[h_idx],
            lose_int_s[h_idx],
            params.animal_patrol[h_idx],
            state.animal_move_int_sampled[h_idx],
            agent_pos,
            state.obs_pos,
            obs_block_for_animals,    # merged: obs_blocking | obs_blocks_animals
            params.obs_hides_agent,   # for agent_hidden (byte-parity with old internal params.obs_hides_agent)
            hunt_key,
            grid_height=params.height,
            grid_width=params.width,
            obs_active=state.obs_active,  # NEW: inactive obstacles transparent
            attack_range_s=state.animal_attack_range_sampled[h_idx],
            attack_success_rate=params.animal_attack_success_rate[h_idx],
            hunt_attack_delay=state.animal_attack_delay_sampled[h_idx],
            hunt_active=state.animal_active[h_idx],
            has_attack_feature=params.has_attack_feature,
        )
        # Scatter back
        new_pos     = new_pos.at[h_idx].set(new_hunt_pos)
        new_state   = new_state.at[h_idx].set(new_hunt_state)
        new_stamina = new_stamina.at[h_idx].set(new_hunt_stamina)
        new_mt      = new_mt.at[h_idx].set(new_hunt_mt)
        new_at      = new_at.at[h_idx].set(new_hunt_at)

    # ── Branch B: WANDER subset ──────────────────────────────────────────────
    if len(params.wander_idx) > 0:
        w_idx = jnp.array(params.wander_idx, dtype=jnp.int32)

        new_wand_pos, new_wand_mt = _wander_step(
            animal_pos[w_idx],
            animal_mt[w_idx],
            params.animal_patrol[w_idx],
            state.animal_move_int_sampled[w_idx],
            state.obs_pos,
            obs_block_for_animals,    # merged: obs_blocking | obs_blocks_animals
            wander_key,
            grid_height=params.height,
            grid_width=params.width,
            obs_active=state.obs_active,  # NEW: inactive obstacles transparent
        )
        new_pos = new_pos.at[w_idx].set(new_wand_pos)
        new_mt  = new_mt.at[w_idx].set(new_wand_mt)

    # ── Branch C: STATIC — pass-through, no PRNG draws ──────────────────────
    # (No scatter needed; positions stay as-is.)

    # ── Ghost-predator fix: re-park inactive slots off-grid every step ────────
    # Without this gate, the hunt/wander movement updates above un-park inactive
    # slots (animal_active=False) from their off-grid sentinel (height, width):
    # the hard-grid clip inside _hunt_step / _wander_step forces the position into
    # [0, h-1]×[0, w-1], so after step 1 the inactive slot lands at (h-1, w-1)
    # and visually chases the agent (with 0 damage — all harm gates already check
    # animal_active — but the slot appears on-grid).
    # Fix: after all subset movement updates, re-park any inactive slot back to the
    # off-grid sentinel.  jnp.where on a static boolean mask is vmap-safe and
    # recompile-safe.  For all-active configs animal_active is all-True, so
    # jnp.where(True, new_pos, off_grid) == new_pos — byte-identical, no-op.
    _off_grid = jnp.array([params.height, params.width], dtype=jnp.int32)
    new_pos = jnp.where(state.animal_active[:, None], new_pos, _off_grid[None, :])

    return new_pos, new_state, new_stamina, new_mt, new_at

@jax.jit
def jax_step(state: EnvState, action: int, params: EnvParams) -> tuple[EnvState, jnp.ndarray, jnp.ndarray, dict]:
    """Orchestrates a full environment step in JAX."""
    
    # 0. Split key for random events
    # Preserve today's 6-way split byte-for-byte (rename predator_key→hunt_key, neutral_key→wander_key).
    key, respawn_key, hunt_key, wander_key, damage_key, property_key = jax.random.split(state.key, 6)

    # 1. Resource Regeneration (before agent moves)
    # Pass res_allocated so inactive (never-existed) slots cannot revive (fix for
    # resource-revival blocker: PER_EPISODE_ENV_VARIANCE 2026-06-23).
    new_active, new_reg_timer, new_cons_count, respawn_mask = update_resources(
        state.res_active, state.res_reg_timer, state.res_cons_count, params,
        res_allocated=state.res_allocated
    )

    # Displace resources that just respawned
    num_res = params.res_type.shape[0]
    res_keys = jax.random.split(respawn_key, num_res)

    def sample_res_pos(rk, area):
        return jax.random.randint(rk, (2,), area[:2], area[2:])

    new_potential_pos = jax.vmap(sample_res_pos)(res_keys, params.res_spawn_area)
    # Only update position IF respawn_mask is true for that resource
    res_pos_after_reg = jnp.where(respawn_mask[:, None], new_potential_pos, state.res_pos)

    # Re-sample chemical property for respawned resources
    noise = jax.random.normal(property_key, shape=params.res_property.shape)
    new_sampled_prop = jnp.clip(params.res_property + params.res_property_std * noise, 0.0, 1.0)
    res_property_sampled_after_reg = jnp.where(
        respawn_mask[:, None], new_sampled_prop, state.res_property_sampled
    )

    # Re-sample visual property for respawned resources (independent stream: fold_in with 0x7150A1).
    visual_property_key_step = jax.random.fold_in(property_key, 0x7150A1)
    vis_noise = jax.random.normal(visual_property_key_step, shape=params.res_visual_property.shape)
    new_sampled_vis_prop = jnp.clip(
        params.res_visual_property + params.res_visual_property_std * vis_noise, 0.0, None
    )
    res_visual_property_sampled_after_reg = jnp.where(
        respawn_mask[:, None], new_sampled_vis_prop, state.res_visual_property_sampled
    )

    # 2. Agent Movement
    new_agent_pos, just_collided = move_agent(state.agent_pos, action, state.obs_pos, params.obs_blocking, params,
                                              obs_active=state.obs_active)

    # 3. Unified Animal Update (per-subset call pattern — B1 fix)
    # hunt_key feeds hunt subset (N_pred draw shapes, byte-identical to old predator_key).
    # wander_key feeds wander subset (N_neutral draw shapes, byte-identical to old neutral_key).
    new_animal_pos, new_animal_state, new_animal_stamina, new_animal_mt, new_animal_at = update_animals(
        state, new_agent_pos, params, hunt_key, wander_key
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
    
    # Hiding Predator interaction (Auto)
    is_hiding_predator = params.res_type == 1
    # Sample damage for each resource interaction
    sampled_res_damage = jax.random.uniform(damage_key, (params.res_type.shape[0],), 
                                           minval=params.res_damage[:, 0], 
                                           maxval=params.res_damage[:, 1])
    damage_res = jnp.sum(jnp.where(jnp.logical_and(interact_resource, is_hiding_predator), sampled_res_damage, 0.0))
    
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
        jnp.logical_and(interact_resource, is_hiding_predator),
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
    
    # Predator Damage (unified — B5 fix: use at_damaging for damage + hit_predator)
    # AND with animal_active so inactive (off-grid) animals cannot bite.
    at_animal = jnp.all(new_animal_pos == new_agent_pos, axis=-1)         # POST-step positions
    at_damaging = jnp.logical_and(at_animal, params.animal_is_damaging & state.animal_active)
    # Sample animal damage (preserving today's draw shape = all N animals, using damage_key)
    sampled_pred_damage = jax.random.uniform(damage_key, (params.animal_damage.shape[0],),
                                             minval=params.animal_damage[:, 0],
                                             maxval=params.animal_damage[:, 1])
    damage_pred = jnp.sum(jnp.where(at_damaging, sampled_pred_damage, 0.0))

    # Trigger Attack Delay for damaging animals that hit the agent.
    # Use per-episode sampled value (degenerate range = scalar for backward compat).
    new_animal_at = jnp.where(at_damaging, state.animal_attack_delay_sampled, new_animal_at)

    # Strike-and-retreat: opt-in animals lose all stamina on contact, which makes
    # the existing hunt cycle disengage (HUNT→RETURN), retreat to patrol centre,
    # recover, and re-engage. No-op (where(False, ...)) for animals that did not
    # opt in or did not contact this step; draws no PRNG → byte-parity preserved.
    new_animal_stamina = jnp.where(
        at_animal & params.animal_disengage_on_contact, 0.0, new_animal_stamina
    )

    # Rock/Obstacle Damage
    # 1. Overlap damage (non-blocking rocks at current pos) — AND with obs_active
    at_obs = jnp.all(state.obs_pos == new_agent_pos, axis=-1)
    # Sample obstacle damage
    sampled_obs_damage = jax.random.uniform(damage_key, (params.obs_damage.shape[0],),
                                           minval=params.obs_damage[:, 0],
                                           maxval=params.obs_damage[:, 1])
    damage_obs_overlap = jnp.sum(jnp.where(
        jnp.logical_and(at_obs, jnp.logical_not(params.obs_blocking)) & state.obs_active,
        sampled_obs_damage, 0.0))

    # 2. Collision damage (blocking rocks) — AND with obs_active
    at_attempted_obs = jnp.all(state.obs_pos == attempted_pos, axis=-1)
    damage_obs_collision = jnp.where(just_collided, jnp.max(jnp.where(at_attempted_obs & state.obs_active, sampled_obs_damage, 0.0), initial=0.0), 0.0)

    # Calculate collision NOC intensity for sensing — inactive obstacles emit no noc
    collision_noc = jnp.where(just_collided, jnp.max(jnp.where(at_attempted_obs & state.obs_active, params.obs_nociception, 0.0), initial=0.0), 0.0)
    
    total_damage = damage_res + damage_pred + damage_obs_overlap + damage_obs_collision
    
    # 5. Body Update
    # rested is always action 4 if enabled
    rested = jnp.logical_and(params.rest_action_enabled, action == 4)
    
    damage_hiding_predator = damage_res
    
    # hit_neutral uses PRE-step animal positions (B5 fix: preserves today's pre/post asymmetry).
    # Today's code: hit_predator uses new_pred_pos (post-move); hit_neutral uses state.neutral_pos (pre-move).
    # We reproduce this exactly:
    #   at_damaging (above) → POST-step → hit_predator
    #   at_neutral_pre → PRE-step state.animal_pos masked by ~animal_is_damaging → hit_neutral
    at_neutral_pre = (
        jnp.logical_and(
            jnp.all(state.animal_pos == new_agent_pos, axis=-1),
            (~params.animal_is_damaging) & state.animal_active
        )
        if state.animal_pos.shape[0] > 0 else jnp.zeros(0, dtype=jnp.bool_)
    )

    info = {
        'ate_food': ate_food,
        'damage': total_damage,
        'damage_hiding_predator': damage_hiding_predator,
        'damage_predator': damage_pred,
        'damage_obstacle': damage_obs_overlap + damage_obs_collision,
        'rested': rested,
        'hit_hiding_predator': jnp.any(jnp.logical_and(interact_resource, is_hiding_predator)),
        'hit_predator': jnp.any(at_damaging),
        'hit_neutral': jnp.any(at_neutral_pre) if state.animal_pos.shape[0] > 0 else jnp.array(False),
    }
    
    new_satiation, new_nutrition, new_injury, next_injury_buffer, next_nociception_history, new_rest_streak, done = update_body(state, info, params)
    # `done` here is REAL DEATH only (starvation / over-eating / injury). update_body does not know
    # about the step clock, so it never fires on a timeout. Capture it BEFORE the truncation merge
    # below so the death_penalty can be gated on real death and NOT on surviving to the step limit.
    # Finding B — see docs/develop/active/diagnosis/v3_pipeline_correctness_diagnosis.md and
    # docs/develop/active/issues/FIX_TRUNCATION_TREATED_AS_DEATH.md.
    real_death = done

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
    # `done` below is the EPISODE-END flag: real death OR timeout. It is used ONLY for episode reset,
    # hidden-state reset, and boundary bookkeeping. It must NOT gate the death_penalty — surviving to
    # max_steps (truncation, reason == 1) is the SUCCESS outcome of a survival task and must not be
    # punished like death. The penalty is gated on `real_death` (captured above). Finding B.
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
        # Death penalty gated on REAL DEATH only (starvation / over-eating / injury), NOT on `done`.
        # Timeout / truncation (reason == 1) keeps just the normal homeostatic step value. Finding B.
        reward_homeostatic = jnp.where(real_death, reward_homeostatic - params.death_penalty, reward_homeostatic)
    else:
        reward_extrinsic = jnp.where(ate_food, 1.0, 0.0)
        # Death penalty gated on REAL DEATH only — NOT on timeout. Finding B.
        reward_extrinsic = jnp.where(real_death, -params.death_penalty, reward_extrinsic)
    
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
    # M3 zero-N fallback: preserve today's `if state.pred_pos.shape[0] > 0 else 99.0` pattern.
    dist_to_food = jnp.min(jnp.where(jnp.logical_and(state.res_active, params.res_type == 0), jnp.linalg.norm(state.res_pos - new_agent_pos, axis=-1), 99.0)) if state.res_pos.shape[0] > 0 else 99.0
    dist_to_hiding_predator = jnp.min(jnp.where(jnp.logical_and(state.res_active, params.res_type == 1), jnp.linalg.norm(state.res_pos - new_agent_pos, axis=-1), 99.0)) if state.res_pos.shape[0] > 0 else 99.0

    # Unified animal distances
    dist_per_animal = (
        jnp.linalg.norm(state.animal_pos - new_agent_pos, axis=-1)
        if state.animal_pos.shape[0] > 0
        else jnp.zeros((0,), dtype=jnp.float32)
    )
    # Legacy aliases — kept for one release cycle (C5); computed from per-class masks.
    # Mask inactive animals OUT of the distance min (they should read 99.0 = "absent").
    if len(params.predator_indices) > 0:
        pred_mask = params.animal_is_damaging & state.animal_active
        dist_to_pred = jnp.min(jnp.where(pred_mask, dist_per_animal, 99.0))
        dist_per_predator = dist_per_animal[jnp.array(params.predator_indices, dtype=jnp.int32)]
    else:
        dist_to_pred = 99.0
        dist_per_predator = jnp.zeros((0,), dtype=jnp.float32)

    if len(params.neutral_indices) > 0:
        neutral_mask = (~params.animal_is_damaging) & state.animal_active
        dist_to_neutral = jnp.min(jnp.where(neutral_mask, dist_per_animal, 99.0))
        dist_per_neutral = dist_per_animal[jnp.array(params.neutral_indices, dtype=jnp.int32)]
    else:
        dist_to_neutral = 99.0
        dist_per_neutral = jnp.zeros((0,), dtype=jnp.float32)

    info['dist_to_food'] = dist_to_food
    info['dist_to_pred'] = dist_to_pred
    info['dist_to_neutral'] = dist_to_neutral
    info['dist_to_hiding_predator'] = dist_to_hiding_predator
    info['dist_per_neutral'] = dist_per_neutral
    info['dist_per_predator'] = dist_per_predator
    info['dist_per_animal'] = dist_per_animal

    # Bush occupancy: True iff agent is standing on an obstacle marked hides_agent.
    # Mirrors the agent_hidden computation inside update_predators (line ~150);
    # recomputed here at minimal cost because EnvParams is in scope and we want it on `info`.
    # Uses new_agent_pos (post-step position) — correct for M2's "agent dives into bush" semantics.
    # AND with obs_active: inactive bushes do not conceal the agent.
    agent_in_bush = jnp.any(jnp.logical_and(
        jnp.all(state.obs_pos == new_agent_pos, axis=-1),
        params.obs_hides_agent & state.obs_active
    )) if state.obs_pos.shape[0] > 0 else jnp.array(False)
    info['agent_in_bush'] = agent_in_bush

    # 7. Final State
    new_state = state._replace(
        agent_pos=new_agent_pos,
        current_step=next_step,
        res_pos=res_pos_after_reg,
        res_active=final_active,
        res_allocated=state.res_allocated,  # never mutated: carry through unchanged
        res_reg_timer=next_reg_timer,
        res_cons_count=next_cons_count,
        res_property_sampled=res_property_sampled_after_reg,
        res_visual_property_sampled=res_visual_property_sampled_after_reg,
        # Unified animal fields (positions + state mutate; sampled distributional fields unchanged in step)
        animal_pos=new_animal_pos,
        animal_state=new_animal_state,
        animal_stamina=new_animal_stamina,
        animal_move_timer=new_animal_mt,
        animal_attack_timer=new_animal_at,
        animal_property_sampled=state.animal_property_sampled,
        animal_visual_property_sampled=state.animal_visual_property_sampled,
        animal_detect_sampled=state.animal_detect_sampled,
        animal_max_stamina_sampled=state.animal_max_stamina_sampled,
        animal_recovery_sampled=state.animal_recovery_sampled,
        animal_hunt_thresh_sampled=state.animal_hunt_thresh_sampled,
        animal_lose_interest_sampled=state.animal_lose_interest_sampled,
        animal_move_int_sampled=state.animal_move_int_sampled,
        animal_attack_delay_sampled=state.animal_attack_delay_sampled,
        animal_attack_range_sampled=state.animal_attack_range_sampled,
        # Per-episode activation masks: constant within an episode (set at reset, unchanged by step)
        animal_active=state.animal_active,
        obs_active=state.obs_active,
        obs_visual_property_sampled=state.obs_visual_property_sampled,
        satiation=new_satiation,
        nutrition=new_nutrition,
        injury_level=new_injury,
        injury_buffer=next_injury_buffer,
        nociception_history_buffer=next_nociception_history,
        last_collision_noc=collision_noc,
        rest_streak=new_rest_streak,
        terminated=done,
        key=key,
        last_action=jnp.array(action, dtype=jnp.int32),
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
    """Functional reset for the JAX environment (v2.0 — unified animal entity).

    Placement: vmap sampling + resolve_overlaps_global (per_entity mode) or
    lax.scan over type groups (per_type mode).

    N1 fix: entity concatenation order for the overlap-resolve scan is preserved
      as [res, pred, obs, neutral] (pred = predator-class animals; neutral =
      neutral-class animals). Positions are sliced back in the same order.
    N2 fix: property-key split is 4-way (prop_key_res / prop_key_pred /
      prop_key_obs / prop_key_neutral). The predator-class / neutral-class subsets
      of animal_property are sampled with prop_key_pred / prop_key_neutral
      respectively, then concatenated in predator-first order.
    N3 fix: placement_key split remains 6-way (res, pred, obs, neutral, resolve).
    """
    # Keep the original 5-way outer split byte-for-byte (N3 fix).
    # animal_episode_key is derived from property_key via fold_in so the
    # existing agent_key / placement_key / body_key / property_key streams
    # remain byte-identical to the pre-refactor code.
    key, agent_key, placement_key, body_key, property_key = jax.random.split(key, 5)
    # Derive per-episode animal sampling key without disturbing existing streams.
    animal_episode_key = jax.random.fold_in(property_key, 0xAE1)

    # 1. Agent Position
    random_pos = jax.random.randint(agent_key, (2,), 0, jnp.array([params.height, params.width]))
    agent_pos = jnp.where(params.random_start_pos, random_pos, params.start_pos)

    # 2. Entity Placement
    num_res = params.res_type.shape[0]
    # N1 fix: pred = animals with predator class; neutral = animals with neutral class.
    # Use static index tuples (pytree_node=False) to slice spawn areas per class.
    num_pred_class   = len(params.predator_indices)
    num_neutral_class = len(params.neutral_indices)
    num_obs = params.obs_blocking.shape[0]

    if params.placement_mode == 'per_entity':
        # ── Per-Entity Scan: vmap sample + resolve_overlaps_global ──
        # N3 fix: 6-way split (unchanged from pre-refactor split shape).
        placement_key, res_key, pred_key, obs_key, neutral_key, resolve_key = \
            jax.random.split(placement_key, 6)

        # Sample initial positions (may have overlaps)
        res_keys = jax.random.split(res_key, num_res) if num_res > 0 else jax.random.split(res_key, 1)[:0]
        res_pos = jax.vmap(lambda k, a: jax.random.randint(k, (2,), a[:2], a[2:]))(
            res_keys, params.res_spawn_area) if num_res > 0 else jnp.zeros((0, 2), dtype=jnp.int32)

        # N1 fix: per-class spawn areas from config_loader (pred_spawn_area_for_placement
        # is not stored on params; use params.animal_spawn_area + predator_indices).
        if num_pred_class > 0:
            pred_sa = params.animal_spawn_area[jnp.array(list(params.predator_indices), dtype=jnp.int32)]
            pred_keys = jax.random.split(pred_key, num_pred_class)
            pred_pos = jax.vmap(lambda k, a: jax.random.randint(k, (2,), a[:2], a[2:]))(
                pred_keys, pred_sa)
        else:
            pred_pos = jnp.zeros((0, 2), dtype=jnp.int32)

        obs_keys = jax.random.split(obs_key, num_obs) if num_obs > 0 else jax.random.split(obs_key, 1)[:0]
        obs_pos = jax.vmap(lambda k, a: jax.random.randint(k, (2,), a[:2], a[2:]))(
            obs_keys, params.obs_spawn_area) if num_obs > 0 else jnp.zeros((0, 2), dtype=jnp.int32)

        if num_neutral_class > 0:
            neutral_sa = params.animal_spawn_area[jnp.array(list(params.neutral_indices), dtype=jnp.int32)]
            neutral_keys = jax.random.split(neutral_key, num_neutral_class)
            neutral_pos = jax.vmap(lambda k, a: jax.random.randint(k, (2,), a[:2], a[2:]))(
                neutral_keys, neutral_sa)
        else:
            neutral_pos = jnp.zeros((0, 2), dtype=jnp.int32)

        # N1 fix: concat order [res, pred, obs, neutral] for resolve scan.
        all_positions = jnp.concatenate([res_pos, pred_pos, obs_pos, neutral_pos], axis=0)
        # Build matching spawn-area array (N1 fix: same ordering).
        if num_pred_class > 0:
            pred_sa_all = params.animal_spawn_area[jnp.array(list(params.predator_indices), dtype=jnp.int32)]
        else:
            pred_sa_all = jnp.zeros((0, 4), dtype=jnp.int32)
        if num_neutral_class > 0:
            neutral_sa_all = params.animal_spawn_area[jnp.array(list(params.neutral_indices), dtype=jnp.int32)]
        else:
            neutral_sa_all = jnp.zeros((0, 4), dtype=jnp.int32)
        all_spawn_areas = jnp.concatenate([
            params.res_spawn_area, pred_sa_all,
            params.obs_spawn_area, neutral_sa_all,
        ], axis=0)
        if all_positions.shape[0] > 0:
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
            valid = jnp.arange(params.max_per_type) < count
            for j in range(params.max_per_type):
                occ = jnp.where(valid[j], occ.at[type_flat[j]].set(True), occ)
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

    # 3. Split positions back (N1 fix: same [res, pred, obs, neutral] order).
    res_pos   = all_positions[:num_res]
    pred_pos  = all_positions[num_res:num_res + num_pred_class]
    obs_pos   = all_positions[num_res + num_pred_class:num_res + num_pred_class + num_obs]
    neutral_pos = all_positions[num_res + num_pred_class + num_obs:]

    # 4. Assemble unified animal_pos [N, 2] in predator-first order (N1 fix).
    N = params.animal_property.shape[0]
    if N > 0:
        # Scatter pred/neutral positions back into the unified [N, 2] array
        # using the static index tuples.
        animal_pos_init = jnp.zeros((N, 2), dtype=jnp.int32)
        if num_pred_class > 0:
            p_idx = jnp.array(list(params.predator_indices), dtype=jnp.int32)
            animal_pos_init = animal_pos_init.at[p_idx].set(pred_pos)
        if num_neutral_class > 0:
            n_idx = jnp.array(list(params.neutral_indices), dtype=jnp.int32)
            animal_pos_init = animal_pos_init.at[n_idx].set(neutral_pos)
    else:
        animal_pos_init = jnp.zeros((0, 2), dtype=jnp.int32)

    # 5. Body (Random start support)
    body_key1, body_key2, body_key3 = jax.random.split(body_key, 3)

    if params.random_start_nutrition:
        nutrition = jax.random.uniform(
            body_key2, (),
            minval=params.start_nutrition_low,
            maxval=params.start_nutrition_high)
    else:
        nutrition = params.start_nutrition

    fullness_ratio = jnp.clip(nutrition / params.max_nutrition, 0.0, 1.0)
    satiation = params.max_satiation * jnp.power(fullness_ratio, params.nutrition_to_satiation_scaling_factor)

    if params.random_start_injury:
        injury = jax.random.uniform(
            body_key3, (),
            minval=params.start_injury_low,
            maxval=params.start_injury_high)
    else:
        injury = 0.0

    injury_buffer = jnp.zeros(params.smoothing_duration)
    nociception_history_buffer = jnp.zeros(params.interoceptive_kernel_length)

    # 6. Property sampling (N2 fix: 4-way split, pred/neutral sampled separately).
    prop_key_res, prop_key_pred, prop_key_obs, prop_key_neutral = jax.random.split(property_key, 4)

    def _sample_property(sub_key, mean, std):
        noise = jax.random.normal(sub_key, shape=mean.shape)
        return jnp.clip(mean + std * noise, 0.0, 1.0)

    res_property_sampled = _sample_property(prop_key_res, params.res_property, params.res_property_std)
    obs_property_sampled = _sample_property(prop_key_obs, params.obs_property, params.obs_property_std)

    # N2 fix: sample pred-class and neutral-class properties separately then
    # scatter back into the unified [N, vector_size] array (preserving byte-parity
    # with the pre-refactor prop_key_pred / prop_key_neutral draws).
    if N > 0:
        animal_property_sampled = jnp.zeros_like(params.animal_property)
        if num_pred_class > 0:
            p_idx = jnp.array(list(params.predator_indices), dtype=jnp.int32)
            pred_prop_mean = params.animal_property[p_idx]
            pred_prop_std  = params.animal_property_std[p_idx]
            pred_prop_sampled = _sample_property(prop_key_pred, pred_prop_mean, pred_prop_std)
            animal_property_sampled = animal_property_sampled.at[p_idx].set(pred_prop_sampled)
        if num_neutral_class > 0:
            n_idx = jnp.array(list(params.neutral_indices), dtype=jnp.int32)
            neutral_prop_mean = params.animal_property[n_idx]
            neutral_prop_std  = params.animal_property_std[n_idx]
            neutral_prop_sampled = _sample_property(prop_key_neutral, neutral_prop_mean, neutral_prop_std)
            animal_property_sampled = animal_property_sampled.at[n_idx].set(neutral_prop_sampled)
    else:
        animal_property_sampled = jnp.zeros((0, params.animal_property.shape[-1] if params.animal_property.shape[0] == 0 else params.animal_property.shape[-1]), dtype=jnp.float32)

    # 6b. Visual property sampling — INDEPENDENT stream via fold_in(property_key, 0x7150A1).
    # This constant is unique (not reusing 0xAE1 which is the animal_episode_key constant)
    # so the olfactory draws above are byte-unchanged by construction.
    # Clip at 0.0 only (no upper bound: visual intensities are unbounded; with std=0 → exactly mean).
    visual_property_key = jax.random.fold_in(property_key, 0x7150A1)
    vis_key_res, vis_key_pred, vis_key_obs, vis_key_neutral = jax.random.split(visual_property_key, 4)

    def _sample_visual_property(sub_key, mean, std):
        noise = jax.random.normal(sub_key, shape=mean.shape)
        return jnp.clip(mean + std * noise, 0.0, None)

    res_visual_property_sampled = _sample_visual_property(
        vis_key_res, params.res_visual_property, params.res_visual_property_std)
    obs_visual_property_sampled = _sample_visual_property(
        vis_key_obs, params.obs_visual_property, params.obs_visual_property_std)

    # Animals: same pred/neutral subset pattern as olfactory sampling
    if N > 0:
        animal_visual_property_sampled = jnp.zeros_like(params.animal_visual_property)
        if num_pred_class > 0:
            p_idx = jnp.array(list(params.predator_indices), dtype=jnp.int32)
            pred_vis_mean = params.animal_visual_property[p_idx]
            pred_vis_std  = params.animal_visual_property_std[p_idx]
            pred_vis_sampled = _sample_visual_property(vis_key_pred, pred_vis_mean, pred_vis_std)
            animal_visual_property_sampled = animal_visual_property_sampled.at[p_idx].set(pred_vis_sampled)
        if num_neutral_class > 0:
            n_idx = jnp.array(list(params.neutral_indices), dtype=jnp.int32)
            neutral_vis_mean = params.animal_visual_property[n_idx]
            neutral_vis_std  = params.animal_visual_property_std[n_idx]
            neutral_vis_sampled = _sample_visual_property(vis_key_neutral, neutral_vis_mean, neutral_vis_std)
            animal_visual_property_sampled = animal_visual_property_sampled.at[n_idx].set(neutral_vis_sampled)
    else:
        animal_visual_property_sampled = jnp.zeros(
            (0, params.animal_visual_property.shape[-1]), dtype=jnp.float32)

    # 7b. Per-episode count-range activation masks (NEW — PER_EPISODE_ENV_VARIANCE).
    #
    # For each entity class (res / animal / obs):
    #   - If ALL entries are degenerate (low == high), skip the K-draw entirely so
    #     the PRNG stream is byte-identical to pre-feature code (parity guard).
    #     The mask is all-True (same as the fixed-count semantics).
    #   - If ANY entry has a genuine range (low < high), derive a per-episode key
    #     via fold_in(property_key, <class-specific constant>) and draw K per entry.
    #     Slots [K:count_high] for that entry are set False and parked off-grid.
    #
    # Fold-in constants are chosen to not collide with existing uses
    # (0xAE1 = animal_episode_key, 0x7150A1 = visual_property_key):
    _COUNT_KEY_RES    = 0xC0A1  # resource activation
    _COUNT_KEY_ANIMAL = 0xC0A2  # animal activation
    _COUNT_KEY_OBS    = 0xC0A3  # obstacle activation

    def _build_activation_mask(count_low_arr, count_high_arr, entry_id_arr,
                               has_range: bool, fold_const: int, num_slots: int,
                               rng_key):
        """Return (mask [num_slots bool], positions_park [not used here]).

        When has_range=False (pure Python bool, static), returns all-True without
        any jax.random call so the PRNG stream is untouched.
        When has_range=True, draws K per entry and builds the mask.
        """
        if (not has_range) or num_slots == 0:
            return jnp.ones(num_slots, dtype=jnp.bool_)
        # Derive independent key for count-activation draws
        act_key = jax.random.fold_in(rng_key, fold_const)
        num_entries = count_low_arr.shape[0]
        # Draw K per entry: randint(act_key_i, (), low, high+1) → K_i in [low, high]
        entry_keys = jax.random.split(act_key, num_entries)
        def _draw_k(ek, lo, hi):
            # When lo==hi, uniform([lo, hi+1)) always returns lo (integer draw)
            return jax.random.randint(ek, (), lo, hi + 1)
        K_per_entry = jax.vmap(_draw_k)(entry_keys, count_low_arr, count_high_arr)
        # Build slot mask: slot s is active iff its within-entry rank < K_entry
        # entry_id_arr[s] = which entry slot s belongs to.
        # within_rank[s] = how many slots for the same entry came before slot s
        # = cumcount(entry_id_arr)[s]
        # Computed as: rank[s] = sum_{t < s} (entry_id_arr[t] == entry_id_arr[s])
        # Vectorized: (arange(num_slots)[:, None] > arange(num_slots)[None, :]) &
        #             (entry_id_arr[:, None] == entry_id_arr[None, :])  → too large.
        # Instead, use lax.scan to build cumcount:
        def _cumcount(carry, eid):
            counts = carry
            rank = counts[eid]
            counts = counts.at[eid].add(1)
            return counts, rank
        _, slot_rank = jax.lax.scan(
            _cumcount,
            jnp.zeros(num_entries, dtype=jnp.int32),
            entry_id_arr
        )
        # slot_rank[s] = rank of slot s within its entry (0-based)
        # Slot is active iff slot_rank[s] < K[entry_id[s]]
        slot_K = K_per_entry[entry_id_arr]  # [num_slots] — K for this slot's entry
        mask = slot_rank < slot_K
        return mask

    # Resource activation mask
    num_res_entries_for_mask = params.res_count_low.shape[0]
    res_activation_mask = _build_activation_mask(
        params.res_count_low, params.res_count_high, params.res_entry_id,
        params.has_res_range, _COUNT_KEY_RES, num_res,
        property_key
    )

    # Animal activation mask
    animal_activation_mask = _build_activation_mask(
        params.animal_count_low, params.animal_count_high, params.animal_entry_id,
        params.has_animal_range, _COUNT_KEY_ANIMAL, N,
        property_key
    )

    # Obstacle activation mask
    num_obs_slots_for_mask = params.obs_blocking.shape[0]
    obs_activation_mask = _build_activation_mask(
        params.obs_count_low, params.obs_count_high, params.obs_entry_id,
        params.has_obs_range, _COUNT_KEY_OBS, num_obs,
        property_key
    )

    # Park inactive slots off-grid: position (height, width) is outside [0,h-1]x[0,w-1]
    # so they can never overlap the agent, never collide, and are excluded from sensing.
    _off_grid = jnp.array([params.height, params.width], dtype=jnp.int32)

    if num_res > 0 and params.has_res_range:
        res_pos = jnp.where(res_activation_mask[:, None], res_pos, _off_grid[None, :])
    if N > 0 and params.has_animal_range:
        animal_pos_init = jnp.where(animal_activation_mask[:, None], animal_pos_init, _off_grid[None, :])
    if num_obs > 0 and params.has_obs_range:
        obs_pos = jnp.where(obs_activation_mask[:, None], obs_pos, _off_grid[None, :])

    # 7. Per-episode distributional sampling for the 4 float + 3 integer behavioural fields.
    #    7 independent draws per field — shape (N,) each.
    #    For wander/static entries the ranges are [0, 0] (from _load_animals);
    #    jax.random.uniform([0,0]) = 0.0 exactly for float fields; randint([s,s+1)) = s for int fields.
    if N > 0:
        ep_keys = jax.random.split(animal_episode_key, 7)  # split size UNCHANGED (parity)
        # detection_range: inclusive-integer randint([lo, hi+1)) — mirrors move_interval/attack_delay.
        animal_detect_sampled = jax.random.randint(
            ep_keys[0], (N,), params.animal_detect_low, params.animal_detect_high + 1)
        animal_max_stamina_sampled = jax.random.uniform(
            ep_keys[1], (N,), minval=params.animal_max_stamina_low, maxval=jnp.maximum(params.animal_max_stamina_high, params.animal_max_stamina_low))
        animal_recovery_sampled = jax.random.uniform(
            ep_keys[2], (N,), minval=params.animal_recovery_low, maxval=jnp.maximum(params.animal_recovery_high, params.animal_recovery_low))
        animal_hunt_thresh_sampled = jax.random.uniform(
            ep_keys[3], (N,), minval=params.animal_hunt_thresh_low, maxval=jnp.maximum(params.animal_hunt_thresh_high, params.animal_hunt_thresh_low))
        animal_lose_interest_sampled = jax.random.uniform(
            ep_keys[4], (N,), minval=params.animal_lose_interest_low, maxval=jnp.maximum(params.animal_lose_interest_high, params.animal_lose_interest_low))
        # Integer sampling: randint half-open [lo, hi+1) is inclusive on both ends.
        # Degenerate range (lo == hi): randint([s, s+1)) always yields s — byte-identical to scalar.
        animal_move_int_sampled = jax.random.randint(
            ep_keys[5], (N,), params.animal_move_int_low, params.animal_move_int_high + 1)
        animal_attack_delay_sampled = jax.random.randint(
            ep_keys[6], (N,), params.animal_attack_delay_low, params.animal_attack_delay_high + 1)
        # Jump/pounce feature: attack_range sampled from an INDEPENDENT fold_in
        # stream — NOT part of the size-7 ep_keys split above (widening it to 8
        # would change all seven existing draws for every config; see "PRNG tail
        # is free" / "size-locked at 7" in PREDATOR_JUMP_MECHANISM.md).
        attack_range_key = jax.random.fold_in(animal_episode_key, 0xA77AC7)
        # inclusive-integer randint([lo, hi+1)) — degenerate [s,s] still yields s (parity).
        animal_attack_range_sampled = jax.random.randint(
            attack_range_key, (N,), params.animal_attack_range_low, params.animal_attack_range_high + 1)
    else:
        animal_detect_sampled        = jnp.zeros(0, dtype=jnp.int32)
        animal_max_stamina_sampled   = jnp.zeros(0, dtype=jnp.float32)
        animal_recovery_sampled      = jnp.zeros(0, dtype=jnp.float32)
        animal_hunt_thresh_sampled   = jnp.zeros(0, dtype=jnp.float32)
        animal_lose_interest_sampled = jnp.zeros(0, dtype=jnp.float32)
        animal_move_int_sampled      = jnp.zeros(0, dtype=jnp.int32)
        animal_attack_delay_sampled  = jnp.zeros(0, dtype=jnp.int32)
        animal_attack_range_sampled  = jnp.zeros(0, dtype=jnp.int32)

    state = EnvState(
        agent_pos=agent_pos,
        current_step=jnp.array(0, dtype=jnp.int32),
        res_pos=res_pos,
        # res_active starts from the activation mask (not all-ones) so inactive slots
        # (count_high - K) start inactive. For degenerate-range configs, res_activation_mask
        # is all-True, so this is byte-identical to jnp.ones(num_res) → parity preserved.
        res_active=res_activation_mask,
        # res_allocated is the immutable per-episode allocation mask: set once here,
        # never modified by jax_step.  update_resources gates respawn on it to prevent
        # inactive (never-existed) slots from reviving after step 1 (revival-blocker fix).
        res_allocated=res_activation_mask,
        res_cons_count=jnp.zeros(num_res, dtype=jnp.int32),
        res_reg_timer=jnp.zeros(num_res, dtype=jnp.int32),
        res_property_sampled=res_property_sampled,
        res_visual_property_sampled=res_visual_property_sampled,
        # Unified animal fields
        animal_pos=animal_pos_init,
        animal_state=jnp.zeros(N, dtype=jnp.int32),           # PATROL=0
        animal_stamina=animal_max_stamina_sampled.copy() if N > 0 else jnp.zeros(0, dtype=jnp.float32),
        animal_move_timer=jnp.zeros(N, dtype=jnp.int32),
        animal_attack_timer=jnp.zeros(N, dtype=jnp.int32),
        animal_property_sampled=animal_property_sampled,
        animal_visual_property_sampled=animal_visual_property_sampled,
        animal_detect_sampled=animal_detect_sampled,
        animal_max_stamina_sampled=animal_max_stamina_sampled,
        animal_recovery_sampled=animal_recovery_sampled,
        animal_hunt_thresh_sampled=animal_hunt_thresh_sampled,
        animal_lose_interest_sampled=animal_lose_interest_sampled,
        animal_move_int_sampled=animal_move_int_sampled,
        animal_attack_delay_sampled=animal_attack_delay_sampled,
        animal_attack_range_sampled=animal_attack_range_sampled,
        # Per-episode activation masks (NEW — PER_EPISODE_ENV_VARIANCE)
        animal_active=animal_activation_mask,
        obs_pos=obs_pos,
        obs_property_sampled=obs_property_sampled,
        obs_visual_property_sampled=obs_visual_property_sampled,
        obs_active=obs_activation_mask,
        satiation=jnp.array(satiation, dtype=jnp.float32),
        nutrition=jnp.array(nutrition, dtype=jnp.float32),
        injury_level=jnp.array(injury, dtype=jnp.float32),
        injury_buffer=injury_buffer,
        nociception_history_buffer=nociception_history_buffer,
        last_collision_noc=jnp.array(0.0, dtype=jnp.float32),
        rest_streak=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False, dtype=jnp.bool_),
        key=key,
        last_action=jnp.array(4 if params.rest_action_enabled else 5, dtype=jnp.int32),
    )

    return state



