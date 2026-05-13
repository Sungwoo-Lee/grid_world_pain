"""Per-episode scalar accumulators for the 20 Episode/* WandB keys.

Mirrors the pattern of src/behavior/accumulators.py and
src/behavior/distance_aggregator.py.  Pure numpy — no JAX, no PyTorch.

Both the JAX trainer (future) and the sheeprl bridge import from here.

API:
  EpisodeAccumulatorState   — per-env numpy arrays.
  make_episode_state()      — factory (zeroed arrays).
  episode_reset_env()       — episode-end reset for one env slot.
                              NOTE: episode_counter is NOT reset — it counts
                              cumulative completed episodes across resets.
  episode_step_update()     — accumulate per-step scalars.
  episode_finalise_episode() — compute final Episode/* dict at episode done.
  episode_wandb_keys()      — return the 20 WandB key strings (stable order).

Key mapping (JAX info dict → WandB key):
  reward                   → Episode/Reward (sum), Episode/Reward_Min, Episode/Reward_Max
  step_count (incremented) → Episode/Steps
  episode_counter          → Episode/Number
  termination_reason       → Episode/Term_{MaxSteps,Starvation,Overeating,Injury} (one-hot)
  damage                   → Episode/TotalDamage
  damage_predator          → Episode/DamagePredator
  damage_hiding_predator   → Episode/DamageDanger  (renamed at boundary)
  damage_obstacle          → Episode/DamageObstacle
  ate_food                 → Episode/FoodEaten
  hit_predator             → Episode/PredatorHits
  hit_hiding_predator      → Episode/DangerHits AND Episode/HidingPredatorHits
  rested                   → Episode/RestCount
  event_collided           → Episode/Collisions
  hit_neutral              → Episode/RabbitHits
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Termination reason codes (matches core.py jnp.where chain)
_TERM_MAX_STEPS   = 1
_TERM_STARVATION  = 2
_TERM_OVEREATING  = 3
_TERM_INJURY      = 4


@dataclass
class EpisodeAccumulatorState:
    """Per-env numpy accumulators for the 20 Episode/* keys.

    All arrays have shape ``[num_envs]``.

    ``episode_counter`` survives across ``episode_reset_env`` calls —
    it is only ever incremented (never zeroed) and tracks cumulative
    completed episodes for the lifetime of the wrapper instance.
    """

    num_envs: int

    # Reward statistics (per-episode)
    reward_sum: np.ndarray = field(init=False)
    reward_min: np.ndarray = field(init=False)
    reward_max: np.ndarray = field(init=False)

    # Step count (reset each episode)
    step_count: np.ndarray = field(init=False)

    # Cumulative episode counter (never reset)
    episode_counter: np.ndarray = field(init=False)

    # Damage accumulators
    total_damage:    np.ndarray = field(init=False)
    damage_predator: np.ndarray = field(init=False)
    damage_danger:   np.ndarray = field(init=False)   # source: damage_hiding_predator
    damage_obstacle: np.ndarray = field(init=False)

    # Event counters (bool → int per step)
    food_eaten:          np.ndarray = field(init=False)
    predator_hits:       np.ndarray = field(init=False)
    danger_hits:         np.ndarray = field(init=False)   # source: hit_hiding_predator
    rest_count:          np.ndarray = field(init=False)
    collisions:          np.ndarray = field(init=False)
    rabbit_hits:         np.ndarray = field(init=False)
    hiding_predator_hits: np.ndarray = field(init=False)  # same source as danger_hits

    def __post_init__(self):
        ne = self.num_envs
        self.reward_sum     = np.zeros(ne, dtype=np.float64)
        self.reward_min     = np.full(ne, np.inf, dtype=np.float64)
        self.reward_max     = np.full(ne, -np.inf, dtype=np.float64)
        self.step_count     = np.zeros(ne, dtype=np.int64)
        self.episode_counter = np.zeros(ne, dtype=np.int64)

        self.total_damage    = np.zeros(ne, dtype=np.float64)
        self.damage_predator = np.zeros(ne, dtype=np.float64)
        self.damage_danger   = np.zeros(ne, dtype=np.float64)
        self.damage_obstacle = np.zeros(ne, dtype=np.float64)

        self.food_eaten           = np.zeros(ne, dtype=np.int64)
        self.predator_hits        = np.zeros(ne, dtype=np.int64)
        self.danger_hits          = np.zeros(ne, dtype=np.int64)
        self.rest_count           = np.zeros(ne, dtype=np.int64)
        self.collisions           = np.zeros(ne, dtype=np.int64)
        self.rabbit_hits          = np.zeros(ne, dtype=np.int64)
        self.hiding_predator_hits = np.zeros(ne, dtype=np.int64)


def make_episode_state(num_envs: int) -> EpisodeAccumulatorState:
    """Factory — returns a zero-initialised EpisodeAccumulatorState."""
    return EpisodeAccumulatorState(num_envs=num_envs)


def episode_reset_env(state: EpisodeAccumulatorState, env_idx: int) -> None:
    """Reset per-episode accumulators for env slot *env_idx* at episode start.

    NOTE: ``episode_counter`` is intentionally NOT reset — it is a lifetime
    cumulative counter that records how many complete episodes this env slot
    has seen.
    """
    state.reward_sum[env_idx]  = 0.0
    state.reward_min[env_idx]  = np.inf
    state.reward_max[env_idx]  = -np.inf
    state.step_count[env_idx]  = 0

    state.total_damage[env_idx]    = 0.0
    state.damage_predator[env_idx] = 0.0
    state.damage_danger[env_idx]   = 0.0
    state.damage_obstacle[env_idx] = 0.0

    state.food_eaten[env_idx]           = 0
    state.predator_hits[env_idx]        = 0
    state.danger_hits[env_idx]          = 0
    state.rest_count[env_idx]           = 0
    state.collisions[env_idx]           = 0
    state.rabbit_hits[env_idx]          = 0
    state.hiding_predator_hits[env_idx] = 0


def episode_step_update(
    state: EpisodeAccumulatorState,
    info_np_t: dict,
    done_mask: np.ndarray,
) -> None:
    """Accumulate per-step scalars into state.

    Args:
        state:      EpisodeAccumulatorState to update in place.
        info_np_t:  Per-env scalar/array dict for this step.  Expected keys:
                      ``reward``               — float32 [num_envs]
                      ``damage``               — float32 [num_envs]
                      ``damage_predator``      — float32 [num_envs]
                      ``damage_hiding_predator`` — float32 [num_envs]
                      ``damage_obstacle``      — float32 [num_envs]
                      ``ate_food``             — bool [num_envs]
                      ``hit_predator``         — bool [num_envs]
                      ``hit_hiding_predator``  — bool [num_envs]
                      ``rested``               — bool [num_envs]
                      ``event_collided``       — bool [num_envs]
                      ``hit_neutral``          — bool [num_envs]
        done_mask:  bool [num_envs] — True for envs finishing an episode THIS step.
                    Passed for signature parity; not used inside the step update
                    itself — caller calls episode_reset_env *after* finalization.
    """
    r   = np.asarray(info_np_t['reward'], dtype=np.float64).reshape(-1)
    dmg = np.asarray(info_np_t['damage'], dtype=np.float64).reshape(-1)
    dp  = np.asarray(info_np_t['damage_predator'], dtype=np.float64).reshape(-1)
    dh  = np.asarray(info_np_t['damage_hiding_predator'], dtype=np.float64).reshape(-1)
    do_ = np.asarray(info_np_t['damage_obstacle'], dtype=np.float64).reshape(-1)

    af  = np.asarray(info_np_t['ate_food'], dtype=bool).reshape(-1)
    hp  = np.asarray(info_np_t['hit_predator'], dtype=bool).reshape(-1)
    hhp = np.asarray(info_np_t['hit_hiding_predator'], dtype=bool).reshape(-1)
    rst = np.asarray(info_np_t['rested'], dtype=bool).reshape(-1)
    col = np.asarray(info_np_t['event_collided'], dtype=bool).reshape(-1)
    rbt = np.asarray(info_np_t['hit_neutral'], dtype=bool).reshape(-1)

    state.reward_sum += r
    state.reward_min  = np.minimum(state.reward_min, r)
    state.reward_max  = np.maximum(state.reward_max, r)
    state.step_count += 1

    state.total_damage    += dmg
    state.damage_predator += dp
    state.damage_danger   += dh
    state.damage_obstacle += do_

    state.food_eaten           += af.astype(np.int64)
    state.predator_hits        += hp.astype(np.int64)
    state.danger_hits          += hhp.astype(np.int64)
    state.rest_count           += rst.astype(np.int64)
    state.collisions           += col.astype(np.int64)
    state.rabbit_hits          += rbt.astype(np.int64)
    state.hiding_predator_hits += hhp.astype(np.int64)


def episode_finalise_episode(
    state: EpisodeAccumulatorState,
    env_idx: int,
    terminated_reason: int,
) -> dict:
    """Compute the 20 WandB Episode/* scalars for env slot *env_idx*.

    Increments ``episode_counter[env_idx]`` as a side effect (called once
    per completed episode).

    Args:
        state:             EpisodeAccumulatorState.
        env_idx:           Which env slot is terminating.
        terminated_reason: Integer termination code from info_jax['termination_reason'].
                           0=active (should not be called), 1=max_steps,
                           2=starvation, 3=overeating, 4=injury.

    Returns:
        Flat dict of 20 ``Episode/*`` WandB keys → scalar float values.
    """
    i = env_idx
    state.episode_counter[i] += 1

    r = int(terminated_reason)

    out: dict = {}
    # Reward stats
    out['Episode/Reward']     = float(state.reward_sum[i])
    out['Episode/Reward_Min'] = float(state.reward_min[i]) if not np.isinf(state.reward_min[i]) else 0.0
    out['Episode/Reward_Max'] = float(state.reward_max[i]) if not np.isinf(state.reward_max[i]) else 0.0

    # Episode meta
    out['Episode/Steps']  = float(state.step_count[i])
    out['Episode/Number'] = float(state.episode_counter[i])

    # Termination one-hot
    out['Episode/Term_MaxSteps']  = 1.0 if r == _TERM_MAX_STEPS  else 0.0
    out['Episode/Term_Starvation']= 1.0 if r == _TERM_STARVATION else 0.0
    out['Episode/Term_Overeating']= 1.0 if r == _TERM_OVEREATING else 0.0
    out['Episode/Term_Injury']    = 1.0 if r == _TERM_INJURY      else 0.0

    # Damage
    out['Episode/TotalDamage']    = float(state.total_damage[i])
    out['Episode/DamagePredator'] = float(state.damage_predator[i])
    out['Episode/DamageDanger']   = float(state.damage_danger[i])
    out['Episode/DamageObstacle'] = float(state.damage_obstacle[i])

    # Interaction events
    out['Episode/FoodEaten']          = float(state.food_eaten[i])
    out['Episode/PredatorHits']       = float(state.predator_hits[i])
    out['Episode/DangerHits']         = float(state.danger_hits[i])
    out['Episode/RestCount']          = float(state.rest_count[i])
    out['Episode/Collisions']         = float(state.collisions[i])
    out['Episode/RabbitHits']         = float(state.rabbit_hits[i])
    out['Episode/HidingPredatorHits'] = float(state.hiding_predator_hits[i])

    return out


def episode_wandb_keys() -> list:
    """Return the 20 WandB key strings emitted by this module (stable order).

    Used by the sheeprl aggregator to register metrics at startup.
    """
    return [
        # Reward statistics
        'Episode/Reward',
        'Episode/Reward_Min',
        'Episode/Reward_Max',
        # Episode meta
        'Episode/Steps',
        'Episode/Number',
        # Termination one-hot flags
        'Episode/Term_MaxSteps',
        'Episode/Term_Starvation',
        'Episode/Term_Overeating',
        'Episode/Term_Injury',
        # Damage
        'Episode/TotalDamage',
        'Episode/DamagePredator',
        'Episode/DamageDanger',
        'Episode/DamageObstacle',
        # Interaction events
        'Episode/FoodEaten',
        'Episode/PredatorHits',
        'Episode/DangerHits',
        'Episode/RestCount',
        'Episode/Collisions',
        'Episode/RabbitHits',
        'Episode/HidingPredatorHits',
    ]
