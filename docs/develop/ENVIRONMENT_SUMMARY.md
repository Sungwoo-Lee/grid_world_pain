# GridWorld Pain - Project Reference

> **Last Updated**: 2026-03-12 | **Backend**: 100% JAX/Flax NNX |


## Overview

**GridWorld Pain** is an RL research platform for **Interoceptive AI** — agents balancing external reward-seeking with internal homeostatic regulation (hunger/satiation, injury avoidance). Core pillars: homeostatic RL (drive-reduction reward), sensory fusion (chemical gradients + nociceptors + proprioception + interoception), JAX `vmap` parallelism (128-1000+ envs), and neuromodulated RNN gating.

| Algorithm | Status | Files |
|-----------|--------|-------|
| **RecurrentPPO** | Primary | `recurrent_ppo_network.py`, `recurrent_ppo_trainer.py` |
| **DreamerV3** | Experimental | `dreamer_v3_nnx.py`, `dreamer_v3_trainer.py` |
| **PPO** (FF) | Supporting | `ppo_network.py`, `ppo_trainer.py` |
| **DQN / DRQN** | Supporting | `dqn_network.py`, `drqn_network.py` |
| **Neuromodulated** | Experimental | `neuromodulator.py`, `neuromodulated_*.yaml` |

---

## File Manifest

### Environment (`src/environment/`)
| File | Purpose |
|------|---------|
| `core.py` | `jax_step()`, `jax_reset()`, `update_body`, `update_predators`, `update_neutral_animals`, `update_resources`, `resolve_overlaps_global`, `place_in_area`. Pure functional. |
| `state.py` | `EnvState` and `EnvParams` Flax `struct.dataclass` pytree definitions. |
| `sensor.py` | `get_observation()`, `apply_perceptual_noise()`, `get_observation_breakdown()`. Sensors: `sense_resource`, `sense_collision`, `sense_location`, `sense_extero_nociception`, `sense_visual`. |
| `wrapper.py` | `ParallelEnv` (vmap wrapper for step/reset/obs). `auto_reset_step()` for vmapped step-with-autoreset. |
| `config_loader.py` | YAML → `EnvParams`. Entity count expansion, type-level placement groups, noise config parsing. |
| `renderer.py` | Telemetry dashboard (3-panel: interoception vitals, center arena + minimap, exteroception pods). Icon loading, video export. |
| `grid_world.py` | Duplicate of `renderer.py` (legacy). |

### Models (`src/models/`)
| File | Purpose |
|------|---------|
| `recurrent_ppo_network.py` | `ActorCriticRNN`: Hierarchical `ObservationEncoder` (GroupedMLP → body hub → association hub → fusion) + GRU/LSTM + Actor/Critic. |
| `recurrent_ppo_trainer.py` | PPO: rollout collection, GAE/MC returns, clipped surrogate, entropy regularization. Separate actor/critic optimizers. |
| `dreamer_v3_nnx.py` | RSSM (GRU deter + categorical stoch), Encoder, Decoder, Reward/Continue predictors, Actor, Critic. |
| `dreamer_v3_trainer.py` | World model BPTT, replay buffer, 15-step imagination rollouts for actor/critic. |
| `dreamer_v3_network.py` | DreamerV3 network spec builder. |
| `dreamer_v3_util.py` | `Moments` EMA normalizer, `Ratio` class. |
| `neuromodulator.py` | `HierarchicalNeuromodulator`: percept + memory gating → multiplicative gates per encoder phase. |
| `modulated_gru_cell.py` | GRU cell with multiplicative neuromodulation. |
| `modulated_layer_norm_gru_cell.py` | LayerNorm variant of modulated GRU. |
| `ppo_network.py` / `ppo_trainer.py` | Feed-forward PPO. |
| `dqn_network.py` / `dqn_trainer.py` | DQN with replay buffer. |
| `drqn_network.py` / `drqn_trainer.py` | Recurrent DQN. |

### Utilities (`src/utils/`)
| File | Purpose |
|------|---------|
| `config.py` | `Config`: strict `get_mandatory()`, nested dot-notation, deep `merge()`. |
| `evaluation_core.py` | Shared eval logic (train.py + evaluation.py). |
| `wandb_utils.py` | WandB init, metric logging, video upload. |
| `visualization.py` | Rendering helpers. |

### Configuration (`configs/`)
| Path | Purpose |
|------|---------|
| `environment/default.yaml` | Base env: 10x10 grid, 4 food + 8 danger, 3 predators, 12 rocks, 20 bushes, 5 rabbits. |
| `train/default.yaml` | Training defaults: episodes, num_envs, checkpoint frequency. |
| `models/recurrent_ppo.yaml` | RecurrentPPO: GRU, lr=5e-4/1e-4, gamma=0.95, seq_len=128, hidden=128, hierarchical. |
| `models/dreamer_v3.yaml` | DreamerV3: RSSM deter=512/stoch=32x32, lr=1e-4/3e-5, entropy=3e-4, batch=64. |
| `models/neuromodulated_ppo.yaml` | RecurrentPPO + hierarchical neuromodulation. |
| `models/neuromodulated_dreamer_v3.yaml` | DreamerV3 + neuromodulation. |
| `models/ppo.yaml`, `dqn.yaml`, `drqn.yaml` | Other algorithm configs. |
| `experiment/ablation/survival/01-08_*.yaml` | Survival branch ablation levels. |
| `experiment/ablation/homeostatic/04-08_*.yaml` | Homeostatic branch ablation levels. |

### Entry Points
| File | Purpose |
|------|---------|
| `train.py` | Unified training loop (all algorithms). Config loading, WandB, checkpointing. |
| `evaluation.py` | Checkpoint eval + video rendering + stats. |
| `run_all_experiments.py` | Batch ablation sweep launcher. |

---

## Environment Architecture

### EnvState (mutable per-step)
- **Agent**: `agent_pos [2]`, `current_step`, `last_action`
- **Resources**: `res_pos [N,2]`, `res_active [N]`, `res_cons_count [N]`, `res_reg_timer [N]`
- **Predators**: `pred_pos [P,2]`, `pred_state [P]` (0=patrol, 1=hunt, 2=return), `pred_stamina [P]`, `pred_move_timer [P]`, `pred_attack_timer [P]`
- **Neutral Animals**: `neutral_pos [M,2]`, `neutral_move_timer [M]`
- **Obstacles**: `obs_pos [O,2]`
- **Body**: `satiation`, `nutrition`, `injury_level`, `injury_buffer [smoothing_dur]`, `last_collision_noc`, `rest_streak`
- **Meta**: `terminated`, `key` (PRNGKey)

### EnvParams (immutable per-episode)
- **Grid**: `height`, `width`, `max_steps`, `grid_location_type [H,W]` (0=plain, 1=grass, 2=sand)
- **Resources**: `res_type [N]` (0=food, 1=danger), `res_property [N,5]`, `res_damage [N,2]`, `res_nociception [N]`, `res_spawn_area [N,4]`, `res_max_cons [N]`, `res_reg_delay [N]`
- **Predators**: `pred_property [P,5]`, `pred_nociception [P]`, `pred_damage [P,2]`, `pred_detect [P]`, `pred_patrol [P,4]`, `pred_spawn_area [P,4]`, `pred_move_int [P]`, `pred_max_stamina [P]`, `pred_recovery [P]`, `pred_hunt_thresh [P]`, `pred_attack_delay [P]`, `pred_lose_interest_mult [P]`, `predator_enabled`
- **Obstacles**: `obs_blocking [O]`, `obs_hides_agent [O]`, `obs_damage [O,2]`, `obs_property [O,5]`, `obs_nociception [O]`, `obs_type [O]`, `obstacle_names`, `obs_spawn_area [O,4]`
- **Neutral**: `neutral_property [M,5]`, `neutral_nociception [M]`, `neutral_move_int [M]`, `neutral_patrol [M,4]`, `neutral_spawn_area [M,4]`
- **Placement**: `placement_mode` ("per_entity"/"per_type"), `type_areas [T,4]`, `type_counts [T]`, `type_entity_map [T,max_per_type]`, `max_per_type`, `num_types`, `num_entities`
- **Body**: `max_satiation/nutrition/injury`, `metabolic_cost`, `food_nutrition_gain`, `setpoint`, `start_satiation/nutrition`, `nutrition_to_satiation_scaling_factor`, `recovery_base_rate/accel_rate`, `smoothing_duration`, `death_penalty`, `overeating_death`, `use_homeostatic_reward`, `with_satiation/nutrition/injury`, `random_start_satiation/nutrition/injury/pos`, `start_pos [2]`, `eating_nutrition_cost`, `eating_reward_penalty`
- **Sensors**: `olfactory_enabled`, `nociception_enabled`, `location_sensor_enabled`, `visual_sensor_enabled`, `proprioception_enabled`, `sensor_radius`, `sensor_decay`, `sensor_range`, `visual_sensor_range`, `local_view_size`, `action_dim`, `olfactory_vector_size`, `nociception_size`
- **Actions**: `rest_action_enabled`, `eat_action_enabled`
- **Noise**: `perceptual_noise_enabled`, `noise_modality_order` (tuple from YAML key order), `noise_modes/sigmas/injury_scales/clip_min/clip_max` arrays (padded to 12)

### Core Loop (`jax_step`)

1. **Resource regeneration** — timers tick, respawn where timer=0, reposition within spawn area
2. **Agent movement** — 4 dirs + rest(4) + eat(5), obstacle collision check → `just_collided` flag
3. **Predator update** — patrol→hunt→return state machine, Manhattan pursuit, stamina, obstacle collision via `vmap`, bush concealment suppresses hunting, attack delay
4. **Neutral animal update** — random patrol, obstacle collision via `vmap`
5. **Interaction** — resource consumption (auto or eat-action), damage from: dangers (sampled [min,max]), predators (sampled + triggers attack timer), obstacle overlap + collision. Stores `collision_noc`
6. **Body update** — nutrition decay, food gain minus `eating_nutrition_cost`, satiation = `max_S * (N/max_N)^k`, injury via smoothed ring buffer + exponential streak recovery
7. **Reward** — Survival: +1.0 food / -death_penalty death. Homeostatic: `prev_drive - curr_drive` where `drive = ‖(satiation - setpoint, injury)‖₂`. Both: minus `eating_reward_penalty`
8. **Termination** — codes: 0=active, 1=truncated, 2=starvation, 3=overeating, 4=injury. Without `with_injury`, any damage = instant death
9. **Info dict** — `ate_food`, `damage` (total + per-source), `rested`, `hit_danger`, `hit_predator`, `reward_homeostatic/extrinsic`, `drive_hunger/injury`, `dist_to_food/pred`, `event_collided`, `termination_reason`

### Entity Placement (`jax_reset`)

- **`per_entity`**: vmap-sample → `resolve_overlaps_global()` sequential scan. Faster on small grids.
- **`per_type`**: `lax.scan` over type groups → `place_in_area()` per group. Better for large grids.

Both guarantee overlap-free placement. Supports random start position, nutrition (uniform [max/2, max]), and injury.

---

## Sensory System

### Observation Vector Order (`get_observation()` in `sensor.py`)

| # | Sensor | On? | Dim | Mapping |
|:-:|--------|:---:|-----|---------|
| 1 | **Injury** | Always | 1 | `injury / max_injury` |
| 2 | **Nutrition** | Always | 1 | `nutrition / max_nutrition` |
| 3 | **Satiation** | Always | 1 | `satiation / max_satiation` |
| 4 | **Extero Nociception** | `nociception_enabled` | 1 | Max intensity: danger/predator/obstacle overlap + collision bump |
| 5 | **Olfaction** | `olfactory_enabled` | 5 | `Σ property * decay(dist) * active` across all entities |
| 6 | **Collision** | Always | `2r²+2r+1` | Binary Manhattan diamond (OOB or blocking obstacle) |
| 7 | **Proprioception** | `proprioception_enabled` | `action_dim` | One-hot of `last_action` |
| 8 | **Visual** | `visual_sensor_enabled` | `(2r²+2r+1)*8` | 8-ch Manhattan diamond: Grass/Sand/Plain/Food/Danger/Predator/Rock/Neutral |
| 9 | **Location** | `location_sensor_enabled` | 2 | Normalized (r,c) to [-1, 1] |

### Perceptual Noise (`apply_perceptual_noise()`)

Per-modality Gaussian noise. Mode 0: none. Mode 1: constant `sigma_base`. Mode 2: state-dependent `sigma_base * (1 + injury_scale * norm_injury)`. Order from `noise_modality_order` tuple (YAML key order). Per-modality clipping via `clip_min/max`. Dimension computed by `get_observation_breakdown()`.

---

## Agent Architectures

### RecurrentPPO
```
Obs → ObservationEncoder (hierarchical):
  Phase 1: GroupedMLP — per-sensor MLPs via einsum → [128] each
  Phase 2: Body Hub — MLP([intero + nociception + collision]) → [128]
  Phase 3: Association Hub — MLP([proprioception + olfaction]) → [128]
  Fusion: concat all → Linear → [128]
→ GRU/LSTM (hidden=128) → Actor (logits) + Critic (V)
```
Training: sequence rollouts, GAE/MC returns, clipped PPO, separate Adam (actor 5e-4, critic 1e-4).

### DreamerV3
```
RSSM: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1}) [deter=512], z_t ~ Cat(32x32)
Posterior: q(z|h,obs), Prior: p(z|h)
Decoder → obs, Reward predictor, Continue predictor
Actor: π(a|h,z), Critic: V(h,z) [Moments EMA]
```
Training: replay buffer (100k), world model BPTT, 15-step imagination rollouts.

### Neuromodulation
```
HierarchicalNeuromodulator:
  Percept: MLP([satiation, nutrition, injury, collision]) → φ_p
  Memory:  MLP(prior_hidden) → φ_m
  → γ_unimodal, γ_body, γ_association = softmax(φ_p + φ_m)
  → z_memory = tanh(φ_m), temperature = sigmoid(φ_p)
```
Multiplicative: `encoded_phase *= γ_phase`. Tracked as `modulator/gamma_*`.

---

## Execution Reference

```bash
# Training
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --agent_config configs/models/recurrent_ppo.yaml \
  --config configs/environment/default.yaml \
  --episodes 50000 --num-envs 128 --seed 42 --tag my_experiment

# Evaluation
python evaluation.py --results_dir results/JAX_RecurrentPPO/run_name/ \
  --episodes 100 --checkpoint latest --render --num_envs 16

# Batch Experiments
python run_all_experiments.py --tag ablation_v2 --episodes 50000
```

Key CLI: `--config`, `--agent_config` (required), `--episodes`, `--total-timesteps`, `--num-envs`, `--seed`, `--tag`, `--wandb-project/group/name`, `--checkpoint-frequency`, `--load-checkpoint`, `--device`.

**Checkpointing**: Orbax, every `--checkpoint-frequency` episodes, max 5 retained, strict architecture matching on restore.
