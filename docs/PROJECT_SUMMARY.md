# 🧠 GridWorld Pain Project - Comprehensive System Documentation

> **Project Location**: `/media/nas01/projects/Interoceptive-AI/grid_world_pain`  
> **Last Updated**: 2026-01-28  
> **Purpose**: LLM Agent Context Document

---

## 📖 Executive Overview

**GridWorld Pain** is a sophisticated Reinforcement Learning research platform designed to study **Interoceptive AI** - agents that perceive and act based on internal body states (hunger, pain) rather than just external goals. The project implements a custom grid-based environment with homeostatic regulation and supports multiple state-of-the-art RL algorithms.

### Key Research Concepts
- **Interoception**: Internal body-state awareness (satiation, health)
- **Homeostatic RL**: Reward signals based on drive reduction (maintaining physiological setpoints)
- **Sensory Integration**: Gradient-based olfactory sensors, nociceptors, and collision detection

---

## 🏗️ System Architecture

```mermaid
graph TB
subgraph "Entry Points"
        A[train.py<br/>Torch RL Loop]
        AJ[train_jax.py<br/>JAX Parallel Loop]
        B[evaluation.py<br/>Torch Viz]
        BJ[evaluation_jax.py<br/>JAX Evaluation]
        C[main_jax.py<br/>JAX Demo]
        D[run_all_experiments.py]
    end
    
    subgraph "Environment Layer"
        F[GridWorld<br/>Torch-friendly]
        FJ[JAX-native Core<br/>jax_env/]
        G[InteroceptiveBody]
        H[SensorySystem]
    end
    
    subgraph "Agent Layer"
        I[Torch Agents<br/>DQN, PPO, etc.]
        IJ[JAX Agents<br/>RecurrentPPO]
    end
    
    subgraph "Utilities"
        O[Visualization]
        R[WandB Utils]
        S[Config System]
        T[Evaluation Core]
    end
    
    A --> F & G & H
    AJ --> FJ
    A --> I
    AJ --> IJ
    B --> T
    BJ --> T
    T --> O & R
    D --> A
```

---

## 📁 Project Structure

| Path | Lines | Description |
|------|-------|-------------|
| [train.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/train.py) | 1172 | Torch training with activation capture |
| [train_jax.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/train_jax.py) | 570 | JAX training with massive parallelization |
| [evaluation_jax.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/evaluation_jax.py) | 200 | JAX evaluation and high-speed viz |
| [main_jax.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/main_jax.py) | 150 | JAX-native interactive sandbox |
| [run_all_experiments.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/run_all_experiments.py) | 164 | Parallel training launcher |

### Source Directory (`src/`)

```
src/
├── environment/
│   ├── grid_world.py      # Torch-friendly implementation
│   ├── jax_env/           # JAX-native core
│   │   ├── core.py        # Vectorized GridWorld (Pure JAX)
│   │   ├── sensor.py      # Vectorized sensory logic
│   │   ├── renderer.py    # High-performance JAX renderer
│   │   └── wrapper.py     # ParallelEnv JAX wrapper
│   ├── body.py            # Physio dynamics logic
│   └── sensor.py          # Torch-friendly sensory logic
│
├── models/
│   ├── dqn.py             # Torch DQNAgent
│   ├── ppo.py             # Torch PPOAgent
│   ├── jax_models/        # JAX/Flax NNX models
│   │   └── recurrent_ppo_network.py # ActorCriticRNN
│   └── ...
│
└── utils/
    ├── activation_monitor.py  # Hook-based activation capture (134 lines)
    ├── config.py              # Config class with strict validation (82 lines)
    ├── evaluation_core.py     # Core evaluation logic (429 lines)
    ├── lrp_monitor.py         # Layerwise Relevance Propagation (78 lines)
    ├── state_utils.py         # State preprocessing, frame stacking (103 lines)
    ├── visualization.py       # Video, Q-table, activation viz (634 lines)
    └── wandb_utils.py         # WandB login, video upload (168 lines)
```

---

## 🧪 Ablation Study Framework

The project includes a systematic ablation study framework with 8 progressive levels, branching into two distinct research paths at Level 04.

### Ablation Structure (`configs/ablation/`)

| Level | Name | Motivation/Changes | Branch Availability |
| :--- | :--- | :--- | :--- |
| **01** | Goal Only | Baseline pathfinding: navigate to food. | Survival Only |
| **02** | Danger Only | Introduction of static hazard zones (instant death). | Survival Only |
| **03** | Predator Intro | Introduction of an active predator (instant death). | Survival Only |
| **04** | Nociception | **Branch Point.** Intro of Injury system & Nociceptor sensor. | Both |
| **05** | Olfactory | Intro of Satiation system (Homeostatic branch) & Olfactory sensor. | Both |
| **06** | Proprioception | Addition of Proprioception sensor. | Both |
| **07** | Collision | Addition of Collision sensor. | Both |
| **08** | Location | Addition of Location sensor. | Both |

### Research Branches
1.  **Survival branch**: Focuses on survival without explicit homeostatic drive-reduction rewards (`use_homeostatic_reward: false`).
2.  **Homeostatic branch**: Focuses on internal state maintenance using drive-reduction rewards (`use_homeostatic_reward: true`).

---

## 🌍 Environment Module - Multi-Resource System

The environment has transitioned from a single-resource legacy system to a flexible **Multi-Resource Entity System**.

### Resource Entity Model
Each resource is an object with its own lifecycle:
- **Type**: `food` or `danger`.
- **Properties**: N-dimensional chemical/sensory signatures.
- **Lifecycle**: `max_consumption` (depletion) and `regeneration_delay` (respawning).
- **Spatial**: Configurable `spawn_area` [[min_r, min_c], [max_r, max_c]].

### GridWorld Implementation
The core environment logic resides in [src/environment/grid_world.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/src/environment/grid_world.py).

#### Configurable Multi-Resource System
Instead of fixed positions, the environment now loads a list of resource templates:
```yaml
environment:
  resources:
    - name: "Food Source A"
      type: "food"
      count: 1
      spawn_area: [[0, 0], [5, 5]]
      properties: [1.0, 0.0, 0.0, 0.0, 0.0]
      max_consumption: 3
      regeneration_delay: 50
    - name: "Danger Zone"
      type: "danger"
      count: 10
      spawn_area: [[0, 0], [20, 20]]
      damage: 10
```

### Multi-Predator AI System
GridWorld now supports multiple active predators with complex behaviors:
- **State Machine**: Predators transition between `PATROL`, `HUNT`, `RETURN`, and `IDLE`.
- **Stamina Dynamics**: Predators consume stamina while hunting and must recover before hunting again.
- **Zone Constraints**: Predators can be restricted to specific `spawn_area` and `patrol_area`.
- **Detection Range**: Hunting is triggered when the agent enters a configurable `detection_range`.

```yaml
environment:
  predator_enabled: true
  predators:
    - name: "Alpha Predator"
      count: 1
      move_interval: 3
      damage: 3.0
      detection_range: 5
      max_stamina: 20
      stamina_recovery_rate: 0.1
      hunt_stamina_threshold: 0.5
```

#### Action Space (Dynamic)
| Configuration | Actions | Indices |
|---------------|---------|----------|
| Both enabled | 6 | 0-3: Move, 4: Rest, 5: Eat |
| Eat only | 5 | 0-3: Move, 4: Eat |
| Rest only | 5 | 0-3: Move, 4: Rest |
| Neither | 4 | 0-3: Move only |

---

## 🤖 Agent Implementations - Detailed

### DQNAgent ([src/models/dqn.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/src/models/dqn.py))
Standard Deep Q-Network with experience replay and target networks.

### DRQNAgent ([src/models/drqn.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/src/models/drqn.py))
Deep Recurrent Q-Network (LSTM) for handling partial observability in complex sensor environments.

### PPOAgent ([src/models/ppo.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/src/models/ppo.py))
Proximal Policy Optimization with separate actor-critic MLP networks.

### RecurrentPPOAgent ([src/models/recurrent_ppo.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/src/models/recurrent_ppo.py))
PPO with an LSTM backbone, effective for high-temporal dependency scenarios.

### DreamerV3Agent ([src/models/dreamer_v3.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/src/models/dreamer_v3.py))
State-of-the-art Model-Based RL using the RSSM (Recurrent State Space Model).

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.5.1 | Deep learning framework |
| `numpy` | ≥2.1.2 | Numerical computing |
| `matplotlib` | ≥3.9.2 | Visualization |
| `wandb` | ≥0.18.5 | Experiment tracking |
| `pyyaml` | ≥6.0.2 | Config parsing |

---

## 🚀 Usage Examples

### Running an Ablation Experiment
```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
    --agent_config configs/models/dqn.yaml \
    --config configs/ablation/homeostatic/05_olfactory.yaml \
    --episodes 5000 \
    --tag ablation_run_5
```

### Parallel Batch Execution
```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python run_all_experiments.py \
    --tag final_baseline \
    --config configs/ablation/survival/08_location.yaml \
    --episodes 10000
```
