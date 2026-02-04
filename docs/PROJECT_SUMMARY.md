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
        A[train.py<br/>~1172 lines]
        B[evaluation.py<br/>~599 lines]
        C[main.py<br/>Debug Sandbox]
        D[run_all_experiments.py]
        E[hyperparameter_search.py]
        EA[verify_ablation_levels.py]
    end
    
    subgraph "Environment Layer"
        F[GridWorld<br/>grid_world.py]
        G[InteroceptiveBody<br/>body.py]
        H[SensorySystem<br/>sensor.py]
    end
    
    subgraph "Agent Layer"
        I[DQNAgent]
        J[PPOAgent]
        K[DRQNAgent]
        L[RecurrentPPOAgent]
        M[DreamerV3Agent]
        N[QLearningAgent]
    end
    
    subgraph "Utilities"
        O[Visualization]
        P[ActivationMonitor]
        Q[LRPMonitor]
        R[WandB Utils]
        S[Config System]
        T[Evaluation Core]
    end
    
    A --> F & G & H
    A --> I & J & K & L & M & N
    B --> T
    T --> O & P & Q & R
    D --> A
    E --> A
    EA --> F & H & S
```

---

## 📁 Project Structure

| Path | Lines | Description |
|------|-------|-------------|
| [train.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/train.py) | 1172 | Main training script with full RL loop |
| [evaluation.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/evaluation.py) | 599 | Evaluation and video generation entry point |
| [main.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/main.py) | 390 | Debug sandbox with random agent |
| [run_all_experiments.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/run_all_experiments.py) | 164 | Parallel training launcher |
| [hyperparameter_search.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/hyperparameter_search.py) | 300 | Grid search automation |
| [verify_ablation_levels.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/verify_ablation_levels.py) | 114 | Ablation configuration validator |

### Source Directory (`src/`)

```
src/
├── environment/
│   ├── grid_world.py      # GridWorld class (598 lines)
│   ├── body.py            # InteroceptiveBody class (166 lines)
│   └── sensor.py          # SensorySystem with 5 sensor types (462 lines)
│
├── models/
│   ├── dqn.py             # DQNAgent, DQN network (188 lines)
│   ├── drqn.py            # DRQNAgent with LSTM (321 lines)
│   ├── ppo.py             # PPOAgent, ActorCritic (265 lines)
│   ├── recurrent_ppo.py   # RecurrentPPOAgent with LSTM (346 lines)
│   ├── dreamer_v3.py      # DreamerV3Agent, RSSM world model (876 lines)
│   └── q_learning.py      # Tabular QLearningAgent (148 lines)
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

### GridWorld ([src/environment/grid_world.py](file:///media/nas01/projects/Interoceptive-AI/grid_world_pain/src/environment/grid_world.py))

#### Configurable Resources
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
