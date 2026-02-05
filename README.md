# 🎮 GridWorld Pain

> **A robust, visualization-ready Reinforcement Learning environment for Interoceptive AI research.**

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/status-active-success?style=for-the-badge)

<div align="center">
  <img src="assets/agent_demo.gif" alt="GridWorld Agent Demo" width="600px" />
</div>

---

## 📖 Overview

**GridWorld Pain** is a custom, high-performance implementation of the classic GridWorld environment designed for **Reinforcement Learning (RL)** research and development. Unlike standard implementations, this package places a heavy emphasis on **observability**, **visualization**, and **interoception** (internal body states).

It supports both **Classic Tabular Methods** and **Deep Reinforcement Learning** algorithms.

## ✨ Key Features

- **🚀 Lightweight Core**: Built with pure Python and optimized for speed.
- **⚡ JAX-Native Parallelization**: Fully vectorizable JAX environment supporting thousands of parallel envs on GPU.
- **🤖 Multi-Agent Support**: Includes implementations for **DQN, DRQN, PPO, RecurrentPPO**, **DreamerV3** (Torch), and **RecurrentPPO** (JAX/Flax).
- **🧠 Interoception**: Simulation of internal body states (Satiation, Injury) that drive reward signals (Homeostatic RL).
- **🎥 Professional Visualization**: High-speed rendering in JAX and activation monitoring in PyTorch.
- **🔍 Activation Monitoring**: Visualizes internal neural network activations (Torch) and LRP attributions.
- **📦 Configuration Driven**: Fully YAML-based configuration with strict validation.
- **🧪 Ablation Study Support**: 13+ ablation configurations for systematic research.

---

## 📂 Project Structure

```text
grid_world_pain/
├── configs/                    # ⚙️ Configuration YAMLs
│   ├── ablation/               # 🧪 Homeostatic & Survival Branches
│   ├── environment/            # 🌍 Environment settings
│   ├── models/                 # 🤖 Agent hyperparameters
├── train.py                    # 🧠 Torch Training Script
├── train_jax.py                # ⚡ JAX Parallel Training Script
├── evaluation.py               # 🎬 Torch Evaluation & Visualization
├── evaluation_jax.py           # 🎬 JAX Evaluation script
├── main_jax.py                 # 🏃‍♂️ JAX Console Demo
├── run_all_experiments.py      # 🚀 Parallel Training Launcher
├── src/                        # 🐍 Source Code
│   ├── environment/            # 🌍 GridWorld & JAX-native core
│   ├── models/                 # 🤖 Agent Implementations (Torch & JAX)
│   └── utils/                  # 🛠️ Utilities (Config, Visualization)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.11 or higher
- **Conda**: Recommended for environment management

### 💿 Installation

1.  **Create and activate the Conda environment:**
    ```bash
    conda create -n grid_world_pain python=3.11
    conda activate grid_world_pain
    ```

2.  **Install dependencies:**
    ```bash
    pip install -e .
    ```

---

## 🛠️ Usage

### 1. Training Agents (PyTorch)
Train various RL agents using the `train.py` script. Configuration is handled via YAML files in `configs/`.

**Train DQN with ablation config:**
```bash
python train.py --agent_config configs/models/dqn.yaml --config configs/ablation/survival/01_goal_only.yaml --episodes 1000
```

### 2. High-Performance Training (JAX)
For massive parallelization and GPU acceleration, use the JAX-native pipeline.

**Train RecurrentPPO with 128 parallel envs:**
```bash
python train_jax.py --agent_config configs/models/recurrent_ppo.yaml --config configs/ablation/homeostatic/08_homeostatic.yaml --num-envs 128
```

### 3. Evaluating & Visualizing
After training, generate high-quality videos and verify performance.

**PyTorch Evaluation:**
```bash
python evaluation.py --results_dir results/DQN/my_run --episodes 3
```

**JAX Evaluation:**
```bash
python evaluation_jax.py --results_dir results/JAX_RecurrentPPO/my_run --episodes 3 --render-video
```

**Outputs:**
- Generates `.mp4` videos with real-time sensory visualizations.
- JAX renderer optimized for speed using figure/icon caching.
- Optionally uploads videos to WandB.

### 3. Ablation Study Experiments
Run all algorithms in parallel with a specific ablation configuration:

```bash
python run_all_experiments.py --tag ablation_01 --config configs/ablation/01_goal_only.yaml --episodes 10000
```

**Available Ablation Levels (configs/ablation/):**
| Level | File | Focus |
|-------|------|-------|
| 01 | `01_goal_only.yaml` | Baseline pathfinding, no dynamics |
| 02 | `02_danger_only.yaml` | Static danger zone, health system |
| 03 | `03_dynamic_fixed.yaml` | Dynamic food↔danger transitions |
| 04 | `04_nociception.yaml` | +Pain sensor (nociceptor) |
| 05 | `05_olfactory.yaml` | +Chemical gradient sensing |
| 06 | `06_proprioception.yaml` | +Motor feedback |
| 07 | `07_collision.yaml` | +Directional collision detection |
| 08 | `08_homeostatic.yaml` | +Drive reduction reward |
| 09 | `09_location.yaml` | +Spatial awareness (full model) |

### 4. Verify Ablation Configurations
Verify that all ablation configs are correctly set up:

```bash
python verify_ablation_levels.py
```

---

## 📚 Environment Features

### `GridWorld`
The core environment (`src/environment/grid_world.py`) supports:
- **Dynamic Resources**: Food can transition to Danger (and vice versa) with probabilistic gating.
- **Resource Relocation**: Food periodically changes location during episodes.
- **Configurable Actions**: Eat and Rest actions can be enabled/disabled.

### `InteroceptiveBody`
Simulates the agent's physiological needs (`src/environment/body.py`):
- **Satiation**: Hunger mechanics with metabolism and overeating death.
- **Health**: Physical health that degrades in danger zones.
- **Homeostatic Reward**: Drive reduction towards setpoints.

### `SensorySystem`
Biologically-inspired sensory inputs (`src/environment/sensor.py`):
- **Olfactory**: Gradient-based chemical detection
- **Nociception**: Pain sensor for danger zones
- **Collision**: Directional ray-based wall detection
- **Location**: Centered spatial coordinates
- **Proprioception**: Previous action as motor feedback

---

## 📖 Documentation

For detailed system documentation, see [docs/PROJECT_SUMMARY.md](docs/PROJECT_SUMMARY.md).

---

<div align="center">
    <sub>Built with ❤️ by the Interoceptive AI Team</sub>
</div>
