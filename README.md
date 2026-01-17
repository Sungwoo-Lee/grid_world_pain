# 🎮 GridWorld Pain

> **A robust, visualization-ready Reinforcement Learning environment.**

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/status-active-success?style=for-the-badge)

---

## 📖 Overview

**GridWorld Pain** is a custom, high-performance implementation of the classic GridWorld environment designed for **Reinforcement Learning (RL)** research and development. Unlike standard implementations, this package places a heavy emphasis on **observability**, **visualization**, and **interoception** (internal body states).

It supports both **Classic Tabular Methods** and **Deep Reinforcement Learning** algorithms.

## ✨ Key Features

- **🚀 Lightweight Core**: Built with pure Python and optimized for speed.
- **🤖 Multi-Agent Support**: Includes implementations for **DQN, DRQN, PPO, RecurrentPPO**, and **DreamerV3**, alongside classic Q-Learning.
- **🧠 Interoception**: Simulation of internal body states (Satiation, Health) that drive reward signals (Homeostatic RL).
- **🎥 Built-in Visualization**: Seamless integration with `matplotlib` and `imageio` for generating MP4 replays.
- **📦 Configuration Driven**: Fully YAML-based configuration for easy experimentation.

---

## 📂 Project Structure

```text
grid_world_pain/
├── configs/                    # ⚙️ Configuration YAMLs (Environment & Models)
├── evaluation.py               # 🎬 Evaluation & Visualization Script
├── main.py                     # 🏃‍♂️ Console Demo Entry Point
├── train.py                    # 🧠 RL Training Script
├── README.md                   # 📄 Documentation
├── results/                    # 📂 Training Results & Artifacts
└── src/                        # 🐍 Source Code
    ├── environment/            # 🌍 Environment Logic (GridWorld, Body, Sensory)
    ├── models/                 # 🤖 Agent Implementations (DQN, PPO, etc.)
    └── utils/                  # 🛠️ Utilities (Config, Visualization)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.9 or higher
- **Conda**: Recommended for environment management

### 💿 Installation

1.  **Create and activate the Conda environment:**
    ```bash
    conda create -n grid_world_pain python
    conda activate grid_world_pain
    ```

2.  **Install dependencies:**
    ```bash
    pip install -e .
    ```

---

## 🛠️ Usage

### 1. Training Agents
Train various RL agents using the `train.py` script. Configuration is handled via YAML files in `configs/`.

**Train DQN:**
```bash
python train.py --agent_config configs/models/dqn.yaml --episodes 1000
```

**Train DRQN (Recurrent):**
```bash
python train.py --agent_config configs/models/drqn.yaml --episodes 1000
```

**Train PPO:**
```bash
python train.py --agent_config configs/models/ppo.yaml --episodes 5000
```

**Command Line Overrides:**
You can override common parameters directly:
```bash
python train.py --agent_config configs/models/dqn.yaml --episodes 500 --device cuda:0
```

### 2. Evaluating & Visualizing
After training, use `evaluation.py` to generate videos and verify performance. This script automatically loads the configuration used during training.

```bash
python evaluation.py --results_dir results/DQN/20260117-141318_default --episodes 3
```

**Outputs:**
- Generates `.mp4` videos of the agent's performance in `results/.../videos/`.
- If using Tabular Q-Learning, generates Q-table visualizations.

### 3. Simple Debugging Session
Run the console-based simulation to see the agent move in the GridWorld with random behaviors.

```bash
python main.py
```

---

## 📚 Environment Features

### `GridWorld`
The core environment (`src/environment/grid_world.py`) supports:
- **Resource Relocation**: Food can periodically change location during an episode (`relocate_resource: true`).
- **Random Initialization**: Agents and resources (if enabled) start at random positions on reset.
- **Hazards**: Configurable danger zones (`danger_prob`, `danger_duration`) that affect health.

### `InteroceptiveBody`
Simulates the agent's physiological needs (`src/environment/body.py`):
- **Satiation**: Hunger mechanics constrained by `max_satiation`.
- **Health**: Physical health that degrades in danger zones and recovers over time.
- **Homeostatic Reward**: Rewards are generated based on drive reduction (maintaining variables near setpoints) rather than simple external goals.

---

<div align="center">
    <sub>Built with ❤️ by the Interoceptive AI Team</sub>
</div>
