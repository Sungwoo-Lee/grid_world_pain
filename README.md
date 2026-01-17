# 🎮 GridWorld Pain

> **A robust, visualization-ready Reinforcement Learning environment.**

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/status-active-success?style=for-the-badge)

---

## 📖 Overview

**GridWorld Pain** is a custom, high-performance implementation of the classic GridWorld environment designed for **Reinforcement Learning (RL)** research and development. Unlike standard implementations, this package places a heavy emphasis on **observability** and **visualization**, allowing you to not only train agents but also watch them learn in real-time or export high-quality video demonstrations.

## ✨ Key Features

- **🚀 Lightweight Core**: Built with pure Python and optimized for speed.
- **🤖 RL Ready**: Includes a Q-Learning agent implementation out of the box.
- **🎥 Built-in Visualization**: Seamless integration with `matplotlib` and `imageio` for generating MP4 replays.
- **🧩 Standard Interface**: Familiar API design (`reset`, `step`, `render`) compliant with standard RL paradigms.
- **📦 Modular Architecture**: Clean separation between environment logic, visualization, and execution.

---

## 📂 Project Structure

```text
grid_world_pain/
├── pyproject.toml              # ⚙️ Configuration & Dependencies
├── main.py                     # 🏃‍♂️ Console Demo Entry Point
├── visualize.py                # 🎬 Video Generation Script
├── train_rl.py                 # 🧠 RL Training & Verification Script
├── README.md                   # 📄 Documentation
├── results/                    # 📂 Training Results & Artifacts
└── src/
    └── grid_world_pain/        # 🐍 Source Code
        ├── __init__.py         # Package Exporter
        ├── grid_world.py       # Core Environment Logic
        └── agent.py            # 🤖 Q-Learning Agent
```

---

## 🚀 Getting Started

### Prerequisites

- **Python**: 3.9 or higher
- **Conda**: Recommended for environment management

### 💿 Installation

1.  **Create and activate a fresh Conda environment:**
    ```bash
    conda create -n grid_world_pain python
    conda activate grid_world_pain
    ```

2.  **Install the package in editable mode:**
    ```bash
    pip install -e .
    ```

---

## 🛠️ Usage

### 1. Simple Debugging Session
Run the console-based simulation to see the agent move in the GridWorld with random behaviors and satiation dynamics.

```bash
python main.py
```

### 2. Generate Video Replays
Create a high-quality `.mp4` video of the agent's episode.

```bash
python visualize.py
```

### 3. Training the RL Agent
Train the Q-Learning agent and verify its performance. The script supports command-line arguments for flexibility.

**Run with default settings (100,000 episodes):**
```bash
python train_rl.py
```

**Run a quick debug session (e.g., 500 episodes):**
```bash
python train_rl.py --episodes 500
```

**Outputs in `results/`:**
- `q_table.npy`: The learned Q-values.
- `rl_agent_video.mp4`: A video recording of the trained agent verification episode.
- `q_table_vis.png`: A multi-panel visualization showing the learned policy and values at **Low**, **Mid**, and **High** satiation levels, including the food location.

---

## 📚 API Reference

### `GridWorld`
The core class located in `src/grid_world_pain/grid_world.py`.

| Method | Description |
| :--- | :--- |
| `__init__(width, height, start, goal)` | Initializes the grid dimensions and key coordinates. |
| `reset()` | Resets the environment and returns the initial state. **Now randomizes agent position.** |
| `step(action)` | Executes an action (`Up`, `Right`, `Down`, `Left`) and returns `(state, reward, done)`. |
| `render()` | Prints the grid state to the console. |
| `render_rgb_array(satiation, max_satiation)` | Returns a NumPy array representing the current frame with a satiation bar. |

### Configuration Extensions
- **Resource Relocation**: Resources (food) can now periodically change location (`relocate_resource: true`).
- **Safety/Pain**: Configurable danger zones and health mechanics.

### `InteroceptiveBody`
Simulates the agent's internal physiological state (satiation).

| Method | Description |
| :--- | :--- |
| `reset()` | **Randomizes start satiation** (between 50%-100% max) and returns initial level. |
| `step(info)` | Updates satiation based on metabolism (-1) and eating (+5). |

### `QLearningAgent`
The RL agent located in `src/grid_world_pain/agent.py`.

| Method | Description |
| :--- | :--- |
| `__init__(env, alpha, gamma, epsilon)` | Initializes hyperparameters and Q-table. |
| `choose_action(state)` | Selects an action using epsilon-greedy policy. |
| `update(state, action, reward, next_state)` | Updates Q-values based on experience. |
| `train(episodes)` | Runs the training loop. |
| `save(filepath)` | Saves the Q-table to a file. |

---

<div align="center">
    <sub>Built with ❤️ by the Interoceptive AI Team</sub>
</div>
