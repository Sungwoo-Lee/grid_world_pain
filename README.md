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
├── README.md                   # 📄 Documentation
└── src/
    └── grid_world_pain/        # 🐍 Source Code
        ├── __init__.py         # Package Exporter
        └── grid_world.py       # Core Environment Logic
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

### 1. Simple Text Demo
Run the console-based simulation to see the agent move in ASCII format.

```bash
python main.py
```

### 2. Generate Video Replays
Create a high-quality `.mp4` video of the agent's episode.

```bash
python visualize.py
```

---

## 📚 API Reference

### `GridWorld`
The core class located in `src/grid_world_pain/grid_world.py`.

| Method | Description |
| :--- | :--- |
| `__init__(width, height, start, goal)` | Initializes the grid dimensions and key coordinates. |
| `reset()` | Resets the environment and returns the initial state. |
| `step(action)` | Executes an action (`Up`, `Right`, `Down`, `Left`) and returns `(state, reward, done)`. |
| `render()` | Prints the grid state to the console. |
| `render_rgb_array()` | Returns a NumPy array representing the current frame (for video). |

---

<div align="center">
    <sub>Built with ❤️ by the Interoceptive AI Team</sub>
</div>
