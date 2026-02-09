# Utility Scripts

This directory contains helper scripts for environment maintenance and asset generation.

## 🎥 Recording a Demo
Use `record_env_demo.py` to record a video of the environment using a random policy and default parameters.

**Usage:**
```bash
# From project root
PYTHONPATH=. /home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/record_env_demo.py
```
Outputs to: `assets/temp_demo.mp4`

## 🖼️ Video to GIF Conversion
Use `video_to_gif.py` to convert recorded MP4 files into optimized GIFs for documentation.

**Usage:**
```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/video_to_gif.py assets/temp_demo.mp4 assets/agent_demo.gif 10
```

## 🛑 Requirements
Ensure you are using the mandatory conda environment:
`/home/vncuser/miniconda3/envs/grid_world_pain/`
