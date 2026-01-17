#!/bin/bash
set -e

# Configuration
ENV_NAME="grid_world_pain"
CONFIG_PATH="configs/models/drqn.yaml"
EPISODES=1000
DEVICE="cuda"
RESULTS_BASE_DIR="results"
ASSETS_DIR="assets"
GIF_NAME="agent_demo.gif"

echo "=========================================="
echo "    GridWorld Auto-Demo Generator"
echo "=========================================="

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed."
    exit 1
fi

# Ensure assets directory exists
mkdir -p "$ASSETS_DIR"

# 1. Train a fresh model (Short run for demo purposes)
echo "[1/3] Training DRQN agent for $EPISODES episodes..."
# Note: Assuming conda env is active or we are running inside it. 
# Attempting to run python directly.
python train.py --agent_config "$CONFIG_PATH" --episodes "$EPISODES" --device "$DEVICE" --tag "demo"

# Find the specific results directory (newest in results/DRQN ending with _demo)
# Listing directories, sorting by time, filtering.
RESULTS_DIR=$(ls -td "$RESULTS_BASE_DIR"/DRQN/*_demo | head -n 1)

if [ -z "$RESULTS_DIR" ]; then
    echo "Error: Could not find the results directory for the demo run."
    exit 1
fi

echo "Using results directory: $RESULTS_DIR"

# 2. Evaluate to generate video
echo "[2/3] Evaluating agent..."
python evaluation.py --results_dir "$RESULTS_DIR" --episodes 1

# Locate the video file
# evaluation.py usually names it final_trained_agent.mp4 for the final model
VIDEO_PATH="$RESULTS_DIR/videos/final_trained_agent.mp4"

if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found at $VIDEO_PATH"
    exit 1
fi

# 3. Convert to GIF
echo "[3/3] Converting to GIF..."
ffmpeg -y -v warning -i "$VIDEO_PATH" \
    -vf "fps=10,scale=1200:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
    "$ASSETS_DIR/$GIF_NAME"

echo "=========================================="
echo "Success! Demo GIF saved to:"
echo "$ASSETS_DIR/$GIF_NAME"
echo "=========================================="
