"""
Script for generating a video visualization of an episode in the GridWorld.

This script runs a single episode with a random agent (similar to main.py),
but instead of text output, it captures graphical frames using Matplotlib
and stitches them into an MP4 video file using imageio.
"""

from src.grid_world_pain import GridWorld
import random
import imageio
import numpy as np

def visualize_episode():
    """
    Runs an episode, captures frames, and saves them as a video.
    """
    # Initialize the environment
    env = GridWorld()
    
    # Reset to start state
    state = env.reset()
    
    # List to store image frames
    frames = []

    print("Generating video frames...")
    
    # Capture the initial state frame
    # render_rgb_array returns a numpy array of shape (H, W, 3) representing the RGB image
    frames.append(env.render_rgb_array())
    
    done = False
    step_count = 0
    max_steps = 30 # Limit video length
    
    # Main loop for the episode
    while not done and step_count < max_steps:
        # Choose random action (0-3)
        action = random.randint(0, 3)
        
        # Take step
        state, reward, done = env.step(action)
        
        # Capture frame of the new state
        frames.append(env.render_rgb_array())
        step_count += 1
        
        if done:
            print("Goal reached!")
            # Add a few duplicate frames of the final goal state
            # so the video pauses briefly at the end before stopping
            for _ in range(5):
                frames.append(env.render_rgb_array())
            break
            
    # Save the collected frames as an .mp4 video
    video_filename = 'gridworld_video.mp4'
    print(f"Saving video to {video_filename}...")
    
    # fps=5 means the video will play at 5 frames per second
    imageio.mimsave(video_filename, frames, fps=5)
    print("Done!")

if __name__ == "__main__":
    visualize_episode()
