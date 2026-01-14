"""
Main script for demonstrating the custom GridWorld environment.
This script initializes the environment and runs a single episode with a random agent,
printing the grid state to the console at each step.
"""

from src.grid_world_pain import GridWorld
import random
import time

def main():
    # Initialize the environment with default parameters (5x5 grid)
    env = GridWorld()
    
    # Reset the environment to the start state
    state = env.reset()
    print("Start State:")
    env.render()
    
    done = False
    step_count = 0
    max_steps = 20 # Safety limit to prevent infinite loops if agent gets stuck
    
    # Main loop: Continue taking steps until goal is reached or max steps exceeded
    while not done and step_count < max_steps:
        # Choose a random action:
        # 0=Up, 1=Right, 2=Down, 3=Left
        action = random.randint(0, 3)
        
        # Helper list to print human-readable action names
        action_name = ["Up", "Right", "Down", "Left"][action]
        print(f"Step {step_count + 1}: Action {action_name}")
        
        # Take the step in the environment
        # state: The new position (row, col)
        # reward: The immediate reward received
        # done: Boolean flag, True if the episode is finished (goal reached)
        state, reward, done = env.step(action)
        
        # Visualize the new state in the console
        env.render()
        print(f"Reward: {reward}, Done: {done}\n")
        
        step_count += 1
        
        # Check for success condition
        if done:
            print("Goal Reached!")
            break

if __name__ == "__main__":
    main()
