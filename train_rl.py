
from src.grid_world_pain import GridWorld
from src.grid_world_pain.agent import QLearningAgent
import time
import imageio
import numpy as np

def train_and_visualize():
    """
    Trains the Q-learning agent and visualizes the result.
    """
    # Initialize environment and agent
    env = GridWorld()
    agent = QLearningAgent(env)
    
    print("Training agent...")
    start_time = time.time()
    
    # Train the agent
    episodes = 500
    agent.train(episodes=episodes)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # --- Visualization / Verification ---
    print("\nRunning verification episode...")
    
    # Switch to greedy policy (epsilon = 0)
    agent.epsilon = 0
    
    state = env.reset()
    frames = []
    frames.append(env.render_rgb_array())
    
    done = False
    step_count = 0
    max_steps = 20
    
    print("Path taken:")
    print("Start State")
    
    while not done and step_count < max_steps:
        # Choose best action
        action = agent.choose_action(state)
        
        # Take step
        next_state, reward, done = env.step(action)
        
        # Record frame
        frames.append(env.render_rgb_array())
        
        # Print info
        action_name = ["Up", "Right", "Down", "Left"][action]
        print(f"Step {step_count+1}: Action {action_name}, State {next_state}")
        
        state = next_state
        step_count += 1
        
        if done:
            print("Goal Reached!")
            # Add end frames
            for _ in range(5):
                frames.append(env.render_rgb_array())
            break
            
    if not done:
        print("Failed to reach goal within max steps.")

    # Save results
    import os
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save video
    video_filename = os.path.join(results_dir, "rl_agent_video.mp4")
    imageio.mimsave(video_filename, frames, fps=5)
    print(f"\nVideo saved to {video_filename}")
    
    # Save model
    model_filename = os.path.join(results_dir, "q_table.npy")
    agent.save(model_filename)


if __name__ == "__main__":
    train_and_visualize()
