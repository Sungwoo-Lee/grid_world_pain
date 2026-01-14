
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
    
    # Visualize Q-table
    vis_filename = os.path.join(results_dir, "q_table_vis.png")
    plot_q_table(agent.q_table, vis_filename)

def plot_q_table(q_table, save_path):
    """
    Visualizes the Q-table as a heatmap with action arrows.
    
    Args:
        q_table (numpy.ndarray): The Q-table of shape (height, width, 4).
        save_path (str): Path to save the visualization.
    """
    import matplotlib.pyplot as plt
    
    height, width, _ = q_table.shape
    
    # Calculate best action and max Q-value for each cell
    best_actions = np.argmax(q_table, axis=2)
    max_q_values = np.max(q_table, axis=2)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot heatmap
    cax = ax.imshow(max_q_values, cmap='viridis', interpolation='nearest')
    fig.colorbar(cax, label='Max Q-Value')
    
    # Add arrows indicating best action
    # Actions: 0=Up, 1=Right, 2=Down, 3=Left
    # Corresponding (dx, dy) for plotting arrows
    # Note: In matric coordinates (row, col), Up is (-1, 0), Right is (0, 1), etc.
    # But for plotting with matplotlib (x, y) where x=col, y=row (inverted y-axis by imshow)
    action_deltas = {
        0: (0, -0.3),  # Up (negative y in plot coordinates if origin is top-left)
        1: (0.3, 0),   # Right
        2: (0, 0.3),   # Down
        3: (-0.3, 0)   # Left
    }
    
    for r in range(height):
        for c in range(width):
            action = best_actions[r, c]
            dx, dy = action_deltas[action]
            
            # Draw arrow
            # Note: origin is 'upper' by default for imshow, so y increases downwards
            ax.arrow(c, r, dx, dy, head_width=0.1, head_length=0.1, fc='white', ec='white')
            
    ax.set_title("Learned Policy (Q-Table)")
    ax.set_xticks(np.arange(width))
    ax.set_yticks(np.arange(height))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Q-table visualization saved to {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    train_and_visualize()
