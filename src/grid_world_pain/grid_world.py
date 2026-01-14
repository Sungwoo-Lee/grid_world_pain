
import numpy as np
import matplotlib.pyplot as plt

class GridWorld:
    """
    A simple 2D GridWorld environment for Reinforcement Learning.
    
    The agent moves in a grid of size (height x width).
    The goal is to reach a specific target cell.
    
    Coordinates: (row, col)
    - (0, 0) is the top-left corner.
    - (height-1, width-1) is the bottom-right corner.
    
    Actions:
    - 0: Up
    - 1: Right
    - 2: Down
    - 3: Left
    
    Rewards:
    - -1 for each step (to encourage shortest path).
    - 0 when the goal is reached.
    """
    
    def __init__(self, height=5, width=5, start=(0, 0), goal=(4, 4)):
        """
        Initializes the GridWorld environment.

        Args:
            height (int): Number of rows in the grid.
            width (int): Number of columns in the grid.
            start (tuple): Start position (row, col).
            goal (tuple): Goal position (row, col).
        """
        self.height = height
        self.width = width
        self.start = start
        self.goal = goal
        self.agent_pos = start
        
    def reset(self):
        """
        Resets the agent to the start position.
        
        Returns:
            tuple: The initial state (agent position).
        """
        self.agent_pos = self.start
        return self.agent_pos
    
    def step(self, action):
        """
        Moves the agent based on the chosen action.
        
        Args:
            action (int): The action to take (0=Up, 1=Right, 2=Down, 3=Left).
            
        Returns:
            tuple: A tuple containing:
                - next_state (tuple): The new (row, col) position of the agent.
                - reward (int): The reward received (-1 per step, 0 at goal).
                - done (bool): True if the goal is reached, False otherwise.
        """
        row, col = self.agent_pos
        
        if action == 0:   # Up
            row = max(0, row - 1)
        elif action == 1: # Right
            col = min(self.width - 1, col + 1)
        elif action == 2: # Down
            row = min(self.height - 1, row + 1)
        elif action == 3: # Left
            col = max(0, col - 1)
            
        self.agent_pos = (row, col)
        
        done = (self.agent_pos == self.goal)
        reward = 0 if done else -1
        
        return self.agent_pos, reward, done

    def render(self):
        """
        Prints a simple ASCII representation of the grid to the console.
        
        'A' represents the Agent.
        'G' represents the Goal.
        '.' represents an empty cell.
        """
        for r in range(self.height):
            line = ""
            for c in range(self.width):
                if (r, c) == self.agent_pos:
                    line += "A "
                elif (r, c) == self.goal:
                    line += "G "
                else:
                    line += ". "
            print(line)
        print()

    def render_rgb_array(self):
        """
        Renders the grid as an RGB image using Matplotlib.
        
        This method creates a figure, draws the grid, agent, and goal,
        and converts the plot into a numpy array suitable for video generation.
        
        Returns:
            numpy.ndarray: An RGB image array of shape (height, width, 3).
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Setup the grid axes
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xticks(np.arange(0, self.width + 1, 1))
        ax.set_yticks(np.arange(0, self.height + 1, 1))
        ax.grid(True, color='black')
        ax.set_aspect('equal')
        
        # Invert y-axis to match array indexing (0,0 at top-left)
        # In matrix coordinates, row increases downwards.
        ax.invert_yaxis()
        
        # Draw Agent (Red Circle)
        # We add 0.5 to coordinates to center the shape in the cell
        agent_circle = plt.Circle((self.agent_pos[1] + 0.5, self.agent_pos[0] + 0.5), 0.3, color='red', label='Agent')
        ax.add_patch(agent_circle)
        
        # Draw Goal (Green Square)
        goal_rect = plt.Rectangle((self.goal[1], self.goal[0]), 1, 1, color='green', alpha=0.5, label='Goal')
        ax.add_patch(goal_rect)
        
        # Remove axis ticks/labels for a cleaner look
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        plt.legend(loc='upper right')
        
        # Draw the canvas and convert to numpy array
        fig.canvas.draw()
        # buffer_rgba returns a buffer representation of the image
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        
        plt.close(fig) # Close the figure to prevent memory leaks
        return data[..., :3]  # Return RGB channels, omitting Alpha
