
import matplotlib.pyplot as plt
import numpy as np
from src.environment.grid_world import GridWorld
import os

def verify_overlaps():
    # Setup env
    env = GridWorld(
        height=5, width=5, start=(0, 0), resource_pos=(1, 1),
        with_satiation=True, max_steps=100,
        prob_switch_to_danger=0.5, min_danger_duration=1, damage_amount=1,
        prob_switch_to_food=0.5, min_food_duration=1, relocate_resource=False, relocation_steps=10,
        vector_size=5, food_property=[1]*5, danger_property=[1]*5,
        predator_enabled=True
    )
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Agent + Food
    env.agent_pos = (1, 1)
    env.resource_pos = (1, 1)
    env.resource_state = 'food'
    env.predator_pos = (4, 4)
    img1 = env.render_rgb_array()
    axes[0].imshow(img1)
    axes[0].set_title("Agent + Food (Eating)")
    axes[0].axis('off')
    
    # 2. Agent + Danger
    env.resource_state = 'danger'
    img2 = env.render_rgb_array()
    axes[1].imshow(img2)
    axes[1].set_title("Agent + Danger (Pain)")
    axes[1].axis('off')
    
    # 3. Agent + Predator
    env.resource_pos = (4, 4)
    env.predator_pos = (1, 1)
    img3 = env.render_rgb_array()
    axes[2].imshow(img3)
    axes[2].set_title("Agent + Predator (Clash)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/vncuser/.gemini/antigravity/brain/6c384cfd-2b68-4dd4-9a1f-dffbdcee6f23/overlap_verification.png')
    print("Verification image saved.")

if __name__ == "__main__":
    verify_overlaps()
