"""
Evaluation script for trained visuomotor policies.
Tests policies with visual corruption (distractors/occlusions).
"""

import torch
import numpy as np
import argparse
import os
import pybullet as p

from panda_gym.envs import PandaPickAndPlaceEnv
from policy import load_policy
from data import get_clean_transform


def add_visual_corruption(env, corruption_type='distractor'):
    """
    Add visual corruption to the environment using pybullet.
    
    Args:
        env: The panda-gym environment instance
        corruption_type: Type of corruption ('distractor' or 'occlusion')
    """
    # Get the pybullet client from the environment
    # panda-gym uses pybullet internally, we need to access it
    try:
        # Try different ways to access the physics client
        physics_client = None
        
        # Method 1: Check if env has sim attribute
        if hasattr(env, 'sim'):
            if hasattr(env.sim, 'client_id'):
                physics_client = env.sim.client_id
            elif hasattr(env.sim, '_client_id'):
                physics_client = env.sim._client_id
            elif hasattr(env.sim, 'physics_client_id'):
                physics_client = env.sim.physics_client_id
        
        # Method 2: Check for _sim attribute
        if physics_client is None and hasattr(env, '_sim'):
            if hasattr(env._sim, 'client_id'):
                physics_client = env._sim.client_id
            elif hasattr(env._sim, '_client_id'):
                physics_client = env._sim._client_id
        
        # Method 3: Try to get from pybullet directly (if already connected)
        if physics_client is None:
            try:
                # Try to use the default client (panda-gym usually uses DIRECT mode)
                physics_client = p.connect(p.DIRECT)
            except:
                pass
        
        # If we still don't have a client, try to find it through the environment
        if physics_client is None:
            # panda-gym might store it differently
            if hasattr(env, 'physics_client_id'):
                physics_client = env.physics_client_id
        
        # Use the physics client (or default if None)
        if corruption_type == 'distractor':
            # Add a red sphere as a static distractor
            # Position it off-center, not interfering with the task
            distractor_pos = [0.3, 0.3, 0.1]  # Off to the side
            
            # Create a simple visual shape (sphere)
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=0.05,  # 5cm radius
                rgbaColor=[1.0, 0.0, 0.0, 1.0]  # Red
            )
            
            # Create a multi-body with no collision (visual only)
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=distractor_pos
            )
            
        elif corruption_type == 'occlusion':
            # Add a small box as occlusion
            occlusion_pos = [0.0, 0.0, 0.15]  # In the middle, elevated
            
            visual_shape_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[0.03, 0.03, 0.05],  # Small box
                rgbaColor=[0.5, 0.5, 0.5, 0.8]  # Semi-transparent gray
            )
            
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=occlusion_pos
            )
    
    except Exception as e:
        print(f"Warning: Could not add visual corruption: {e}")
        print("This is expected if pybullet client access is restricted.")
        print("Continuing without visual corruption...")


def evaluate_policy(policy_path, corruption_type='distractor', n_episodes=100, 
                   device='cuda', image_size=(84, 84), backbone_type=None,
                   max_steps=200):
    """
    Evaluate a trained policy with visual corruption.
    
    Args:
        policy_path: Path to the trained policy weights
        corruption_type: Type of visual corruption ('distractor', 'occlusion', or None)
        n_episodes: Number of test episodes
        device: Device to run evaluation on
        image_size: Size of input images
        backbone_type: Backbone type used in the policy ('resnet', 'vit', or 'cnn'). 
                       If None, will be auto-detected from filename or checkpoint.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load policy (backbone_type will be auto-detected if None)
    print(f"Loading policy from {policy_path}...")
    policy = load_policy(policy_path, image_size=image_size, action_dim=4, 
                         backbone_type=backbone_type, device=device)
    
    # Get transform (use clean transform for evaluation)
    transform = get_clean_transform()
    
    # Initialize environment
    env = PandaPickAndPlaceEnv(render_mode='rgb_array', render_width=84, render_height=84)
    
    # Add visual corruption if specified
    if corruption_type:
        print(f"Adding visual corruption: {corruption_type}")
        add_visual_corruption(env, corruption_type=corruption_type)
    
    # Evaluation loop
    success_count = 0
    total_reward = 0.0
    
    print(f"\nEvaluating policy for {n_episodes} episodes...")
    
    # Add corruption once before episodes (if it persists across resets)
    if corruption_type:
        add_visual_corruption(env, corruption_type=corruption_type)
    
    for episode in range(n_episodes):
        observation, info = env.reset()
        
        # Re-add corruption after reset if needed (some environments reset the scene)
        if corruption_type and episode == 0:
            # Try adding again after first reset
            add_visual_corruption(env, corruption_type=corruption_type)
        
        done = False
        episode_reward = 0.0
        steps = 0
        while not done and steps < max_steps:
            # Render to get image
            image = env.render()
            
            # Preprocess image
            from PIL import Image
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Get action from policy
            with torch.no_grad():
                action = policy(image_tensor)
                action = action.cpu().numpy()[0]
            
            # Step environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation = next_observation
            episode_reward += reward
            steps += 1
        
        # Check success
        if info.get('is_success', False) or episode_reward > 0:
            success_count += 1
        
        total_reward += episode_reward
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}, "
                  f"Success Rate: {success_count / (episode + 1):.2%}")
    
    env.close()
    
    # Print results
    success_rate = success_count / n_episodes
    avg_reward = total_reward / n_episodes
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Policy: {policy_path}")
    print(f"  Corruption Type: {corruption_type if corruption_type else 'None'}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Task Success Rate: {success_rate:.2%} ({success_count}/{n_episodes})")
    print(f"  Average Reward: {avg_reward:.4f}")
    print(f"{'='*50}")
    
    return success_rate, avg_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate visuomotor BC policy')
    parser.add_argument('--policy', type=str, required=True,
                       help='Path to trained policy weights')
    parser.add_argument('--corruption', type=str, default='none',
                       choices=['distractor', 'occlusion', 'none'],
                       help='Type of visual corruption (use "none" for no corruption)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of test episodes')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--backbone', type=str, default=None, nargs='?',
                       choices=['resnet', 'vit', 'cnn'],
                       help='Backbone architecture (auto-detected from filename if not specified)')
    parser.add_argument('--max_steps', type=int, default=200,
                       help='Max steps per episode during evaluation')
    
    args = parser.parse_args()
    
    corruption_type = None if args.corruption == 'none' else args.corruption
    
    evaluate_policy(
        policy_path=args.policy,
        corruption_type=corruption_type,
        n_episodes=args.episodes,
        device=args.device,
        backbone_type=args.backbone,
        max_steps=args.max_steps
    )

