"""
Evaluation script for trained visuomotor policies.
Tests policies with visual corruption (distractors/occlusions).
"""

import torch
import numpy as np
import argparse
import os
import pybullet as p
from PIL import Image

from panda_gym.envs import PandaReachEnv
from policy import load_policy
from data import get_clean_transform
from action_utils import spherical_to_cartesian


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
    policy = load_policy(policy_path, image_size=image_size, action_dim=3,  # Reach task: 3D actions 
                         backbone_type=backbone_type, device=device)
    
    # Get transform (use clean transform for evaluation)
    transform = get_clean_transform()
    
    # Initialize environment
    env = PandaReachEnv(render_mode='rgb_array', render_width=84, render_height=84)
    
    # Add visual corruption if specified
    if corruption_type:
        print(f"Adding visual corruption: {corruption_type}")
        add_visual_corruption(env, corruption_type=corruption_type)
    
    # Evaluation loop
    success_count = 0
    total_reward = 0.0
    
    # Store trajectories for visualization with success/failure labels
    trajectories_for_viz = []  # List of (images, is_success, episode_idx)
    
    print(f"\nEvaluating policy for {n_episodes} episodes...")
    
    # Add corruption once before episodes (if it persists across resets)
    if corruption_type:
        add_visual_corruption(env, corruption_type=corruption_type)
    
    episode = 0
    skipped_episodes = 0
    max_skip_attempts = 1000  # Prevent infinite loops
    
    while episode < n_episodes and skipped_episodes < max_skip_attempts:
        observation, info = env.reset()
        
        # Check if object and target positions overlap (trivial success case)
        if isinstance(observation, dict):
            object_pos = np.array(observation['achieved_goal'])
            target_pos = np.array(observation['desired_goal'])
        else:
            obs_array = np.array(observation)
            object_pos = obs_array[7:10] if len(obs_array) >= 10 else np.zeros(3)
            target_pos = obs_array[10:13] if len(obs_array) >= 13 else np.zeros(3)
        
        # Check if object and target are too close (within 2cm threshold)
        distance = np.linalg.norm(object_pos - target_pos)
        if distance < 0.02:
            skipped_episodes += 1
            if skipped_episodes % 10 == 0:
                print(f"Skipped {skipped_episodes} trivial episodes (object at target location)...")
            continue
        
        # Re-add corruption after reset if needed (some environments reset the scene)
        if corruption_type and episode == 0:
            # Try adding again after first reset
            add_visual_corruption(env, corruption_type=corruption_type)
        
        done = False
        episode_reward = 0.0
        steps = 0
        episode_images = []  # Store images for this episode
        hidden_state = None  # Initialize GRU hidden state for this episode
        
        while not done and steps < max_steps:
            # Render to get image
            image = env.render()
            
            # Store raw image for visualization (before transform)
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image_copy = (image * 255).astype(np.uint8)
                else:
                    image_copy = image.copy()
                episode_images.append(image_copy)
            
            # Extract proprioceptive state from observation (only gripper position for reach task)
            state_tensor = None
            if hasattr(policy, 'state_dim') and policy.state_dim > 0:
                if isinstance(observation, dict):
                    obs_array = np.array(observation['observation'])
                    gripper_pos = obs_array[:3]  # end-effector position
                else:
                    obs_array = np.array(observation)
                    gripper_pos = obs_array[:3] if len(obs_array) >= 3 else np.zeros(3)
                
                # Proprioceptive state: only gripper_pos (3D) - no gripper_width needed for reach task
                proprio_state = gripper_pos
                state_tensor = torch.FloatTensor(proprio_state).unsqueeze(0).to(device)
            
            # Preprocess image for policy
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image_pil = Image.fromarray(image)
            
            image_tensor = transform(image_pil).unsqueeze(0).to(device)
            
            # Get action from policy
            with torch.no_grad():
                # Handle GRU if enabled (maintain hidden state across steps in episode)
                if hasattr(policy, 'use_gru') and policy.use_gru:
                    action_output, hidden_state = policy(image_tensor, state_tensor, hidden=hidden_state)
                    action_output = action_output.cpu().numpy()[0]
                else:
                    action_output = policy(image_tensor, state_tensor)
                    action_output = action_output.cpu().numpy()[0]
            
            # Check if model was trained with spherical coordinates
            use_spherical = False
            if hasattr(policy, 'use_spherical'):
                use_spherical = policy.use_spherical
            else:
                # Try to detect from checkpoint metadata
                try:
                    checkpoint = torch.load(policy_path, map_location='cpu')
                    if 'use_spherical' in checkpoint.get('metadata', {}):
                        use_spherical = checkpoint['metadata']['use_spherical']
                    elif 'use_spherical' in checkpoint:
                        use_spherical = checkpoint['use_spherical']
                except:
                    pass
            
            # Convert based on coordinate system
            if use_spherical:
                # Model outputs are in tanh range [-1, 1], need to scale to spherical ranges
                # Magnitude: sigmoid of first component -> [0, 1]
                # Theta: tanh of second component -> [-1, 1] -> scale to [0, π]
                # Phi: tanh of third component -> [-1, 1] -> scale to [-π, π]
                magnitude = 1.0 / (1.0 + np.exp(-action_output[0]))  # sigmoid: [0, 1]
                theta = (action_output[1] + 1.0) * 0.5 * np.pi  # [0, π]
                phi = action_output[2] * np.pi  # [-π, π]
                
                # Convert to Cartesian
                action_spherical_scaled = np.array([magnitude, theta, phi])
                action = spherical_to_cartesian(action_spherical_scaled)
            else:
                # Direct Cartesian output
                action = action_output
            
            # Debug: print action stats for first few episodes (before step)
            if episode < 3:
                action_magnitude = np.linalg.norm(action)  # Magnitude of action (3D for reach)
                if use_spherical:
                    print(f"  Episode {episode + 1}, Step {steps}: spherical = {action_spherical_scaled}, "
                          f"cartesian = {action}, magnitude = {action_magnitude:.4f}")
                else:
                    print(f"  Episode {episode + 1}, Step {steps}: cartesian = {action}, magnitude = {action_magnitude:.4f}")
            
            # Step environment (actions are 3D Cartesian for reach task)
            next_observation, reward, terminated, truncated, info = env.step(action)
            
            # Debug: print reward after step
            if episode < 3:
                print(f"    -> reward = {reward:.3f}")
            done = terminated or truncated
            observation = next_observation
            episode_reward += reward
            steps += 1
        
        # Check success
        is_success = info.get('is_success', False) or episode_reward > 0
        if is_success:
            success_count += 1
        
        # Store trajectory for visualization with success label
        if len(episode_images) > 0:
            trajectories_for_viz.append((episode_images, is_success, episode))
        
        total_reward += episode_reward
        episode += 1  # Increment episode counter after valid episode
        
        if episode % 10 == 0:
            print(f"Episode {episode}/{n_episodes} (skipped {skipped_episodes} trivial), "
                  f"Success Rate: {success_count / episode:.2%}")
    
    env.close()
    
    # Print results
    if episode > 0:
        success_rate = success_count / episode
        avg_reward = total_reward / episode
    else:
        success_rate = 0.0
        avg_reward = 0.0
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Policy: {policy_path}")
    print(f"  Corruption Type: {corruption_type if corruption_type else 'None'}")
    print(f"  Episodes: {episode} (skipped {skipped_episodes} trivial cases)")
    print(f"  Task Success Rate: {success_rate:.2%} ({success_count}/{episode})")
    print(f"  Average Reward: {avg_reward:.4f}")
    print(f"{'='*50}")
    
    # Save trajectories with metadata for later visualization
    eval_trajectories_file = 'eval_trajectories_reach.pkl'
    import pickle
    with open(eval_trajectories_file, 'wb') as f:
        pickle.dump(trajectories_for_viz, f)
    print(f"\nSaved {len(trajectories_for_viz)} evaluation trajectories to {eval_trajectories_file}")
    
    # Create preview images from evaluation episodes
    if len(trajectories_for_viz) > 0:
        print(f"\nCreating preview images from first {min(5, len(trajectories_for_viz))} episodes...")
        os.makedirs('eval_previews_reach', exist_ok=True)
        
        for ep_idx, (trajectory, _, _) in enumerate(trajectories_for_viz[:5]):
            create_eval_preview(trajectory, 
                              f'eval_previews_reach/episode_{ep_idx+1:03d}.png',
                              grid_size=(4, 4))
        print(f"Saved preview images to eval_previews_reach/ directory")
    
    return success_rate, avg_reward, trajectories_for_viz


def create_eval_preview(images, output_file, grid_size=(4, 4)):
    """
    Create a preview grid image from evaluation episode images.
    Similar to create_preview.py but works with a list of images directly.
    
    Args:
        images: List of numpy arrays (images from the episode)
        output_file: Path to save the preview image
        grid_size: Tuple of (rows, cols) for the grid (default: (4, 4))
    """
    if len(images) == 0:
        return
    
    grid_rows, grid_cols = grid_size
    total_grid_frames = grid_rows * grid_cols
    
    # Select evenly spaced frames from the trajectory
    num_frames = len(images)
    if num_frames <= total_grid_frames:
        # Use all frames if trajectory is shorter than grid
        selected_indices = list(range(num_frames))
        # Pad with last frame if needed
        while len(selected_indices) < total_grid_frames:
            selected_indices.append(num_frames - 1)
    else:
        # Select evenly spaced frames
        step = num_frames / total_grid_frames
        selected_indices = [int(i * step) for i in range(total_grid_frames)]
    
    # Extract and process images
    processed_images = []
    for idx in selected_indices[:total_grid_frames]:
        image = images[idx]
        
        # Convert to numpy array if not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Handle dtype - ensure uint8
        if image.dtype != np.uint8:
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Handle image shape
        if len(image.shape) == 2:
            # Grayscale, convert to RGB
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                # RGBA, convert to RGB
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError(f"Unexpected image shape: {image.shape}, expected (H, W, 3) or (H, W, 4)")
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}, expected 2D or 3D array")
        
        processed_images.append(image)
    
    # Create grid
    img_height, img_width = processed_images[0].shape[:2]
    grid_image = np.zeros((grid_rows * img_height, grid_cols * img_width, 3), dtype=np.uint8)
    
    for i, img in enumerate(processed_images):
        row = i // grid_cols
        col = i % grid_cols
        y_start = row * img_height
        y_end = y_start + img_height
        x_start = col * img_width
        x_end = x_start + img_width
        grid_image[y_start:y_end, x_start:x_end] = img
    
    # Save image
    pil_image = Image.fromarray(grid_image)
    pil_image.save(output_file)


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
    
    success_rate, avg_reward, _ = evaluate_policy(
        policy_path=args.policy,
        corruption_type=corruption_type,
        n_episodes=args.episodes,
        device=args.device,
        backbone_type=args.backbone,
        max_steps=args.max_steps,
    )

