"""
Expert data collection script for PandaPickAndPlace-v3 environment.
Implements a scripted expert policy and collects demonstration trajectories.
"""

import numpy as np
import pickle
from panda_gym.envs import PandaPickAndPlaceEnv


def get_expert_action(observation, env, state):
    """
    Scripted expert policy for PandaPickAndPlace-v3.
    
    State machine:
    1. Reach for the cube's position
    2. Move down to grasp
    3. Close gripper
    4. Move up to the target (lift) position
    
    Args:
        observation: Current environment observation (dict or array)
        env: The environment instance
        state: Dictionary tracking the current state of the expert policy
        
    Returns:
        action: 4-dimensional action vector [x, y, z, gripper]
    """
    # Handle panda-gym observation format (dict with 'observation', 'achieved_goal', 'desired_goal')
    if isinstance(observation, dict):
        # Extract observation array and goal positions
        obs_array = np.array(observation['observation'])
        gripper_pos = obs_array[:3]  # First 3 elements are gripper position
        object_pos = np.array(observation['achieved_goal'])  # Object position
        target_pos = np.array(observation['desired_goal'])  # Target position
        # Gripper state might be in obs_array, but we'll infer from action history
        gripper_state = obs_array[9] if len(obs_array) > 9 else 1.0
    else:
        # Fallback for array observations
        obs_array = np.array(observation)
        if len(obs_array) >= 9:
            gripper_pos = obs_array[:3]
            object_pos = obs_array[3:6]
            target_pos = obs_array[6:9]
            gripper_state = obs_array[9] if len(obs_array) > 9 else 1.0
        else:
            gripper_pos = obs_array[:3] if len(obs_array) >= 3 else np.zeros(3)
            object_pos = gripper_pos.copy()
            target_pos = gripper_pos.copy()
            gripper_state = 1.0
    
    # Initialize state if needed
    if state['phase'] == 'init':
        state['phase'] = 'reach'
        state['target_reached'] = False
        state['grasped'] = False
        state['lifted'] = False
    
    action = np.zeros(4)
    
    # Phase 1: Reach for the cube's position (move gripper above the cube)
    if state['phase'] == 'reach':
        target_above_cube = object_pos.copy()
        target_above_cube[2] += 0.1  # 10cm above the cube
        
        # Move towards the position above the cube
        direction = target_above_cube - gripper_pos
        distance = np.linalg.norm(direction)
        
        if distance < 0.02:  # Close enough
            state['phase'] = 'move_down'
        else:
            # Normalize direction and scale to action space [-1, 1]
            # Use a scaling factor to convert position differences to actions
            direction = direction / (distance + 1e-6)
            # Scale direction to reasonable action values (max 0.3 per step)
            max_step = 0.3
            action[:3] = np.clip(direction * min(distance / max_step, 1.0), -1.0, 1.0)
            action[3] = 1.0  # Keep gripper open
    
    # Phase 2: Move down to grasp
    elif state['phase'] == 'move_down':
        target_grasp = object_pos.copy()
        target_grasp[2] = object_pos[2] + 0.02  # Slightly above the cube center
        
        direction = target_grasp - gripper_pos
        distance = np.linalg.norm(direction)
        
        if distance < 0.015:
            state['phase'] = 'grasp'
        else:
            direction = direction / (distance + 1e-6)
            # Scale for precision movement
            max_step = 0.2
            action[:3] = np.clip(direction * min(distance / max_step, 1.0), -1.0, 1.0)
            action[3] = 1.0  # Keep gripper open
    
    # Phase 3: Close gripper
    elif state['phase'] == 'grasp':
        action[:3] = np.zeros(3)  # Don't move
        action[3] = -1.0  # Close gripper
        
        # Check if we've grasped (gripper closed for a few steps)
        if gripper_state < 0.5:  # Gripper is closed
            state['grasp_steps'] = state.get('grasp_steps', 0) + 1
            if state['grasp_steps'] > 5:  # Wait a few steps
                state['phase'] = 'lift'
                state['grasped'] = True
    
    # Phase 4: Move up to the target (lift) position
    elif state['phase'] == 'lift':
        # First lift up
        if not state['lifted']:
            target_lift = gripper_pos.copy()
            target_lift[2] = object_pos[2] + 0.15  # Lift 15cm up
            
            direction = target_lift - gripper_pos
            distance = np.linalg.norm(direction)
            
            if distance < 0.02:
                state['lifted'] = True
            else:
                direction = direction / (distance + 1e-6)
                max_step = 0.3
                action[:3] = np.clip(direction * min(distance / max_step, 1.0), -1.0, 1.0)
                action[3] = -1.0  # Keep gripper closed
        else:
            # Move towards target position
            target_place = target_pos.copy()
            target_place[2] = target_pos[2] + 0.1  # Above target
            
            direction = target_place - gripper_pos
            distance = np.linalg.norm(direction)
            
            if distance < 0.02:
                # Move down to place
                target_final = target_pos.copy()
                target_final[2] = target_pos[2] + 0.02
                
                direction = target_final - gripper_pos
                distance = np.linalg.norm(direction)
                
                if distance < 0.015:
                    action[3] = 1.0  # Open gripper to release
                else:
                    direction = direction / (distance + 1e-6)
                    max_step = 0.2
                    action[:3] = np.clip(direction * min(distance / max_step, 1.0), -1.0, 1.0)
                    action[3] = -1.0  # Keep gripper closed
            else:
                direction = direction / (distance + 1e-6)
                max_step = 0.3
                action[:3] = np.clip(direction * min(distance / max_step, 1.0), -1.0, 1.0)
                action[3] = -1.0  # Keep gripper closed
    
    return action


def collect_demonstrations(n_episodes=100, output_file='demonstrations.pkl'):
    """
    Collect expert demonstrations from the PandaPickAndPlace-v3 environment.
    
    Note: rgb_array rendering is slow (~50-100ms per frame). Each episode may take
    10-20 seconds depending on task complexity. For 100 episodes, expect ~15-30 minutes.
    
    Args:
        n_episodes: Number of episodes to collect
        output_file: Path to save the collected demonstrations
    """
    # Initialize environment
    # Note: rgb_array rendering is slow. Consider using 'human' mode for faster collection
    # or reduce render frequency if you don't need every frame
    env = PandaPickAndPlaceEnv(render_mode='rgb_array', render_width=84, render_height=84)
    
    all_trajectories = []
    successful_episodes = 0
    
    print(f"Collecting {n_episodes} expert demonstrations...")
    print("Note: rgb_array rendering is slow (~50-100ms/frame). This will take 15-30 minutes for 100 episodes.")
    print("Progress will be shown after each episode completes.\n")
    
    import time
    start_time = time.time()
    
    for episode in range(n_episodes):
        observation, info = env.reset()
        trajectory = []
        state = {'phase': 'init', 'grasp_steps': 0}
        done = False
        steps = 0
        max_steps = 200  # Maximum steps per episode
        
        while not done and steps < max_steps:
            # Get expert action
            action = get_expert_action(observation, env, state)
            
            # Render to get image (this is the bottleneck - rgb_array rendering is slow)
            # We need images for BC training, so we must render every step
            # But we can skip rendering for failed episodes to save time
            image = env.render()
            # Store (image, action) tuple - use array() for better performance than .copy()
            trajectory.append((np.array(image), np.array(action)))
            
            # Step environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation = next_observation
            steps += 1
            
            # Early termination if clearly failed (saves time)
            if steps > 50 and reward < -0.5:
                break
        
        # Check if episode was successful
        elapsed = time.time() - start_time
        avg_time_per_ep = elapsed / (episode + 1)
        remaining_eps = n_episodes - (episode + 1)
        eta_seconds = avg_time_per_ep * remaining_eps
        eta_min = int(eta_seconds // 60)
        eta_sec = int(eta_seconds % 60)
        
        if info.get('is_success', False) or reward > 0:
            all_trajectories.append(trajectory)
            successful_episodes += 1
            print(f"Episode {episode + 1}/{n_episodes}: ✓ Success ({steps} steps, {avg_time_per_ep:.1f}s/ep, ETA: {eta_min}m{eta_sec}s, Success rate: {100*successful_episodes/(episode+1):.1f}%)")
        else:
            print(f"Episode {episode + 1}/{n_episodes}: ✗ Failed ({steps} steps, {avg_time_per_ep:.1f}s/ep, ETA: {eta_min}m{eta_sec}s)")
    
    env.close()
    
    print(f"\nCollection complete!")
    print(f"Successful episodes: {successful_episodes}/{n_episodes}")
    print(f"Total trajectories: {len(all_trajectories)}")
    
    # Save trajectories
    with open(output_file, 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    print(f"Saved demonstrations to {output_file}")
    
    return all_trajectories


if __name__ == '__main__':
    # Collect demonstrations
    # Note: rgb_array rendering is the main bottleneck (~50-100ms per frame)
    # Each episode takes ~10-20 seconds. For 100 episodes, expect 15-30 minutes total.
    collect_demonstrations(n_episodes=100)

