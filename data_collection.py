"""
Expert data collection script for PandaPickAndPlace-v3 environment.
Implements a scripted expert policy and collects demonstration trajectories.
Test change
"""

import numpy as np
import pickle
import argparse
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
        obs_array = np.array(observation['observation'])
        gripper_pos = obs_array[:3]  # end-effector position
        gripper_width = obs_array[6] if len(obs_array) > 6 else 0.0  # finger opening
        object_pos = np.array(observation['achieved_goal'])
        target_pos = np.array(observation['desired_goal'])
    else:
        obs_array = np.array(observation)
        gripper_pos = obs_array[:3] if len(obs_array) >= 3 else np.zeros(3)
        gripper_width = obs_array[6] if len(obs_array) > 6 else 0.0
        object_pos = obs_array[7:10] if len(obs_array) >= 10 else gripper_pos.copy()
        target_pos = obs_array[10:13] if len(obs_array) >= 13 else gripper_pos.copy()
    
    # Initialize state if needed
    if state['phase'] == 'init':
        state['phase'] = 'reach'
        state['target_reached'] = False
        state['grasped'] = False
        state['lifted'] = False
        state['grasp_steps'] = 0
    
    action = np.zeros(4)
    
    # Phase 1: Reach for the cube's position (move gripper above the cube)
    if state['phase'] == 'reach':
        target = np.array([object_pos[0], object_pos[1], 0.25])
        direction = target - gripper_pos
        distance = np.linalg.norm(direction)
        if distance < 0.02:
            state['phase'] = 'move_down'
        else:
            direction = direction / (distance + 1e-6)
            action[:3] = np.clip(direction, -1.0, 1.0)
            action[3] = 1.0
    
    # Phase 2: Move down to grasp
    elif state['phase'] == 'move_down':
        target = np.array([object_pos[0], object_pos[1], 0.035])
        direction = target - gripper_pos
        distance = np.linalg.norm(direction)
        if distance < 0.008:
            state['phase'] = 'grasp'
        else:
            direction = direction / (distance + 1e-6)
            action[:3] = np.clip(direction * 0.5, -1.0, 1.0)
            action[3] = 1.0
    
    # Phase 3: Close gripper
    elif state['phase'] == 'grasp':
        action[:3] = np.zeros(3)
        action[3] = -1.0  # close gripper
        state['grasp_steps'] += 1
        # consider grasped once fingers are mostly closed or after a few steps
        if gripper_width < 0.05 or state['grasp_steps'] > 12:
            state['phase'] = 'lift'
            state['grasped'] = True
    
    # Phase 4: Move up to the target (lift) position
    elif state['phase'] == 'lift':
        if not state['lifted']:
            target = np.array([gripper_pos[0], gripper_pos[1], 0.25])
            direction = target - gripper_pos
            distance = np.linalg.norm(direction)
            if distance < 0.02:
                state['lifted'] = True
            else:
                direction = direction / (distance + 1e-6)
                action[:3] = np.clip(direction, -1.0, 1.0)
                action[3] = -1.0
        elif not state.get('at_target_xy', False):
            target = np.array([target_pos[0], target_pos[1], 0.25])
            direction = target - gripper_pos
            distance = np.linalg.norm(direction)
            if distance < 0.02:
                state['at_target_xy'] = True
            else:
                direction = direction / (distance + 1e-6)
                action[:3] = np.clip(direction, -1.0, 1.0)
                action[3] = -1.0
        elif not state.get('placing', False):
            target = np.array([target_pos[0], target_pos[1], 0.05])
            direction = target - gripper_pos
            distance = np.linalg.norm(direction)
            if distance < 0.01:
                state['placing'] = True
            else:
                direction = direction / (distance + 1e-6)
                action[:3] = np.clip(direction * 0.5, -1.0, 1.0)
                action[3] = -1.0
        else:
            action[:3] = np.zeros(3)
            action[3] = 1.0  # open
    
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
            success_rate = 100 * successful_episodes / (episode + 1)
            print(
                f"Episode {episode + 1}/{n_episodes}: [OK] "
                f"({steps} steps, {avg_time_per_ep:.1f}s/ep, ETA: {eta_min}m{eta_sec}s, "
                f"Success rate: {success_rate:.1f}%)"
            )
        else:
            print(
                f"Episode {episode + 1}/{n_episodes}: [FAIL] "
                f"({steps} steps, {avg_time_per_ep:.1f}s/ep, ETA: {eta_min}m{eta_sec}s)"
            )
    
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
    parser = argparse.ArgumentParser(description="Collect expert demonstrations for PandaPickAndPlace-v3.")
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to collect')
    parser.add_argument('--output', type=str, default='demonstrations.pkl', help='Output pickle filename')
    args = parser.parse_args()

    # Note: rgb_array rendering is the main bottleneck (~50-100ms per frame)
    # Each episode takes ~10-20 seconds. For 100 episodes, expect 15-30 minutes total.
    collect_demonstrations(n_episodes=args.episodes, output_file=args.output)

