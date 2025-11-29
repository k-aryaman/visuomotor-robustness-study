"""
Expert data collection script for PandaReach-v3 environment.
Implements a scripted expert policy and collects demonstration trajectories.
"""

import numpy as np
import pickle
import argparse
from panda_gym.envs import PandaReachEnv


def get_expert_action(observation, env, state):
    """
    Scripted expert policy for PandaReach-v3.
    
    Simple policy: Move end-effector directly toward the target position.
    
    Args:
        observation: Current environment observation (dict with 'observation', 'achieved_goal', 'desired_goal')
        env: The environment instance
        state: Dictionary tracking the current state of the expert policy (unused for reach)
        
    Returns:
        action: 3-dimensional action vector [dx, dy, dz] (no gripper control for reach task)
    """
    # Handle panda-gym observation format (dict with 'observation', 'achieved_goal', 'desired_goal')
    if isinstance(observation, dict):
        obs_array = np.array(observation['observation'])
        gripper_pos = obs_array[:3]  # end-effector position
        target_pos = np.array(observation['desired_goal'])
    else:
        obs_array = np.array(observation)
        gripper_pos = obs_array[:3] if len(obs_array) >= 3 else np.zeros(3)
        target_pos = obs_array[3:6] if len(obs_array) >= 6 else gripper_pos.copy()
    
    # Simple policy: move directly toward target
    direction = target_pos - gripper_pos
    distance = np.linalg.norm(direction)
    
    if distance < 0.01:  # Very close to target, use small action
        action = direction * 10.0  # Scale up small distances
    else:
        # Normalize direction and scale by distance (closer = smaller steps)
        direction = direction / (distance + 1e-6)
        # Scale action magnitude: closer to target = smaller steps
        max_step = 0.1  # Maximum step size
        step_size = min(max_step, distance * 0.5)  # Scale down as we approach
        action = direction * step_size
    
    # Clip to action space bounds [-1, 1]
    action = np.clip(action, -1.0, 1.0)
    
    return action


def collect_demonstrations(n_episodes=100, output_file='demonstrations.pkl'):
    """
    Collect expert demonstrations from the PandaReach-v3 environment.
    
    Note: rgb_array rendering is slow (~50-100ms per frame). Each episode may take
    5-10 seconds depending on task complexity. For 100 episodes, expect ~10-20 minutes.
    
    Args:
        n_episodes: Number of episodes to collect
        output_file: Path to save the collected demonstrations
    """
    # Initialize environment
    # Note: rgb_array rendering is slow. Consider using 'human' mode for faster collection
    # or reduce render frequency if you don't need every frame
    env = PandaReachEnv(render_mode='rgb_array', render_width=84, render_height=84)
    
    all_trajectories = []
    successful_episodes = 0
    
    print(f"Collecting {n_episodes} expert demonstrations...")
    print("Note: rgb_array rendering is slow (~50-100ms/frame). This will take 10-20 minutes for 100 episodes.")
    print("Progress will be shown after each episode completes.\n")
    
    import time
    start_time = time.time()
    
    for episode in range(n_episodes):
        observation, info = env.reset()
        trajectory = []
        state = {}  # No complex state machine needed for reach task
        
        # Store initial end-effector position (box start) and target position
        if isinstance(observation, dict):
            start_pos = np.array(observation['achieved_goal']).copy()  # Initial end-effector position
            target_pos = np.array(observation['desired_goal']).copy()  # Target position
        else:
            obs_array = np.array(observation)
            start_pos = obs_array[:3].copy() if len(obs_array) >= 3 else np.zeros(3)
            target_pos = obs_array[3:6].copy() if len(obs_array) >= 6 else np.zeros(3)
        
        done = False
        steps = 0
        max_steps = 200  # Maximum steps per episode
        
        while not done and steps < max_steps:
            # Get expert action
            action = get_expert_action(observation, env, state)
            
            # Render to get image (this is the bottleneck - rgb_array rendering is slow)
            # We need images for BC training, so we must render every step
            image = env.render()
            
            # Extract proprioceptive state from observation (only gripper position for reach task)
            if isinstance(observation, dict):
                obs_array = np.array(observation['observation'])
                gripper_pos = obs_array[:3]  # end-effector position
            else:
                obs_array = np.array(observation)
                gripper_pos = obs_array[:3] if len(obs_array) >= 3 else np.zeros(3)
            
            # Proprioceptive state: only gripper_pos (3D) - no gripper_width for reach task
            proprio_state = gripper_pos.copy()
            
            # Store (image, proprio_state, action) tuple
            trajectory.append((np.array(image), np.array(proprio_state), np.array(action)))
            
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
            # Store trajectory with metadata: (trajectory, metadata)
            # metadata contains start_pos and target_pos for reference (even if not used later)
            trajectory_with_metadata = {
                'trajectory': trajectory,
                'start_pos': start_pos,
                'target_pos': target_pos
            }
            all_trajectories.append(trajectory_with_metadata)
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
    parser = argparse.ArgumentParser(description="Collect expert demonstrations for PandaReach-v3.")
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to collect')
    parser.add_argument('--output', type=str, default='demonstrations.pkl', help='Output pickle filename')
    args = parser.parse_args()

    # Note: rgb_array rendering is the main bottleneck (~50-100ms per frame)
    # Each episode takes ~5-10 seconds. For 100 episodes, expect 10-20 minutes total.
    collect_demonstrations(n_episodes=args.episodes, output_file=args.output)

