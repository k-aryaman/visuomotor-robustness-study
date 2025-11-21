import numpy as np
import pickle
from panda_gym.envs import PandaPickAndPlaceEnv
import time
import os
import cv2

def get_expert_action(observation, env, state):
    """
    Scripted expert policy for PandaPickAndPlace-v3.
    Performs: reach -> move_down -> grasp -> lift -> place
    """
    # Extract positions
    obs_array = observation['observation'] if isinstance(observation, dict) else observation
    obs_array = np.array(obs_array)
    
    gripper_pos = obs_array[:3]
    object_pos = obs_array[3:6]
    target_pos = obs_array[6:9]
    gripper_state = obs_array[9] if len(obs_array) > 9 else 1.0  # 1=open, -1=closed

    # Initialize state
    if 'phase' not in state:
        state.update({'phase': 'reach', 'grasped': False, 'lifted': False, 'grasp_steps': 0})

    action = np.zeros(4)

    # -------------------
    # Phase 1: Reach above cube
    if state['phase'] == 'reach':
        target = object_pos.copy()
        target[2] += 0.1  # above cube
        direction = target - gripper_pos
        distance = np.linalg.norm(direction)

        if distance < 0.02:
            state['phase'] = 'move_down'
        else:
            action[:3] = np.clip(direction * 5.0, -1, 1)
            action[3] = 1.0  # keep open

    # -------------------
    # Phase 2: Move down to grasp
    elif state['phase'] == 'move_down':
        target = object_pos.copy()
        target[2] += 0.02
        direction = target - gripper_pos
        distance = np.linalg.norm(direction)

        if distance < 0.015:
            state['phase'] = 'grasp'
        else:
            action[:3] = np.clip(direction * 5.0, -1, 1)
            action[3] = 1.0

    # -------------------
    # Phase 3: Close gripper
    elif state['phase'] == 'grasp':
        action[:3] = 0
        action[3] = -1.0  # close
        state['grasp_steps'] += 1
        if state['grasp_steps'] >= 5:
            state['phase'] = 'lift'
            state['grasped'] = True

    # -------------------
    # Phase 4: Lift object
    elif state['phase'] == 'lift':
        target = gripper_pos.copy()
        target[2] += 0.15
        direction = target - gripper_pos
        distance = np.linalg.norm(direction)

        if distance < 0.02:
            state['phase'] = 'move_to_target'
            state['lifted'] = True
        else:
            action[:3] = np.clip(direction * 3.0, -1, 1)
            action[3] = -1.0

    # -------------------
    # Phase 5: Move above target
    elif state['phase'] == 'move_to_target':
        target = target_pos.copy()
        target[2] += 0.1
        direction = target - gripper_pos
        distance = np.linalg.norm(direction)

        if distance < 0.02:
            state['phase'] = 'move_down_target'
        else:
            action[:3] = np.clip(direction * 3.0, -1, 1)
            action[3] = -1.0

    # -------------------
    # Phase 6: Move down to place
    elif state['phase'] == 'move_down_target':
        target = target_pos.copy()
        target[2] += 0.02
        direction = target - gripper_pos
        distance = np.linalg.norm(direction)

        if distance < 0.015:
            state['phase'] = 'release'
        else:
            action[:3] = np.clip(direction * 3.0, -1, 1)
            action[3] = -1.0

    # -------------------
    # Phase 7: Release object
    elif state['phase'] == 'release':
        action[:3] = 0
        action[3] = 1.0
        state['phase'] = 'done'

    return action

def collect_demonstrations(n_episodes=100, output_file='demonstrations.pkl', video_dir='videos'):
    """
    Collect expert demonstrations from PandaPickAndPlace-v3 and save videos of successful episodes.
    
    Args:
        n_episodes: Number of episodes to collect
        output_file: Path to save the collected demonstrations
        video_dir: Directory to save videos of successful episodes
    """
    # Initialize environment
    env = PandaPickAndPlaceEnv(render_mode='rgb_array', render_width=480, render_height=480)
    
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    all_trajectories = []
    successful_episodes = 0
    
    print(f"Collecting {n_episodes} expert demonstrations...\n")
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        observation, info = env.reset()
        trajectory = []
        state = {'phase': 'init', 'grasp_steps': 0}
        done = False
        steps = 0
        max_steps = 200
        
        while not done and steps < max_steps:
            action = get_expert_action(observation, env, state)
            image = env.render()  # RGB frame
            trajectory.append((np.array(image), np.array(action)))
            
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation = next_observation
            steps += 1
            
            # Early termination if clearly failed
            if steps > 50 and reward < -0.5:
                break
        
        # Check if episode was successful
        if info.get('is_success', False) or reward > 0:
            all_trajectories.append(trajectory)
            successful_episodes += 1
            
        # Save video of this trajectory
        successString = "Success" if info.get('is_success', False) or reward > 0 else "Failed"
        video_path = os.path.join(video_dir, f"episode_{episode+1:03d}_{successString}.mp4")
        height, width, _ = trajectory[0][0].shape
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        
        for frame, _ in trajectory:
            # Convert RGB to BGR for OpenCV
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video_writer.release()
            
        if successString == "Success":
            print(f"Episode {episode + 1}/{n_episodes}: ✓ Success ({steps} steps, video saved)")
        else:
            print(f"Episode {episode + 1}/{n_episodes}: ✗ Failed ({steps} steps)")
    
    env.close()
    
    # Save demonstrations
    with open(output_file, 'wb') as f:
        pickle.dump(all_trajectories, f)
    
    print(f"\nCollection complete! Successful episodes: {successful_episodes}/{n_episodes}")
    print(f"Saved demonstrations to {output_file}")
    print(f"Videos saved in folder: {video_dir}")
    
    return all_trajectories
if __name__ == '__main__':
    collect_demonstrations(n_episodes=50)  # reduce for testing