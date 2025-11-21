import numpy as np
import pickle
from panda_gym.envs import PandaPickAndPlaceEnv
import cv2
import os
import time

def get_expert_action(observation, env, state):
    # --- same as your original get_expert_action ---
    # Keep all your existing code, but add debug printing:
    if isinstance(observation, dict):
        obs_array = np.array(observation['observation'])
        gripper_pos = obs_array[:3]
        object_pos = np.array(observation['achieved_goal'])
        target_pos = np.array(observation['desired_goal'])
        gripper_state = obs_array[9] if len(obs_array) > 9 else 1.0
    else:
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

    # Initialize state
    if state['phase'] == 'init':
        state['phase'] = 'reach'
        state['target_reached'] = False
        state['grasped'] = False
        state['lifted'] = False

    action = np.zeros(4)

    # --- existing phase logic ---
    # ... keep all your phase handling logic ...

    # Debug printing for visualization
    print(f"Phase: {state['phase']}, Gripper: {gripper_pos}, Cube: {object_pos}, Target: {target_pos}, Gripper state: {gripper_state}")

    return action

def collect_demonstrations(n_episodes=100, output_file='demonstrations.pkl', human_render=False, save_video=True):
    """
    Collect expert demonstrations and optionally visualize/save video.
    """
    render_mode = 'human' if human_render else 'rgb_array'
    env = PandaPickAndPlaceEnv(render_mode=render_mode, render_width=84, render_height=84)
    
    all_trajectories = []
    successful_episodes = 0

    os.makedirs("videos", exist_ok=True)

    start_time = time.time()

    for episode in range(n_episodes):
        observation, info = env.reset()
        trajectory = []
        state = {'phase': 'init', 'grasp_steps': 0}
        done = False
        steps = 0
        max_steps = 200

        frames = []

        while not done and steps < max_steps:
            action = get_expert_action(observation, env, state)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            if render_mode == 'rgb_array':
                frame = env.render()
                frames.append(frame)
                trajectory.append((np.array(frame), np.array(action)))
            else:
                env.render()
                trajectory.append((None, np.array(action)))  # images not saved

            observation = next_observation
            steps += 1

            if steps > 50 and reward < -0.5:
                break

        # Save video for this episode
        if save_video and frames:
            h, w, _ = frames[0].shape
            out = cv2.VideoWriter(f'videos/episode_{episode+1:03d}.mp4',
                                  cv2.VideoWriter_fourcc(*'mp4v'), 20, (w,h))
            for f in frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()

        if info.get('is_success', False) or reward > 0:
            all_trajectories.append(trajectory)
            successful_episodes += 1
            print(f"Episode {episode+1}/{n_episodes}: ✓ Success ({steps} steps, success rate: {100*successful_episodes/(episode+1):.1f}%)")
        else:
            print(f"Episode {episode+1}/{n_episodes}: ✗ Failed ({steps} steps)")

    env.close()

    print(f"\nCollection complete! Successful episodes: {successful_episodes}/{n_episodes}")
    with open(output_file, 'wb') as f:
        pickle.dump(all_trajectories, f)
    print(f"Saved demonstrations to {output_file}")

    return all_trajectories

if __name__ == '__main__':
    # Set human_render=True to see the GUI and debug visually
    collect_demonstrations(n_episodes=10, human_render=True, save_video=True)