"""
Evaluate a trained policy and print per-episode distances to target.

Usage example:
    python evaluate_print_distances.py --policy models/policy_pixel_aug_resnet.pth --episodes 100 --out_csv distances.csv

This script uses the same observation parsing logic as `evaluate.py` to compute final distances.
"""

import argparse
import torch
import numpy as np
from panda_gym.envs import PandaPickAndPlaceEnv
from policy import load_policy
from data import get_clean_transform
from evaluate import add_visual_corruption
from PIL import Image
import pybullet as p
import signal


def _timeout_handler(signum, frame):
    raise TimeoutError("operation timed out")


def extract_final_positions(observation):
    """Extract final object and target positions from observation (dict or array).

    Returns (final_object_pos (3,), final_target_pos (3,)).
    """
    if isinstance(observation, dict):
        final_object_pos = np.array(observation.get('achieved_goal', observation.get('observation', [0, 0, 0])[:3]))
        final_target_pos = np.array(observation.get('desired_goal', [0, 0, 0]))
    else:
        obs_array = np.array(observation)
        if len(obs_array) >= 13:
            final_object_pos = obs_array[7:10]
            final_target_pos = obs_array[10:13]
        elif len(obs_array) >= 10:
            final_object_pos = obs_array[7:10]
            final_target_pos = np.zeros(3)
        elif len(obs_array) >= 3:
            final_object_pos = obs_array[:3]
            final_target_pos = np.zeros(3)
        else:
            final_object_pos = np.zeros(3)
            final_target_pos = np.zeros(3)
    return final_object_pos, final_target_pos


def evaluate_and_print(policy_path, corruption=None, n_episodes=100, device='cpu', max_steps=200, out_csv=''):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", flush=True)

    # Load policy
    try:
        policy = load_policy(policy_path, image_size=(84, 84), action_dim=4, device=device)
    except Exception as e:
        print(f"Error loading policy: {e}", flush=True)
        raise

    transform = get_clean_transform()

    print("Ensuring pybullet is connected in DIRECT mode (headless)...", flush=True)
    try:
        # If pybullet isn't connected, this will raise; connect in DIRECT mode
        p.getConnectionInfo()
        print("pybullet already connected", flush=True)
    except Exception:
        try:
            p.connect(p.DIRECT)
            print("Connected pybullet in DIRECT mode", flush=True)
        except Exception as e:
            print(f"Warning: failed to connect pybullet DIRECT: {e}", flush=True)

    print("Creating environment...", flush=True)
    env = PandaPickAndPlaceEnv(render_mode='rgb_array', render_width=84, render_height=84)
    print("Environment created.", flush=True)

    if corruption:
        print(f"Adding visual corruption: {corruption}", flush=True)
        try:
            add_visual_corruption(env, corruption_type=corruption)
        except Exception as e:
            print(f"Warning: add_visual_corruption failed: {e}", flush=True)

    results = []

    for ep in range(1, n_episodes + 1):
        print(f"Starting episode {ep}/{n_episodes}", flush=True)
        try:
            # Use alarm timeout to avoid hanging indefinitely in env.reset()
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(30)  # seconds
            observation, info = env.reset()
            signal.alarm(0)
        except TimeoutError:
            print(f"Timeout during env.reset() on episode {ep}", flush=True)
            break
        except Exception as e:
            print(f"Error during env.reset() on episode {ep}: {e}", flush=True)
            break
        # Try re-adding corruption after reset in case scene cleared
        if corruption:
            try:
                add_visual_corruption(env, corruption_type=corruption)
            except Exception:
                pass

        done = False
        steps = 0
        episode_reward = 0.0

        while not done and steps < max_steps:
            try:
                image = env.render()
            except Exception as e:
                print(f"Error during env.render() on episode {ep}, step {steps}: {e}", flush=True)
                done = True
                break

            # Prepare image tensor
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image_proc = (image * 255).astype(np.uint8)
                else:
                    image_proc = image
                pil = Image.fromarray(image_proc)
            else:
                pil = image

            img_tensor = transform(pil).unsqueeze(0).to(device)

            with torch.no_grad():
                action = policy(img_tensor)
                action = action.cpu().numpy()[0]

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observation = next_observation
            episode_reward += reward
            steps += 1

        final_object_pos, final_target_pos = extract_final_positions(observation)
        final_distance = float(np.linalg.norm(final_object_pos - final_target_pos))
        is_success = bool(info.get('is_success', False) or episode_reward > 0)

        results.append({'episode': ep, 'distance': final_distance, 'success': is_success, 'reward': episode_reward})
        status = 'SUCCESS' if is_success else 'FAIL'
        print(f"Episode {ep}: {status}, distance={final_distance:.4f} m, reward={episode_reward:.2f}, steps={steps}", flush=True)

    env.close()

    # Optionally save CSV
    if out_csv:
        import csv
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['episode', 'distance', 'success', 'reward'])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"Saved per-episode distances to {out_csv}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate policy and print per-episode distances')
    parser.add_argument('--policy', type=str, required=True, help='Path to trained policy (.pth)')
    parser.add_argument('--corruption', type=str, default=None, choices=['distractor', 'occlusion', 'none'], help='Visual corruption type')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to evaluate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--max_steps', type=int, default=200, help='Max steps per episode')
    parser.add_argument('--out_csv', type=str, default='', help='Optional CSV to save per-episode distances')

    args = parser.parse_args()

    corruption = None if args.corruption == 'none' else args.corruption

    evaluate_and_print(args.policy, corruption=corruption, n_episodes=args.episodes, device=args.device, max_steps=args.max_steps, out_csv=args.out_csv)
