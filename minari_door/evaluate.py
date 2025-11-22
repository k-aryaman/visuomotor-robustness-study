"""
Evaluation script for trained offline RL policies on door-opening task.
Tests policies with various perturbations to assess robustness.
"""

import torch
import numpy as np
import argparse
import os
import json
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
try:
    import gymnasium_robotics
    gymnasium_robotics.register_robotics_envs()
except ImportError:
    print("gymnasium-robotics not installed. Install with: pip install gymnasium-robotics")
    
import minari
from tqdm import tqdm

from policy import create_policy, load_policy


class ObservationPerturbator:
    """
    Applies various perturbations to observations during evaluation.
    """
    
    def __init__(
        self,
        perturbation_type: str,
        strength: float = 0.1,
        obs_mean: Optional[np.ndarray] = None,
        obs_std: Optional[np.ndarray] = None
    ):
        self.perturbation_type = perturbation_type
        self.strength = strength
        self.obs_mean = obs_mean if obs_mean is not None else 0.0
        self.obs_std = obs_std if obs_std is not None else 1.0
        
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Apply perturbation to observation."""
        if self.perturbation_type == 'none' or self.perturbation_type is None:
            return obs
        
        obs = obs.copy()
        
        if self.perturbation_type == 'gaussian':
            noise = np.random.normal(0, self.strength, obs.shape)
            obs = obs + noise * self.obs_std
            
        elif self.perturbation_type == 'uniform':
            noise = np.random.uniform(-self.strength, self.strength, obs.shape)
            obs = obs + noise * self.obs_std
            
        elif self.perturbation_type == 'dropout':
            mask = np.random.random(obs.shape) > self.strength
            obs = obs * mask
            
        elif self.perturbation_type == 'adversarial':
            # Simple adversarial: flip sign of random subset
            mask = np.random.random(obs.shape) < self.strength
            obs = np.where(mask, -obs, obs)
            
        elif self.perturbation_type == 'stuck':
            # Simulate stuck sensors - some values don't update
            if not hasattr(self, '_stuck_values'):
                self._stuck_values = obs.copy()
                self._stuck_mask = np.random.random(obs.shape) < self.strength
            obs = np.where(self._stuck_mask, self._stuck_values, obs)
            
        elif self.perturbation_type == 'delay':
            # Observation delay - use previous observation
            if not hasattr(self, '_obs_buffer'):
                self._obs_buffer = [obs.copy() for _ in range(5)]
            self._obs_buffer.append(obs.copy())
            delay_steps = int(self.strength * 5) + 1
            delayed_obs = self._obs_buffer[-delay_steps]
            self._obs_buffer = self._obs_buffer[-10:]  # Keep buffer bounded
            return delayed_obs
            
        elif self.perturbation_type == 'bias':
            # Constant bias on observations
            bias = self.strength * self.obs_std
            obs = obs + bias
            
        elif self.perturbation_type == 'scale':
            # Random scaling of observations
            scales = 1 + np.random.uniform(-self.strength, self.strength, obs.shape)
            obs = obs * scales
            
        return obs
    
    def reset(self):
        """Reset any stateful perturbation state."""
        if hasattr(self, '_stuck_values'):
            delattr(self, '_stuck_values')
        if hasattr(self, '_stuck_mask'):
            delattr(self, '_stuck_mask')
        if hasattr(self, '_obs_buffer'):
            delattr(self, '_obs_buffer')


def process_observation(obs, obs_mean=None, obs_std=None):
    """
    Process raw observation from environment.
    Handles both dict and array observations.
    """
    if isinstance(obs, dict):
        # Concatenate observation components for goal-conditioned envs
        components = []
        if 'observation' in obs:
            components.append(obs['observation'])
        if 'achieved_goal' in obs:
            components.append(obs['achieved_goal'])
        if 'desired_goal' in obs:
            components.append(obs['desired_goal'])
        obs = np.concatenate(components)
    
    obs = np.array(obs, dtype=np.float32)
    
    # Normalize if stats provided
    if obs_mean is not None and obs_std is not None:
        obs = (obs - obs_mean) / (obs_std + 1e-8)
    
    return obs


def evaluate_policy(
    policy_path: str,
    n_episodes: int = 100,
    perturbation_type: str = 'none',
    perturbation_strength: float = 0.1,
    device: str = 'cuda',
    render: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Evaluate a trained policy on the door-opening task.
    
    Args:
        policy_path: Path to trained policy checkpoint
        n_episodes: Number of evaluation episodes
        perturbation_type: Type of observation perturbation
        perturbation_strength: Strength of perturbation
        device: Device to run policy on
        render: Whether to render environment
        verbose: Whether to print progress
        
    Returns:
        results: Dictionary of evaluation metrics
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load policy and config
    checkpoint = torch.load(policy_path, map_location=device)
    metadata = checkpoint.get('metadata', {})
    
    # Load config if available
    config_path = os.path.join(os.path.dirname(policy_path), 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        obs_mean = np.array(config.get('obs_mean', None))
        obs_std = np.array(config.get('obs_std', None))
    else:
        obs_mean = None
        obs_std = None
    
    # Get dimensions
    obs_dim = metadata.get('obs_dim', 39)  # Default door env obs dim
    action_dim = metadata.get('action_dim', 28)  # Default door env action dim
    policy_type = metadata.get('policy_type', 'mlp')
    
    # Create and load policy
    policy = create_policy(policy_type, obs_dim, action_dim)
    policy.load_state_dict(checkpoint['state_dict'])
    policy.to(device)
    policy.eval()
    
    # Create environment
    render_mode = 'human' if render else None
    try:
        env = gym.make('AdroitHandDoor-v1', render_mode=render_mode)
    except:
        print("Could not create AdroitHandDoor-v1 environment.")
        print("Make sure gymnasium-robotics is installed: pip install gymnasium-robotics")
        return {}
    
    # Create perturbator
    perturbator = ObservationPerturbator(
        perturbation_type=perturbation_type,
        strength=perturbation_strength,
        obs_mean=obs_mean,
        obs_std=obs_std
    )
    
    # Evaluation metrics
    successes = []
    returns = []
    episode_lengths = []
    
    for episode in tqdm(range(n_episodes), desc='Evaluating', disable=not verbose):
        obs, info = env.reset()
        perturbator.reset()
        
        done = False
        episode_return = 0.0
        step = 0
        max_steps = 200
        
        while not done and step < max_steps:
            # Process observation
            processed_obs = process_observation(obs, obs_mean, obs_std)
            
            # Apply perturbation
            perturbed_obs = perturbator(processed_obs)
            
            # Get action from policy
            obs_tensor = torch.tensor(perturbed_obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy(obs_tensor)
                action = action.cpu().numpy()[0]
            
            # Clip action to valid range
            action = np.clip(action, -1.0, 1.0)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            step += 1
            
            if render:
                env.render()
        
        # Record metrics
        success = info.get('is_success', False) or info.get('success', False)
        successes.append(float(success))
        returns.append(episode_return)
        episode_lengths.append(step)
    
    env.close()
    
    # Compute statistics
    results = {
        'success_rate': np.mean(successes),
        'success_std': np.std(successes),
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'mean_episode_length': np.mean(episode_lengths),
        'std_episode_length': np.std(episode_lengths),
        'n_episodes': n_episodes,
        'perturbation_type': perturbation_type,
        'perturbation_strength': perturbation_strength,
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Evaluation Results:")
        print(f"  Policy: {policy_path}")
        print(f"  Perturbation: {perturbation_type} (strength={perturbation_strength})")
        print(f"  Episodes: {n_episodes}")
        print(f"  Success Rate: {results['success_rate']:.2%} ± {results['success_std']:.2%}")
        print(f"  Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
        print(f"  Mean Episode Length: {results['mean_episode_length']:.1f}")
        print(f"{'='*50}")
    
    return results


def robustness_sweep(
    policy_path: str,
    perturbation_types: List[str] = None,
    strengths: List[float] = None,
    n_episodes: int = 50,
    device: str = 'cuda',
    output_file: str = 'robustness_results.json'
) -> Dict:
    """
    Sweep over perturbation types and strengths to assess robustness.
    
    Args:
        policy_path: Path to trained policy
        perturbation_types: List of perturbation types to test
        strengths: List of perturbation strengths to test
        n_episodes: Episodes per configuration
        device: Device to run on
        output_file: File to save results
        
    Returns:
        all_results: Dictionary of all results
    """
    if perturbation_types is None:
        perturbation_types = ['none', 'gaussian', 'uniform', 'dropout', 'bias', 'scale', 'adversarial']
    
    if strengths is None:
        strengths = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    
    all_results = {}
    
    for pert_type in perturbation_types:
        all_results[pert_type] = {}
        
        for strength in strengths:
            if pert_type == 'none' and strength > 0:
                continue
                
            print(f"\nEvaluating: {pert_type}, strength={strength}")
            
            results = evaluate_policy(
                policy_path=policy_path,
                n_episodes=n_episodes,
                perturbation_type=pert_type,
                perturbation_strength=strength,
                device=device,
                verbose=False
            )
            
            all_results[pert_type][strength] = results
            
            print(f"  Success Rate: {results['success_rate']:.2%}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    return all_results


def compare_policies(
    policy_paths: List[str],
    policy_names: List[str] = None,
    perturbation_type: str = 'gaussian',
    strengths: List[float] = None,
    n_episodes: int = 50,
    device: str = 'cuda',
    output_file: str = 'comparison_results.json'
) -> Dict:
    """
    Compare multiple policies under perturbations.
    
    Args:
        policy_paths: List of paths to trained policies
        policy_names: Names for each policy
        perturbation_type: Type of perturbation to apply
        strengths: List of perturbation strengths
        n_episodes: Episodes per configuration
        device: Device to run on
        output_file: File to save results
        
    Returns:
        all_results: Dictionary of comparison results
    """
    if policy_names is None:
        policy_names = [f'policy_{i}' for i in range(len(policy_paths))]
    
    if strengths is None:
        strengths = [0.0, 0.1, 0.2, 0.3, 0.5]
    
    all_results = {name: {} for name in policy_names}
    
    for policy_path, policy_name in zip(policy_paths, policy_names):
        print(f"\n{'='*50}")
        print(f"Evaluating: {policy_name}")
        print(f"{'='*50}")
        
        for strength in strengths:
            print(f"\nPerturbation strength: {strength}")
            
            results = evaluate_policy(
                policy_path=policy_path,
                n_episodes=n_episodes,
                perturbation_type=perturbation_type if strength > 0 else 'none',
                perturbation_strength=strength,
                device=device,
                verbose=False
            )
            
            all_results[policy_name][strength] = results
            print(f"  Success Rate: {results['success_rate']:.2%}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*60}")
    print("Summary Comparison (Success Rate)")
    print(f"{'='*60}")
    
    header = f"{'Policy':<20}"
    for s in strengths:
        header += f"  {s:.1f}"
    print(header)
    print("-" * 60)
    
    for name in policy_names:
        row = f"{name:<20}"
        for s in strengths:
            if s in all_results[name]:
                sr = all_results[name][s]['success_rate']
                row += f"  {sr:.2f}"
            else:
                row += f"  N/A"
        print(row)
    
    print(f"\nResults saved to {output_file}")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate door-opening policy')
    
    parser.add_argument('--policy', type=str, required=True,
                       help='Path to trained policy')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'sweep', 'compare'],
                       help='Evaluation mode')
    
    # Single evaluation arguments
    parser.add_argument('--perturbation', type=str, default='none',
                       choices=['none', 'gaussian', 'uniform', 'dropout', 
                               'adversarial', 'stuck', 'delay', 'bias', 'scale'],
                       help='Perturbation type')
    parser.add_argument('--strength', type=float, default=0.1,
                       help='Perturbation strength')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    
    # Comparison arguments
    parser.add_argument('--compare_policies', type=str, nargs='+',
                       help='Additional policies for comparison')
    parser.add_argument('--policy_names', type=str, nargs='+',
                       help='Names for compared policies')
    
    # General arguments
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--render', action='store_true',
                       help='Render environment')
    parser.add_argument('--output', type=str, default='eval_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        results = evaluate_policy(
            policy_path=args.policy,
            n_episodes=args.episodes,
            perturbation_type=args.perturbation,
            perturbation_strength=args.strength,
            device=args.device,
            render=args.render
        )
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
            
    elif args.mode == 'sweep':
        results = robustness_sweep(
            policy_path=args.policy,
            n_episodes=args.episodes,
            device=args.device,
            output_file=args.output
        )
        
    elif args.mode == 'compare':
        if args.compare_policies is None:
            print("Error: --compare_policies required for compare mode")
            exit(1)
            
        all_policies = [args.policy] + args.compare_policies
        
        if args.policy_names:
            names = args.policy_names
        else:
            names = [os.path.basename(os.path.dirname(p)) for p in all_policies]
        
        results = compare_policies(
            policy_paths=all_policies,
            policy_names=names,
            perturbation_type=args.perturbation,
            n_episodes=args.episodes,
            device=args.device,
            output_file=args.output
        )
