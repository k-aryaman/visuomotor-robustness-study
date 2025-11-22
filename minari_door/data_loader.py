"""
Data loading and perturbation script for Minari door-opening dataset.
Loads the D4RL door dataset and applies various perturbations for robustness training.
"""

import numpy as np
import pickle
import minari
from typing import Dict, List, Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader
import argparse


class DoorDataset(Dataset):
    """
    PyTorch Dataset for door-opening demonstrations from Minari.
    Supports observation perturbations for robustness training.
    """
    
    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        transform: Optional[Callable] = None,
        perturbation_type: Optional[str] = None,
        perturbation_strength: float = 0.1
    ):
        """
        Args:
            observations: Array of observations (N, obs_dim)
            actions: Array of actions (N, action_dim)
            transform: Optional transform to apply to observations
            perturbation_type: Type of perturbation ('gaussian', 'dropout', 'adversarial', None)
            perturbation_strength: Strength of perturbation (0.0 to 1.0)
        """
        self.observations = observations.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.transform = transform
        self.perturbation_type = perturbation_type
        self.perturbation_strength = perturbation_strength
        
        # Store original stats for normalization
        self.obs_mean = np.mean(self.observations, axis=0)
        self.obs_std = np.std(self.observations, axis=0) + 1e-8
        self.action_mean = np.mean(self.actions, axis=0)
        self.action_std = np.std(self.actions, axis=0) + 1e-8
        
    def __len__(self) -> int:
        return len(self.observations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obs = self.observations[idx].copy()
        action = self.actions[idx].copy()
        
        # Apply perturbation
        if self.perturbation_type is not None:
            obs = self._apply_perturbation(obs)
        
        # Apply transform
        if self.transform is not None:
            obs = self.transform(obs)
        
        return torch.tensor(obs, dtype=torch.float32), torch.tensor(action, dtype=torch.float32)
    
    def _apply_perturbation(self, obs: np.ndarray) -> np.ndarray:
        """Apply perturbation to observation."""
        if self.perturbation_type == 'gaussian':
            # Add Gaussian noise
            noise = np.random.normal(0, self.perturbation_strength * self.obs_std, obs.shape)
            obs = obs + noise
            
        elif self.perturbation_type == 'dropout':
            # Randomly zero out some observation dimensions
            mask = np.random.random(obs.shape) > self.perturbation_strength
            obs = obs * mask
            
        elif self.perturbation_type == 'uniform':
            # Add uniform noise
            noise = np.random.uniform(
                -self.perturbation_strength * self.obs_std,
                self.perturbation_strength * self.obs_std,
                obs.shape
            )
            obs = obs + noise
            
        elif self.perturbation_type == 'swap':
            # Randomly swap observation dimensions
            n_swaps = int(self.perturbation_strength * len(obs))
            for _ in range(n_swaps):
                i, j = np.random.choice(len(obs), 2, replace=False)
                obs[i], obs[j] = obs[j], obs[i]
                
        elif self.perturbation_type == 'scale':
            # Randomly scale observation dimensions
            scales = 1 + np.random.uniform(
                -self.perturbation_strength,
                self.perturbation_strength,
                obs.shape
            )
            obs = obs * scales
            
        return obs
    
    def normalize(self, normalize_obs: bool = True, normalize_actions: bool = False):
        """Normalize observations and/or actions."""
        if normalize_obs:
            self.observations = (self.observations - self.obs_mean) / self.obs_std
        if normalize_actions:
            self.actions = (self.actions - self.action_mean) / self.action_std


def load_minari_door_dataset(
    dataset_name: str = "D4RL/door/human-v2",
    download: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load the door-opening dataset from Minari.
    
    Args:
        dataset_name: Name of the Minari dataset
        download: Whether to download if not available locally
        
    Returns:
        observations: All observations
        actions: All actions
        rewards: All rewards
        terminals: Terminal flags
        info: Dataset metadata
    """
    print(f"Loading Minari dataset: {dataset_name}")
    
    # Download if necessary
    if download:
        try:
            minari.download_dataset(dataset_name)
        except Exception as e:
            print(f"Dataset may already exist or download failed: {e}")
    
    # Load dataset
    dataset = minari.load_dataset(dataset_name)
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []
    
    # Iterate through episodes
    for episode in dataset.iterate_episodes():
        # Get observations - handle dict observations (goal-conditioned envs)
        if isinstance(episode.observations, dict):
            # Concatenate observation, achieved_goal, and desired_goal
            obs = episode.observations.get('observation', episode.observations.get('obs'))
            achieved = episode.observations.get('achieved_goal', np.zeros((len(obs), 0)))
            desired = episode.observations.get('desired_goal', np.zeros((len(obs), 0)))
            
            if obs is not None:
                observations = np.concatenate([obs, achieved, desired], axis=-1) if achieved.size > 0 else obs
            else:
                # Fallback: concatenate all values
                observations = np.concatenate([v for v in episode.observations.values()], axis=-1)
        else:
            observations = episode.observations
        
        actions = episode.actions
        rewards = episode.rewards
        
        # Create terminal flags
        terminals = np.zeros(len(rewards), dtype=bool)
        terminals[-1] = True  # Last step is terminal
        
        all_observations.append(observations[:-1])  # Exclude last obs (no action)
        all_actions.append(actions)
        all_rewards.append(rewards)
        all_terminals.append(terminals)
    
    # Concatenate all episodes
    all_observations = np.concatenate(all_observations, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_rewards = np.concatenate(all_rewards, axis=0)
    all_terminals = np.concatenate(all_terminals, axis=0)
    
    info = {
        'dataset_name': dataset_name,
        'num_episodes': dataset.total_episodes,
        'num_steps': dataset.total_steps,
        'obs_dim': all_observations.shape[-1],
        'action_dim': all_actions.shape[-1],
    }
    
    print(f"Loaded {info['num_episodes']} episodes, {info['num_steps']} steps")
    print(f"Observation dim: {info['obs_dim']}, Action dim: {info['action_dim']}")
    
    return all_observations, all_actions, all_rewards, all_terminals, info


def create_perturbed_dataset(
    observations: np.ndarray,
    actions: np.ndarray,
    perturbation_type: str = 'gaussian',
    perturbation_strength: float = 0.1,
    normalize: bool = True
) -> DoorDataset:
    """
    Create a PyTorch dataset with perturbations applied.
    
    Args:
        observations: Original observations
        actions: Original actions
        perturbation_type: Type of perturbation
        perturbation_strength: Strength of perturbation
        normalize: Whether to normalize data
        
    Returns:
        DoorDataset instance
    """
    dataset = DoorDataset(
        observations=observations,
        actions=actions,
        perturbation_type=perturbation_type,
        perturbation_strength=perturbation_strength
    )
    
    if normalize:
        dataset.normalize(normalize_obs=True, normalize_actions=False)
    
    return dataset


def save_processed_data(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    info: Dict,
    output_file: str = 'door_demonstrations.pkl'
):
    """Save processed data to pickle file."""
    data = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'terminals': terminals,
        'info': info
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved processed data to {output_file}")


def load_processed_data(input_file: str = 'door_demonstrations.pkl') -> Dict:
    """Load processed data from pickle file."""
    with open(input_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded data from {input_file}")
    return data


class PerturbationScheduler:
    """
    Scheduler for curriculum learning with perturbations.
    Gradually increases perturbation strength during training.
    """
    
    def __init__(
        self,
        initial_strength: float = 0.0,
        final_strength: float = 0.3,
        warmup_epochs: int = 10,
        total_epochs: int = 50
    ):
        self.initial_strength = initial_strength
        self.final_strength = final_strength
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
    def get_strength(self, epoch: int) -> float:
        """Get perturbation strength for current epoch."""
        if epoch < self.warmup_epochs:
            return self.initial_strength
        
        progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        
        return self.initial_strength + progress * (self.final_strength - self.initial_strength)


def get_dataloaders(
    dataset_name: str = "D4RL/door/human-v2",
    batch_size: int = 256,
    perturbation_type: Optional[str] = None,
    perturbation_strength: float = 0.1,
    train_split: float = 0.9,
    normalize: bool = True,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Get train and validation dataloaders.
    
    Args:
        dataset_name: Minari dataset name
        batch_size: Batch size
        perturbation_type: Type of perturbation for training data
        perturbation_strength: Strength of perturbation
        train_split: Fraction of data for training
        normalize: Whether to normalize data
        num_workers: Number of dataloader workers
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        info: Dataset info
    """
    # Load data
    observations, actions, rewards, terminals, info = load_minari_door_dataset(dataset_name)
    
    # Split data
    n_samples = len(observations)
    n_train = int(n_samples * train_split)
    indices = np.random.permutation(n_samples)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    # Create datasets
    train_dataset = DoorDataset(
        observations=observations[train_indices],
        actions=actions[train_indices],
        perturbation_type=perturbation_type,
        perturbation_strength=perturbation_strength
    )
    
    val_dataset = DoorDataset(
        observations=observations[val_indices],
        actions=actions[val_indices],
        perturbation_type=None,  # No perturbation for validation
        perturbation_strength=0.0
    )
    
    if normalize:
        # Use training stats for normalization
        train_dataset.normalize(normalize_obs=True)
        
        # Apply same normalization to validation
        val_dataset.obs_mean = train_dataset.obs_mean
        val_dataset.obs_std = train_dataset.obs_std
        val_dataset.observations = (val_dataset.observations - train_dataset.obs_mean) / train_dataset.obs_std
    
    # Store normalization stats in info
    info['obs_mean'] = train_dataset.obs_mean
    info['obs_std'] = train_dataset.obs_std
    info['action_mean'] = train_dataset.action_mean
    info['action_std'] = train_dataset.action_std
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and process Minari door dataset')
    parser.add_argument('--dataset', type=str, default='D4RL/door/human-v2',
                       choices=['D4RL/door/human-v2', 'D4RL/door/expert-v2', 'D4RL/door/cloned-v2'],
                       help='Minari dataset name')
    parser.add_argument('--output', type=str, default='door_demonstrations.pkl',
                       help='Output file path')
    parser.add_argument('--perturbation', type=str, default=None,
                       choices=['gaussian', 'dropout', 'uniform', 'swap', 'scale', None],
                       help='Perturbation type to preview')
    parser.add_argument('--strength', type=float, default=0.1,
                       help='Perturbation strength')
    
    args = parser.parse_args()
    
    # Load dataset
    observations, actions, rewards, terminals, info = load_minari_door_dataset(args.dataset)
    
    # Save processed data
    save_processed_data(observations, actions, rewards, terminals, info, args.output)
    
    # Preview perturbation if specified
    if args.perturbation:
        print(f"\nPreviewing {args.perturbation} perturbation with strength {args.strength}:")
        dataset = create_perturbed_dataset(
            observations[:100],
            actions[:100],
            perturbation_type=args.perturbation,
            perturbation_strength=args.strength
        )
        
        # Show example
        orig_obs = observations[0]
        perturbed_obs, _ = dataset[0]
        
        print(f"Original obs (first 5 dims): {orig_obs[:5]}")
        print(f"Perturbed obs (first 5 dims): {perturbed_obs[:5].numpy()}")
        print(f"Difference: {np.abs(orig_obs[:5] - perturbed_obs[:5].numpy())}")
