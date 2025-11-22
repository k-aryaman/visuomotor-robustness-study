"""
Training script for offline reinforcement learning on door-opening task.
Supports behavior cloning with various perturbation strategies for robustness.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
from typing import Dict, Optional, Tuple
import json
from tqdm import tqdm

from data_loader import (
    get_dataloaders,
    DoorDataset,
    PerturbationScheduler,
    load_minari_door_dataset
)
from policy import create_policy, MLPPolicy


def train_behavior_cloning(
    train_loader: DataLoader,
    val_loader: DataLoader,
    policy: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    n_epochs: int = 100,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    perturbation_scheduler: Optional[PerturbationScheduler] = None,
    save_dir: str = 'models',
    save_freq: int = 10,
    log_freq: int = 1,
    early_stopping_patience: int = 20
) -> Dict:
    """
    Train a behavior cloning policy.
    
    Args:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        policy: Policy network
        optimizer: Optimizer
        device: Device to train on
        n_epochs: Number of training epochs
        scheduler: Optional learning rate scheduler
        perturbation_scheduler: Optional perturbation strength scheduler
        save_dir: Directory to save checkpoints
        save_freq: Save checkpoint every N epochs
        log_freq: Log metrics every N epochs
        early_stopping_patience: Stop if no improvement for N epochs
        
    Returns:
        history: Training history
    """
    os.makedirs(save_dir, exist_ok=True)
    
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'perturbation_strength': []
    }
    
    for epoch in range(n_epochs):
        # Update perturbation strength if using curriculum
        if perturbation_scheduler is not None:
            strength = perturbation_scheduler.get_strength(epoch)
            train_loader.dataset.perturbation_strength = strength
            history['perturbation_strength'].append(strength)
        
        # Training phase
        policy.train()
        train_losses = []
        
        for obs, actions in tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}', leave=False):
            obs = obs.to(device)
            actions = actions.to(device)
            
            optimizer.zero_grad()
            pred_actions = policy(obs)
            loss = criterion(pred_actions, actions)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        policy.eval()
        val_losses = []
        
        with torch.no_grad():
            for obs, actions in val_loader:
                obs = obs.to(device)
                actions = actions.to(device)
                
                pred_actions = policy(obs)
                loss = criterion(pred_actions, actions)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)
        
        # Learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step(avg_val_loss)
        
        # Logging
        if (epoch + 1) % log_freq == 0:
            pert_str = f", Pert: {history['perturbation_strength'][-1]:.3f}" if perturbation_scheduler else ""
            print(f"Epoch {epoch+1}/{n_epochs} - "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Val Loss: {avg_val_loss:.6f}, "
                  f"LR: {current_lr:.6f}{pert_str}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'state_dict': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'metadata': {
                    'policy_type': 'mlp',
                    'obs_dim': policy.obs_dim,
                    'action_dim': policy.action_dim,
                }
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_policy.pth'))
        else:
            patience_counter += 1
        
        # Save periodic checkpoint
        if (epoch + 1) % save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'state_dict': policy.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'metadata': {
                    'policy_type': 'mlp',
                    'obs_dim': policy.obs_dim,
                    'action_dim': policy.action_dim,
                }
            }
            torch.save(checkpoint, os.path.join(save_dir, f'policy_epoch_{epoch+1}.pth'))
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    checkpoint = {
        'epoch': n_epochs - 1,
        'state_dict': policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'metadata': {
            'policy_type': 'mlp',
            'obs_dim': policy.obs_dim,
            'action_dim': policy.action_dim,
        }
    }
    torch.save(checkpoint, os.path.join(save_dir, 'final_policy.pth'))
    
    # Save training history
    with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.6f}")
    
    return history


def train_with_perturbation_curriculum(
    dataset_name: str = "D4RL/door/human-v2",
    policy_type: str = 'mlp',
    hidden_dims: Tuple[int, ...] = (256, 256, 256),
    perturbation_type: str = 'gaussian',
    initial_strength: float = 0.0,
    final_strength: float = 0.3,
    warmup_epochs: int = 20,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    weight_decay: float = 1e-5,
    device: str = 'cuda',
    save_dir: str = 'models'
) -> Dict:
    """
    Train with curriculum learning on perturbation strength.
    
    Args:
        dataset_name: Minari dataset name
        policy_type: Type of policy network
        hidden_dims: Hidden layer dimensions
        perturbation_type: Type of perturbation
        initial_strength: Starting perturbation strength
        final_strength: Final perturbation strength
        warmup_epochs: Epochs before starting perturbation curriculum
        n_epochs: Total training epochs
        batch_size: Batch size
        lr: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to train on
        save_dir: Directory to save models
        
    Returns:
        history: Training history
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, val_loader, info = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=batch_size,
        perturbation_type=perturbation_type,
        perturbation_strength=initial_strength,
        normalize=True
    )
    
    obs_dim = info['obs_dim']
    action_dim = info['action_dim']
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create policy
    policy = create_policy(
        policy_type=policy_type,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims
    )
    policy.to(device)
    
    # Count parameters
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy parameters: {n_params:,}")
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(policy.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    # Perturbation scheduler
    perturbation_scheduler = PerturbationScheduler(
        initial_strength=initial_strength,
        final_strength=final_strength,
        warmup_epochs=warmup_epochs,
        total_epochs=n_epochs
    )
    
    # Save configuration
    config = {
        'dataset_name': dataset_name,
        'policy_type': policy_type,
        'hidden_dims': hidden_dims,
        'perturbation_type': perturbation_type,
        'initial_strength': initial_strength,
        'final_strength': final_strength,
        'warmup_epochs': warmup_epochs,
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': weight_decay,
        'obs_dim': obs_dim,
        'action_dim': action_dim,
        'obs_mean': info['obs_mean'].tolist(),
        'obs_std': info['obs_std'].tolist()
    }
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Train
    history = train_behavior_cloning(
        train_loader=train_loader,
        val_loader=val_loader,
        policy=policy,
        optimizer=optimizer,
        device=device,
        n_epochs=n_epochs,
        scheduler=scheduler,
        perturbation_scheduler=perturbation_scheduler,
        save_dir=save_dir
    )
    
    return history


def train_clean_baseline(
    dataset_name: str = "D4RL/door/human-v2",
    policy_type: str = 'mlp',
    hidden_dims: Tuple[int, ...] = (256, 256, 256),
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = 'cuda',
    save_dir: str = 'models/clean'
) -> Dict:
    """
    Train a clean baseline without any perturbations.
    """
    return train_with_perturbation_curriculum(
        dataset_name=dataset_name,
        policy_type=policy_type,
        hidden_dims=hidden_dims,
        perturbation_type=None,  # No perturbation
        initial_strength=0.0,
        final_strength=0.0,
        warmup_epochs=0,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        save_dir=save_dir
    )


def train_robust_policy(
    dataset_name: str = "D4RL/door/human-v2",
    policy_type: str = 'mlp',
    hidden_dims: Tuple[int, ...] = (256, 256, 256),
    perturbation_type: str = 'gaussian',
    perturbation_strength: float = 0.2,
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = 'cuda',
    save_dir: str = 'models/robust'
) -> Dict:
    """
    Train with constant perturbation (no curriculum).
    """
    return train_with_perturbation_curriculum(
        dataset_name=dataset_name,
        policy_type=policy_type,
        hidden_dims=hidden_dims,
        perturbation_type=perturbation_type,
        initial_strength=perturbation_strength,
        final_strength=perturbation_strength,
        warmup_epochs=0,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        save_dir=save_dir
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train offline RL policy for door opening')
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='D4RL/door/human-v2',
                       choices=['D4RL/door/human-v2', 'D4RL/door/expert-v2', 'D4RL/door/cloned-v2'],
                       help='Minari dataset name')
    
    # Training mode
    parser.add_argument('--mode', type=str, default='curriculum',
                       choices=['clean', 'robust', 'curriculum'],
                       help='Training mode')
    
    # Policy arguments
    parser.add_argument('--policy', type=str, default='mlp',
                       choices=['mlp', 'residual', 'gaussian', 'ensemble', 'transformer'],
                       help='Policy architecture')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 256, 256],
                       help='Hidden layer dimensions')
    
    # Perturbation arguments
    parser.add_argument('--perturbation', type=str, default='gaussian',
                       choices=['gaussian', 'dropout', 'uniform', 'swap', 'scale'],
                       help='Perturbation type')
    parser.add_argument('--init_strength', type=float, default=0.0,
                       help='Initial perturbation strength')
    parser.add_argument('--final_strength', type=float, default=0.3,
                       help='Final perturbation strength')
    parser.add_argument('--warmup', type=int, default=20,
                       help='Warmup epochs before perturbation curriculum')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    
    # Set save directory based on mode
    if args.mode == 'clean':
        save_dir = os.path.join(args.save_dir, 'clean')
    elif args.mode == 'robust':
        save_dir = os.path.join(args.save_dir, f'robust_{args.perturbation}')
    else:
        save_dir = os.path.join(args.save_dir, f'curriculum_{args.perturbation}')
    
    # Train based on mode
    if args.mode == 'clean':
        history = train_clean_baseline(
            dataset_name=args.dataset,
            policy_type=args.policy,
            hidden_dims=tuple(args.hidden_dims),
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            save_dir=save_dir
        )
    elif args.mode == 'robust':
        history = train_robust_policy(
            dataset_name=args.dataset,
            policy_type=args.policy,
            hidden_dims=tuple(args.hidden_dims),
            perturbation_type=args.perturbation,
            perturbation_strength=args.final_strength,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            save_dir=save_dir
        )
    else:  # curriculum
        history = train_with_perturbation_curriculum(
            dataset_name=args.dataset,
            policy_type=args.policy,
            hidden_dims=tuple(args.hidden_dims),
            perturbation_type=args.perturbation,
            initial_strength=args.init_strength,
            final_strength=args.final_strength,
            warmup_epochs=args.warmup,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=args.device,
            save_dir=save_dir
        )
    
    print(f"\nTraining complete! Models saved to {save_dir}")
