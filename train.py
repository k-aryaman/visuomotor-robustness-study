"""
Training script for visuomotor behavior cloning.
Trains policies under different visual regimes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os

from policy import VisuomotorBCPolicy
from data import DemonstrationDataset, get_clean_transform, get_pixel_aug_transform


def train_policy(regime='pixel_aug', n_epochs=50, batch_size=32, lr=1e-3, 
                 demonstrations_file='demonstrations.pkl', device='cuda'):
    """
    Train a behavior cloning policy.
    
    Args:
        regime: Visual regime ('clean' or 'pixel_aug')
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        demonstrations_file: Path to demonstrations file
        device: Device to train on ('cuda' or 'cpu')
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get appropriate transform
    if regime == 'clean':
        transform = get_clean_transform()
        model_name = 'policy_clean.pth'
    elif regime == 'pixel_aug':
        transform = get_pixel_aug_transform()
        model_name = 'policy_pixel_aug.pth'
    else:
        raise ValueError(f"Unknown regime: {regime}")
    
    print(f"Training with {regime} visual regime...")
    
    # Load dataset
    dataset = DemonstrationDataset(demonstrations_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Initialize policy
    policy = VisuomotorBCPolicy(image_size=(84, 84), action_dim=4, use_resnet=True)
    policy.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Training loop
    policy.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, actions) in enumerate(dataloader):
            images = images.to(device)
            actions = actions.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predicted_actions = policy(images)
            
            # Compute loss
            loss = criterion(predicted_actions, actions)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch + 1}/{n_epochs}, Average Loss: {avg_loss:.6f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', model_name)
    torch.save(policy.state_dict(), model_path)
    print(f"\nSaved model to {model_path}")
    
    return policy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train visuomotor BC policy')
    parser.add_argument('--regime', type=str, default='pixel_aug', 
                       choices=['clean', 'pixel_aug'],
                       help='Visual regime for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--data', type=str, default='demonstrations.pkl',
                       help='Path to demonstrations file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    train_policy(
        regime=args.regime,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        demonstrations_file=args.data,
        device=args.device
    )

