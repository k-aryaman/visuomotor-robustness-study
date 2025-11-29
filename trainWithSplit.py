"""
Training script for visuomotor behavior cloning.
Trains policies under different visual regimes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import argparse
import os

from policy import VisuomotorBCPolicy
from data import DemonstrationDataset, get_clean_transform, get_pixel_aug_transform


def train_policy(regime='pixel_aug', n_epochs=50, batch_size=32, lr=1e-3, 
                 demonstrations_file='demonstrations.pkl', device='cuda', backbone_type='resnet',
                 loss_log_file=None, val_split=0.1):
    """
    Train a behavior cloning policy.
    
    Args:
        regime: Visual regime ('clean' or 'pixel_aug')
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        demonstrations_file: Path to demonstrations file
        device: Device to train on ('cuda' or 'cpu')
        backbone_type: Backbone type ('resnet', 'vit', or 'cnn')
        loss_log_file: Optional CSV path to log per-epoch train/val loss
        val_split: Fraction of data to hold out for validation (0.0-1.0)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get appropriate transform
    if regime == 'clean':
        transform = get_clean_transform()
        model_name = f'policy_clean_{backbone_type}.pth'
    elif regime == 'pixel_aug':
        transform = get_pixel_aug_transform()
        model_name = f'policy_pixel_aug_{backbone_type}.pth'
    else:
        raise ValueError(f"Unknown regime: {regime}")
    
    print(f"Training with {regime} visual regime and {backbone_type} backbone...")
    
    # Load dataset
    dataset = DemonstrationDataset(demonstrations_file, transform=transform)
    # Held-out validation split
    if val_split < 0 or val_split >= 1:
        raise ValueError(f"val_split must be in [0, 1). Got {val_split}")
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Dataset split -> train: {train_size}, val: {val_size}")
    else:
        train_dataset, val_dataset = dataset, None
        print(f"Dataset split -> train: {train_size}, val: 0 (no validation)")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2) if val_dataset else None
    
    # Initialize policy
    policy = VisuomotorBCPolicy(image_size=(84, 84), action_dim=4, backbone_type=backbone_type)
    policy.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Training loop
    policy.train()
    epoch_losses = []
    for epoch in range(n_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, actions) in enumerate(train_loader):
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
        
        avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Validation pass (no gradient)
        avg_val_loss = None
        if val_loader:
            policy.eval()
            val_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for images, actions in val_loader:
                    images = images.to(device)
                    actions = actions.to(device)
                    predicted_actions = policy(images)
                    val_loss = criterion(predicted_actions, actions)
                    val_total += val_loss.item()
                    val_batches += 1
            avg_val_loss = val_total / val_batches if val_batches > 0 else 0.0
            policy.train()
        
        if avg_val_loss is not None:
            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_train_loss:.6f}")
        
        epoch_losses.append((epoch + 1, avg_train_loss, avg_val_loss))
    
    # Optionally write losses to a log file (epoch, train_loss, val_loss)
    if loss_log_file:
        os.makedirs(os.path.dirname(loss_log_file), exist_ok=True) if os.path.dirname(loss_log_file) else None
        with open(loss_log_file, 'w', encoding='utf-8') as f:
            f.write("epoch,train_loss,val_loss\n")
            for ep, train_loss, val_loss in epoch_losses:
                val_str = f"{val_loss:.6f}" if val_loss is not None else ""
                f.write(f"{ep},{train_loss:.6f},{val_str}\n")
        print(f"Saved loss log to {loss_log_file}")
    
    # Save model with metadata
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', model_name)
    # Save state_dict with metadata for easier loading
    checkpoint = {
        'state_dict': policy.state_dict(),
        'backbone_type': backbone_type,
        'image_size': (84, 84),
        'action_dim': 4,
        'metadata': {
            'backbone_type': backbone_type,
            'regime': regime,
            'image_size': (84, 84),
            'action_dim': 4
        }
    }
    torch.save(checkpoint, model_path)
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
    parser.add_argument('--backbone', type=str, default='resnet',
                       choices=['resnet', 'vit', 'cnn'],
                       help='Backbone architecture (resnet, vit, or cnn)')
    parser.add_argument('--loss_log', type=str, default=None,
                       help='Optional path to save per-epoch loss (CSV with epoch,train_loss,val_loss)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Fraction of data for validation (0 disables validation)')
    
    args = parser.parse_args()
    
    train_policy(
        regime=args.regime,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        demonstrations_file=args.data,
        device=args.device,
        backbone_type=args.backbone,
        loss_log_file=args.loss_log,
        val_split=args.val_split
    )

