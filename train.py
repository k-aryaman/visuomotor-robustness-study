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
import numpy as np

from policy import VisuomotorBCPolicy
from policy_act import ACTPolicy
from data import DemonstrationDataset, get_clean_transform, get_pixel_aug_transform
from action_utils import cartesian_to_spherical, spherical_to_cartesian


def train_policy(regime='pixel_aug', n_epochs=50, batch_size=32, lr=1e-3, 
                 demonstrations_file='demonstrations_push.pkl', device='cuda', backbone_type='resnet',
                 resume_from_checkpoint=None, weight_decay=1e-4, lr_decay=0.95, output_dir=None,
                 use_gru=False, gru_hidden_dim=128, use_act=False, act_chunk_size=10, act_seq_len=1,
                 use_spherical=False):
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
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set up output directory
    if output_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"training_runs_push/{regime}_{backbone_type}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    print(f"Output directory: {output_dir}")
    
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
    if use_act:
        print(f"Using ACT (Action Chunking with Transformers): chunk_size={act_chunk_size}, seq_len={act_seq_len}")
    
    # Load dataset
    dataset = DemonstrationDataset(demonstrations_file, transform=transform)
    
    # For ACT, we need sequences from trajectories, so we'll handle batching differently
    if use_act:
        # ACT needs sequences from trajectories, so we'll create a custom collate function
        from torch.utils.data import Dataset
        import pickle
        import numpy as np
        from PIL import Image
        
        class ACTSequenceDataset(Dataset):
            """Dataset that provides observation sequences and action chunks for ACT training."""
            def __init__(self, demonstrations_file, transform, seq_len=1, chunk_size=10):
                self.transform = transform
                self.seq_len = seq_len
                self.chunk_size = chunk_size
                
                # Load trajectories
                with open(demonstrations_file, 'rb') as f:
                    self.trajectories = pickle.load(f)
                
                # Check if using proprioception
                self.use_proprioception = len(self.trajectories[0][0]) == 3 if len(self.trajectories) > 0 and len(self.trajectories[0]) > 0 else False
                
                # Create samples: for each trajectory, create (seq_len observation, chunk_size action) pairs
                self.samples = []
                for traj_idx, trajectory in enumerate(self.trajectories):
                    if len(trajectory) < seq_len + chunk_size:
                        continue  # Skip trajectories that are too short
                    
                    # For each valid starting position
                    for start_idx in range(len(trajectory) - seq_len - chunk_size + 1):
                        # Get observation sequence
                        obs_seq = trajectory[start_idx:start_idx + seq_len]
                        # Get action chunk (actions from start_idx to start_idx + chunk_size)
                        action_chunk = trajectory[start_idx + seq_len - 1:start_idx + seq_len - 1 + chunk_size]
                        
                        # Extract actions from action chunk
                        if self.use_proprioception:
                            actions = [item[2] for item in action_chunk]  # (image, state, action)
                        else:
                            actions = [item[1] for item in action_chunk]  # (image, action)
                        
                        self.samples.append((traj_idx, start_idx, obs_seq, actions))
                
                print(f"Created {len(self.samples)} ACT training samples from {len(self.trajectories)} trajectories")
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                traj_idx, start_idx, obs_seq, actions = self.samples[idx]
                
                # Process observation sequence
                images = []
                states = []
                for obs in obs_seq:
                    if self.use_proprioception:
                        image, state, _ = obs
                    else:
                        image, _ = obs
                        state = None
                    
                    # Convert image
                    if isinstance(image, np.ndarray):
                        if image.dtype != np.uint8:
                            image = (image * 255).astype(np.uint8)
                        image = Image.fromarray(image)
                    
                    if self.transform:
                        image = self.transform(image)
                    
                    images.append(image)
                    if state is not None:
                        states.append(torch.FloatTensor(state))
                
                # Stack images: (seq_len, 3, H, W)
                images = torch.stack(images)
                
                # Stack states if available: (seq_len, state_dim)
                if self.use_proprioception:
                    states = torch.stack(states)
                else:
                    states = None
                
                # Stack actions: (chunk_size, action_dim)
                actions = torch.stack([torch.FloatTensor(a) for a in actions])
                
                return images, states, actions
        
        dataset = ACTSequenceDataset(demonstrations_file, transform, seq_len=act_seq_len, chunk_size=act_chunk_size)
        # Use num_workers=0 for ACT to avoid pickling issues with local class
        # Split into train and validation (80/20)
        total_size = len(dataset)
        val_size = int(0.2 * total_size)
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"Split dataset: {train_size} train, {val_size} validation samples")
    else:
        # Split into train and validation (80/20)
        total_size = len(dataset)
        val_size = int(0.2 * total_size)
        train_size = total_size - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        print(f"Split dataset: {train_size} train, {val_size} validation samples")
    
    # Check if dataset uses proprioception
    use_proprioception = dataset.use_proprioception
    # For push task: only gripper_pos (3D), no gripper_width
    state_dim = 3 if use_proprioception else 0
    
    if use_proprioception:
        print(f"Using proprioceptive state (dim={state_dim})")
    else:
        print("Using image-only (no proprioception)")
    
    # Initialize policy
    if use_act:
        policy = ACTPolicy(
            image_size=(84, 84),
            action_dim=3,  # Push task uses 3D actions (x, y, z) - no gripper
            chunk_size=act_chunk_size,
            backbone_type=backbone_type,
            state_dim=state_dim,
            d_model=512,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dropout=0.1
        )
    else:
        policy = VisuomotorBCPolicy(image_size=(84, 84), action_dim=3, backbone_type=backbone_type,  # Push task uses 3D actions 
                                    state_dim=state_dim, use_gru=use_gru, gru_hidden_dim=gru_hidden_dim)
    policy.to(device)
    
    # Loss and optimizer with weight decay (L2 regularization)
    # Use Huber loss (smooth L1) - less sensitive to outliers than MSE, helps with action magnitude
    # Also supports magnitude-weighted loss if needed
    use_weighted_loss = True  # Set to False to use standard Huber loss
    use_spherical_coords = use_spherical  # Convert actions to spherical coordinates (magnitude, theta, phi)
    use_no_motion_penalty = False if use_spherical_coords else True  # Disabled when using spherical coords (magnitude is explicit)
    no_motion_threshold = 0.01  # Threshold for "no motion" (actions with magnitude < this are considered zero)
    no_motion_penalty_weight = 0.5  # Weight for no-motion penalty term
    
    if use_spherical_coords:
        print("Using spherical coordinate representation: (magnitude, theta, phi)")
        print("  - Magnitude: action strength [0, 1]")
        print("  - Theta: elevation angle from z-axis [0, π]")
        print("  - Phi: azimuth angle in xy-plane [-π, π]")
        print("  WARNING: Zero-magnitude actions may cause instability in theta/phi")
    else:
        print("Using Cartesian coordinate representation: (dx, dy, dz)")
    
    if use_weighted_loss:
        criterion = nn.SmoothL1Loss(reduction='none')  # Huber loss, get per-sample loss
    else:
        criterion = nn.SmoothL1Loss()  # Standard Huber loss
    optimizer = optim.Adam(policy.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler (ReduceLROnPlateau - reduces LR when loss plateaus)
    # This is better than fixed exponential decay as it adapts to actual training progress
    # threshold: minimum relative change to qualify as an improvement (0.01 = 1% improvement)
    # With threshold=0.01 and threshold_mode='rel', improvements < 1% don't count
    # patience=2: reduce LR after 2 epochs without meaningful improvement
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6, 
        threshold=0.01, threshold_mode='rel'  # 1% relative improvement threshold
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        policy.load_state_dict(checkpoint['state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            print(f"Resuming from epoch {start_epoch}")
        if 'loss' in checkpoint:
            print(f"Previous loss: {checkpoint['loss']:.6f}")
    
    # Training loop
    policy.train()
    total_batches = len(dataloader)
    print(f"Total batches per epoch: {total_batches}")
    print("Starting training...\n")
    
    # Track best validation loss for model saving
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(start_epoch, n_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if use_act:
                # ACT: batch is (images_seq, states_seq, actions_chunk)
                # images_seq: (batch, seq_len, 3, H, W)
                # states_seq: (batch, seq_len, state_dim) or None
                # actions_chunk: (batch, chunk_size, action_dim)
                images_seq, states_seq, actions_chunk = batch
                images_seq = images_seq.to(device)
                if states_seq is not None:
                    states_seq = states_seq.to(device)
                actions_chunk = actions_chunk.to(device)
                
                # Forward pass: ACT outputs (batch, chunk_size, action_dim) in spherical coords
                optimizer.zero_grad()
                predicted_actions_spherical = policy(images_seq, states_seq)
                
                # Convert expert actions from Cartesian to spherical
                if use_spherical_coords:
                    # Reshape for conversion: (batch, chunk_size, 3) -> (batch * chunk_size, 3)
                    batch_size, chunk_size, _ = actions_chunk.shape
                    actions_flat = actions_chunk.view(-1, 3)
                    actions_spherical_flat = cartesian_to_spherical(actions_flat)
                    actions_spherical = actions_spherical_flat.view(batch_size, chunk_size, 3)
                    
                    # Model outputs are already in spherical space (magnitude, theta, phi)
                    # For magnitude: use sigmoid or ReLU to ensure non-negative
                    # For angles: they're already in valid ranges from tanh + scaling
                    # Scale predicted actions: tanh outputs [-1, 1], need to map appropriately
                    # Magnitude: [0, 1] -> scale to [0, max_magnitude] (max is typically 1.0 for normalized actions)
                    # Theta: [-1, 1] -> [0, π]
                    # Phi: [-1, 1] -> [-π, π]
                    predicted_magnitude = torch.sigmoid(predicted_actions_spherical[..., 0])  # [0, 1]
                    predicted_theta = (predicted_actions_spherical[..., 1] + 1.0) * 0.5 * np.pi  # [0, π]
                    predicted_phi = predicted_actions_spherical[..., 2] * np.pi  # [-π, π]
                    predicted_actions_spherical = torch.stack([predicted_magnitude, predicted_theta, predicted_phi], dim=-1)
                    
                    # Use spherical coordinates for loss
                    actions_for_loss = actions_spherical
                    predicted_for_loss = predicted_actions_spherical
                else:
                    actions_for_loss = actions_chunk
                    predicted_for_loss = predicted_actions_spherical
                
                # Loss: compare predicted chunk with target chunk
                if use_weighted_loss:
                    # Per-sample loss for weighting
                    per_sample_loss = criterion(predicted_for_loss, actions_for_loss)  # (batch, chunk_size, action_dim)
                    
                    if use_spherical_coords:
                        # Weight by magnitude (first dimension in spherical)
                        expert_action_mags = actions_for_loss[..., 0:1]  # (batch, chunk_size, 1)
                    else:
                        # Weight by action magnitude (all dimensions for push task)
                        expert_action_mags = torch.norm(actions_for_loss, dim=2, keepdim=True)  # (batch, chunk_size, 1)
                    action_weights = 1.0 + 2.0 * expert_action_mags  # (batch, chunk_size, 1)
                    loss = (per_sample_loss * action_weights).mean()
                else:
                    loss = criterion(predicted_for_loss, actions_for_loss)
            else:
                # Standard BC: single action prediction
                if use_proprioception:
                    images, states, actions = batch
                    images = images.to(device)
                    states = states.to(device)
                    actions = actions.to(device)
                else:
                    images, actions = batch
                    images = images.to(device)
                    actions = actions.to(device)
                    states = None
                
                # Forward pass: outputs in spherical coords (magnitude, theta, phi)
                optimizer.zero_grad()
                if hasattr(policy, 'use_gru') and policy.use_gru:
                    predicted_actions_spherical, _ = policy(images, states, hidden=None)
                else:
                    predicted_actions_spherical = policy(images, states)
                
                # Convert expert actions from Cartesian to spherical
                if use_spherical_coords:
                    actions_spherical = cartesian_to_spherical(actions)
                    
                    # Scale predicted actions: tanh outputs [-1, 1], need to map appropriately
                    # Magnitude: [0, 1] -> scale to [0, max_magnitude]
                    # Theta: [-1, 1] -> [0, π]
                    # Phi: [-1, 1] -> [-π, π]
                    predicted_magnitude = torch.sigmoid(predicted_actions_spherical[..., 0])  # [0, 1]
                    predicted_theta = (predicted_actions_spherical[..., 1] + 1.0) * 0.5 * np.pi  # [0, π]
                    predicted_phi = predicted_actions_spherical[..., 2] * np.pi  # [-π, π]
                    predicted_actions_spherical = torch.stack([predicted_magnitude, predicted_theta, predicted_phi], dim=-1)
                    
                    # Use spherical coordinates for loss
                    actions_for_loss = actions_spherical
                    predicted_for_loss = predicted_actions_spherical
                else:
                    actions_for_loss = actions
                    predicted_for_loss = predicted_actions_spherical
                
                # Compute loss
                if use_weighted_loss:
                    # Per-sample loss for weighting
                    per_sample_loss = criterion(predicted_for_loss, actions_for_loss)  # Shape: (batch_size, action_dim)
                    
                    if use_spherical_coords:
                        # Weight by magnitude (first dimension in spherical)
                        expert_action_mags = actions_for_loss[..., 0:1]  # (batch_size, 1)
                    else:
                        # Weight loss by expert action magnitude to encourage matching large actions
                        expert_action_mags = torch.norm(actions_for_loss, dim=1, keepdim=True)  # Action magnitude (3D for push)
                    # Scale weights: small actions get weight 1.0, large actions get weight up to 3.0
                    action_weights = 1.0 + 2.0 * expert_action_mags  # Range: [1.0, 3.0]
                    # Apply weights to loss (expand to match action_dim)
                    loss = (per_sample_loss * action_weights.unsqueeze(1)).mean()
                else:
                    # Standard loss
                    loss = criterion(predicted_for_loss, actions_for_loss)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                current_avg = total_loss / num_batches
                print(f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_idx + 1}/{total_batches}, Current Avg Loss: {current_avg:.6f}", flush=True)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Validation phase
        policy.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_batch in val_dataloader:
                if use_act:
                    images_seq, states_seq, actions_chunk = val_batch
                    images_seq = images_seq.to(device)
                    if states_seq is not None:
                        states_seq = states_seq.to(device)
                    actions_chunk = actions_chunk.to(device)
                    
                    predicted_actions_spherical = policy(images_seq, states_seq)
                    
                    # Convert expert actions from Cartesian to spherical
                    if use_spherical_coords:
                        batch_size, chunk_size, _ = actions_chunk.shape
                        actions_flat = actions_chunk.view(-1, 3)
                        actions_spherical_flat = cartesian_to_spherical(actions_flat)
                        actions_spherical = actions_spherical_flat.view(batch_size, chunk_size, 3)
                        
                        # Scale predicted actions
                        predicted_magnitude = torch.sigmoid(predicted_actions_spherical[..., 0])
                        predicted_theta = (predicted_actions_spherical[..., 1] + 1.0) * 0.5 * np.pi
                        predicted_phi = predicted_actions_spherical[..., 2] * np.pi
                        predicted_actions_spherical = torch.stack([predicted_magnitude, predicted_theta, predicted_phi], dim=-1)
                        
                        actions_for_loss = actions_spherical
                        predicted_for_loss = predicted_actions_spherical
                    else:
                        actions_for_loss = actions_chunk
                        predicted_for_loss = predicted_actions_spherical
                    
                    if use_weighted_loss:
                        per_sample_loss = criterion(predicted_for_loss, actions_for_loss)
                        if use_spherical_coords:
                            expert_action_mags = actions_for_loss[..., 0:1]
                        else:
                            expert_action_mags = torch.norm(actions_for_loss, dim=2, keepdim=True)
                        action_weights = 1.0 + 2.0 * expert_action_mags
                        batch_loss = (per_sample_loss * action_weights).mean()
                    else:
                        batch_loss = criterion(predicted_for_loss, actions_for_loss)
                else:
                    if use_proprioception:
                        images, states, actions = val_batch
                        images = images.to(device)
                        states = states.to(device)
                        actions = actions.to(device)
                    else:
                        images, actions = val_batch
                        images = images.to(device)
                        actions = actions.to(device)
                        states = None
                    
                    if hasattr(policy, 'use_gru') and policy.use_gru:
                        predicted_actions_spherical, _ = policy(images, states, hidden=None)
                    else:
                        predicted_actions_spherical = policy(images, states)
                    
                    # Convert expert actions from Cartesian to spherical
                    if use_spherical_coords:
                        actions_spherical = cartesian_to_spherical(actions)
                        
                        # Scale predicted actions
                        predicted_magnitude = torch.sigmoid(predicted_actions_spherical[..., 0])
                        predicted_theta = (predicted_actions_spherical[..., 1] + 1.0) * 0.5 * np.pi
                        predicted_phi = predicted_actions_spherical[..., 2] * np.pi
                        predicted_actions_spherical = torch.stack([predicted_magnitude, predicted_theta, predicted_phi], dim=-1)
                        
                        actions_for_loss = actions_spherical
                        predicted_for_loss = predicted_actions_spherical
                    else:
                        actions_for_loss = actions
                        predicted_for_loss = predicted_actions_spherical
                    
                    if use_weighted_loss:
                        per_sample_loss = criterion(predicted_for_loss, actions_for_loss)
                        if use_spherical_coords:
                            expert_action_mags = actions_for_loss[..., 0:1]
                        else:
                            expert_action_mags = torch.norm(actions_for_loss, dim=1, keepdim=True)
                        action_weights = 1.0 + 2.0 * expert_action_mags
                        batch_loss = (per_sample_loss * action_weights.unsqueeze(1)).mean()
                    else:
                        batch_loss = criterion(predicted_for_loss, actions_for_loss)
                
                val_loss += batch_loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        policy.train()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}", flush=True)
        
        # Update learning rate (ReduceLROnPlateau uses validation loss)
        old_lr = current_lr
        scheduler.step(avg_val_loss)  # Use validation loss for scheduling
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"  -> Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}", flush=True)
        
        # Save checkpoint after each epoch
        # Remove .pth extension from model_name if present, then add epoch and .pth
        checkpoint_name = model_name.replace('.pth', '') if model_name.endswith('.pth') else model_name
        checkpoint_path = os.path.join(output_dir, 'models', f'{checkpoint_name}_epoch_{epoch + 1}.pth')
        checkpoint = {
            'state_dict': policy.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'backbone_type': backbone_type,
            'image_size': (84, 84),
            'action_dim': 3,  # Push task: 3D actions (x, y, z)
            'state_dim': state_dim,
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'val_loss': avg_val_loss,
            'use_act': use_act,
            'use_spherical': use_spherical_coords,
            'metadata': {
                'backbone_type': backbone_type,
                'regime': regime,
                'image_size': (84, 84),
                'action_dim': 3,  # Push task: 3D actions (x, y, z)
                'state_dim': state_dim,
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'use_act': use_act,
                'use_spherical': use_spherical_coords,
            }
        }
        if use_act:
            checkpoint['act_chunk_size'] = act_chunk_size
            checkpoint['act_seq_len'] = act_seq_len
            checkpoint['metadata']['act_chunk_size'] = act_chunk_size
            checkpoint['metadata']['act_seq_len'] = act_seq_len
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}", flush=True)
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            best_model_path = os.path.join(output_dir, 'models', f'{checkpoint_name}_best.pth')
            torch.save(checkpoint, best_model_path)
            print(f"  -> New best model! Val loss: {avg_val_loss:.6f} (saved to {best_model_path})", flush=True)
    
    # Save final model with metadata
    model_path = os.path.join(output_dir, 'models', model_name)
    # Save state_dict with metadata for easier loading
    checkpoint = {
        'state_dict': policy.state_dict(),
        'backbone_type': backbone_type,
        'image_size': (84, 84),
        'action_dim': 3,  # Push task: 3D actions
        'state_dim': state_dim,
        'use_act': use_act,
        'use_spherical': use_spherical_coords,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'metadata': {
            'backbone_type': backbone_type,
            'regime': regime,
            'image_size': (84, 84),
            'action_dim': 3,  # Push task: 3D actions (x, y, z)
            'state_dim': state_dim,
            'use_act': use_act,
            'use_spherical': use_spherical_coords,
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
        }
    }
    if use_act:
        checkpoint['act_chunk_size'] = act_chunk_size
        checkpoint['act_seq_len'] = act_seq_len
        checkpoint['metadata']['act_chunk_size'] = act_chunk_size
        checkpoint['metadata']['act_seq_len'] = act_seq_len
    torch.save(checkpoint, model_path)
    print(f"\nSaved final model to {model_path}")
    print(f"Best model was at epoch {best_epoch} with validation loss: {best_val_loss:.6f}")
    
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
    parser.add_argument('--data', type=str, default='demonstrations_push.pkl',
                       help='Path to demonstrations file')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--backbone', type=str, default='resnet',
                       choices=['resnet', 'vit', 'cnn'],
                       help='Backbone architecture (resnet, vit, or cnn)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., models/policy_clean_resnet_epoch_22.pth)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay (L2 regularization) coefficient (default: 1e-4)')
    parser.add_argument('--lr_decay', type=float, default=0.95,
                       help='[DEPRECATED] Not used with ReduceLROnPlateau scheduler. LR reduces automatically when loss plateaus.')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for models and logs (default: creates timestamped directory)')
    parser.add_argument('--use_gru', action='store_true',
                       help='Enable GRU for temporal modeling (lightweight RNN layer)')
    parser.add_argument('--gru_hidden_dim', type=int, default=128,
                       help='Hidden dimension for GRU (default: 128, lightweight)')
    parser.add_argument('--use_act', action='store_true',
                       help='Use ACT (Action Chunking with Transformers) instead of standard BC')
    parser.add_argument('--act_chunk_size', type=int, default=10,
                       help='Number of actions to predict in each chunk for ACT (default: 10)')
    parser.add_argument('--act_seq_len', type=int, default=1,
                       help='Number of observation timesteps to use as input for ACT (default: 1)')
    parser.add_argument('--use_spherical', action='store_true',
                       help='Use spherical coordinates (magnitude, theta, phi) instead of Cartesian (dx, dy, dz)')
    
    args = parser.parse_args()
    
    train_policy(
        regime=args.regime,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        demonstrations_file=args.data,
        device=args.device,
        backbone_type=args.backbone,
        resume_from_checkpoint=args.resume,
        weight_decay=args.weight_decay,
        lr_decay=args.lr_decay,
        output_dir=args.output_dir,
        use_gru=args.use_gru,
        gru_hidden_dim=args.gru_hidden_dim,
        use_act=args.use_act,
        act_chunk_size=args.act_chunk_size,
        act_seq_len=args.act_seq_len,
        use_spherical=args.use_spherical
    )

