"""
PyTorch policy model for visuomotor behavior cloning.
Implements a CNN-MLP or ViT-MLP architecture for processing RGB images and outputting actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VisuomotorBCPolicy(nn.Module):
    """
    CNN-MLP or ViT-MLP architecture for visuomotor behavior cloning.
    
    Backbone: 4-layer CNN, pre-trained ResNet-18, or pre-trained Vision Transformer (ViT)
    Head: MLP that takes flattened image features and outputs continuous actions
    """
    
    def __init__(self, image_size=(84, 84), action_dim=3, backbone_type='resnet', feature_dim=512, use_resnet=None, state_dim=0, use_gru=False, gru_hidden_dim=128):  # Default 3 for push task
        """
        Initialize the policy network.
        
        Args:
            image_size: Tuple of (height, width) for input images
            action_dim: Dimension of action space (3 for push task, 4 for pick-and-place)
            backbone_type: Backbone type ('resnet', 'vit', or 'cnn')
            feature_dim: Dimension of feature vector before MLP head
            use_resnet: Deprecated - use backbone_type instead. If provided, maps to 'resnet' or 'cnn'
            state_dim: Dimension of proprioceptive state (0 if not using proprioception, 10 for full state)
        """
        super(VisuomotorBCPolicy, self).__init__()
        
        self.image_size = image_size
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.use_gru = use_gru
        self.gru_hidden_dim = gru_hidden_dim if use_gru else 0
        
        # Handle backward compatibility with use_resnet parameter
        if use_resnet is not None:
            backbone_type = 'resnet' if use_resnet else 'cnn'
        
        self.backbone_type = backbone_type
        
        if backbone_type == 'resnet':
            # Use pre-trained ResNet-18 as backbone
            resnet = models.resnet18(weights='IMAGENET1K_V1')
            # Remove the final fully connected layer
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            # ResNet-18 outputs 512 features after global average pooling
            backbone_output_dim = 512
        elif backbone_type == 'vit':
            # Use pre-trained Vision Transformer (ViT-B/16)
            # ViT-B/16 outputs 768 features from the [CLS] token
            vit = models.vit_b_16(weights='IMAGENET1K_V1')
            # Store components needed for feature extraction
            self.vit_conv_proj = vit.conv_proj
            self.vit_encoder = vit.encoder
            self.vit_class_token = vit.class_token
            # Store original pos_embed for interpolation
            self.vit_pos_embed_original = vit.encoder.pos_embedding
            # Will interpolate pos_embed on first forward pass
            self.vit_pos_embed_interpolated = None
            backbone_output_dim = 768
        elif backbone_type == 'cnn':
            # Simple 4-layer CNN with minimal dropout (feature extraction needs less regularization)
            self.backbone = nn.Sequential(
                # Input: 3 x 84 x 84
                nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
                nn.ReLU(),
                # No dropout in CNN - let it learn features freely
                # 32 x 21 x 21
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                # No dropout in CNN - let it learn features freely
                # 64 x 11 x 11
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # 64 x 11 x 11
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                # 64 x 11 x 11
                nn.AdaptiveAvgPool2d((1, 1)),
                # 64 x 1 x 1
            )
            backbone_output_dim = 64
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}. Must be 'resnet', 'vit', or 'cnn'")
        
        # Optional GRU for temporal modeling (lightweight RNN)
        if use_gru:
            # GRU input: backbone features + proprioceptive state
            gru_input_dim = backbone_output_dim + state_dim
            self.gru = nn.GRU(
                input_size=gru_input_dim,
                hidden_size=gru_hidden_dim,
                num_layers=1,  # Single layer for lightweight
                batch_first=True  # (batch, seq, features)
            )
            # MLP head takes GRU hidden state
            head_input_dim = gru_hidden_dim
        else:
            self.gru = None
            # MLP head takes backbone features + state directly
            head_input_dim = backbone_output_dim + state_dim
        
        # MLP head for action prediction with higher dropout (decision-making can tolerate more regularization)
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Higher dropout in MLP head for regularization
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3),  # Higher dropout in MLP head for regularization
            nn.Linear(feature_dim, action_dim),
            nn.Tanh()  # Normalize actions to [-1, 1] range
        )
    
    def forward(self, image, state=None, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            image: Tensor of shape (batch_size, 3, height, width) or (batch_size, seq_len, 3, height, width) if using GRU
            state: Optional tensor of shape (batch_size, state_dim) or (batch_size, seq_len, state_dim) for proprioceptive state
            hidden: Optional GRU hidden state for temporal modeling (batch_size, 1, hidden_dim)
            
        Returns:
            action: Tensor of shape (batch_size, action_dim) or (batch_size, seq_len, action_dim)
            hidden: Updated GRU hidden state (if using GRU)
        """
        # Extract features from image
        if self.backbone_type == 'vit':
            # ViT forward pass: project to patches, add class token, encode, extract [CLS]
            # Project image to patches: (B, C, H, W) -> (B, C, H', W')
            x = self.vit_conv_proj(image)
            n, c, h, w = x.shape
            num_patches = h * w
            # Flatten patches: (B, C, H, W) -> (B, H*W, C)
            x = x.reshape(n, c, num_patches).permute(0, 2, 1)
            
            # Expand class token and add to sequence
            batch_class_token = self.vit_class_token.expand(n, -1, -1)
            x = torch.cat([batch_class_token, x], dim=1)
            
            # Interpolate encoder's positional embeddings to match sequence length
            seq_len = x.shape[1]
            if self.vit_pos_embed_interpolated is None or self.vit_pos_embed_interpolated.shape[1] != seq_len:
                pos_embed = self.vit_pos_embed_original  # Shape: (1, 197, 768) for ViT-B/16
                if pos_embed.shape[1] != seq_len:
                    # Interpolate positional embeddings using 2D interpolation
                    # Original: 197 tokens (1 CLS + 14x14 patches)
                    # Get CLS token pos embed and patch pos embeds separately
                    cls_pos_embed = pos_embed[:, 0:1, :]  # (1, 1, 768)
                    patch_pos_embed = pos_embed[:, 1:, :]  # (1, 196, 768)
                    
                    # Reshape patch pos embeds to 2D grid (14, 14, 768) -> (1, 768, 14, 14)
                    patch_pos_embed_2d = patch_pos_embed.reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
                    
                    # Calculate target grid size (sqrt of num_patches)
                    target_size = int(num_patches ** 0.5)
                    # Interpolate to target size
                    patch_pos_embed_interp = F.interpolate(
                        patch_pos_embed_2d, 
                        size=(target_size, target_size), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    # Reshape back: (1, 768, H, W) -> (1, H*W, 768)
                    patch_pos_embed_interp = patch_pos_embed_interp.permute(0, 2, 3, 1).reshape(1, target_size * target_size, -1)
                    
                    # Concatenate CLS token back
                    self.vit_pos_embed_interpolated = torch.cat([cls_pos_embed, patch_pos_embed_interp], dim=1)
                else:
                    self.vit_pos_embed_interpolated = pos_embed
                
                # Update encoder's positional embedding
                self.vit_encoder.pos_embedding = nn.Parameter(self.vit_pos_embed_interpolated)
            
            # Pass through encoder (it will add positional embeddings internally)
            x = self.vit_encoder(x)
            
            # Extract [CLS] token (first token) - shape: (batch_size, hidden_dim)
            features = x[:, 0]
        else:
            features = self.backbone(image)
            # Flatten features
            if self.backbone_type == 'resnet':
                # ResNet already has global average pooling
                features = features.view(features.size(0), -1)
            else:  # CNN
                features = features.view(features.size(0), -1)
        
        # Concatenate proprioceptive state if available
        if state is not None and self.state_dim > 0:
            features = torch.cat([features, state], dim=-1)  # Use -1 for last dim to handle both 2D and 3D
        
        # Apply GRU if enabled (for temporal modeling)
        if self.use_gru:
            # Handle both single timestep and sequence inputs
            if len(features.shape) == 2:
                # Single timestep: (batch_size, feature_dim) -> (batch_size, 1, feature_dim)
                features = features.unsqueeze(1)
            
            # Initialize hidden state if not provided
            if hidden is None:
                batch_size = features.shape[0]
                hidden = torch.zeros(1, batch_size, self.gru_hidden_dim, 
                                    device=features.device, dtype=features.dtype)
            
            # GRU forward: (batch, seq_len, input_dim) -> (batch, seq_len, hidden_dim)
            gru_out, hidden = self.gru(features, hidden)
            
            # Use last timestep output (or squeeze if single timestep)
            if gru_out.shape[1] == 1:
                features = gru_out.squeeze(1)  # (batch, hidden_dim)
            else:
                features = gru_out[:, -1, :]  # (batch, hidden_dim) - last timestep
        else:
            # No GRU - use features directly
            if len(features.shape) == 3:
                # If somehow got 3D without GRU, take last timestep
                features = features[:, -1, :]
        
        # Predict action
        action = self.head(features)
        
        if self.use_gru:
            return action, hidden
        else:
            return action


def load_policy(model_path, image_size=(84, 84), action_dim=3, backbone_type=None, use_resnet=None, device='cpu'):  # Default 3 for push task
    """
    Load a trained policy from a checkpoint.
    
    Args:
        model_path: Path to the saved model weights
        image_size: Tuple of (height, width) for input images
        action_dim: Dimension of action space
        backbone_type: Backbone type ('resnet', 'vit', or 'cnn'). If None, will be inferred from filename or checkpoint.
        use_resnet: Deprecated - use backbone_type instead
        device: Device to load the model on
        
    Returns:
        model: Loaded policy model
    """
    # Load checkpoint first
    checkpoint = torch.load(model_path, map_location=device)
    
    # Auto-detect backbone type if not provided
    if backbone_type is None:
        # Try to infer from filename first
        import os
        filename = os.path.basename(model_path).lower()
        if '_vit' in filename:
            backbone_type = 'vit'
        elif '_cnn' in filename:
            backbone_type = 'cnn'
        elif '_resnet' in filename:
            backbone_type = 'resnet'
        else:
            # Try loading from checkpoint metadata
            if isinstance(checkpoint, dict) and 'backbone_type' in checkpoint:
                backbone_type = checkpoint['backbone_type']
            elif isinstance(checkpoint, dict) and 'metadata' in checkpoint:
                backbone_type = checkpoint['metadata'].get('backbone_type', 'resnet')
            else:
                # Default to resnet if we can't determine
                backbone_type = 'resnet'
                print(f"Warning: Could not infer backbone type from filename or checkpoint. Defaulting to 'resnet'.")
    
    # Handle backward compatibility
    if use_resnet is not None:
        backbone_type = 'resnet' if use_resnet else 'cnn'
    
    # Handle both old format (just state_dict) and new format (with metadata)
    state_dim = 0  # Default: no proprioception
    use_spherical = False  # Default: Cartesian coordinates
    use_gru = False  # Default: no GRU
    gru_hidden_dim = 128  # Default GRU hidden dimension
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Use metadata if available (overrides filename inference)
        if 'backbone_type' in checkpoint:
            backbone_type = checkpoint['backbone_type']
        elif 'metadata' in checkpoint and 'backbone_type' in checkpoint['metadata']:
            backbone_type = checkpoint['metadata'].get('backbone_type', backbone_type)
        # Get state_dim from checkpoint if available
        if 'state_dim' in checkpoint:
            state_dim = checkpoint['state_dim']
        elif 'metadata' in checkpoint and 'state_dim' in checkpoint['metadata']:
            state_dim = checkpoint['metadata'].get('state_dim', 0)
        # Get use_spherical from checkpoint if available
        if 'use_spherical' in checkpoint:
            use_spherical = checkpoint['use_spherical']
        elif 'metadata' in checkpoint and 'use_spherical' in checkpoint['metadata']:
            use_spherical = checkpoint['metadata'].get('use_spherical', False)
        # Get use_gru from checkpoint if available
        if 'use_gru' in checkpoint:
            use_gru = checkpoint['use_gru']
        elif 'metadata' in checkpoint and 'use_gru' in checkpoint['metadata']:
            use_gru = checkpoint['metadata'].get('use_gru', False)
        # Get gru_hidden_dim from checkpoint if available
        if 'gru_hidden_dim' in checkpoint:
            gru_hidden_dim = checkpoint['gru_hidden_dim']
        elif 'metadata' in checkpoint and 'gru_hidden_dim' in checkpoint['metadata']:
            gru_hidden_dim = checkpoint['metadata'].get('gru_hidden_dim', 128)
    else:
        # Old format: just state_dict
        state_dict = checkpoint
    
    # Auto-detect GRU from state_dict if not in checkpoint metadata
    # Check if GRU weights are present in state_dict
    if not use_gru:
        gru_keys = [k for k in state_dict.keys() if 'gru' in k.lower()]
        if gru_keys:
            use_gru = True
            print(f"Detected GRU in checkpoint (keys: {len(gru_keys)} GRU parameters)")
            # Infer gru_hidden_dim from head input size
            if 'head.0.weight' in state_dict:
                head_input_size = state_dict['head.0.weight'].shape[1]
                if head_input_size != (512 + state_dim):  # If not backbone+state, must be GRU output
                    gru_hidden_dim = head_input_size
                    print(f"Inferred gru_hidden_dim={gru_hidden_dim} from head input size")
    
    model = VisuomotorBCPolicy(image_size=image_size, action_dim=action_dim, 
                               backbone_type=backbone_type, use_resnet=None, 
                               state_dim=state_dim, use_gru=use_gru, gru_hidden_dim=gru_hidden_dim)
    
    # For ViT, remove interpolated pos_embedding from state_dict if present
    # (we'll interpolate it on the fly during forward pass)
    if backbone_type == 'vit':
        keys_to_remove = [k for k in state_dict.keys() if 'vit_encoder.pos_embedding' in k or 'vit_pos_embed_interpolated' in k]
        for k in keys_to_remove:
            del state_dict[k]
    
    # Load state dict with strict=False to handle size mismatches
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    # Store use_spherical flag for evaluation
    model.use_spherical = use_spherical
    return model

