"""
ACT (Action Chunking with Transformers) policy for visuomotor behavior cloning.
Predicts sequences of actions (chunks) instead of single actions per timestep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


class ACTPolicy(nn.Module):
    """
    Action Chunking with Transformers (ACT) policy.
    
    Architecture:
    - Visual Encoder (CNN/ResNet): Extracts features from images
    - Transformer Encoder: Processes sequence of observations (visual + proprioceptive)
    - Transformer Decoder: Predicts sequence of future actions
    - Action Head: Maps decoder output to action space
    
    Input: Sequence of (image, state) pairs
    Output: Sequence of actions (chunk)
    """
    
    def __init__(self, 
                 image_size=(84, 84),
                 action_dim=4,
                 chunk_size=10,
                 backbone_type='resnet',
                 state_dim=4,
                 d_model=512,
                 nhead=8,
                 num_encoder_layers=4,
                 num_decoder_layers=4,
                 dim_feedforward=2048,
                 dropout=0.1):
        """
        Initialize ACT policy.
        
        Args:
            image_size: Tuple of (height, width) for input images
            action_dim: Dimension of action space (default 4 for panda-gym)
            chunk_size: Number of actions to predict in each chunk
            backbone_type: Visual backbone type ('resnet' or 'cnn')
            state_dim: Dimension of proprioceptive state (default 4: gripper_pos + width)
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            num_decoder_layers: Number of transformer decoder layers
            dim_feedforward: Dimension of feedforward network in transformer
            dropout: Dropout rate
        """
        super(ACTPolicy, self).__init__()
        
        self.image_size = image_size
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.d_model = d_model
        self.backbone_type = backbone_type
        
        # Visual encoder (backbone)
        if backbone_type == 'resnet':
            # Use pre-trained ResNet-18 as backbone
            resnet = models.resnet18(weights='IMAGENET1K_V1')
            # Remove the final fully connected layer
            self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])
            visual_feature_dim = 512  # ResNet-18 output
        elif backbone_type == 'cnn':
            # Simple CNN backbone (same as in VisuomotorBCPolicy)
            self.visual_encoder = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            visual_feature_dim = 64
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")
        
        # Project visual features to d_model
        self.visual_proj = nn.Linear(visual_feature_dim, d_model)
        
        # Project proprioceptive state to d_model
        if state_dim > 0:
            self.state_proj = nn.Linear(state_dim, d_model)
        
        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder: processes sequence of observations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Transformer decoder: predicts sequence of actions
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Action query embeddings (learnable embeddings for each action in chunk)
        self.action_queries = nn.Parameter(torch.randn(chunk_size, d_model))
        
        # Action head: maps decoder output to action space
        self.action_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, action_dim),
            nn.Tanh()  # Normalize actions to [-1, 1]
        )
        
    def forward(self, images, states=None, query_actions=None):
        """
        Forward pass through ACT policy.
        
        Args:
            images: Tensor of shape (batch_size, seq_len, 3, height, width) or (batch_size, 3, height, width)
                   If single timestep, will be expanded to sequence
            states: Optional tensor of shape (batch_size, seq_len, state_dim) or (batch_size, state_dim)
                   Proprioceptive state for each timestep
            query_actions: Optional tensor for teacher forcing during training
                          Shape: (batch_size, chunk_size, action_dim)
        
        Returns:
            actions: Tensor of shape (batch_size, chunk_size, action_dim)
                    Predicted action chunk
        """
        # Handle single timestep input (for inference)
        if len(images.shape) == 4:
            # (batch_size, 3, H, W) -> (batch_size, 1, 3, H, W)
            images = images.unsqueeze(1)
            single_timestep = True
        else:
            single_timestep = False
        
        batch_size, seq_len = images.shape[0], images.shape[1]
        
        # Extract visual features for each timestep
        # Reshape to process all images at once: (batch * seq_len, 3, H, W)
        images_flat = images.view(batch_size * seq_len, *images.shape[2:])
        visual_features_flat = self.visual_encoder(images_flat)  # (batch * seq_len, C, 1, 1)
        visual_features_flat = visual_features_flat.view(batch_size * seq_len, -1)  # (batch * seq_len, C)
        visual_features_flat = self.visual_proj(visual_features_flat)  # (batch * seq_len, d_model)
        visual_features = visual_features_flat.view(batch_size, seq_len, self.d_model)  # (batch, seq_len, d_model)
        
        # Process proprioceptive state if available
        if states is not None and self.state_dim > 0:
            if len(states.shape) == 2:
                # Single timestep: (batch, state_dim) -> (batch, 1, state_dim)
                states = states.unsqueeze(1)
            state_features = self.state_proj(states)  # (batch, seq_len, d_model)
            # Combine visual and state features (add or concatenate)
            # Using addition for simplicity (both projected to same dimension)
            encoder_input = visual_features + state_features
        else:
            encoder_input = visual_features
        
        # Add positional encoding
        encoder_input = self.pos_encoder(encoder_input)  # (batch, seq_len, d_model)
        
        # Transformer encoder: process observation sequence
        encoder_output = self.transformer_encoder(encoder_input)  # (batch, seq_len, d_model)
        
        # Prepare action queries (learnable embeddings for each action in chunk)
        action_queries = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, chunk_size, d_model)
        
        # Transformer decoder: predict action chunk
        # encoder_output: (batch, seq_len, d_model) - memory for cross-attention
        # action_queries: (batch, chunk_size, d_model) - queries for self-attention
        decoder_output = self.transformer_decoder(
            tgt=action_queries,
            memory=encoder_output
        )  # (batch, chunk_size, d_model)
        
        # Map decoder output to actions
        actions = self.action_head(decoder_output)  # (batch, chunk_size, action_dim)
        
        return actions


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    Adds sinusoidal positional encodings to input sequences.
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

