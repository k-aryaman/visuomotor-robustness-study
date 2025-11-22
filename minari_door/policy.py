"""
Policy networks for offline reinforcement learning on door-opening task.
Includes MLP and Transformer-based architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class MLPPolicy(nn.Module):
    """
    Multi-layer perceptron policy for behavior cloning.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256, 256),
        activation: str = 'relu',
        dropout: float = 0.0,
        layer_norm: bool = True
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Build layers
        layers = []
        in_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())  # Actions typically in [-1, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.fc2(x)
        return x + residual


class ResidualMLPPolicy(nn.Module):
    """
    Residual MLP policy for more stable training on complex tasks.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(obs)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x)
        return torch.tanh(x)


class GaussianPolicy(nn.Module):
    """
    Gaussian policy that outputs mean and log_std.
    Useful for stochastic policies and uncertainty estimation.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared backbone
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)
        
    def forward(
        self, 
        obs: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        features = self.backbone(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        if deterministic:
            return torch.tanh(mean), None
        
        std = torch.exp(log_std)
        noise = torch.randn_like(mean)
        action = mean + std * noise
        
        # Log probability calculation
        log_prob = -0.5 * (((action - mean) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))
        log_prob = log_prob.sum(-1, keepdim=True)
        
        # Squashing correction
        log_prob -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(-1, keepdim=True)
        
        return torch.tanh(action), log_prob
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        action, _ = self.forward(obs, deterministic=deterministic)
        return action


class EnsemblePolicy(nn.Module):
    """
    Ensemble of policies for uncertainty estimation and robustness.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_members: int = 5,
        hidden_dims: Tuple[int, ...] = (256, 256)
    ):
        super().__init__()
        
        self.num_members = num_members
        self.members = nn.ModuleList([
            MLPPolicy(obs_dim, action_dim, hidden_dims)
            for _ in range(num_members)
        ])
        
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns mean action and uncertainty (std across ensemble).
        """
        actions = torch.stack([member(obs) for member in self.members], dim=0)
        mean_action = actions.mean(dim=0)
        std_action = actions.std(dim=0)
        return mean_action, std_action
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        use_uncertainty: bool = False,
        uncertainty_threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Get action, optionally using uncertainty for conservative behavior.
        """
        mean_action, std_action = self.forward(obs)
        
        if use_uncertainty:
            # Scale down action magnitude when uncertain
            uncertainty = std_action.mean(dim=-1, keepdim=True)
            scale = torch.clamp(1.0 - uncertainty / uncertainty_threshold, 0.1, 1.0)
            return mean_action * scale
        
        return mean_action


class TransformerPolicy(nn.Module):
    """
    Transformer-based policy for handling sequential dependencies.
    Useful when observation includes history.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        self.input_proj = nn.Linear(obs_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs shape: (batch, obs_dim) or (batch, seq_len, obs_dim)
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # Add sequence dimension
        
        x = self.input_proj(obs)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Pool over sequence
        return self.output_proj(x)


def create_policy(
    policy_type: str,
    obs_dim: int,
    action_dim: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create policy networks.
    
    Args:
        policy_type: Type of policy ('mlp', 'residual', 'gaussian', 'ensemble', 'transformer')
        obs_dim: Observation dimension
        action_dim: Action dimension
        **kwargs: Additional arguments for specific policy types
    """
    if policy_type == 'mlp':
        return MLPPolicy(obs_dim, action_dim, **kwargs)
    elif policy_type == 'residual':
        return ResidualMLPPolicy(obs_dim, action_dim, **kwargs)
    elif policy_type == 'gaussian':
        return GaussianPolicy(obs_dim, action_dim, **kwargs)
    elif policy_type == 'ensemble':
        return EnsemblePolicy(obs_dim, action_dim, **kwargs)
    elif policy_type == 'transformer':
        return TransformerPolicy(obs_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def load_policy(
    checkpoint_path: str,
    device: str = 'cuda'
) -> Tuple[nn.Module, Dict]:
    """
    Load a trained policy from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        policy: Loaded policy network
        metadata: Training metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get architecture info
    metadata = checkpoint.get('metadata', {})
    policy_type = metadata.get('policy_type', 'mlp')
    obs_dim = metadata.get('obs_dim')
    action_dim = metadata.get('action_dim')
    
    # Create policy
    policy = create_policy(policy_type, obs_dim, action_dim)
    policy.load_state_dict(checkpoint['state_dict'])
    policy.to(device)
    policy.eval()
    
    return policy, metadata


if __name__ == '__main__':
    # Test policies
    obs_dim = 39  # Door env observation dim
    action_dim = 28  # Door env action dim
    batch_size = 32
    
    obs = torch.randn(batch_size, obs_dim)
    
    print("Testing policy architectures...")
    
    # MLP
    mlp = MLPPolicy(obs_dim, action_dim)
    action = mlp(obs)
    print(f"MLP output shape: {action.shape}")
    
    # Residual MLP
    res_mlp = ResidualMLPPolicy(obs_dim, action_dim)
    action = res_mlp(obs)
    print(f"Residual MLP output shape: {action.shape}")
    
    # Gaussian
    gaussian = GaussianPolicy(obs_dim, action_dim)
    action, log_prob = gaussian(obs)
    print(f"Gaussian output shape: {action.shape}, log_prob shape: {log_prob.shape}")
    
    # Ensemble
    ensemble = EnsemblePolicy(obs_dim, action_dim, num_members=3)
    action, std = ensemble(obs)
    print(f"Ensemble output shape: {action.shape}, std shape: {std.shape}")
    
    # Transformer
    transformer = TransformerPolicy(obs_dim, action_dim)
    action = transformer(obs)
    print(f"Transformer output shape: {action.shape}")
    
    print("\nAll policy tests passed!")
