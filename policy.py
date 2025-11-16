"""
PyTorch policy model for visuomotor behavior cloning.
Implements a CNN-MLP architecture for processing RGB images and outputting actions.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class VisuomotorBCPolicy(nn.Module):
    """
    CNN-MLP architecture for visuomotor behavior cloning.
    
    Backbone: 4-layer CNN or pre-trained ResNet-18
    Head: MLP that takes flattened image features and outputs continuous actions
    """
    
    def __init__(self, image_size=(84, 84), action_dim=4, use_resnet=True, feature_dim=512):
        """
        Initialize the policy network.
        
        Args:
            image_size: Tuple of (height, width) for input images
            action_dim: Dimension of action space (default 4 for panda-gym)
            use_resnet: If True, use ResNet-18 backbone; else use simple CNN
            feature_dim: Dimension of feature vector before MLP head
        """
        super(VisuomotorBCPolicy, self).__init__()
        
        self.image_size = image_size
        self.action_dim = action_dim
        self.use_resnet = use_resnet
        
        if use_resnet:
            # Use pre-trained ResNet-18 as backbone
            resnet = models.resnet18(pretrained=True)
            # Remove the final fully connected layer
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            # ResNet-18 outputs 512 features after global average pooling
            backbone_output_dim = 512
        else:
            # Simple 4-layer CNN
            self.backbone = nn.Sequential(
                # Input: 3 x 84 x 84
                nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2),
                nn.ReLU(),
                # 32 x 21 x 21
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
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
        
        # MLP head for action prediction
        self.head = nn.Sequential(
            nn.Linear(backbone_output_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_dim),
            nn.Tanh()  # Normalize actions to [-1, 1] range
        )
    
    def forward(self, image):
        """
        Forward pass through the network.
        
        Args:
            image: Tensor of shape (batch_size, 3, height, width)
            
        Returns:
            action: Tensor of shape (batch_size, action_dim)
        """
        # Extract features from image
        features = self.backbone(image)
        
        # Flatten features
        if self.use_resnet:
            # ResNet already has global average pooling
            features = features.view(features.size(0), -1)
        else:
            features = features.view(features.size(0), -1)
        
        # Predict action
        action = self.head(features)
        
        return action


def load_policy(model_path, image_size=(84, 84), action_dim=4, use_resnet=True, device='cpu'):
    """
    Load a trained policy from a checkpoint.
    
    Args:
        model_path: Path to the saved model weights
        image_size: Tuple of (height, width) for input images
        action_dim: Dimension of action space
        use_resnet: Whether to use ResNet backbone
        device: Device to load the model on
        
    Returns:
        model: Loaded policy model
    """
    model = VisuomotorBCPolicy(image_size=image_size, action_dim=action_dim, use_resnet=use_resnet)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

