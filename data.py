"""
PyTorch Dataset class for loading demonstration data.
Includes transform objects for different visual regimes.
"""

import pickle
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class DemonstrationDataset(Dataset):
    """
    PyTorch Dataset for loading demonstration trajectories.
    """
    
    def __init__(self, demonstrations_file='demonstrations.pkl', transform=None):
        """
        Initialize the dataset.
        
        Args:
            demonstrations_file: Path to the pickle file containing trajectories
            transform: torchvision.transforms object to apply to images
        """
        self.transform = transform
        
        # Load demonstrations
        with open(demonstrations_file, 'rb') as f:
            self.trajectories = pickle.load(f)
        
        # Flatten trajectories into individual (image, action) pairs
        self.data = []
        for trajectory in self.trajectories:
            for image, action in trajectory:
                self.data.append((image, action))
        
        print(f"Loaded {len(self.trajectories)} trajectories")
        print(f"Total (image, action) pairs: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single (image, action) pair.
        
        Args:
            idx: Index of the sample
            
        Returns:
            image: Transformed image tensor
            action: Action tensor
        """
        image, action = self.data[idx]
        
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            # Ensure image is in the correct format (H, W, C) and uint8
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert action to tensor
        action = torch.FloatTensor(action)
        
        return image, action


def get_clean_transform():
    """
    Get transform for clean visual regime (standard normalization only).
    
    Returns:
        transform: torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_pixel_aug_transform():
    """
    Get transform for pixel augmentation regime.
    Includes: normalization + RandomResizedCrop, ColorJitter, and GaussianBlur.
    
    Returns:
        transform: torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((84, 84)),
        transforms.RandomResizedCrop(size=84, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


if __name__ == '__main__':
    # Test the dataset
    print("Testing DemonstrationDataset...")
    
    # Test clean transform
    clean_transform = get_clean_transform()
    clean_dataset = DemonstrationDataset('demonstrations.pkl', transform=clean_transform)
    print(f"Clean dataset size: {len(clean_dataset)}")
    
    if len(clean_dataset) > 0:
        image, action = clean_dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Action shape: {action.shape}")
        print(f"Action: {action}")
    
    # Test pixel augmentation transform
    pixel_aug_transform = get_pixel_aug_transform()
    aug_dataset = DemonstrationDataset('demonstrations.pkl', transform=pixel_aug_transform)
    print(f"\nPixel augmentation dataset size: {len(aug_dataset)}")
    
    if len(aug_dataset) > 0:
        image, action = aug_dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Action shape: {action.shape}")

