"""
Utility functions for converting between Cartesian and spherical (polar) action coordinates.
"""

import torch
import numpy as np


def cartesian_to_spherical(cartesian, min_magnitude=1e-4):
    """
    Convert 3D Cartesian coordinates (dx, dy, dz) to spherical coordinates (magnitude, theta, phi).
    
    Spherical coordinates:
    - r (magnitude): sqrt(dx² + dy² + dz²), clamped to minimum to avoid angle instability
    - θ (theta): elevation angle from z-axis [0, π]
    - φ (phi): azimuth angle in xy-plane [-π, π]
    
    Args:
        cartesian: Tensor or array of shape (..., 3) with (dx, dy, dz)
        min_magnitude: Minimum magnitude to use when action is near zero (default: 1e-4)
                      This prevents angle instability when magnitude is exactly zero
        
    Returns:
        spherical: Tensor or array of shape (..., 3) with (magnitude, theta, phi)
    """
    if isinstance(cartesian, torch.Tensor):
        dx, dy, dz = cartesian[..., 0], cartesian[..., 1], cartesian[..., 2]
        
        # Magnitude with minimum clamp to avoid angle instability
        r = torch.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)  # Add small epsilon to avoid division by zero
        r = torch.clamp(r, min=min_magnitude)  # Clamp minimum magnitude
        
        # Theta: elevation angle from z-axis [0, π]
        # theta = arccos(z/r) when r > 0
        theta = torch.acos(torch.clamp(dz / r, -1.0, 1.0))
        
        # Phi: azimuth angle in xy-plane [-π, π]
        phi = torch.atan2(dy, dx)
        
        return torch.stack([r, theta, phi], dim=-1)
    else:
        # NumPy version
        cartesian = np.asarray(cartesian)
        dx, dy, dz = cartesian[..., 0], cartesian[..., 1], cartesian[..., 2]
        
        r = np.sqrt(dx**2 + dy**2 + dz**2 + 1e-8)
        r = np.clip(r, min_magnitude, None)  # Clamp minimum magnitude
        
        theta = np.arccos(np.clip(dz / r, -1.0, 1.0))
        phi = np.arctan2(dy, dx)
        
        return np.stack([r, theta, phi], axis=-1)


def spherical_to_cartesian(spherical):
    """
    Convert spherical coordinates (magnitude, theta, phi) to 3D Cartesian (dx, dy, dz).
    
    Args:
        spherical: Tensor or array of shape (..., 3) with (magnitude, theta, phi)
        
    Returns:
        cartesian: Tensor or array of shape (..., 3) with (dx, dy, dz)
    """
    if isinstance(spherical, torch.Tensor):
        r, theta, phi = spherical[..., 0], spherical[..., 1], spherical[..., 2]
        
        dx = r * torch.sin(theta) * torch.cos(phi)
        dy = r * torch.sin(theta) * torch.sin(phi)
        dz = r * torch.cos(theta)
        
        return torch.stack([dx, dy, dz], dim=-1)
    else:
        # NumPy version
        spherical = np.asarray(spherical)
        r, theta, phi = spherical[..., 0], spherical[..., 1], spherical[..., 2]
        
        dx = r * np.sin(theta) * np.cos(phi)
        dy = r * np.sin(theta) * np.sin(phi)
        dz = r * np.cos(theta)
        
        return np.stack([dx, dy, dz], axis=-1)

