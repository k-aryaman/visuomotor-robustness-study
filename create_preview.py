"""
Script to create preview images from demonstration trajectories.
Generates a grid of frames showing a sample trajectory.
"""

import pickle
import numpy as np
from PIL import Image
import argparse
import os


def create_preview_grid(demonstrations_file, output_file, trajectory_idx=0, grid_size=(4, 4)):
    """
    Create a preview grid image from a demonstration trajectory.
    
    Args:
        demonstrations_file: Path to the pickle file containing trajectories
        output_file: Path to save the preview image
        trajectory_idx: Index of the trajectory to visualize (default: 0)
        grid_size: Tuple of (rows, cols) for the grid (default: (4, 4))
    """
    # Load demonstrations
    with open(demonstrations_file, 'rb') as f:
        trajectories = pickle.load(f)
    
    if len(trajectories) == 0:
        print(f"Error: No trajectories found in {demonstrations_file}")
        return
    
    if trajectory_idx >= len(trajectories):
        print(f"Warning: trajectory_idx {trajectory_idx} >= {len(trajectories)}. Using trajectory 0.")
        trajectory_idx = 0
    
    trajectory = trajectories[trajectory_idx]
    num_frames = len(trajectory)
    grid_rows, grid_cols = grid_size
    total_grid_frames = grid_rows * grid_cols
    
    # Select evenly spaced frames from the trajectory
    if num_frames <= total_grid_frames:
        # Use all frames if trajectory is shorter than grid
        selected_indices = list(range(num_frames))
        # Pad with last frame if needed
        while len(selected_indices) < total_grid_frames:
            selected_indices.append(num_frames - 1)
    else:
        # Select evenly spaced frames
        step = num_frames / total_grid_frames
        selected_indices = [int(i * step) for i in range(total_grid_frames)]
    
    # Extract images - following data_collection.py format:
    # trajectory.append((np.array(image), np.array(action)))
    # where image comes from env.render() with render_mode='rgb_array', render_width=84, render_height=84
    images = []
    for idx in selected_indices[:total_grid_frames]:
        image, _ = trajectory[idx]
        
        # Convert to numpy array if not already (should already be from data_collection.py)
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Handle dtype - env.render() with rgb_array typically returns uint8 (0-255)
        if image.dtype != np.uint8:
            # If float in [0, 1], scale to uint8
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Handle image shape - rgb_array should be (H, W, 3) or (H, W, 4) for RGBA
        if len(image.shape) == 2:
            # Grayscale, convert to RGB
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                # RGBA, convert to RGB
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError(f"Unexpected image shape: {image.shape}, expected (H, W, 3) or (H, W, 4)")
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}, expected 2D or 3D array")
        
        images.append(image)
    
    # Create grid
    img_height, img_width = images[0].shape[:2]
    grid_image = np.zeros((grid_rows * img_height, grid_cols * img_width, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        row = i // grid_cols
        col = i % grid_cols
        y_start = row * img_height
        y_end = y_start + img_height
        x_start = col * img_width
        x_end = x_start + img_width
        grid_image[y_start:y_end, x_start:x_end] = img
    
    # Save image
    pil_image = Image.fromarray(grid_image)
    pil_image.save(output_file)
    print(f"Saved preview grid to {output_file}")
    print(f"  Trajectory {trajectory_idx} with {num_frames} frames")
    print(f"  Grid size: {grid_rows}x{grid_cols}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create preview grid from demonstration trajectories")
    parser.add_argument('--input', type=str, default='demonstrations.pkl',
                       help='Input pickle file with trajectories')
    parser.add_argument('--output', type=str, default='demo_preview.png',
                       help='Output preview image filename')
    parser.add_argument('--trajectory', type=int, default=0,
                       help='Index of trajectory to visualize (default: 0)')
    parser.add_argument('--grid_rows', type=int, default=4,
                       help='Number of rows in grid (default: 4)')
    parser.add_argument('--grid_cols', type=int, default=4,
                       help='Number of columns in grid (default: 4)')
    
    args = parser.parse_args()
    
    create_preview_grid(
        demonstrations_file=args.input,
        output_file=args.output,
        trajectory_idx=args.trajectory,
        grid_size=(args.grid_rows, args.grid_cols)
    )

