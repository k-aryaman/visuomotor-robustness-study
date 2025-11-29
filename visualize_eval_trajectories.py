"""
Script to visualize success and failure trajectories from evaluation.
Uses create_preview functionality to show multiple success and failure examples.
"""

import pickle
import argparse
import os
from create_preview import create_preview_grid
import numpy as np


def visualize_eval_trajectories(eval_trajectories_file, output_dir='eval_visualizations', max_examples=1):
    """
    Visualize multiple success and failure trajectories from evaluation.
    
    Args:
        eval_trajectories_file: Path to pickle file with evaluation trajectories
        output_dir: Directory to save visualization images
        max_examples: Maximum number of success and failure examples to visualize (default: 1)
    """
    # Load evaluation trajectories
    with open(eval_trajectories_file, 'rb') as f:
        trajectories = pickle.load(f)
    
    if len(trajectories) == 0:
        print(f"Error: No trajectories found in {eval_trajectories_file}")
        return
    
    # Find success and failure trajectories
    success_trajectories = []  # List of (images, episode_idx)
    failure_trajectories = []   # List of (images, episode_idx)
    
    for traj_data in trajectories:
        images, is_success, episode_idx = traj_data
        if is_success:
            if len(success_trajectories) < max_examples:
                success_trajectories.append((images, episode_idx))
        else:
            if len(failure_trajectories) < max_examples:
                failure_trajectories.append((images, episode_idx))
        
        # Stop if we have enough of both
        if len(success_trajectories) >= max_examples and len(failure_trajectories) >= max_examples:
            break
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize success trajectories
    if len(success_trajectories) > 0:
        print(f"Found {len(success_trajectories)} success example(s)")
        for idx, (images, episode_idx) in enumerate(success_trajectories):
            # Convert list of images to format: list of (image,) tuples
            success_trajectory = [(img,) for img in images]
            success_output = os.path.join(output_dir, f'success_trajectory_{idx}.png')
            create_success_failure_preview(success_trajectory, success_output, 
                                          title=f"Success Episode {episode_idx}")
            print(f"  Saved to {success_output}")
    else:
        print("Warning: No successful episodes found")
    
    # Visualize failure trajectories
    if len(failure_trajectories) > 0:
        print(f"Found {len(failure_trajectories)} failure example(s)")
        for idx, (images, episode_idx) in enumerate(failure_trajectories):
            failure_trajectory = [(img,) for img in images]
            failure_output = os.path.join(output_dir, f'failure_trajectory_{idx}.png')
            create_success_failure_preview(failure_trajectory, failure_output,
                                          title=f"Failure Episode {episode_idx}")
            print(f"  Saved to {failure_output}")
    else:
        print("Warning: No failed episodes found")
    
    print(f"\nTotal visualizations saved: {len(success_trajectories) + len(failure_trajectories)}")


def create_success_failure_preview(trajectory, output_file, grid_size=(4, 4), title=""):
    """
    Create a preview grid from a trajectory (list of (image,) tuples).
    Similar to create_preview_grid but handles the simpler format.
    """
    from PIL import Image, ImageDraw, ImageFont
    
    num_frames = len(trajectory)
    grid_rows, grid_cols = grid_size
    total_grid_frames = grid_rows * grid_cols
    
    # Select evenly spaced frames
    if num_frames <= total_grid_frames:
        selected_indices = list(range(num_frames))
        while len(selected_indices) < total_grid_frames:
            selected_indices.append(num_frames - 1)
    else:
        step = num_frames / total_grid_frames
        selected_indices = [int(i * step) for i in range(total_grid_frames)]
    
    # Extract and process images
    images = []
    for idx in selected_indices[:total_grid_frames]:
        frame = trajectory[idx]
        image = frame[0]  # Just get the image
        
        # Convert to numpy array if not already
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Handle dtype
        if image.dtype != np.uint8:
            if image.dtype == np.float32 or image.dtype == np.float64:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Handle image shape
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3:
            if image.shape[2] == 4:
                image = image[:, :, :3]
            elif image.shape[2] != 3:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        
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
    
    # Convert to PIL and add title if provided
    pil_image = Image.fromarray(grid_image)
    if title:
        # Add title text at the top
        draw = ImageDraw.Draw(pil_image)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
        # Draw text with background
        bbox = draw.textbbox((0, 0), title, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        # Draw background rectangle
        draw.rectangle([(0, 0), (text_width + 10, text_height + 10)], fill=(0, 0, 0, 200))
        draw.text((5, 5), title, fill=(255, 255, 255), font=font)
    
    pil_image.save(output_file)
    print(f"Saved preview grid to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize success and failure trajectories from evaluation")
    parser.add_argument('--input', type=str, default='eval_trajectories.pkl',
                       help='Input pickle file with evaluation trajectories')
    parser.add_argument('--output_dir', type=str, default='eval_visualizations',
                       help='Output directory for visualization images')
    parser.add_argument('--examples', type=int, default=1,
                       help='Maximum number of success and failure examples to visualize (default: 1)')
    
    args = parser.parse_args()
    
    visualize_eval_trajectories(args.input, args.output_dir, max_examples=args.examples)

