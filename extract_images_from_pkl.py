"""
Extract images from demonstration pickle file.
Reconstructs all images and saves them to a directory.
"""

import pickle
import numpy as np
from PIL import Image
import argparse
import os
from tqdm import tqdm


def extract_images(pkl_file, output_dir, max_trajectories=None, max_images_per_traj=None):
    """
    Extract images from pickle file and save them.
    
    Args:
        pkl_file: Path to pickle file
        output_dir: Directory to save images
        max_trajectories: Maximum number of trajectories to process (None = all)
        max_images_per_traj: Maximum images per trajectory (None = all)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Found {len(trajectories)} trajectories")
    
    total_images = 0
    traj_count = min(len(trajectories), max_trajectories) if max_trajectories else len(trajectories)
    
    for traj_idx in tqdm(range(traj_count), desc="Processing trajectories"):
        trajectory = trajectories[traj_idx]
        num_images = min(len(trajectory), max_images_per_traj) if max_images_per_traj else len(trajectory)
        
        for img_idx in range(num_images):
            image, state, action = trajectory[img_idx]
            
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                # Ensure uint8 format
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = image.astype(np.uint8)
                
                # Save image
                img = Image.fromarray(image)
                filename = f"traj{traj_idx:04d}_img{img_idx:04d}.png"
                img.save(os.path.join(output_dir, filename))
                total_images += 1
    
    print(f"\n✅ Extracted {total_images} images saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract images from demonstration pickle file")
    parser.add_argument("--input", type=str, default="demonstrations_push.pkl",
                       help="Input pickle file")
    parser.add_argument("--output", type=str, default="extracted_images/",
                       help="Output directory for images")
    parser.add_argument("--max_trajectories", type=int, default=None,
                       help="Maximum number of trajectories to process (None = all)")
    parser.add_argument("--max_images_per_traj", type=int, default=None,
                       help="Maximum images per trajectory (None = all)")
    
    args = parser.parse_args()
    
    extract_images(args.input, args.output, args.max_trajectories, args.max_images_per_traj)

