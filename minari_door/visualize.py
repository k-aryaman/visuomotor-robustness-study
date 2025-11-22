"""
Visualization script for generating videos of door-opening demonstrations.
Supports both clean and perturbed observation playback.
"""

import numpy as np
import argparse
import os
from typing import Optional, List, Dict, Tuple
import gymnasium as gym
try:
    import gymnasium_robotics
    gymnasium_robotics.register_robotics_envs()
except ImportError:
    print("Warning: gymnasium-robotics not installed. Install with: pip install gymnasium-robotics")

import minari
from tqdm import tqdm

# Video writing
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: cv2 not installed. Install with: pip install opencv-python")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("Warning: imageio not installed. Install with: pip install imageio[ffmpeg]")


class DemonstrationVisualizer:
    """
    Visualize demonstrations from Minari dataset with optional perturbations.
    """
    
    def __init__(
        self,
        dataset_name: str = "D4RL/door/human-v2",
        render_width: int = 640,
        render_height: int = 480,
        fps: int = 30
    ):
        self.dataset_name = dataset_name
        self.render_width = render_width
        self.render_height = render_height
        self.fps = fps
        
        # Load dataset
        print(f"Loading dataset: {dataset_name}")
        try:
            self.dataset = minari.load_dataset(dataset_name)
        except FileNotFoundError:
            print(f"Dataset not found locally. Downloading...")
            minari.download_dataset(dataset_name)
            self.dataset = minari.load_dataset(dataset_name)
        
        print(f"Loaded {self.dataset.total_episodes} episodes, {self.dataset.total_steps} steps")
        
        # Create environment for rendering
        self.env = gym.make('AdroitHandDoor-v1', render_mode='rgb_array')
        
    def _apply_perturbation(
        self,
        obs: np.ndarray,
        perturbation_type: str,
        strength: float,
        obs_std: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply perturbation to observation."""
        if perturbation_type == 'none' or perturbation_type is None:
            return obs
        
        obs = obs.copy()
        if obs_std is None:
            obs_std = np.abs(obs) + 1e-6
        
        if perturbation_type == 'gaussian':
            noise = np.random.normal(0, strength, obs.shape)
            obs = obs + noise * obs_std
            
        elif perturbation_type == 'uniform':
            noise = np.random.uniform(-strength, strength, obs.shape)
            obs = obs + noise * obs_std
            
        elif perturbation_type == 'dropout':
            mask = np.random.random(obs.shape) > strength
            obs = obs * mask
            
        elif perturbation_type == 'scale':
            scales = 1 + np.random.uniform(-strength, strength, obs.shape)
            obs = obs * scales
            
        elif perturbation_type == 'bias':
            obs = obs + strength * obs_std
            
        return obs
    
    def _process_observation(self, obs) -> np.ndarray:
        """Process observation from episode data."""
        if isinstance(obs, dict):
            components = []
            if 'observation' in obs:
                components.append(np.array(obs['observation']))
            if 'achieved_goal' in obs:
                components.append(np.array(obs['achieved_goal']))
            if 'desired_goal' in obs:
                components.append(np.array(obs['desired_goal']))
            return np.concatenate(components)
        return np.array(obs)
    
    def _add_text_overlay(
        self,
        frame: np.ndarray,
        text_lines: List[str],
        position: str = 'top-left'
    ) -> np.ndarray:
        """Add text overlay to frame."""
        if not HAS_CV2:
            return frame
        
        frame = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        line_height = 20
        
        if position == 'top-left':
            x, y = 10, 25
        elif position == 'top-right':
            x, y = frame.shape[1] - 200, 25
        elif position == 'bottom-left':
            x, y = 10, frame.shape[0] - len(text_lines) * line_height
        else:
            x, y = 10, 25
        
        # Add semi-transparent background
        overlay = frame.copy()
        bg_height = len(text_lines) * line_height + 10
        bg_width = 250
        cv2.rectangle(overlay, (x - 5, y - 20), (x + bg_width, y + bg_height - 20), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        for i, line in enumerate(text_lines):
            cv2.putText(frame, line, (x, y + i * line_height), font, font_scale, color, thickness)
        
        return frame
    
    def generate_episode_video(
        self,
        episode_idx: int = 0,
        output_path: str = 'episode.mp4',
        perturbation_type: Optional[str] = None,
        perturbation_strength: float = 0.0,
        show_info: bool = True
    ) -> str:
        """
        Generate video for a single episode.
        
        Args:
            episode_idx: Index of episode to visualize
            output_path: Path to save video
            perturbation_type: Type of perturbation to apply
            perturbation_strength: Strength of perturbation
            show_info: Whether to show info overlay
            
        Returns:
            output_path: Path to saved video
        """
        if not HAS_CV2 and not HAS_IMAGEIO:
            raise ImportError("Either cv2 or imageio required for video generation")
        
        # Get episode data
        episodes = list(self.dataset.iterate_episodes())
        if episode_idx >= len(episodes):
            raise ValueError(f"Episode {episode_idx} not found. Dataset has {len(episodes)} episodes.")
        
        episode = episodes[episode_idx]
        
        # Process observations
        if isinstance(episode.observations, dict):
            obs_key = 'observation' if 'observation' in episode.observations else list(episode.observations.keys())[0]
            observations = episode.observations[obs_key]
        else:
            observations = episode.observations
        
        actions = episode.actions
        rewards = episode.rewards
        
        # Compute observation statistics for perturbation
        obs_std = np.std(observations, axis=0) + 1e-6
        
        # Reset environment
        self.env.reset()
        
        frames = []
        total_reward = 0
        
        print(f"Generating video for episode {episode_idx}...")
        
        for step in tqdm(range(len(actions)), desc="Rendering"):
            # Get observation (potentially perturbed)
            obs = observations[step]
            if perturbation_type and perturbation_strength > 0:
                perturbed_obs = self._apply_perturbation(
                    obs, perturbation_type, perturbation_strength, obs_std
                )
            else:
                perturbed_obs = obs
            
            # Get action
            action = actions[step]
            
            # Step environment
            _, reward, _, _, _ = self.env.step(action)
            total_reward += reward
            
            # Render frame
            frame = self.env.render()
            
            # Resize if needed
            if frame.shape[0] != self.render_height or frame.shape[1] != self.render_width:
                if HAS_CV2:
                    frame = cv2.resize(frame, (self.render_width, self.render_height))
            
            # Add info overlay
            if show_info and HAS_CV2:
                pert_str = f"{perturbation_type} ({perturbation_strength:.2f})" if perturbation_type else "None"
                info_lines = [
                    f"Episode: {episode_idx}",
                    f"Step: {step}/{len(actions)}",
                    f"Reward: {rewards[step]:.3f}",
                    f"Total: {total_reward:.3f}",
                    f"Perturbation: {pert_str}"
                ]
                frame = self._add_text_overlay(frame, info_lines)
            
            frames.append(frame)
        
        # Save video
        self._save_video(frames, output_path)
        
        print(f"Saved video to {output_path}")
        return output_path
    
    def generate_comparison_video(
        self,
        episode_idx: int = 0,
        output_path: str = 'comparison.mp4',
        perturbation_types: List[str] = None,
        perturbation_strength: float = 0.2,
        show_info: bool = True
    ) -> str:
        """
        Generate side-by-side comparison video showing clean vs perturbed.
        
        Args:
            episode_idx: Index of episode to visualize
            output_path: Path to save video
            perturbation_types: List of perturbation types to compare
            perturbation_strength: Strength of perturbation
            show_info: Whether to show info overlay
            
        Returns:
            output_path: Path to saved video
        """
        if not HAS_CV2:
            raise ImportError("cv2 required for comparison video")
        
        if perturbation_types is None:
            perturbation_types = ['none', 'gaussian', 'dropout']
        
        # Get episode data
        episodes = list(self.dataset.iterate_episodes())
        episode = episodes[episode_idx]
        
        if isinstance(episode.observations, dict):
            obs_key = 'observation' if 'observation' in episode.observations else list(episode.observations.keys())[0]
            observations = episode.observations[obs_key]
        else:
            observations = episode.observations
        
        actions = episode.actions
        rewards = episode.rewards
        obs_std = np.std(observations, axis=0) + 1e-6
        
        # Calculate grid layout
        n_views = len(perturbation_types)
        if n_views <= 2:
            grid_rows, grid_cols = 1, n_views
        elif n_views <= 4:
            grid_rows, grid_cols = 2, 2
        else:
            grid_rows = 2
            grid_cols = (n_views + 1) // 2
        
        cell_width = self.render_width // grid_cols
        cell_height = self.render_height // grid_rows
        
        frames = []
        
        print(f"Generating comparison video for episode {episode_idx}...")
        
        for step in tqdm(range(len(actions)), desc="Rendering"):
            # Create combined frame
            combined = np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)
            
            for idx, pert_type in enumerate(perturbation_types):
                # Reset and replay to current step
                self.env.reset()
                for s in range(step + 1):
                    self.env.step(actions[s])
                
                # Render
                frame = self.env.render()
                frame = cv2.resize(frame, (cell_width, cell_height))
                
                # Add label
                if show_info:
                    strength_str = f" ({perturbation_strength:.1f})" if pert_type != 'none' else ""
                    label = f"{pert_type}{strength_str}"
                    cv2.putText(frame, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 1)
                    cv2.putText(frame, f"Step: {step}", (5, 40), cv2.FONT_HERSHEY_SIMPLEX,
                               0.4, (255, 255, 255), 1)
                
                # Place in grid
                row = idx // grid_cols
                col = idx % grid_cols
                y_start = row * cell_height
                x_start = col * cell_width
                combined[y_start:y_start+cell_height, x_start:x_start+cell_width] = frame
            
            frames.append(combined)
        
        self._save_video(frames, output_path)
        print(f"Saved comparison video to {output_path}")
        return output_path
    
    def generate_multiple_episodes(
        self,
        output_dir: str = 'videos',
        n_episodes: int = 5,
        perturbation_type: Optional[str] = None,
        perturbation_strength: float = 0.0
    ) -> List[str]:
        """
        Generate videos for multiple episodes.
        
        Args:
            output_dir: Directory to save videos
            n_episodes: Number of episodes to visualize
            perturbation_type: Type of perturbation
            perturbation_strength: Strength of perturbation
            
        Returns:
            output_paths: List of paths to saved videos
        """
        os.makedirs(output_dir, exist_ok=True)
        
        n_available = self.dataset.total_episodes
        n_episodes = min(n_episodes, n_available)
        
        output_paths = []
        
        for i in range(n_episodes):
            pert_str = f"_{perturbation_type}_{perturbation_strength}" if perturbation_type else "_clean"
            output_path = os.path.join(output_dir, f"episode_{i}{pert_str}.mp4")
            
            self.generate_episode_video(
                episode_idx=i,
                output_path=output_path,
                perturbation_type=perturbation_type,
                perturbation_strength=perturbation_strength
            )
            output_paths.append(output_path)
        
        return output_paths
    
    def generate_perturbation_sweep(
        self,
        episode_idx: int = 0,
        output_dir: str = 'videos/sweep',
        perturbation_type: str = 'gaussian',
        strengths: List[float] = None
    ) -> List[str]:
        """
        Generate videos with different perturbation strengths.
        
        Args:
            episode_idx: Episode to visualize
            output_dir: Directory to save videos
            perturbation_type: Type of perturbation
            strengths: List of perturbation strengths
            
        Returns:
            output_paths: List of paths to saved videos
        """
        if strengths is None:
            strengths = [0.0, 0.1, 0.2, 0.3, 0.5]
        
        os.makedirs(output_dir, exist_ok=True)
        output_paths = []
        
        for strength in strengths:
            pert = None if strength == 0 else perturbation_type
            output_path = os.path.join(output_dir, f"episode_{episode_idx}_{perturbation_type}_{strength:.2f}.mp4")
            
            self.generate_episode_video(
                episode_idx=episode_idx,
                output_path=output_path,
                perturbation_type=pert,
                perturbation_strength=strength
            )
            output_paths.append(output_path)
        
        return output_paths
    
    def _save_video(self, frames: List[np.ndarray], output_path: str):
        """Save frames as video file."""
        if len(frames) == 0:
            print("Warning: No frames to save")
            return
        
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        if HAS_CV2:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
        elif HAS_IMAGEIO:
            imageio.mimsave(output_path, frames, fps=self.fps)
        
        else:
            raise ImportError("Either cv2 or imageio required for video saving")
    
    def close(self):
        """Close environment."""
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description='Generate videos of door-opening demonstrations')
    
    parser.add_argument('--dataset', type=str, default='D4RL/door/human-v2',
                       choices=['D4RL/door/human-v2', 'D4RL/door/expert-v2', 'D4RL/door/cloned-v2'],
                       help='Minari dataset name')
    
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'multiple', 'comparison', 'sweep'],
                       help='Visualization mode')
    
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode index to visualize')
    parser.add_argument('--n_episodes', type=int, default=5,
                       help='Number of episodes for multiple mode')
    
    parser.add_argument('--perturbation', type=str, default=None,
                       choices=['none', 'gaussian', 'uniform', 'dropout', 'scale', 'bias'],
                       help='Perturbation type')
    parser.add_argument('--strength', type=float, default=0.1,
                       help='Perturbation strength')
    
    parser.add_argument('--output_dir', type=str, default='videos',
                       help='Output directory for videos')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for single video')
    
    parser.add_argument('--width', type=int, default=640,
                       help='Video width')
    parser.add_argument('--height', type=int, default=480,
                       help='Video height')
    parser.add_argument('--fps', type=int, default=30,
                       help='Video FPS')
    
    parser.add_argument('--no_info', action='store_true',
                       help='Disable info overlay')
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = DemonstrationVisualizer(
        dataset_name=args.dataset,
        render_width=args.width,
        render_height=args.height,
        fps=args.fps
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        if args.mode == 'single':
            output = args.output or os.path.join(args.output_dir, f'episode_{args.episode}.mp4')
            viz.generate_episode_video(
                episode_idx=args.episode,
                output_path=output,
                perturbation_type=args.perturbation if args.perturbation != 'none' else None,
                perturbation_strength=args.strength,
                show_info=not args.no_info
            )
            
        elif args.mode == 'multiple':
            viz.generate_multiple_episodes(
                output_dir=args.output_dir,
                n_episodes=args.n_episodes,
                perturbation_type=args.perturbation if args.perturbation != 'none' else None,
                perturbation_strength=args.strength
            )
            
        elif args.mode == 'comparison':
            output = args.output or os.path.join(args.output_dir, f'comparison_episode_{args.episode}.mp4')
            viz.generate_comparison_video(
                episode_idx=args.episode,
                output_path=output,
                perturbation_types=['none', 'gaussian', 'dropout', 'scale'],
                perturbation_strength=args.strength,
                show_info=not args.no_info
            )
            
        elif args.mode == 'sweep':
            viz.generate_perturbation_sweep(
                episode_idx=args.episode,
                output_dir=os.path.join(args.output_dir, 'sweep'),
                perturbation_type=args.perturbation or 'gaussian',
                strengths=[0.0, 0.1, 0.2, 0.3, 0.5]
            )
            
    finally:
        viz.close()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
