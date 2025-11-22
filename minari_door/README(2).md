# Door Opening Offline RL with Robustness Training

This project provides scripts for offline reinforcement learning on the door-opening task using the Minari dataset, with support for various perturbation strategies to improve robustness.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
  - [Apple Silicon Mac Users (IMPORTANT)](#apple-silicon-mac-users-important)
  - [Standard Installation](#standard-installation)
- [Available Datasets](#available-datasets)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Full Pipeline](#full-pipeline)
  - [Clean Pipeline](#clean-pipeline-no-perturbations)
  - [Perturbed Pipeline](#perturbed-pipeline-with-robustness-training)
- [Visualization](#visualization)
- [Perturbation Types](#perturbation-types)
- [Policy Architectures](#policy-architectures)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

This project enables:
- Loading door-opening demonstrations from the Minari D4RL datasets
- Training behavior cloning policies with optional observation perturbations
- Evaluating policy robustness under various perturbation types
- Generating videos of demonstrations (requires MuJoCo)

The key idea is that training with perturbations (domain randomization or curriculum learning) helps policies generalize better to observation noise and sensor failures during deployment.

---

## Installation

### Apple Silicon Mac Users (IMPORTANT)

**MuJoCo requires a native ARM64 Python build.** If you're on an Apple Silicon Mac (M1/M2/M3), you must ensure your conda environment uses ARM64 Python, NOT x86_64 through Rosetta.

#### Check your current Python architecture:

```bash
python -c "import platform; print(platform.machine())"
```

- If it prints `arm64` → You're good, proceed with standard installation
- If it prints `x86_64` → You need to create a native ARM64 environment

#### Creating a native ARM64 conda environment:

```bash
# Deactivate any current environment
conda deactivate

# Create new environment with ARM64 architecture
CONDA_SUBDIR=osx-arm64 conda create -n door_rl python=3.9

# Activate the environment
conda activate door_rl

# Verify it's ARM64
python -c "import platform; print(platform.machine())"
# Should print: arm64

# Set the subdir permanently for this environment
conda config --env --set subdir osx-arm64
```

#### If your entire conda installation is x86_64:

You may need to install a native ARM64 Miniconda:

```bash
# Download ARM64 Miniconda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install (choose a different location if you have existing conda)
bash Miniconda3-latest-MacOSX-arm64.sh

# Restart terminal and create environment
conda create -n door_rl python=3.9
conda activate door_rl
```

### Standard Installation

Once you have the correct Python architecture:

```bash
# Activate your environment
conda activate door_rl

# Install core dependencies (IMPORTANT: numpy<2 for compatibility)
pip install "numpy<2"

# Install Minari and HuggingFace support for downloading datasets
pip install minari "minari[hf]"

# Install Gymnasium and MuJoCo (required for visualization/evaluation)
pip install gymnasium "gymnasium[mujoco]"
pip install gymnasium-robotics

# Install PyTorch (for training)
pip install torch torchvision

# Install visualization dependencies
pip install opencv-python imageio[ffmpeg]

# Install other utilities
pip install tqdm scipy matplotlib seaborn
```

#### Complete installation commands (copy-paste ready):

```bash
# All in one block
pip install "numpy<2"
pip install minari "minari[hf]"
pip install gymnasium "gymnasium[mujoco]"
pip install gymnasium-robotics
pip install torch torchvision
pip install opencv-python "imageio[ffmpeg]"
pip install tqdm scipy matplotlib seaborn
```

#### Verify installation:

```bash
# Test MuJoCo
python -c "import mujoco; print('MuJoCo OK')"

# Test Gymnasium Robotics
python -c "import gymnasium_robotics; print('Gymnasium Robotics OK')"

# Test Minari
python -c "import minari; print('Minari OK')"

# Test all together
python -c "import mujoco; import gymnasium_robotics; import minari; print('All imports OK!')"
```

---

## Available Datasets

The following Minari D4RL door-opening datasets are available:

| Dataset | Episodes | Steps | Size | Description |
|---------|----------|-------|------|-------------|
| `D4RL/door/human-v2` | 25 | 6,729 | 7.1 MB | Human demonstrations (small, good for testing) |
| `D4RL/door/expert-v2` | 5,000 | 1,000,000 | 1.1 GB | Expert policy rollouts (best quality) |
| `D4RL/door/cloned-v2` | 4,356 | 1,000,000 | 1.1 GB | Behavior cloned policy rollouts |

**Recommendation:** Start with `D4RL/door/human-v2` for quick testing, then use `D4RL/door/expert-v2` for final training.

### Download a dataset:

```bash
# Using Minari CLI
minari download D4RL/door/human-v2

# Or via Python
python -c "import minari; minari.download_dataset('D4RL/door/human-v2')"
```

### List available datasets:

```bash
# List remote datasets
minari list remote

# List locally downloaded datasets
minari list local
```

---

## Project Structure

```
door_offline_rl/
├── data_loader.py      # Data loading and perturbation utilities
├── policy.py           # Neural network architectures
├── train.py            # Training script
├── evaluate.py         # Evaluation script with robustness testing
├── visualize.py        # Video generation (requires MuJoCo)
├── requirements.txt    # Dependencies
├── README.md           # This file
└── models/             # Saved models (created during training)
    ├── clean/
    ├── robust_gaussian/
    └── curriculum_gaussian/
```

---

## Quick Start

```bash
# 1. Download data
python data_loader.py --dataset D4RL/door/human-v2 --output door_data.pkl

# 2. Train a clean baseline
python train.py --mode clean --dataset D4RL/door/human-v2 --epochs 50

# 3. Evaluate
python evaluate.py --policy models/clean/best_policy.pth --episodes 50
```

---

## Full Pipeline

### Clean Pipeline (No Perturbations)

This trains a standard behavior cloning policy without any data augmentation.

#### Step 1: Download and preprocess data

```bash
python data_loader.py \
    --dataset D4RL/door/expert-v2 \
    --output data/door_expert.pkl
```

#### Step 2: Train clean policy

```bash
python train.py \
    --mode clean \
    --dataset D4RL/door/expert-v2 \
    --policy mlp \
    --hidden_dims 256 256 256 \
    --epochs 100 \
    --batch_size 256 \
    --lr 3e-4 \
    --device cuda \
    --save_dir models/clean
```

For CPU training (if no GPU):
```bash
python train.py \
    --mode clean \
    --dataset D4RL/door/expert-v2 \
    --epochs 100 \
    --device cpu \
    --save_dir models/clean
```

#### Step 3: Evaluate clean policy

```bash
# Evaluate without perturbations
python evaluate.py \
    --policy models/clean/best_policy.pth \
    --perturbation none \
    --episodes 100 \
    --output results/clean_eval.json

# Test robustness (will likely perform poorly)
python evaluate.py \
    --policy models/clean/best_policy.pth \
    --perturbation gaussian \
    --strength 0.2 \
    --episodes 100 \
    --output results/clean_robustness.json
```

---

### Perturbed Pipeline (With Robustness Training)

This trains policies that are robust to observation perturbations.

#### Option A: Constant Perturbation (Domain Randomization)

Train with a fixed perturbation strength throughout:

```bash
python train.py \
    --mode robust \
    --dataset D4RL/door/expert-v2 \
    --policy mlp \
    --hidden_dims 256 256 256 \
    --perturbation gaussian \
    --final_strength 0.2 \
    --epochs 100 \
    --batch_size 256 \
    --lr 3e-4 \
    --save_dir models/robust_gaussian
```

You can also try different perturbation types:

```bash
# Dropout perturbation (simulates missing sensors)
python train.py \
    --mode robust \
    --dataset D4RL/door/expert-v2 \
    --perturbation dropout \
    --final_strength 0.2 \
    --save_dir models/robust_dropout

# Scale perturbation (simulates calibration errors)
python train.py \
    --mode robust \
    --dataset D4RL/door/expert-v2 \
    --perturbation scale \
    --final_strength 0.2 \
    --save_dir models/robust_scale
```

#### Option B: Curriculum Learning (Recommended)

Gradually increase perturbation strength during training:

```bash
python train.py \
    --mode curriculum \
    --dataset D4RL/door/expert-v2 \
    --policy mlp \
    --hidden_dims 256 256 256 \
    --perturbation gaussian \
    --init_strength 0.0 \
    --final_strength 0.3 \
    --warmup 20 \
    --epochs 100 \
    --batch_size 256 \
    --lr 3e-4 \
    --save_dir models/curriculum_gaussian
```

**Explanation of curriculum parameters:**
- `--init_strength 0.0`: Start with no perturbation
- `--final_strength 0.3`: End with 30% perturbation strength
- `--warmup 20`: Train for 20 epochs before starting to increase perturbation

#### Step 3: Evaluate robust policy

```bash
# Single perturbation evaluation
python evaluate.py \
    --policy models/curriculum_gaussian/best_policy.pth \
    --perturbation gaussian \
    --strength 0.2 \
    --episodes 100

# Full robustness sweep (all perturbation types and strengths)
python evaluate.py \
    --policy models/curriculum_gaussian/best_policy.pth \
    --mode sweep \
    --episodes 50 \
    --output results/curriculum_robustness_sweep.json
```

#### Step 4: Compare clean vs robust policies

```bash
python evaluate.py \
    --policy models/clean/best_policy.pth \
    --mode compare \
    --compare_policies models/robust_gaussian/best_policy.pth models/curriculum_gaussian/best_policy.pth \
    --policy_names clean robust curriculum \
    --perturbation gaussian \
    --episodes 50 \
    --output results/policy_comparison.json
```

---

## Visualization

**Note:** Video generation requires MuJoCo, which needs ARM64 Python on Apple Silicon Macs. See the installation section for details.

#### Generate video of a single episode:

```bash
python visualize.py \
    --mode single \
    --episode 0 \
    --output videos/episode_0.mp4
```

#### Generate video with perturbation:

```bash
python visualize.py \
    --mode single \
    --episode 0 \
    --perturbation gaussian \
    --strength 0.2 \
    --output videos/episode_0_perturbed.mp4
```

#### Generate multiple episode videos:

```bash
# Clean demonstrations
python visualize.py \
    --mode multiple \
    --n_episodes 5 \
    --output_dir videos/clean

# Perturbed demonstrations
python visualize.py \
    --mode multiple \
    --n_episodes 5 \
    --perturbation gaussian \
    --strength 0.2 \
    --output_dir videos/perturbed
```

#### Side-by-side comparison video:

```bash
python visualize.py \
    --mode comparison \
    --episode 0 \
    --strength 0.2 \
    --output videos/comparison.mp4
```

#### Perturbation strength sweep:

```bash
python visualize.py \
    --mode sweep \
    --episode 0 \
    --perturbation gaussian \
    --output_dir videos/sweep
```

### Alternative: Use Google Colab for Visualization

If you can't get MuJoCo working locally, use Google Colab:

```python
# In a Colab notebook
!pip install minari "minari[hf]" gymnasium "gymnasium[mujoco]" gymnasium-robotics opencv-python

# Upload visualize.py and run
!python visualize.py --mode single --episode 0
```

---

## Perturbation Types

| Type | Description | Simulates |
|------|-------------|-----------|
| `gaussian` | Add Gaussian noise to observations | Sensor noise |
| `uniform` | Add uniform noise to observations | Bounded uncertainty |
| `dropout` | Randomly zero out observation dimensions | Missing/failed sensors |
| `scale` | Randomly scale observation dimensions | Calibration errors |
| `bias` | Add constant offset to observations | Sensor drift |
| `adversarial` | Randomly flip signs of observations | Worst-case testing |
| `stuck` | Some sensor values don't update | Sensor failure |
| `delay` | Observations are delayed by N steps | Communication latency |

### Recommended perturbation strengths:

| Strength | Effect |
|----------|--------|
| 0.05 | Mild noise, minimal impact |
| 0.1 | Moderate noise, noticeable impact |
| 0.2 | Strong noise, significant impact |
| 0.3 | Very strong noise, challenging |
| 0.5 | Extreme noise, very challenging |

---

## Policy Architectures

| Type | Description | Best For |
|------|-------------|----------|
| `mlp` | Standard MLP with layer norm | General use (default) |
| `residual` | Residual connections | Deeper networks |
| `gaussian` | Stochastic policy with uncertainty | When you need uncertainty estimates |
| `ensemble` | Multiple policies | Robust uncertainty estimation |
| `transformer` | Transformer architecture | Sequential dependencies |

### Example: Train with different architectures

```bash
# MLP (default)
python train.py --mode curriculum --policy mlp --hidden_dims 256 256 256

# Residual MLP
python train.py --mode curriculum --policy residual --save_dir models/residual

# Ensemble (5 policies)
python train.py --mode curriculum --policy ensemble --save_dir models/ensemble
```

---

## Troubleshooting

### "MuJoCo is not installed" or "_ARRAY_API not found"

This usually means NumPy version incompatibility:

```bash
pip uninstall numpy
pip install "numpy<2"
```

### "x86_64 build of Python on Apple Silicon"

You're running Intel Python through Rosetta. Create a native ARM64 environment:

```bash
# Check current architecture
python -c "import platform; print(platform.machine())"

# If x86_64, create ARM64 environment
CONDA_SUBDIR=osx-arm64 conda create -n door_rl_arm64 python=3.9
conda activate door_rl_arm64
conda config --env --set subdir osx-arm64

# Reinstall all packages
pip install "numpy<2" minari "minari[hf]" gymnasium "gymnasium[mujoco]" gymnasium-robotics torch opencv-python tqdm scipy
```

### "Dataset not found" errors

Install HuggingFace support and download the dataset:

```bash
pip install "minari[hf]"
minari download D4RL/door/human-v2
```

### "No module named 'panda_gym'" or package not found

Make sure you're using conda's pip, not system pip:

```bash
# Check which pip
which pip

# Should show something like: /Users/you/miniconda3/envs/door_rl/bin/pip
# NOT: /usr/bin/pip or /Library/Frameworks/Python.framework/...

# Use conda's pip explicitly if needed
$CONDA_PREFIX/bin/pip install <package>
```

### CUDA out of memory

Reduce batch size:

```bash
python train.py --batch_size 128  # or 64, or 32
```

### Training is slow

- Use GPU: `--device cuda`
- Reduce hidden dims: `--hidden_dims 128 128`
- Use fewer epochs for testing: `--epochs 10`
- Use smaller dataset first: `--dataset D4RL/door/human-v2`

### "huggingface_hub is not installed"

```bash
pip install "minari[hf]"
# or
pip install huggingface_hub
```

---

## Results Analysis

Load and plot results:

```python
import json
import matplotlib.pyplot as plt

# Load comparison results
with open('results/policy_comparison.json', 'r') as f:
    results = json.load(f)

# Plot success rate vs perturbation strength
for policy_name, data in results.items():
    strengths = sorted([float(s) for s in data.keys()])
    success_rates = [data[str(s)]['success_rate'] for s in strengths]
    plt.plot(strengths, success_rates, marker='o', label=policy_name)

plt.xlabel('Perturbation Strength')
plt.ylabel('Success Rate')
plt.title('Policy Robustness Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('robustness_comparison.png', dpi=150)
plt.show()
```

---

## Citation

If you use this code, please cite:

```bibtex
@software{minari,
  author = {Younis, Omar G. and Perez-Vicente, Rodrigo and Balis, John U. and others},
  title = {Minari},
  url = {https://github.com/Farama-Foundation/Minari},
  year = {2024}
}

@software{gymnasium_robotics,
  author = {Rodrigo de Lazcano and Kallinteris Andreas and Jun Jet Tai and others},
  title = {Gymnasium Robotics},
  url = {http://github.com/Farama-Foundation/Gymnasium-Robotics},
  year = {2024}
}
```

---

## License

MIT License
