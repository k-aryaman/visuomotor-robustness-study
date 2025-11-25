# Visuomotor Policy Robustness Study

This project implements a Behavior Cloning (BC) study comparing the generalization ability of policies trained under different visual regimes on the panda-gym PandaPickAndPlace-v3 environment.

## Installation

### Step 1: Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### Step 2: Install Dependencies

**For macOS (with Python 3.13+):**

If you encounter compilation errors when installing pybullet, set the following environment variables:

```bash
export CXXFLAGS="-I$(xcrun --show-sdk-path)/usr/include/c++/v1"
export CPPFLAGS="-I$(xcrun --show-sdk-path)/usr/include/c++/v1"
pip install -r requirements.txt
```

**For other systems or Python versions:**

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install panda-gym pybullet torch torchvision numpy Pillow
```

**Note:** Always activate your virtual environment before running any scripts:
```bash
source venv/bin/activate  # macOS/Linux
```

## Project Structure

- `data_collection.py`: Expert policy and demonstration collection script
- `policy.py`: PyTorch policy architecture (supports CNN, ResNet-18, or ViT-B/16 backbones)
- `data.py`: PyTorch Dataset class with data transforms
- `train.py`: Training script for behavior cloning
- `evaluate.py`: Evaluation script with visual corruption testing

## Usage

### 1. Collect Expert Demonstrations

**Important:** Make sure your virtual environment is activated first!

```bash
source venv/bin/activate  # macOS/Linux
python data_collection.py
```

This will generate `demonstrations.pkl` containing expert trajectories.

### 2. Train a Policy

**Important:** Make sure your virtual environment is activated first!

The training script supports multiple backbone architectures and visual regimes:

**Basic usage (default: ResNet-18 with pixel augmentation):**
```bash
source venv/bin/activate  # macOS/Linux
python train.py
```

**Train with Vision Transformer (ViT):**
```bash
python train.py --backbone vit --regime pixel_aug
```

**Train with simple CNN:**
```bash
python train.py --backbone cnn --regime clean
```

**Full training options:**
```bash
python train.py \
    --backbone resnet \      # Options: resnet, vit, cnn
    --regime pixel_aug \     # Options: clean, pixel_aug
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-3 \
    --data demonstrations.pkl \
    --device cuda
```

This will train a policy using the collected demonstrations and save the model weights. Model files are named based on the regime and backbone (e.g., `policy_pixel_aug_vit.pth`).

### 3. Evaluate a Policy

**Important:** Make sure your virtual environment is activated first!

**Evaluate without corruption (default):**
```bash
source venv/bin/activate  # macOS/Linux
python evaluate.py --policy models/policy_pixel_aug_resnet.pth
```

**Evaluate with visual corruption:**
```bash
python evaluate.py --policy models/policy_pixel_aug_resnet.pth --corruption distractor
```

**Full evaluation options:**
```bash
python evaluate.py \
    --policy models/policy_pixel_aug_resnet.pth \
    --corruption none \        # Options: none, distractor, occlusion
    --backbone resnet \         # Optional: resnet, vit, cnn (auto-detected if not specified)
    --episodes 100 \
    --device cuda
```

This will evaluate the trained policy and report the task success rate. By default, evaluation runs without corruption. The backbone type is automatically detected from the model filename (e.g., `_vit`, `_resnet`, `_cnn`) or from checkpoint metadata, so you typically don't need to specify `--backbone` unless you want to override the auto-detection.

## Backbone Architectures

The policy supports three different backbone architectures for visual feature extraction:

- **ResNet-18** (default): Pre-trained ResNet-18 from ImageNet, outputs 512-dimensional features. Good balance of performance and efficiency.
- **Vision Transformer (ViT)**: Pre-trained ViT-B/16 from ImageNet, outputs 768-dimensional features. May require more data but can capture long-range dependencies.
- **Simple CNN**: 4-layer convolutional network, outputs 64-dimensional features. Lightweight baseline for comparison.

All backbones are followed by the same MLP head that maps features to 4D continuous actions (x, y, z, gripper).

## Visual Regimes

The project supports training under different visual regimes:
- **Clean**: Standard normalization only
- **Pixel Augmentation**: Normalization + RandomResizedCrop, ColorJitter, GaussianBlur
- **Domain Randomization**: (To be implemented - modify data_collection.py)
- **Distractors**: (To be implemented - modify data_collection.py)

## Current Baseline (PandaPickAndPlace)

- **Data**: `demonstrations_rgb.pkl` with 80 scripted-expert episodes (about 2,650 frame/action pairs). Each sample is an 84x84 RGB image plus a 4D action `[dx, dy, dz, gripper]`.
- **Model**: `VisuomotorBCPolicy` with ResNet-18 backbone (ImageNet init) and MLP head (Tanh output), trained for 50 epochs on the clean transform, batch size 64, device=cpu. Saved at `models/policy_clean_resnet_baseline_50ep.pth`.
- **Evaluation (clean)**: `evaluate.py --policy models/policy_clean_resnet_baseline_50ep.pth --corruption none --episodes 10 --max_steps 200 --device cpu` produced `eval_clean_50ep.txt` with 20% success (2/10) and average reward -162.5. Increase episodes and use GPU for a steadier estimate.

## Next Steps

To implement Domain Randomization and Distractor Objects datasets:
1. Modify `data_collection.py` to use `p.changeVisualShape` for domain randomization
2. Use `p.loadURDF` to add distractor objects during data generation
3. Create separate data collection scripts or add flags to control the visual regime

