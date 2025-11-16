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
- `policy.py`: PyTorch CNN-MLP policy architecture
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

```bash
source venv/bin/activate  # macOS/Linux
python train.py
```

This will train a policy using the collected demonstrations and save the model weights.

### 3. Evaluate a Policy

**Important:** Make sure your virtual environment is activated first!

```bash
source venv/bin/activate  # macOS/Linux
python evaluate.py --policy models/policy_pixel_aug.pth --corruption distractor
```

This will evaluate the trained policy with visual corruption and report the task success rate.

## Visual Regimes

The project supports training under different visual regimes:
- **Clean**: Standard normalization only
- **Pixel Augmentation**: Normalization + RandomResizedCrop, ColorJitter, GaussianBlur
- **Domain Randomization**: (To be implemented - modify data_collection.py)
- **Distractors**: (To be implemented - modify data_collection.py)

## Next Steps

To implement Domain Randomization and Distractor Objects datasets:
1. Modify `data_collection.py` to use `p.changeVisualShape` for domain randomization
2. Use `p.loadURDF` to add distractor objects during data generation
3. Create separate data collection scripts or add flags to control the visual regime

