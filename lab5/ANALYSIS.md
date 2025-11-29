# Lab5 Behavior Cloning Analysis

## Overview
This appears to be a homework/template for behavior cloning on ManiSkill's PushCube-v1 task. The code has blanks for students to fill in.

## Key Architecture Differences

### 1. **Observation Space: State-Based (Not Visual)**
- Uses `obs_mode="state"` - direct state observations
- No visual backbone (CNN/ResNet/ViT)
- Just an MLP on state features
- **Your approach**: RGB images → CNN/ResNet → MLP

### 2. **Training Strategy: Iteration-Based (Not Epoch-Based)**
- Trains for **1,000,000 iterations** (not epochs)
- Uses `IterationBasedBatchSampler` to resample batches
- Batch size: **1024** (much larger than your 32)
- **Your approach**: Epoch-based with smaller batches

### 3. **Network Architecture**
From `bc_utils.py` requirements:
- At least 2 hidden layers
- 10K-1M parameters
- Simple MLP (no temporal modeling like GRU)
- **Your approach**: CNN backbone + optional GRU + MLP head

### 4. **Loss Function**
- Simple **MSE loss** (blank in code, but standard BC uses MSE)
- No Huber loss, no magnitude weighting
- **Your approach**: Huber loss with magnitude weighting

### 5. **Data Normalization**
- Normalizes observations to mean=0, std=1
- Computes stats from training data
- **Your approach**: ImageNet normalization for images, but no state normalization

### 6. **Learning Rate**
- Fixed LR: **3e-4** (no scheduler mentioned)
- **Your approach**: 1e-3 with ReduceLROnPlateau scheduler

### 7. **Evaluation**
- Evaluates every 1000 iterations
- 100 parallel evaluation episodes
- Saves checkpoints based on best eval metrics
- **Your approach**: Evaluates every epoch

## Key Takeaways for Your Project

### What They're Doing Right:
1. **State normalization** - Could help your proprioceptive state
2. **Larger batch size** - Might help with training stability
3. **Iteration-based training** - More flexible than epoch-based
4. **Simple MSE loss** - Sometimes simpler is better

### What You're Doing Better:
1. **Visual observations** - More realistic, better for sim-to-real
2. **Huber loss** - More robust to outliers
3. **Learning rate scheduling** - Adaptive to training progress
4. **Temporal modeling (GRU)** - Better for sequential tasks
5. **Magnitude-weighted loss** - Emphasizes important actions

## Recommendations

1. **Add state normalization** for your proprioceptive features:
   ```python
   # Normalize gripper position to mean=0, std=1
   state_mean = states.mean(axis=0)
   state_std = states.std(axis=0)
   normalized_states = (states - state_mean) / (state_std + 1e-8)
   ```

2. **Consider larger batch size** if memory allows (64-128 instead of 32)

3. **Try simple MSE loss** as a baseline to compare with Huber loss

4. **Keep your visual approach** - it's more realistic and challenging

5. **Keep GRU** - temporal modeling is valuable for sequential tasks

