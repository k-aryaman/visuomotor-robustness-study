# Model Architecture Walkthrough

## Overview
The codebase implements **Behavioral Cloning (BC)** for visuomotor control, training a neural network to imitate expert demonstrations. There are two main architectures:

1. **VisuomotorBCPolicy** (`policy.py`): Standard BC with single-action prediction
2. **ACTPolicy** (`policy_act.py`): Action Chunking with Transformers for sequence prediction

Both support:
- Multiple visual backbones (CNN, ResNet-18, ViT)
- Optional proprioceptive state (gripper position)
- Optional GRU for temporal modeling (BC only)
- Spherical coordinate representation for actions

---

## Architecture 1: VisuomotorBCPolicy (Standard BC)

### Input Pipeline

**Data Format:**
- **Image**: RGB image (84×84×3) → normalized to [0, 1] and ImageNet-normalized
- **State** (optional): 3D proprioceptive state `[gripper_x, gripper_y, gripper_z]`
- **Action**: 3D action `[dx, dy, dz]` (end-effector movement)

**Data Flow:**
```
Trajectory → (image, state, action) pairs → Dataset → DataLoader → Batches
```

### Forward Pass (Step-by-Step)

#### Step 1: Visual Feature Extraction (Backbone)

**Option A: Simple CNN**
```
Input: (batch, 3, 84, 84)
  ↓
Conv2d(3→32, kernel=8, stride=4) + ReLU
  → (batch, 32, 21, 21)
  ↓
Conv2d(32→64, kernel=4, stride=2) + ReLU
  → (batch, 64, 11, 11)
  ↓
Conv2d(64→64, kernel=3, stride=1) + ReLU
  → (batch, 64, 11, 11)
  ↓
Conv2d(64→64, kernel=3, stride=1) + ReLU
  → (batch, 64, 11, 11)
  ↓
AdaptiveAvgPool2d(1, 1)
  → (batch, 64, 1, 1)
  ↓
Flatten
  → (batch, 64)  ← Visual features
```

**Option B: ResNet-18 (Pretrained)**
```
Input: (batch, 3, 84, 84)
  ↓
ResNet-18 backbone (ImageNet pretrained)
  → Global Average Pooling
  → (batch, 512)  ← Visual features
```

**Option C: Vision Transformer (ViT-B/16)**
```
Input: (batch, 3, 84, 84)
  ↓
Patch Projection (16×16 patches)
  → (batch, num_patches, embed_dim)
  ↓
Add [CLS] token + Positional Embeddings
  → (batch, num_patches+1, 768)
  ↓
Transformer Encoder (12 layers)
  → (batch, num_patches+1, 768)
  ↓
Extract [CLS] token (first token)
  → (batch, 768)  ← Visual features
```

#### Step 2: Concatenate Proprioceptive State (if available)
```
Visual features: (batch, feature_dim)
State: (batch, 3)  [gripper_x, gripper_y, gripper_z]
  ↓
Concatenate
  → (batch, feature_dim + 3)
```

#### Step 3: Optional GRU for Temporal Modeling
```
If use_gru=True:
  Input: (batch, feature_dim + 3)
    ↓
  Unsqueeze to sequence: (batch, 1, feature_dim + 3)
    ↓
  GRU (1 layer, hidden_dim=128)
    → (batch, 1, 128)
    ↓
  Squeeze: (batch, 128)
    ↓
  Use last hidden state
    → (batch, 128)  ← Temporal features
Else:
  → (batch, feature_dim + 3)  ← Direct features
```

#### Step 4: MLP Head (Action Prediction)
```
Input: (batch, head_input_dim)
  ↓
Linear(head_input_dim → 512) + ReLU
  → (batch, 512)
  ↓
Dropout(0.3)
  ↓
Linear(512 → 512) + ReLU
  → (batch, 512)
  ↓
Dropout(0.3)
  ↓
Linear(512 → 3) + Tanh
  → (batch, 3)  ← Actions in [-1, 1] range
```

### Training Process

#### Loss Function: Huber Loss (Smooth L1)
- Less sensitive to outliers than MSE
- Helps with action magnitude learning

#### Coordinate System Options

**Cartesian (default):**
- Model outputs: `[dx, dy, dz]` directly in [-1, 1]
- Loss: Compare predicted vs. expert actions directly

**Spherical (`--use_spherical`):**
- **Expert actions**: Converted from Cartesian to spherical
  ```
  (dx, dy, dz) → (magnitude, theta, phi)
  - magnitude: sqrt(dx² + dy² + dz²) [clamped to min=1e-4]
  - theta: elevation from z-axis [0, π]
  - phi: azimuth in xy-plane [-π, π]
  ```

- **Model outputs**: Predicted in spherical space
  ```
  Model raw output: [tanh_out₁, tanh_out₂, tanh_out₃]  (all in [-1, 1])
    ↓
  Scale to spherical ranges:
    magnitude = sigmoid(tanh_out₁)  → [0, 1]
    theta = (tanh_out₂ + 1) * 0.5 * π  → [0, π]
    phi = tanh_out₃ * π  → [-π, π]
  ```

- **Loss**: Computed in spherical space
  - Magnitude-weighted loss: `weight = 1.0 + 2.0 * expert_magnitude`
  - Encourages matching large-magnitude actions more

#### Magnitude-Weighted Loss
```python
per_sample_loss = HuberLoss(predicted, expert)  # (batch, 3)
expert_magnitude = ||expert||  # (batch, 1)
weight = 1.0 + 2.0 * expert_magnitude  # Emphasize large actions
loss = (per_sample_loss * weight).mean()
```

#### Learning Rate Scheduling
- **ReduceLROnPlateau**: Reduces LR when validation loss plateaus
- Patience: 2 epochs
- Threshold: 1% relative improvement
- Factor: 0.5 (halve LR)

---

## Architecture 2: ACTPolicy (Action Chunking with Transformers)

### Key Difference
- **BC**: Predicts 1 action per timestep
- **ACT**: Predicts a **chunk** of `chunk_size` actions from a sequence of observations

### Forward Pass

#### Step 1: Visual Encoding (Same as BC)
```
For each timestep in sequence:
  Image → Visual Encoder (CNN/ResNet) → (batch, seq_len, visual_feature_dim)
```

#### Step 2: Project to Transformer Dimension
```
Visual features: (batch, seq_len, visual_feature_dim)
  ↓
Linear(visual_feature_dim → d_model=512)
  → (batch, seq_len, 512)

State features: (batch, seq_len, state_dim)
  ↓
Linear(state_dim → d_model=512)
  → (batch, seq_len, 512)
  ↓
Add: visual_features + state_features
  → (batch, seq_len, 512)
```

#### Step 3: Positional Encoding
```
Add sinusoidal positional encodings
  → (batch, seq_len, 512)
```

#### Step 4: Transformer Encoder
```
Input: (batch, seq_len, 512)  [observation sequence]
  ↓
Transformer Encoder (4 layers, 8 heads)
  → (batch, seq_len, 512)  [encoded observations]
```

#### Step 5: Action Queries (Learnable Embeddings)
```
Action queries: (chunk_size, 512)  [learnable parameters]
  Expand to batch: (batch, chunk_size, 512)
```

#### Step 6: Transformer Decoder
```
Queries: (batch, chunk_size, 512)  [action queries]
Memory: (batch, seq_len, 512)  [encoded observations]
  ↓
Transformer Decoder (4 layers, 8 heads)
  Cross-attention: queries attend to observations
  → (batch, chunk_size, 512)  [decoded action features]
```

#### Step 7: Action Head
```
Input: (batch, chunk_size, 512)
  ↓
Linear(512 → 2048) + ReLU + Dropout
  → (batch, chunk_size, 2048)
  ↓
Linear(2048 → 1024) + ReLU + Dropout
  → (batch, chunk_size, 1024)
  ↓
Linear(1024 → 3) + Tanh
  → (batch, chunk_size, 3)  ← Chunk of actions
```

### Training Process

**Data Format:**
- **Input sequence**: `seq_len` consecutive observations
- **Output chunk**: `chunk_size` consecutive actions

**Loss:**
- Compare predicted chunk `(batch, chunk_size, 3)` with expert chunk
- Same coordinate system options (Cartesian or spherical)
- Same magnitude-weighted loss

---

## Evaluation Process

### During Evaluation

1. **Load Policy**: Restore weights from checkpoint
2. **For each episode**:
   - Reset environment
   - For each step:
     - Get observation (image + state)
     - Forward pass through policy → action
     - **If spherical**: Convert back to Cartesian
       ```
       (magnitude, theta, phi) → (dx, dy, dz)
       ```
     - Step environment with action
     - Check success condition

### Spherical Coordinate Conversion (Evaluation)

```python
# Model output (spherical, scaled):
magnitude = sigmoid(raw[0])  # [0, 1]
theta = (raw[1] + 1) * 0.5 * π  # [0, π]
phi = raw[2] * π  # [-π, π]

# Convert to Cartesian:
dx = magnitude * sin(theta) * cos(phi)
dy = magnitude * sin(theta) * sin(phi)
dz = magnitude * cos(theta)
```

---

## Key Design Decisions

### 1. **Spherical Coordinates**
- **Why**: Separates magnitude and direction learning
- **Benefit**: Model can learn "how much to move" and "where to move" separately
- **Trade-off**: More complex, potential instability at zero magnitude

### 2. **Magnitude-Weighted Loss**
- **Why**: Expert actions have varying magnitudes
- **Benefit**: Emphasizes matching large movements (critical for task success)
- **Weight**: `1.0 + 2.0 * expert_magnitude`

### 3. **Huber Loss (Smooth L1)**
- **Why**: Less sensitive to outliers than MSE
- **Benefit**: Better for action regression, encourages larger actions

### 4. **Dropout Strategy**
- **CNN backbone**: No dropout (let it learn features freely)
- **MLP head**: 0.3 dropout (regularize decision-making)

### 5. **GRU for Temporal Modeling**
- **Why**: Capture temporal dependencies in observations
- **Benefit**: Can learn to smooth actions, handle sequential patterns
- **Trade-off**: Adds complexity, requires maintaining hidden state

### 6. **ACT vs. Standard BC**
- **ACT**: Predicts action sequences, better for long-horizon tasks
- **BC**: Simpler, faster, good for reactive control
- **Choice**: Depends on task horizon and complexity

---

## Data Flow Summary

### Training (Standard BC)
```
Trajectory → Dataset → DataLoader
  ↓
Batch: (images, states, actions)
  ↓
Policy Forward:
  images → Backbone → features
  features + states → [GRU] → head → actions
  ↓
Loss: HuberLoss(predicted_actions, expert_actions)
  [with optional spherical conversion + magnitude weighting]
  ↓
Backward → Update weights
```

### Training (ACT)
```
Trajectory → ACTSequenceDataset
  ↓
Batch: (images_seq, states_seq, actions_chunk)
  ↓
Policy Forward:
  images_seq → Visual Encoder → visual_features
  visual_features + states_seq → Transformer Encoder
  Action Queries → Transformer Decoder → Action Head
  → actions_chunk
  ↓
Loss: HuberLoss(predicted_chunk, expert_chunk)
  ↓
Backward → Update weights
```

### Evaluation
```
Environment → Observation (image + state)
  ↓
Policy Forward → Action (spherical or Cartesian)
  ↓
[If spherical: Convert to Cartesian]
  ↓
Environment Step → Next observation
  ↓
Repeat until success/failure
```

---

## Model Parameters

### VisuomotorBCPolicy (CNN)
- **CNN backbone**: ~50K parameters
- **MLP head**: ~270K parameters
- **Total**: ~320K parameters

### VisuomotorBCPolicy (ResNet-18)
- **ResNet-18**: ~11M parameters (pretrained, fine-tuned)
- **MLP head**: ~270K parameters
- **Total**: ~11.3M parameters

### ACTPolicy
- **Visual encoder**: ~11M (ResNet) or ~50K (CNN)
- **Transformer**: ~15M parameters
- **Total**: ~26M (ResNet) or ~15M (CNN)

---

## Current Configuration (Push Task)

- **Action space**: 3D `[dx, dy, dz]` (no gripper)
- **State space**: 3D `[gripper_x, gripper_y, gripper_z]` (no gripper width)
- **Image size**: 84×84 RGB
- **Backbone**: CNN (default) or ResNet-18
- **Coordinate system**: Spherical (optional, `--use_spherical`)
- **Loss**: Huber loss with magnitude weighting
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau

