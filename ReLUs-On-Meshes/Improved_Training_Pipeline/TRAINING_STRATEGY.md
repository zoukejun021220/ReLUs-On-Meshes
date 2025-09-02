# Conservative Training Strategy for ReLU Mesh Segmentation

## Overview
This training strategy prioritizes stability and gradual convergence through:
1. Smoothness-only warmup to form initial patches
2. Very conservative lambda_contour ramping based on contour loss stalls
3. Temperature-aware learning rate scaling to prevent late-stage NaN

## Stage Progression (300k steps)

### Stage 0: Smoothness + Area + Normal Warmup (0-5k steps, 1.67%)
- **Purpose**: Form initial smooth, balanced, axis-aligned patches without boundary constraints
- **Weights**: 
  - λ_smooth=1.0 (high smoothness)
  - λ_contour=0.001 (no boundary alignment)
  - λ_area=1.0 (equal-sized patches)
  - λ_normal_align=0.5 (strong axis alignment)
  - λ_normal_disp=0.2 (encourage planarity)
- **Learning Rate**: 1e-3 (high LR to quickly establish patches)
- **Features**:
  - Hard pins to anchor the field
  - High smoothness + area balance + normal alignment
  - Area loss uses variance + entropy during warmup (stronger gradients than KL)
  - TV regularization for additional smoothing
  - No temperature increases (β_contour=1.0, β_area=2.0)
  - Strong normal alignment to establish axis-oriented patches early

### Stage A1: Early (5k-18k steps, 1.67-6%)
- **Purpose**: Begin gentle introduction of constraints
- **Weights**: λ_smooth=0.5, λ_contour=0.01, λ_area=0.1
- **Features**:
  - Switch to soft pins
  - Very low contour weight
  - Temperature increases allowed
  - β starts at 0.5 (very low)

### Stage A2: Coarse (18k-45k steps, 6-15%)
- **Purpose**: Gradual constraint increase
- **Weights**: λ_smooth=0.4, λ_contour=0.02, λ_area=0.3
- **Features**:
  - Still strong smoothness
  - Contour weight remains low
  - Moderate area balance

### Stage A3: Shape Formation (45k-120k steps, 15-40%)
- **Purpose**: Main patch formation phase
- **Weights**: λ_smooth=0.3, λ_contour=0.05, λ_area=0.8
- **Features**:
  - Strong area balance
  - Low base contour weight
  - Stall detector can start increasing λ_contour

### Stage B: Refinement (120k-240k steps, 40-80%)
- **Purpose**: Boundary refinement
- **Weights**: λ_smooth=0.2, λ_contour=0.1, λ_area=0.5
- **Features**:
  - Balanced weights
  - Cosine LR decay
  - Adaptive λ_contour based on stalls

### Stage C: Final (240k-300k steps, 80-100%)
- **Purpose**: Final sharpening
- **Weights**: λ_smooth=0.1, λ_contour=0.2, λ_area=0.3
- **Features**:
  - Hard pins return
  - Low smoothness
  - Maximum temperature allowed

## Conservative Lambda_Contour Ramping

### Stall Detection Criteria
- **Monitors**: Contour loss improvement only (not total loss)
- **Window**: 500 steps for moving average
- **Patience**: 1500 steps without improvement
- **Activation Requirements**:
  1. After 20k steps minimum
  2. Area deviation < 0.05 (patches formed)
  3. Contour loss > 0.01 (room for improvement)
  4. Contour loss stalled for 1500 steps

### Ramping Strategy
- **Increment**: Only 2% per stall (multiplier = 1.02)
- **Frequency**: Can adjust every 750 steps
- **Maximum**: λ_contour capped at 5.0
- **No automatic growth**: Only increases on stalls

## Temperature Control

### Beta Progression
- **Start**: βc=0.5, βa=0.5 (very low)
- **Step size**: βc+=0.3, βa+=0.1 per update
- **Maximum**: βc=8.0, βa=4.0
- **Requirements**: 
  - Area deviation < 0.05
  - Boundary fraction > 2%
  - Contour improvement > 0.5%

### Temperature-Aware LR Scaling
```python
temp_scale = min(1.0, 2.0/βc) * min(1.0, 1.5/βa)
scaled_lr = max(1e-5, base_lr * temp_scale)
```
- At βc=8, βa=4: LR is scaled to ~1/8 of base
- Prevents instability at high temperatures

## NaN Prevention

### Safe Normalization
- Custom `safe_normalize` with clamped denominator
- eps=1e-6 (not 1e-12) for all normalizations
- Gradient projection into triangle plane

### Degenerate Triangle Handling
- Mask triangles with Gram det ≤ 1e-10
- Zero gradients for degenerate triangles
- Down-weight edges adjacent to degenerate triangles

### Defensive Gradient Handling
- `torch.nan_to_num_` on gradients
- Clip gradients to [-1, 1] after NaN
- Halve LR temporarily after NaN events

## Normal Axis Alignment (New)

### Purpose
Encourage patches to align with coordinate axes for more regular, axis-oriented segmentation.

### Components
1. **Axis Alignment Loss**: Penalizes deviation of each patch's mean normal from its target axis
2. **Normal Dispersion Loss**: Encourages planarity within each patch (low normal variance)

### Schedule
- **Steps 0-5k (Warmup)**: λ_align=0.5, λ_disp=0.2 (strong early alignment)
- **Steps 5k-30k**: Disabled (let boundaries form naturally)
- **Steps 30k-60k**: λ_align=0.05, λ_disp=0.02 (gentle reintroduction)
- **Steps 60k+**: λ_align=0.2, λ_disp=0.1 (moderate enforcement)

### 5-Patch Option
Use `--use-5-patch-prior` to target 5 patches instead of 6:
- Replaces uniform area prior with [1/5, 1/5, 1/5, 1/5, 1/5, ε]
- Allows one channel (typically -Z) to remain nearly empty

## Improved Pin/Channel Alignment

### Pin Selection Method
- **Normal-based selection**: Picks vertices whose normals best align with coordinate axes
- **Order preservation**: Maintains [+X, -X, +Y, -Y, +Z, -Z] mapping to channels 0-5
- **No reordering**: Avoids torch.unique() which scrambles the channel mapping

### Pin Enforcement Strategy
- **Stage 0 (0-5k)**: Hard projection after each step
- **Stage A1 (5k-18k)**: λ_pin=1.0 (very high soft penalty)
- **Stage A2 (18k-45k)**: λ_pin=0.5 (moderate penalty)
- **Stage A3+ (45k+)**: λ_pin=0.01-0.1 (low penalty)

### Channel Mapping Verification
The training script now prints pin mappings at step 0 to verify correct channel assignment.

## Key Insights

1. **Patches First**: High smoothness + area balance with hard pins establishes initial segmentation
2. **Contour Later**: Only align boundaries after patches have formed
3. **Conservative Growth**: 2% increments prevent sudden instabilities
4. **Temperature Scaling**: Reduced LR at high β prevents gradient explosions
5. **Stall-Based**: Only increase λ_contour when actually needed
6. **Axis Alignment**: Normal losses guide patches toward coordinate axes after initial formation
7. **Pin Integrity**: Strong pin enforcement early prevents channel drift/mixing

## Recommended Command
```bash
# For 6 axis-aligned patches:
python train_improved.py \
    --mesh "path/to/mesh.vtk" \
    --n-steps 300000 \
    --use-soft-pairs \
    --checkpoint-freq 5000 \
    --log-freq 500

# For 5 axis-aligned patches:
python train_improved.py \
    --mesh "path/to/mesh.vtk" \
    --n-steps 300000 \
    --use-soft-pairs \
    --use-5-patch-prior \
    --checkpoint-freq 5000 \
    --log-freq 500
```

The `--use-soft-pairs` flag is recommended for better stability at triple junctions.
The `--use-5-patch-prior` flag allows one channel to remain nearly empty for 5-patch segmentation.