# Improved ReLU Mesh Segmentation Training Pipeline

This directory contains a completely redesigned training pipeline that addresses all convergence issues identified in the original implementation.

## Key Improvements

### 1. Intrinsic Contour Alignment (Replaces 3D SVD Plane Fitting)
- **Problem**: Original method fits global 3D planes to boundary points, which fights the surface geometry on non-convex shapes
- **Solution**: Work in 2D triangle tangent planes, align boundary tangents locally
- **Benefits**: No SVD instability, respects surface intrinsic geometry, much more stable gradients

### 2. Cotangent Laplacian Smoothness (Replaces Unnormalized Edge Differences)
- **Problem**: Original smoothness term ignores edge lengths, causing anisotropic artifacts
- **Solution**: Use proper cotangent weights from discrete differential geometry
- **Benefits**: Isotropic smoothing, better behavior on irregular meshes

### 3. Progress-Gated Temperature Scheduling (Replaces Time-Based Ramping)
- **Problem**: Linear ramp to β=40 can cause premature hardening before boundaries form
- **Solution**: Increase β only when area distribution is balanced and boundaries exist
- **Benefits**: Maintains gradients throughout training, prevents early saturation

### 4. KL Divergence Area Balance (Replaces L1 Loss)
- **Problem**: L1 distance has vanishing gradients when regions are very small/large
- **Solution**: Use KL divergence to uniform distribution
- **Benefits**: Always provides meaningful gradients, better for imbalanced initializations

### 5. Soft Pin Constraints with Annealing (Replaces Hard Projection)
- **Problem**: Hard projection after each step creates discontinuous optimization
- **Solution**: Soft quadratic penalty, only harden in final 10% of training
- **Benefits**: Smooth optimization landscape, better gradient flow

### 6. Two-Stage Training Schedule
- **Stage A (0-60%)**: Low β, high smoothness, establish coarse segmentation
- **Stage B (60-90%)**: Higher β, strong alignment, refine boundaries  
- **Stage C (90-100%)**: Final hardening with hard pins

## Installation

```bash
# Ensure you have PyTorch installed
pip install torch numpy scipy matplotlib

# Optional but recommended for visualization
pip install pyvista
```

## Usage

### Basic Training

```bash
python train_improved.py --mesh /path/to/mesh.vtk --n-steps 100000
```

### Full Options

```bash
python train_improved.py \
    --mesh /path/to/mesh.vtk \
    --output-dir results/experiment1 \
    --n-steps 100000 \
    --n-channels 6 \
    --device cuda \
    --checkpoint-freq 5000 \
    --log-freq 500 \
    --seed 42
```

### Visualization

```bash
python visualization/visualize_results.py results/experiment1
```

## File Structure

```
Improved_Training_Pipeline/
├── utils/
│   └── mesh_preprocessing.py    # Mesh loading, cotangent weights, preprocessing
├── losses/
│   └── improved_losses.py       # All improved loss functions
├── optimization/
│   └── temperature_control.py   # Progress-gated scheduling, monitoring
├── visualization/
│   └── visualize_results.py     # Result visualization and analysis
├── train_improved.py            # Main training script
└── README.md                    # This file
```

## Why the Original Method Didn't Converge

### 1. **Geometry Mismatch**
The original contour alignment loss tries to fit a single 3D plane to boundary points scattered across a curved surface. On complex meshes, this creates contradictory gradients as the optimizer tries to satisfy an impossible constraint.

### 2. **Premature Hardening**
The linear temperature schedule (β → 40) hardens the softmax/sigmoid functions before boundaries are established. This causes gradients to vanish early, freezing the optimization in a poor local minimum.

### 3. **Scale Mismatches**
The contour loss scales with O(C²) channel pairs and mesh size, while smoothness scales with edges. Without proper normalization, one loss dominates and prevents balanced optimization.

### 4. **Numerical Instabilities**
SVD on sparse/degenerate point sets produces unstable gradients. The covariance matrix often has poor conditioning, causing the plane normal to flip randomly during training.

### 5. **Non-Smooth Constraints**
Hard projection of pinned vertices creates a non-differentiable operation the optimizer cannot account for, injecting large residuals at each step.

## Expected Results

With these improvements, you should see:

1. **Stable Convergence**: Loss decreases smoothly without plateaus
2. **Balanced Areas**: All 6 regions maintain roughly equal area throughout training
3. **Planar Boundaries**: Boundaries between regions are straight/planar in the surface
4. **No Gradient Issues**: Gradients remain healthy (no vanishing/exploding) throughout

## Monitoring Convergence

The training script logs several key metrics:

- **Area Deviation**: Max deviation from uniform distribution (should decrease)
- **Boundary Length**: Estimated total boundary length (should stabilize)
- **Temperature Values**: β values (should increase based on progress, not time)
- **Loss Components**: Individual loss terms (all should decrease together)

## Troubleshooting

### If loss plateaus early:
- Check if β is increasing too fast (reduce `step_up_contour/area`)
- Increase Stage A duration (adjust stage fractions in `TwoStageScheduler`)
- Reduce initial β values

### If boundaries are wobbly:
- Increase `lambda_contour` in Stage B
- Ensure cotangent weights are computed correctly (check for degenerate triangles)
- Try increasing mesh resolution

### If regions disappear:
- Check area fractions plot - ensure KL loss is working
- Reduce `beta_area_max` to prevent over-hardening
- Verify pinned vertices are well-separated

## Citation

If you use this improved pipeline, please cite:

```
@misc{relu_mesh_improved,
  title={Improved Training Pipeline for ReLU Mesh Segmentation},
  author={Your Name},
  year={2024},
  note={Based on "ReLUs on Meshes" by Kerry Zou}
}
```