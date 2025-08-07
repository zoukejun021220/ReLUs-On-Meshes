# Improved ReLU-on-Meshes Implementation

This directory contains an improved implementation of the ReLU-on-Meshes algorithm that addresses the convergence issues identified in the original approach.

## Key Improvements

### 1. **Intrinsic Contour Alignment (Replaces 3D Plane Fitting)**
- **Problem**: Global 3D plane fitting via SVD was unstable and fought against surface geometry
- **Solution**: Local 2D alignment in triangle tangent planes
- **Benefits**: No SVD instability, respects surface geometry, stable gradients

### 2. **Cotangent Laplacian Smoothness**
- **Problem**: Simple edge differences ignored mesh geometry, causing anisotropic artifacts
- **Solution**: Proper cotangent-weighted Laplacian
- **Benefits**: Geometry-aware smoothing, better behavior on irregular meshes

### 3. **Progress-Gated Temperature Control**
- **Problem**: Fixed β schedule caused gradient vanishing before boundaries formed
- **Solution**: Increase β only when area balance and boundary length criteria are met
- **Benefits**: Maintains gradients throughout optimization, prevents premature hardening

### 4. **KL Divergence for Area Balance**
- **Problem**: L1 loss had flat gradients for small/large regions
- **Solution**: KL divergence to uniform distribution
- **Benefits**: Consistent gradients regardless of region size

### 5. **Two-Stage Optimization**
- **Stage A (0-60%)**: Low β, focus on smoothness and basic segmentation
- **Stage B (60-100%)**: Refine boundaries with increasing β and contour weight
- **Benefits**: Stable progression from coarse to fine segmentation

### 6. **Soft Pinning with Annealing**
- **Problem**: Hard projection after each step created discontinuous gradients
- **Solution**: Quadratic penalty that hardens only in final 10% of training
- **Benefits**: Smooth optimization landscape, better convergence

## Usage

```python
from improved_implementation import MeshSegmentationTrainer
import torch

# Load your mesh
vertices = torch.tensor(vertices_np, dtype=torch.float32)
faces = torch.tensor(faces_np, dtype=torch.int64)

# Create trainer
trainer = MeshSegmentationTrainer(
    verts=vertices,
    faces=faces,
    n_channels=6
)

# Set pinned vertices (optional)
trainer.set_pinned_vertices(pinned_indices)

# Train
trainer.train(
    n_steps=100000,
    print_every=1000,
    checkpoint_dir="checkpoints"
)

# Get results
f_values = trainer.get_field_values()
```

## File Structure

- `mesh_utils.py`: Mesh preprocessing (edges, adjacency, cotangent weights)
- `losses.py`: Improved loss functions (intrinsic alignment, KL area, cotangent smoothness)
- `temperature_control.py`: Progress-gated β scheduling and adaptive weights
- `trainer.py`: Main training loop with two-stage optimization
- `visualization.py`: Visualization utilities with mesh subdivision
- `example_usage.py`: Complete examples on sphere and complex meshes

## Why Your Method Wasn't Converging

Based on your report (Sections 4.6-4.7), the main issues were:

1. **Geometric Mismatch**: Fitting 3D planes to boundaries on curved surfaces created conflicting gradients
2. **Numerical Instability**: SVD on sparse/degenerate point sets (mentioned in 4.5.11)
3. **Premature Hardening**: Fixed β schedule caused gradients to vanish before boundaries formed
4. **Loss Interference**: Unnormalized losses with O(C²) pair interactions overwhelmed other terms

The improved implementation directly addresses each of these issues with the changes listed above.

## Recommended Parameters

### For Simple Meshes (Sphere, Smooth Surfaces)
- `n_steps`: 50,000
- `initial_lr`: 1e-3
- `stage_transition`: 0.6
- `beta_update_every`: 400

### For Complex Meshes (Kitty, Angel, etc.)
- `n_steps`: 100,000-200,000
- `initial_lr`: 5e-4
- `stage_transition`: 0.7
- `beta_update_every`: 500
- `grad_clip`: 3.0

## Monitoring Convergence

The trainer logs several metrics to diagnose convergence:
- Individual loss components
- Temperature evolution
- Area fractions
- Estimated boundary length

Use `plot_training_history(trainer.history)` to visualize these metrics.