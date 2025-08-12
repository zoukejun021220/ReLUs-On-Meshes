# SVD-Based Training Implementation Summary

This document summarizes the changes made to implement the SVD-based training schedule from PASTE.md.

## Files Modified

### 1. **losses/svd_contour_loss.py** (New File)
- Implemented `fit_plane_weighted()`: Fits planes to weighted 3D points using eigendecomposition/SVD
- Implemented `plane_loss()`: Computes weighted MSE loss for point-to-plane distance
- Implemented `contour_alignment_svd()`: Main SVD-based contour alignment loss function
- Features:
  - Numerical stability through covariance symmetrization and jitter
  - EMA smoothing for temporal consistency
  - Sign consistency checking for plane normals
  - Plane memory for persistent plane parameters across iterations

### 2. **optimization/temperature_control.py**
- Updated `TempController` class:
  - Changed initial beta_contour from 1.0 to 0.8 (SVD schedule)
  - Increased beta_contour_max from 8.0 to 16.0 for sharper final boundaries
  - Reduced step_up_contour from 0.3 to 0.2 for more gradual increases

- Replaced `TwoStageScheduler` stages with SVD-based 5-stage schedule:
  - **Stage 0: Warm Start** (0-5k): High smoothness, no SVD loss, beta_c: 0.8→1.8
  - **Stage 1: Frozen SVD Planes** (5k-30k): Low SVD weight (0.1), update planes every K=20-50 steps
  - **Stage 2: Plane Trust Ramp** (30k-70k): Increase SVD weight (0.3), beta_c: 1.8→4.0
  - **Stage 3: Lock Normals** (70k-120k): Strong SVD (0.5), only update offsets, beta_c→8.0
  - **Stage 4: Crisp Snap** (120k-300k): Very high SVD (0.8), minimal smoothness, beta_c→12-16

### 3. **train_improved.py**
- Added `use_svd_contour` parameter to `train_mesh_segmentation()`
- Added command-line argument `--use-svd-contour`
- Integrated SVD loss computation:
  - Added plane memory management
  - Conditional plane updates based on stage and K_update parameter
  - SVD-specific handling for missing edge_weights
- Updated loss computation to handle SVD mode:
  - Modified margin sharpening, Potts smoothness, and boundary length losses to check for edge_weights
- Added SVD-specific logging to display active pairs count

## Usage

To use the SVD-based training approach:

```bash
python train_improved.py --mesh-path <path_to_mesh> --use-svd-contour
```

## Key Implementation Details

1. **Plane Fitting**: Uses weighted covariance matrix with eigendecomposition for stable plane fitting
2. **Temporal Consistency**: EMA smoothing prevents plane jitter between updates
3. **Update Schedule**: Planes are updated every K steps (K=20-50) to balance stability and accuracy
4. **Stage-Specific Behavior**: Different stages have different update frequencies and trust levels for planes
5. **Numerical Stability**: Covariance symmetrization and jitter prevent degenerate cases

## Benefits of SVD Approach

1. **Geometric Consistency**: Planes provide global geometric constraints for boundaries
2. **Stability**: Frozen/slow plane updates prevent oscillations
3. **Straight Seams**: SVD naturally produces straight boundary lines
4. **Reduced Triple Points**: Plane constraints discourage unnecessary boundary intersections