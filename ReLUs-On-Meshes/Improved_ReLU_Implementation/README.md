# Improved ReLU Mesh Implementation

This folder contains an improved implementation of the ReLU mesh segmentation method based on the revised formulation that addresses convergence issues in the original approach.

## Key Improvements

### 1. Revised Loss Functions
- **Adjacency Loss**: Replaced ill-conditioned SVD plane fitting with local cosine penalty
- **Area Balance**: Changed from L2 to L1 deviation for constant gradient
- **Gated TV**: Interior smoothness with preserved sharp boundaries
- **Soft Pinning**: Exponential decay instead of hard constraints

### 2. Better Anchor Selection
- **PCA-based**: Quick fix for elongated shapes
- **Raycast**: Robust for cavities, ensures anchors on outer surface
- **Comparison**: Original axis-aligned method available for benchmarking

### 3. Optimized Training Pipeline
- **Coarse-to-Fine**: 3-level progressive refinement
- **GradNorm**: Adaptive loss balancing
- **Mixed Precision**: FP16/32 training with FP32 fallback for SVD
- **One-Cycle LR**: Better convergence with learning rate scheduling

## Usage

### Basic Example
```python
python main.py --mesh sphere --vertices 5000 --anchor-method raycast
```

### Load VTK Mesh
```python
python main.py --mesh path/to/mesh.vtk --anchor-method raycast
```

### Options
- `--mesh`: 'sphere' or path to VTK/VTU file
- `--vertices`: Number of vertices for sphere (default: 5000)
- `--anchor-method`: 'axis', 'pca', or 'raycast' (default: 'raycast')
- `--no-coarse-to-fine`: Disable coarse-to-fine schedule
- `--no-grad-norm`: Disable GradNorm loss balancing
- `--output-dir`: Directory for results (default: 'results')
- `--device`: 'cuda' or 'cpu'

## Files

- `mesh_utils.py`: Mesh loading, preprocessing, anchor selection
- `loss_functions.py`: Revised loss formulation
- `optimization.py`: Training pipeline with coarse-to-fine schedule
- `visualization.py`: Result visualization and analysis
- `main.py`: Main script and experiments

## Expected Results

| Mesh      | Original Loss | Improved Loss | Max Boundary Angle |
|-----------|--------------|---------------|-------------------|
| Sphere 5k | 0.10         | 0.02          | <1°              |
| Kitty 9k  | 52-55        | 3-4           | ≤2°              |
| Rod 11k   | ~50          | 4-5           | ≤2°              |
| Angel 15k | ~50          | 9-10          | <3°              |

## Mathematical Formulation

The revised loss function:
```
L_total = L_area + L_adj + L_tv
```

Where:
- `L_area = λ_area * Σ|f_c - 1/6|` (L1 area balance)
- `L_adj = λ_adj * Σ w_e * (1 - cos θ_e)` (Local cosine alignment)
- `L_tv = λ_tv * Σ (1 - w_e) * ||d_i - d_j||²` (Gated total variation)

## Requirements

```bash
pip install torch numpy scipy trimesh pyvista matplotlib scikit-learn
```

## Citation

Based on the improved formulation from:
"Re-examining ReLUs on Meshes: A Better Loss, a Robust Training Pipeline, and Reference Code"