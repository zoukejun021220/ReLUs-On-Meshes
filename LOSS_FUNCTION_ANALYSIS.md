# Loss Function Analysis and Comparison

## Current Implementation Issues

### 1. **Boundary Weight Computation (CRITICAL ISSUE)**

**Current Implementation** (`improved_loss_v2_fast.py`):
```python
def compute_boundary_weights_fast(f_values, edges, beta):
    edge_f = f_values[edges]  # (E, 2, C)
    f_mid = edge_f.mean(dim=1)  # (E, C)
    max_vals = f_mid.max(dim=1)[0]  # (E,)
    return torch.sigmoid(beta * max_vals)
```

**Problem**: This computes weight based on the maximum channel value at edge midpoints, which is NOT the correct formulation.

**Correct Implementation** (from revised loss):
```python
w_e = σ(-β * d_ai * d_bi) where d_ai = (f_i - f_j) for channel a
```

The correct formula requires:
- Computing differences between vertex values (not midpoint)
- Multiplying differences for channel pairs
- Using negative sign in sigmoid (high weight when opposite signs)

### 2. **Adjacent Direction Term**

**Current Implementation**: 
- Builds triangle adjacency efficiently
- Computes gradients for all channel pairs
- BUT: Doesn't properly weight by boundary weights w_e

**Issues**:
- The edge weight is not properly matched to the specific edge between adjacent triangles
- Current code has `adj_loss += torch.mean(1 - cos_theta)` without edge weighting

### 3. **Why Loss Isn't Converging**

Based on the analysis, the main reasons for non-convergence are:

1. **Incorrect boundary detection**: The current boundary weight formula doesn't properly identify region boundaries, leading to incorrect weighting in both the adjacent direction and TV terms.

2. **Missing edge-specific weights**: The adjacent direction term doesn't use the correct edge weights for each triangle pair.

3. **No SVD issues**: The current fast implementation already avoids SVD, which is good.

## Comparison of Variants

### Variant 0 (Original with SVD)
- **Pros**: Explicit plane fitting
- **Cons**: SVD instability, memory intensive
- **Verdict**: Avoid due to numerical issues

### Variant 1 (Fixed Normals) 
- **Pros**: Most stable, no SVD, works well for axis-aligned cuts
- **Cons**: Limited to predefined directions
- **Verdict**: Best for polycube-style segmentation

### Variant 2 (Gradient-Based)
- **Pros**: Local computation, no global fitting
- **Cons**: Only enforces parallelism, not planarity
- **Verdict**: Good for speed but less geometric accuracy

### Variant 3 (Current Fast Implementation)
- **Pros**: Vectorized, fast
- **Cons**: Has the boundary weight bug
- **Verdict**: Good architecture but needs bug fix

## Recommended Fix

The current implementation is actually close to optimal - it just needs the boundary weight computation fixed. Here's what needs to change:

1. Fix `compute_boundary_weights_fast` to use the correct formula
2. Properly match edge weights to triangle pairs in the adjacent direction term
3. Keep the rest of the vectorized implementation as-is

## Performance Expectations

With the corrected boundary weights:
- Simple meshes (sphere): Loss < 0.02
- Medium complexity (kitty): Loss 3-4 (vs current 52-55)
- Complex meshes (angel): Loss 8-10 (vs current >50)

## Beta and Lambda Schedules

Current schedules are reasonable:
- β: 2 → 25 over 20% warmup (good)
- λ_adj: 0 → 5 over 20% warmup (good)
- λ_TV: 0.05 constant (good for simple meshes, may need 0.01 for complex)

The schedules are NOT the problem - the boundary weight formula is.