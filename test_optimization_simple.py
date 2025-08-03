#!/usr/bin/env python3
"""
Simple test script to verify optimization runs without hanging.
"""

import numpy as np
import torch
import time
from relus_mesh_optimization_improved import optimize_relu_mesh
from mesh_optimization_helpers import auto_select_pins

# Create a simple test mesh (tetrahedron)
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0.5, 0.866, 0],
    [0.5, 0.289, 0.816]
], dtype=np.float32)

faces = np.array([
    [0, 1, 2],
    [0, 1, 3],
    [1, 2, 3],
    [2, 0, 3]
], dtype=np.int32)

print(f"Test mesh: {len(vertices)} vertices, {len(faces)} faces")

# Select anchor vertices
pinned_indices = auto_select_pins(vertices, method='bbox')
print(f"Pinned vertices: {pinned_indices}")

# Run optimization with minimal iterations
print("\nRunning optimization test...")
start_time = time.time()

try:
    results = optimize_relu_mesh(
        vertices=vertices,
        faces=faces,
        pinned_indices=pinned_indices,
        n_iters=10,  # Just 10 iterations for testing
        lr_vertex=0.01,
        lr_offset=0.1,
        print_every=1,  # Print every iteration
        save_path=None  # Don't save
    )
    
    print(f"\nTest completed in {time.time() - start_time:.2f} seconds")
    print(f"Final loss: {results['best_loss']:.6f}")
    print("SUCCESS: Optimization runs without hanging!")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()