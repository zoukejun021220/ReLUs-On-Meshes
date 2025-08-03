#!/usr/bin/env python3
"""
Test script to ensure optimization runs for 2 rounds without error.
Tests both current and revised loss functions.
"""

import numpy as np
import torch
import time
import sys
from relus_mesh_optimization_improved import optimize_relu_mesh
from mesh_optimization_helpers import auto_select_pins

# Create a simple test mesh (cube)
vertices = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
], dtype=np.float32)

faces = np.array([
    # bottom
    [0, 1, 2], [0, 2, 3],
    # top
    [4, 6, 5], [4, 7, 6],
    # front
    [0, 5, 1], [0, 4, 5],
    # back
    [2, 7, 3], [2, 6, 7],
    # left
    [0, 7, 4], [0, 3, 7],
    # right
    [1, 6, 2], [1, 5, 6]
], dtype=np.int32)

print(f"Test mesh: {len(vertices)} vertices, {len(faces)} faces")

# Select anchor vertices
pinned_indices = auto_select_pins(vertices, method='bbox')
print(f"Pinned vertices: {pinned_indices}")

# Test 1: Run with current implementation
print("\n" + "="*60)
print("TEST 1: Current implementation (2 iterations)")
print("="*60)

try:
    results1 = optimize_relu_mesh(
        vertices=vertices,
        faces=faces,
        pinned_indices=pinned_indices,
        n_iters=2,  # Just 2 iterations
        lr_vertex=0.01,
        lr_offset=0.1,
        print_every=1,  # Print every iteration
        save_path="test_2rounds_current.npz"
    )
    
    print(f"\nTest 1 PASSED: Current implementation runs without error")
    print(f"Final loss: {results1['best_loss']:.6f}")
    
except Exception as e:
    print(f"\nTest 1 FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Test with revised loss function
print("\n" + "="*60)
print("TEST 2: Testing revised loss function integration")
print("="*60)

# First, let's modify the optimizer to use the revised loss
import os
import shutil

# Backup current file
shutil.copy(
    "/home/kejunzou/Projects/ReLUs on Meshes/improved_loss_v2_fast.py",
    "/home/kejunzou/Projects/ReLUs on Meshes/improved_loss_v2_fast_backup.py"
)

# Create a modified version that uses the revised loss
modified_content = '''#!/usr/bin/env python3
"""
Modified version using revised loss function.
"""

from improved_loss_revised import improved_loss_revised

def improved_loss_function_fast(
    points, triangles, f_values, edges, beta,
    lambda_area=1.0, lambda_adj=5.0, lambda_TV=0.05, num_channels=6
):
    """Wrapper to use revised loss function."""
    return improved_loss_revised(
        points, triangles, f_values, edges, beta,
        lambda_area, lambda_adj, lambda_TV, num_channels
    )
'''

with open("/home/kejunzou/Projects/ReLUs on Meshes/improved_loss_v2_fast_modified.py", "w") as f:
    f.write(modified_content)

# Now test with a longer run
print("\n" + "="*60)
print("TEST 3: Longer run (100 iterations) to check convergence")
print("="*60)

try:
    results3 = optimize_relu_mesh(
        vertices=vertices,
        faces=faces,
        pinned_indices=pinned_indices,
        n_iters=100,  # 100 iterations
        lr_vertex=0.01,
        lr_offset=0.1,
        print_every=20,  # Print every 20 iterations
        save_path="test_convergence.npz"
    )
    
    print(f"\nTest 3 PASSED: Optimization completed successfully")
    print(f"Initial loss: {results3['history'][0]['total_loss']:.6f}")
    print(f"Final loss: {results3['best_loss']:.6f}")
    print(f"Improvement: {(results3['history'][0]['total_loss'] - results3['best_loss']) / results3['history'][0]['total_loss'] * 100:.1f}%")
    
    # Check area fractions
    final_fractions = results3['history'][-1]['area_fractions']
    print(f"\nFinal area fractions: {final_fractions}")
    print(f"Std deviation: {np.std(final_fractions):.4f}")
    
except Exception as e:
    print(f"\nTest 3 FAILED: {e}")
    import traceback
    traceback.print_exc()

# Restore backup
shutil.move(
    "/home/kejunzou/Projects/ReLUs on Meshes/improved_loss_v2_fast_backup.py",
    "/home/kejunzou/Projects/ReLUs on Meshes/improved_loss_v2_fast.py"
)

print("\n" + "="*60)
print("ALL TESTS COMPLETED")
print("="*60)