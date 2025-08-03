"""
Test script to run only 2 iterations using axis-aligned anchor selection.
"""
import numpy as np
import torch
import time
from pathlib import Path

from mesh_utils import load_mesh_from_vtk, pick_axis_aligned_anchors
from optimization import CoarseToFineSchedule, TrainingConfig, optimize_mesh_segmentation

# Monkey patch the schedule to run only 2 iterations
original_init = CoarseToFineSchedule.__init__

def new_init(self):
    self.stages = [
        TrainingConfig(level=0, num_faces=-1, steps=2,  # Only 2 steps!
                     beta_start=10.0, beta_end=10.0,
                     lambda_adj_start=5.0, lambda_adj_end=5.0),
    ]

CoarseToFineSchedule.__init__ = new_init

def test_2_iterations(mesh_path):
    """Run only 2 iterations to test the pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load mesh
    print(f"Loading mesh from {mesh_path}...")
    vertices, faces = load_mesh_from_vtk(mesh_path)
    print(f"Mesh loaded: {len(vertices)} vertices, {len(faces)} faces")
    
    # Select anchor vertices using axis-aligned method (simpler, no raycasting)
    print("Selecting anchor vertices using axis-aligned method...")
    pinned_indices = pick_axis_aligned_anchors(vertices)
    
    print("Pinned vertices:")
    region_names = ["Top", "Bottom", "Front", "Back", "Right", "Left"]
    for i, idx in enumerate(pinned_indices):
        print(f"  {region_names[i]}: vertex {idx} at {vertices[idx]}")
    
    # Run optimization with modified schedule
    print("\nStarting optimization (2 iterations only)...")
    start_time = time.time()
    
    try:
        f_values, history = optimize_mesh_segmentation(
            vertices, faces, pinned_indices,
            use_coarse_to_fine=False,  # Disable to run only one stage
            use_grad_norm=False,  # Disable for simplicity
            device=device
        )
        
        elapsed_time = time.time() - start_time
        print(f"\nTest completed successfully in {elapsed_time:.1f} seconds")
        print(f"Final field values shape: {f_values.shape}")
        print(f"History length: {len(history['total_loss'])}")
        
        # Print the losses from both iterations
        print("\nLoss values:")
        for i in range(len(history['total_loss'])):
            print(f"  Iteration {i+1}: {history['total_loss'][i]:.4f}")
        
        return f_values, history
        
    except Exception as e:
        print(f"\nError during optimization: {e}")
        print("Stopping the program...")
        raise

if __name__ == "__main__":
    mesh_path = "../Piecewise Linear Mesh 3D/l1-poly-dat/hex/kitty/orig.tet.vtk"
    f_values, history = test_2_iterations(mesh_path)
    print("\nProgram completed successfully!")