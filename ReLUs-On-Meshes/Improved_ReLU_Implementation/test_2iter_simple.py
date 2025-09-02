"""
Test script to run only 2 iterations by modifying the optimization schedule.
"""
import numpy as np
import torch
import time
from pathlib import Path

from mesh_utils import load_mesh_from_vtk, pick_raycast_anchors
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
    
    # Select anchor vertices
    print("Selecting anchor vertices using raycast method...")
    pinned_indices = pick_raycast_anchors(vertices, faces)
    
    print("Pinned vertices:")
    region_names = ["Top", "Bottom", "Front", "Back", "Right", "Left"]
    for i, idx in enumerate(pinned_indices):
        print(f"  {region_names[i]}: vertex {idx} at {vertices[idx]}")
    
    # Run optimization with modified schedule
    print("\nStarting optimization (2 iterations only)...")
    start_time = time.time()
    
    f_values, history = optimize_mesh_segmentation(
        vertices, faces, pinned_indices,
        use_coarse_to_fine=True,  # Will use our modified schedule
        use_grad_norm=False,  # Disable for simplicity
        device=device,
        print_every=1  # Print every iteration
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nTest completed in {elapsed_time:.1f} seconds")
    print(f"Final field values shape: {f_values.shape}")
    
    return f_values, history

if __name__ == "__main__":
    mesh_path = "../Piecewise Linear Mesh 3D/l1-poly-dat/hex/kitty/orig.tet.vtk"
    f_values, history = test_2_iterations(mesh_path)