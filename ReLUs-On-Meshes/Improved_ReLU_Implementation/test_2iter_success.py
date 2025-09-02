"""
Test script to run only 2 iterations of the improved ReLU mesh segmentation.
This version runs successfully without errors.
"""
import numpy as np
import torch
import time
from pathlib import Path

from mesh_utils import load_mesh_from_vtk, pick_axis_aligned_anchors
from optimization import optimize_mesh_segmentation, TrainingConfig

# Monkey patch TrainingConfig to run only 2 steps
original_init = TrainingConfig.__init__

def new_init(self, level, num_faces, steps, beta_start, beta_end, 
             lambda_adj_start, lambda_adj_end, lr_max=5e-3, 
             lambda_area=1.0, lambda_tv=0.1):
    self.level = level
    self.num_faces = num_faces
    self.steps = 2  # Force only 2 steps!
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.lambda_adj_start = lambda_adj_start
    self.lambda_adj_end = lambda_adj_end
    self.lr_max = lr_max
    self.lambda_area = lambda_area
    self.lambda_tv = lambda_tv

TrainingConfig.__init__ = new_init

def test_2_iterations(mesh_path):
    """Run only 2 iterations to test the pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load mesh
    print(f"\nLoading mesh from {mesh_path}...")
    vertices, faces = load_mesh_from_vtk(mesh_path)
    print(f"Mesh loaded: {len(vertices)} vertices, {len(faces)} faces")
    
    # Select anchor vertices using axis-aligned method
    print("\nSelecting anchor vertices using axis-aligned method...")
    pinned_indices = pick_axis_aligned_anchors(vertices)
    
    print("\nPinned vertices:")
    region_names = ["Top", "Bottom", "Front", "Back", "Right", "Left"]
    for i, idx in enumerate(pinned_indices):
        print(f"  {region_names[i]}: vertex {idx} at {vertices[idx]}")
    
    # Run optimization
    print("\n" + "="*50)
    print("Starting optimization (2 iterations only)...")
    print("="*50 + "\n")
    
    start_time = time.time()
    
    f_values, history = optimize_mesh_segmentation(
        vertices, faces, pinned_indices,
        use_coarse_to_fine=False,  # Use direct training
        use_grad_norm=False,  # Disable for simplicity
        device=device
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*50)
    print(f"✓ Test completed successfully!")
    print("="*50)
    print(f"\nExecution time: {elapsed_time:.2f} seconds")
    print(f"Final field values shape: {f_values.shape}")
    print(f"History keys: {list(history.keys())}")
    
    return f_values, history

if __name__ == "__main__":
    mesh_path = "../Piecewise Linear Mesh 3D/l1-poly-dat/hex/kitty/orig.tet.vtk"
    print("ReLU Mesh Segmentation - 2 Iteration Test")
    print("=========================================")
    
    f_values, history = test_2_iterations(mesh_path)
    
    print("\n✓ Program completed successfully!")
    print("The optimization ran for exactly 2 iterations without errors.")
    print("\nYou can now safely stop the program.")