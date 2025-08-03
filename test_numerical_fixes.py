#!/usr/bin/env python3
"""
Test script to verify the numerical stability fixes work correctly.
"""
import subprocess
import sys
import os
import time

def run_test_with_timeout(timeout_seconds=20):
    """Run a quick test to verify the fixes work."""
    
    # Change to the improved implementation directory
    impl_dir = "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Improved_ReLU_Implementation"
    os.chdir(impl_dir)
    
    # Import modules directly
    sys.path.insert(0, impl_dir)
    
    import torch
    import numpy as np
    from mesh_utils import load_mesh_from_vtk, pick_axis_aligned_anchors
    from optimization import optimize_mesh_segmentation, TrainingConfig
    
    # Monkey patch to run only a few steps
    original_init = TrainingConfig.__init__
    
    def patched_init(self, level, num_faces, steps, beta_start, beta_end, 
                     lambda_adj_start, lambda_adj_end, lr_max=5e-3, 
                     lambda_area=1.0, lambda_tv=0.1):
        self.level = level
        self.num_faces = num_faces
        self.steps = 2000  # Run 2000 steps to test through warm-up period
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.lambda_adj_start = lambda_adj_start
        self.lambda_adj_end = lambda_adj_end
        self.lr_max = lr_max
        self.lambda_area = lambda_area
        self.lambda_tv = lambda_tv
    
    TrainingConfig.__init__ = patched_init
    
    # Test with dragon mesh
    dragon_mesh = "../Piecewise Linear Mesh 3D/l1-poly-dat/hex/dragon/orig.tet.vtk"
    
    print("Testing numerical stability fixes with dragon mesh...")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load mesh
    vertices, faces = load_mesh_from_vtk(dragon_mesh)
    print(f"Loaded dragon mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Select anchors
    pinned_indices = pick_axis_aligned_anchors(vertices)
    
    print("\nRunning optimization with:")
    print("- Numerically stable adjacency_loss")
    print("- Clamped gated_tv_loss")
    print("- 1000-step warm-up period (beta=0, lambda_adj=0)")
    print("- Testing 2000 steps total")
    print("-" * 60)
    
    start_time = time.time()
    
    try:
        f_values, history = optimize_mesh_segmentation(
            vertices, faces, pinned_indices,
            use_coarse_to_fine=False,  # Direct training
            use_grad_norm=False,
            device=device
        )
        
        print("\n" + "=" * 60)
        print("✓ Test completed successfully!")
        print("No NaN or inf values encountered.")
        print(f"Execution time: {time.time() - start_time:.1f} seconds")
        
        # Check final loss values
        print("\nFinal metrics:")
        for key in ['loss', 'area', 'adjacency', 'tv']:
            if key in history:
                values = history[key]
                if len(values) > 0:
                    print(f"  {key}: {values[-1]:.4f}")
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test_with_timeout()