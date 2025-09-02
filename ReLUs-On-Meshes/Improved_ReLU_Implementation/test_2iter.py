"""
Test script to run only 2 iterations of the improved ReLU mesh segmentation.
"""
import numpy as np
import torch
import time
from pathlib import Path

from mesh_utils import load_mesh_from_vtk, pick_raycast_anchors
from loss_functions import compute_total_loss
from mesh_utils import precompute_mesh_data

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
    
    # Precompute mesh data
    print("\nPrecomputing mesh data...")
    mesh_data = precompute_mesh_data(vertices, faces, device)
    
    # Initialize field values
    num_vertices = len(vertices)
    f_values = torch.randn(num_vertices, 6, device=device) * 0.01
    
    # Pin vertices
    for ch, v_idx in enumerate(pinned_indices):
        f_values[v_idx] = -1.0
        f_values[v_idx, ch] = 1.0
    
    f_values.requires_grad_(True)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW([f_values], lr=1e-3, weight_decay=1e-4)
    
    # Run 2 iterations
    print("\nRunning 2 iterations...")
    for iter in range(2):
        print(f"\nIteration {iter + 1}/2")
        start_time = time.time()
        
        optimizer.zero_grad()
        
        # Compute loss
        loss_dict = compute_total_loss(
            f_values,
            mesh_data['vertices_torch'],
            mesh_data['faces_torch'],
            mesh_data['edges_torch'],
            mesh_data['edge2face_torch'],
            mesh_data['face_areas_torch'],
            mesh_data['B_torch'],
            beta=10.0,
            lambda_area=1.0,
            lambda_adj=5.0,
            lambda_tv=0.1,
            return_components=True
        )
        
        total_loss = loss_dict['total']
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([f_values], max_norm=5.0)
        
        optimizer.step()
        
        # Re-pin vertices
        with torch.no_grad():
            for ch, v_idx in enumerate(pinned_indices):
                f_values[v_idx] = -1.0
                f_values[v_idx, ch] = 1.0
        
        iter_time = time.time() - start_time
        
        print(f"  Loss: {total_loss.item():.4f}")
        print(f"  - Area: {loss_dict['area'].item():.4f}")
        print(f"  - Adjacency: {loss_dict['adjacency'].item():.4f}")
        print(f"  - TV: {loss_dict['tv'].item():.4f}")
        print(f"  Time: {iter_time:.2f}s")
    
    print("\nTest completed successfully!")
    return f_values.detach().cpu().numpy()

if __name__ == "__main__":
    mesh_path = "../Piecewise Linear Mesh 3D/l1-poly-dat/hex/kitty/orig.tet.vtk"
    f_values = test_2_iterations(mesh_path)
    print(f"\nFinal field values shape: {f_values.shape}")