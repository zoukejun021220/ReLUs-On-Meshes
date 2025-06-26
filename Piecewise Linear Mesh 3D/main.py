import numpy as np
import math
import pyvista as pv
import vtk
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from Meshsetup import create_icosphere_mesh, load_volume_tet_mesh_and_extract_surface
from MeshParamCalculation import find_axis_vertices, build_pinned_axes_6
from optimization import optimization
from visualization import visualize_segmentation
from SinOptimization import optimization as sin_optimization

def main():
    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # Step 1: Create the sphere mesh
    # can specify either target_points
    target_points = 5000  # Specify desired number of points (approximate)
   
    
    print(f"Creating icosphere mesh with target of {target_points} points...")
    start_time = time.time()
    #vertices_np, faces_np = create_icosphere_mesh(target_points=target_points, radius=1.0)
    vertices_np, faces_np = load_volume_tet_mesh_and_extract_surface("Piecewise Linear Mesh 3D\l1-poly-dat\hex\kitty\orig.tet.vtk")
    plane_offsets = nn.Parameter(torch.zeros(6, device=device))
    elapsed = time.time() - start_time
    print(f"Created sphere mesh in {elapsed:.2f}s with {len(vertices_np)} vertices and {len(faces_np)} faces")
    
    # Step 2: Choose vertices to pin for 6 regions
    print("Finding vertices to pin...")
    start_time = time.time()
    pinned_indices = find_axis_vertices(vertices_np)
    pin_map = {v_idx: ch for ch, v_idx in enumerate(pinned_indices)}
    pinned_axes_array = build_pinned_axes_6(vertices_np, pinned_indices)
    pinned_axes_torch = torch.tensor(pinned_axes_array, dtype=torch.float32).to(device)
    region_names = ["Top", "Bottom", "Front", "Back", "Right", "Left"]
    elapsed = time.time() - start_time
    print(f"Found pin vertices in {elapsed:.2f}s")
    
    print("Pinning vertices for 6 regions:")
    for i, (name, idx) in enumerate(zip(region_names, pinned_indices)):
        print(f"  {name}: vertex {idx} at position {vertices_np[idx]}")
    
    # Step 3: Optimize the 6-channel scalar field with contour alignment loss
    print("\nStarting optimization...")
    n_iters = 100000  
    f_optimized, mesh, loss_history , savepath= sin_optimization(
        vertices_np=vertices_np,
        faces_np=faces_np,
        pinned_indices=pinned_indices,
        pinned_axes=pinned_axes_torch,
        n_iters=n_iters,
        lr=1e-3,  # Learning rate for the optimizer
        beta=1.0,  # Initial beta for the shock phase
        target_beta=15.0,  # Target beta for the refine phase
        beta_schedule=True,
        num_phases=20,
        lambda_contour_initial=0.1,
        lambda_contour_final=4.0,
        lambda_smooth=0.1,
        lambda_area_initial=0.2,
        lambda_area_final=100.0,  
        enable_early_stopping=False,
        patience=2000,
        lr_max_factor=10,
        lr_min_factor=0.05,
        print_every=100,
        


    )
    
    # Step 4: Visualize the result with softmax instead of sigmoid
    print("\nVisualizing result...")
    visualize_segmentation(
        vertices_np=vertices_np,
        faces_np=faces_np,
        f_values=f_optimized,
        subdivisions=3
       
        
       
    )
    
    total_elapsed = time.time() - total_start_time
    print(f"Total execution time: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()