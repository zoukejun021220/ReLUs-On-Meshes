#!/usr/bin/env python3
"""
Diagnose why loss values are too small.
"""

import torch
import numpy as np
import sys
sys.path.append('/home/kejunzou/Projects/ReLUs on Meshes')

from loss_groundup import MeshLossGroundUp, build_edge_topology, face_gradient_mats


def load_vtk_manual(filename):
    """Manual VTK ASCII parser."""
    vertices = []
    cells = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('POINTS'):
            parts = line.split()
            num_points = int(parts[1])
            i += 1
            
            point_data = []
            while len(point_data) < num_points * 3:
                values = lines[i].strip().split()
                point_data.extend([float(v) for v in values])
                i += 1
            
            vertices = np.array(point_data).reshape(-1, 3)
            continue
            
        if line.startswith('CELLS'):
            parts = line.split()
            num_cells = int(parts[1])
            i += 1
            
            for _ in range(num_cells):
                cell_line = lines[i].strip().split()
                cell_size = int(cell_line[0])
                if cell_size == 4:
                    cells.append([int(cell_line[j]) for j in range(1, 5)])
                i += 1
            continue
            
        i += 1
    
    # Extract surface
    face_set = set()
    for tet in cells:
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]]))
        ]
        for face in faces:
            if face in face_set:
                face_set.remove(face)
            else:
                face_set.add(face)
    
    triangles = np.array(list(face_set))
    return vertices, triangles


def diagnose_mesh(mesh_path):
    """Diagnose loss scaling issues."""
    print(f"\nDiagnosing: {mesh_path}")
    print("="*70)
    
    # Load mesh
    vertices, faces = load_vtk_manual(mesh_path)
    print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Convert to torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    verts = torch.from_numpy(vertices).float()
    faces_tensor = torch.from_numpy(faces).long()
    
    # Compute mesh scale
    bbox_min = verts.min(dim=0)[0]
    bbox_max = verts.max(dim=0)[0]
    mesh_scale = (bbox_max - bbox_min).max().item()
    print(f"Mesh scale (bbox diagonal): {mesh_scale:.3f}")
    
    # Compute face areas
    v0, v1, v2 = verts[faces_tensor[:,0]], verts[faces_tensor[:,1]], verts[faces_tensor[:,2]]
    face_areas = 0.5 * torch.linalg.norm(torch.cross(v1-v0, v2-v0, dim=1), dim=1)
    
    print(f"\nFace areas:")
    print(f"  Min: {face_areas.min().item():.6f}")
    print(f"  Max: {face_areas.max().item():.6f}")
    print(f"  Mean: {face_areas.mean().item():.6f}")
    print(f"  Total: {face_areas.sum().item():.6f}")
    
    # Build topology
    edges, edge2face, _ = build_edge_topology(faces_tensor)
    print(f"\nTopology: {len(edges)} edges")
    
    # Test different initializations
    N = len(vertices)
    C = 6
    
    print("\n" + "="*50)
    print("Testing different field initializations:")
    print("="*50)
    
    # Test 1: Random initialization
    print("\n1. Random initialization (randn * 0.1):")
    f_values = torch.randn(N, C, device=device) * 0.1
    
    # Initialize loss module
    mesh_loss = MeshLossGroundUp(verts, faces_tensor, device)
    
    # Compute loss components (with lambda_adj=0 like at iter 0)
    loss, parts = mesh_loss.compute_loss(f_values, beta=2.0, 
                                         lambda_adj=0.0, lambda_tv=0.05, lambda_area=1.0)
    
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  Area loss: {parts['area']:.6f}")
    print(f"  Adj loss: {parts['adj']:.6f}")
    print(f"  TV loss: {parts['tv']:.6f}")
    
    # Check area fractions
    Pv = torch.softmax(2.0 * f_values, dim=1)
    Pf = Pv[faces_tensor.to(device)].mean(dim=1)
    face_areas_device = face_areas.to(device)
    area_fractions = (Pf.T * face_areas_device).sum(dim=1) / face_areas_device.sum()
    print(f"  Area fractions: {area_fractions.cpu().numpy()}")
    print(f"  Target: {1.0/C:.6f}")
    print(f"  Deviations: {torch.abs(area_fractions - 1.0/C).cpu().numpy()}")
    
    # Test 2: Larger initialization
    print("\n2. Larger initialization (randn * 1.0):")
    f_values = torch.randn(N, C, device=device) * 1.0
    
    loss, parts = mesh_loss.compute_loss(f_values, beta=2.0, 
                                         lambda_adj=5.0, lambda_tv=0.05, lambda_area=1.0)
    
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  Area loss: {parts['area']:.6f}")
    print(f"  Adj loss: {parts['adj']:.6f}")
    print(f"  TV loss: {parts['tv']:.6f}")
    
    # Test 3: Very imbalanced initialization
    print("\n3. Imbalanced initialization (channel 0 = 5.0, others = -1.0):")
    f_values = torch.ones(N, C, device=device) * -1.0
    f_values[:, 0] = 5.0
    
    loss, parts = mesh_loss.compute_loss(f_values, beta=2.0, 
                                         lambda_adj=5.0, lambda_tv=0.05, lambda_area=1.0)
    
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  Area loss: {parts['area']:.6f}")
    print(f"  Adj loss: {parts['adj']:.6f}")
    print(f"  TV loss: {parts['tv']:.6f}")
    
    Pv = torch.softmax(2.0 * f_values, dim=1)
    Pf = Pv[faces_tensor].mean(dim=1)
    area_fractions = (Pf.T * face_areas).sum(dim=1) / face_areas.sum()
    print(f"  Area fractions: {area_fractions.cpu().numpy()}")
    
    # Test with higher beta
    print("\n4. Random init with beta=25.0:")
    f_values = torch.randn(N, C, device=device) * 0.1
    
    loss, parts = mesh_loss.compute_loss(f_values, beta=25.0, 
                                         lambda_adj=5.0, lambda_tv=0.05, lambda_area=1.0)
    
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  Area loss: {parts['area']:.6f}")
    print(f"  Adj loss: {parts['adj']:.6f}")
    print(f"  TV loss: {parts['tv']:.6f}")
    
    # Check what happens with actual optimization targets
    print("\n5. Near-optimal initialization (each channel gets 1/6 of vertices):")
    f_values = torch.zeros(N, C, device=device)
    vertices_per_channel = N // C
    for i in range(C):
        start_idx = i * vertices_per_channel
        end_idx = (i + 1) * vertices_per_channel if i < C-1 else N
        f_values[start_idx:end_idx, i] = 5.0
        # Set other channels negative
        for j in range(C):
            if j != i:
                f_values[start_idx:end_idx, j] = -1.0
    
    loss, parts = mesh_loss.compute_loss(f_values, beta=10.0, 
                                         lambda_adj=5.0, lambda_tv=0.05, lambda_area=1.0)
    
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  Area loss: {parts['area']:.6f}")
    print(f"  Adj loss: {parts['adj']:.6f}")
    print(f"  TV loss: {parts['tv']:.6f}")
    
    Pv = torch.softmax(10.0 * f_values, dim=1)
    Pf = Pv[faces_tensor].mean(dim=1)
    area_fractions = (Pf.T * face_areas).sum(dim=1) / face_areas.sum()
    print(f"  Area fractions: {area_fractions.cpu().numpy()}")


if __name__ == "__main__":
    # Test on dragon mesh
    dragon_path = "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Piecewise Linear Mesh 3D/l1-poly-dat/hex/dragon/orig.tet.vtk"
    diagnose_mesh(dragon_path)
    
    # Also test on a smaller mesh if available
    kitty_path = "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Piecewise Linear Mesh 3D/l1-poly-dat/hex/kitty/orig.tet.vtk"
    import os
    if os.path.exists(kitty_path):
        diagnose_mesh(kitty_path)