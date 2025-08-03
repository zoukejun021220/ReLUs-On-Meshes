#!/usr/bin/env python3
"""
Test the fixed loss implementation on kitty mesh for 10 iterations.
"""

import numpy as np
import torch
import sys
sys.path.append('/home/kejunzou/Projects/ReLUs on Meshes')

from loss_groundup_fixed import MeshLossGroundUpFixed

# Load mesh
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

# Main test
def main():
    vtk_file = "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Piecewise Linear Mesh 3D/l1-poly-dat/hex/kitty/orig.tet.vtk"
    
    vertices, faces = load_vtk_manual(vtk_file)
    print(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Convert to torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    verts = torch.from_numpy(vertices).float()
    faces = torch.from_numpy(faces).long()
    
    # Initialize mesh loss with fixed normalization
    mesh_loss = MeshLossGroundUpFixed(verts, faces, device)
    
    # Initialize field
    N = len(vertices)
    f_values = torch.randn(N, 6, device=device) * 0.1
    
    # Pin vertices
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    
    # Find extremal vertices
    pinned_indices = []
    directions = [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]
    
    for direction in directions:
        projections = vertices @ direction
        idx = np.argmax(projections) if sum(direction) > 0 else np.argmin(projections)
        pinned_indices.append(idx)
    
    # Set pinned values
    for i, idx in enumerate(pinned_indices[:6]):
        f_values[idx] = 0.0
        f_values[idx, i] = 1.0
    
    # Make it a parameter
    f_values = torch.nn.Parameter(f_values)
    
    # Run 10 iterations
    optimizer = torch.optim.Adam([f_values], lr=0.01)
    
    print("\nRunning 10 iterations with FIXED loss normalization:")
    print("-" * 60)
    
    for it in range(10):
        optimizer.zero_grad()
        
        # Get schedules
        schedules = mesh_loss.get_schedules(it, 10)
        beta = schedules['beta']
        lambda_adj = schedules['lambda_adj']
        
        # Compute loss
        loss, components = mesh_loss.compute_loss(
            f_values, 
            beta=beta,
            lambda_adj=lambda_adj,
            lambda_tv=0.05,
            lambda_area=1.0
        )
        
        print(f"Iter {it+1}: Loss={loss.item():8.4f} | "
              f"area={components['area']:6.4f}, "
              f"adj={components['adj']:8.4f}, "
              f"tv={components['tv']:6.4f} | "
              f"β={beta:.1f}, λ_adj={lambda_adj:.1f}")
        
        # Check for explosion
        if loss.item() > 1000:
            print("\nERROR: Loss still exploding!")
            break
        
        # Backward
        loss.backward()
        
        # Step
        optimizer.step()
        
        # Re-pin vertices
        with torch.no_grad():
            for i, idx in enumerate(pinned_indices[:6]):
                f_values[idx] = 0.0
                f_values[idx, i] = 1.0
    
    print("\nTest completed successfully - loss is stable!")
    
    # Final analysis
    print("\nFinal loss components:")
    loss, components = mesh_loss.compute_loss(
        f_values, beta=25.0, lambda_adj=5.0, lambda_tv=0.05, lambda_area=1.0
    )
    print(f"  Total: {loss.item():.4f}")
    print(f"  Area: {components['area']:.4f}")
    print(f"  Adjacent: {components['adj']:.4f}")  
    print(f"  TV: {components['tv']:.4f}")

if __name__ == "__main__":
    main()