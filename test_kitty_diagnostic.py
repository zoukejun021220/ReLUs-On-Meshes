#!/usr/bin/env python3
"""
Diagnostic test to understand the loss behavior.
"""

import numpy as np
import torch
import sys
sys.path.append('/home/kejunzou/Projects/ReLUs on Meshes')

from loss_groundup import build_edge_topology, face_gradient_mats, loss_revised

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

# Main diagnostic
def main():
    vtk_file = "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Piecewise Linear Mesh 3D/l1-poly-dat/hex/kitty/orig.tet.vtk"
    
    vertices, faces = load_vtk_manual(vtk_file)
    print(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Convert to torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    verts = torch.from_numpy(vertices).float().to(device)
    faces = torch.from_numpy(faces).long().to(device)
    
    # Build topology
    edges, edge2face, face2edge = build_edge_topology(faces.cpu())
    edges = edges.to(device)
    edge2face = edge2face.to(device)
    
    print(f"Built topology: {len(edges)} edges")
    
    # Face gradient matrices
    B = face_gradient_mats(verts, faces).to(device)
    
    # Face areas
    v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
    face_areas = 0.5 * torch.linalg.norm(torch.cross(v1-v0, v2-v0, dim=1), dim=1)
    
    # Initialize field
    N = len(vertices)
    Fv = torch.randn(N, 6, device=device) * 0.1
    
    # Pin vertices (simplified)
    for i in range(6):
        idx = i * (N // 6)  # Spread out pinned vertices
        Fv[idx] = -1.0
        Fv[idx, i] = 1.0
    
    print("\nAnalyzing loss components with beta=2.0:")
    
    # Test loss with low beta
    loss, parts = loss_revised(
        Fv, beta=2.0, lambda_adj=5.0, lambda_tv=0.05, lambda_area=1.0,
        verts=verts, faces=faces, edges=edges, edge2face=edge2face, 
        B=B, face_areas=face_areas
    )
    
    print(f"Total loss: {loss.item():.4f}")
    print(f"  Area: {parts['area']:.4f}")
    print(f"  Adjacent: {parts['adj']:.4f}")
    print(f"  TV: {parts['tv']:.4f}")
    
    # Analyze boundary weights
    print("\nAnalyzing boundary weights:")
    C = 6
    v_idx0, v_idx1 = edges[:,0], edges[:,1]
    
    # Look at one channel pair
    a, b = 0, 1
    Fa = Fv[:,a]
    Fb = Fv[:,b]
    d = Fa - Fb
    
    d_i = d[v_idx0]
    d_j = d[v_idx1]
    
    # Boundary weight
    w_e = torch.sigmoid(-2.0 * d_i * d_j)
    
    print(f"Channel pair ({a},{b}):")
    print(f"  d_i range: [{d_i.min():.3f}, {d_i.max():.3f}]")
    print(f"  d_j range: [{d_j.min():.3f}, {d_j.max():.3f}]")
    print(f"  d_i*d_j range: [{(d_i*d_j).min():.3f}, {(d_i*d_j).max():.3f}]")
    print(f"  w_e range: [{w_e.min():.3f}, {w_e.max():.3f}]")
    print(f"  w_e mean: {w_e.mean():.3f}")
    print(f"  Edges with w_e > 0.5: {(w_e > 0.5).sum().item()}")
    
    # Check gradient computation
    print("\nChecking gradient computation:")
    
    # Get gradients for one channel difference
    Ff = Fv[faces]  # (F,3,C)
    grads = []
    for c in range(1):  # Just check first channel
        gc = torch.einsum('fij,fj->fi', B, Ff[:,:,c])
        grads.append(gc)
    grads = torch.stack(grads, dim=2)
    
    print(f"Gradient shape: {grads.shape}")
    print(f"Gradient magnitude range: [{grads.norm(dim=1).min():.3f}, {grads.norm(dim=1).max():.3f}]")
    
    # Check adjacent triangles
    f1 = edge2face[:,0]
    f2 = edge2face[:,1]
    valid = (f1>=0) & (f2>=0)
    print(f"Edges with two adjacent faces: {valid.sum().item()} / {len(edges)}")

if __name__ == "__main__":
    main()