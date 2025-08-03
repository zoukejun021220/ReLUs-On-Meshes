#!/usr/bin/env python3
"""
Test script to debug the vertex mapping issue.
"""
import numpy as np
import sys
import os

# Add implementation directory to path
impl_dir = "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Improved_ReLU_Implementation"
sys.path.insert(0, impl_dir)

def test_downsample_mesh():
    """Test the downsample_mesh function with dragon mesh."""
    from mesh_utils import load_mesh_from_vtk
    from optimization import downsample_mesh
    
    # Load dragon mesh
    dragon_path = "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Piecewise Linear Mesh 3D/l1-poly-dat/hex/dragon/orig.tet.vtk"
    
    print("Loading dragon mesh...")
    vertices, faces = load_mesh_from_vtk(dragon_path)
    print(f"Original mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Test downsampling
    print("\nTesting mesh downsampling to 3000 faces...")
    try:
        coarse_vertices, coarse_faces, vertex_mapping = downsample_mesh(
            vertices, faces, 3000
        )
        
        print(f"\nResults:")
        print(f"Coarse mesh: {len(coarse_vertices)} vertices, {len(coarse_faces)} faces")
        print(f"Vertex mapping shape: {vertex_mapping.shape}")
        print(f"Vertex mapping dtype: {vertex_mapping.dtype}")
        print(f"Vertex mapping range: [{vertex_mapping.min()}, {vertex_mapping.max()}]")
        print(f"Original vertices range: [0, {len(vertices)-1}]")
        
        # Check for out of bounds
        out_of_bounds = vertex_mapping >= len(vertices)
        if out_of_bounds.any():
            print(f"\nWARNING: {out_of_bounds.sum()} out-of-bounds indices found!")
            print(f"Out-of-bounds indices: {vertex_mapping[out_of_bounds]}")
        else:
            print("\n✓ All vertex mappings are within bounds!")
            
    except Exception as e:
        print(f"\nError during downsampling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # First check if we can import
    try:
        import trimesh
        print(f"Trimesh version: {trimesh.__version__}")
    except ImportError:
        print("Trimesh not available in this environment")
        
    try:
        import torch
        print(f"PyTorch available: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not available in this environment")
    
    print("\n" + "=" * 60)
    test_downsample_mesh()