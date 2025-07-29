#!/usr/bin/env python3
"""
Test script to verify VTK loading functionality
"""

import sys
import os
from run_optimization import load_mesh

def test_vtk_loading():
    # Test with one of the VTK files from the dataset
    test_file = "ReLUs-On-Meshes/Piecewise Linear Mesh 3D/l1-poly-dat/hex/kitty/orig.tet.vtk"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        print("Please ensure you're running from the correct directory")
        return
    
    try:
        print(f"Loading VTK file: {test_file}")
        vertices, faces = load_mesh(test_file)
        
        print(f"Successfully loaded mesh!")
        print(f"  Vertices: {vertices.shape}")
        print(f"  Faces: {faces.shape}")
        print(f"  Vertex range: [{vertices.min():.3f}, {vertices.max():.3f}]")
        
        # Verify data types
        assert vertices.shape[1] == 3, "Vertices should have 3 coordinates"
        assert faces.shape[1] == 3, "Faces should have 3 indices"
        
        print("\nVTK loading test PASSED!")
        
    except Exception as e:
        print(f"Error loading VTK file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vtk_loading()