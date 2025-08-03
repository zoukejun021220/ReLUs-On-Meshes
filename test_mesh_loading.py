#!/usr/bin/env python3
"""
Test mesh loading with cleaning to debug the issue.
"""
import sys
import os

# Add implementation directory to path
impl_dir = "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Improved_ReLU_Implementation"
sys.path.insert(0, impl_dir)

def test_mesh_loading():
    """Test loading dragon mesh with cleaning."""
    try:
        import pyvista as pv
        print(f"PyVista version: {pv.__version__}")
    except ImportError:
        print("PyVista not available")
        return
        
    dragon_path = "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Piecewise Linear Mesh 3D/l1-poly-dat/hex/dragon/orig.tet.vtk"
    
    print(f"Loading mesh from: {dragon_path}")
    
    # Test manual loading
    mesh = pv.read(dragon_path)
    print(f"Original mesh type: {type(mesh)}")
    print(f"Original mesh: {mesh.n_points} points, {mesh.n_cells} cells")
    
    # Extract surface
    surface = mesh.extract_surface()
    print(f"Surface: {surface.n_points} points, {surface.n_cells} cells")
    
    # Triangulate
    triangulated = surface.triangulate()
    print(f"Triangulated: {triangulated.n_points} points, {triangulated.n_cells} cells")
    
    # Check faces attribute
    print(f"Has faces attribute: {hasattr(triangulated, 'faces')}")
    if hasattr(triangulated, 'faces'):
        print(f"Faces type: {type(triangulated.faces)}")
        print(f"Faces shape: {triangulated.faces.shape if hasattr(triangulated.faces, 'shape') else 'No shape'}")
        print(f"Faces is None: {triangulated.faces is None}")
    
    # Try cleaning
    print("\nTrying mesh cleaning...")
    areas = triangulated.compute_cell_sizes().cell_data['Area']
    print(f"Area stats: min={areas.min()}, max={areas.max()}, mean={areas.mean()}")
    
    # Clean
    cleaned = triangulated.clean(tolerance=1e-12)
    print(f"After clean: {cleaned.n_points} points, {cleaned.n_cells} cells")
    print(f"Has faces after clean: {hasattr(cleaned, 'faces')}")
    
    # Test our function
    print("\nTesting our load function...")
    from mesh_utils import load_mesh_from_vtk
    try:
        vertices, faces = load_mesh_from_vtk(dragon_path, clean_mesh=False)
        print(f"Success without cleaning: {len(vertices)} vertices, {len(faces)} faces")
    except Exception as e:
        print(f"Error without cleaning: {e}")
        
    try:
        vertices, faces = load_mesh_from_vtk(dragon_path, clean_mesh=True)
        print(f"Success with cleaning: {len(vertices)} vertices, {len(faces)} faces")
    except Exception as e:
        print(f"Error with cleaning: {e}")

if __name__ == "__main__":
    test_mesh_loading()