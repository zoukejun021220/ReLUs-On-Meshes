#!/usr/bin/env python3
"""
Test script to run 10 iterations on the kitty mesh to check for bugs.
"""

import numpy as np
import torch
import sys
import os

# Add the project directory to path
sys.path.append('/home/kejunzou/Projects/ReLUs on Meshes')

# Try different import approaches
try:
    # First try the ground-up implementation
    from loss_groundup import MeshLossGroundUp
    use_groundup = True
    print("Using ground-up loss implementation")
except ImportError as e:
    print(f"Could not import ground-up loss: {e}")
    use_groundup = False
    
try:
    # Also try importing the existing implementation
    from relus_mesh_optimization_improved import optimize_relu_mesh
    from mesh_optimization_helpers import auto_select_pins
    use_existing = True
    print("Using existing optimization implementation")
except ImportError as e:
    print(f"Could not import existing implementation: {e}")
    use_existing = False

# VTK loading function
def load_vtk_mesh(filename):
    """Load a tetrahedral mesh from VTK file and extract surface."""
    try:
        import vtk
        from vtk.util import numpy_support
        
        # Read the VTK file
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        
        # Get the mesh
        mesh = reader.GetOutput()
        
        # Extract surface
        surface_filter = vtk.vtkDataSetSurfaceFilter()
        surface_filter.SetInputData(mesh)
        surface_filter.Update()
        
        surface = surface_filter.GetOutput()
        
        # Get points
        points = numpy_support.vtk_to_numpy(surface.GetPoints().GetData())
        
        # Get triangles
        cells = surface.GetPolys()
        triangles = []
        cells.InitTraversal()
        while True:
            cell = vtk.vtkIdList()
            if cells.GetNextCell(cell) == 0:
                break
            if cell.GetNumberOfIds() == 3:
                triangles.append([cell.GetId(i) for i in range(3)])
        
        triangles = np.array(triangles)
        
        return points, triangles
        
    except ImportError:
        print("VTK not available, trying alternative loading method")
        return None, None

def load_vtk_manual(filename):
    """Manual VTK ASCII parser as fallback."""
    vertices = []
    cells = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Find POINTS section
        if line.startswith('POINTS'):
            parts = line.split()
            num_points = int(parts[1])
            i += 1
            
            # Read points
            point_data = []
            while len(point_data) < num_points * 3:
                values = lines[i].strip().split()
                point_data.extend([float(v) for v in values])
                i += 1
            
            vertices = np.array(point_data).reshape(-1, 3)
            continue
            
        # Find CELLS section
        if line.startswith('CELLS'):
            parts = line.split()
            num_cells = int(parts[1])
            i += 1
            
            # Read cells
            for _ in range(num_cells):
                cell_line = lines[i].strip().split()
                cell_size = int(cell_line[0])
                if cell_size == 4:  # Tetrahedron
                    cells.append([int(cell_line[j]) for j in range(1, 5)])
                i += 1
            continue
            
        i += 1
    
    if len(vertices) == 0 or len(cells) == 0:
        return None, None
    
    # Extract surface from tetrahedra
    print(f"Loaded {len(vertices)} vertices and {len(cells)} tetrahedra")
    
    # Extract unique triangular faces
    face_set = set()
    for tet in cells:
        # Each tetrahedron has 4 triangular faces
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]]))
        ]
        for face in faces:
            if face in face_set:
                face_set.remove(face)  # Interior face
            else:
                face_set.add(face)  # Boundary face
    
    triangles = np.array(list(face_set))
    
    return vertices, triangles

# Main test
def main():
    vtk_file = "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Piecewise Linear Mesh 3D/l1-poly-dat/hex/kitty/orig.tet.vtk"
    
    # Check if file exists
    if not os.path.exists(vtk_file):
        print(f"Error: File not found: {vtk_file}")
        return
    
    print(f"Loading mesh from: {vtk_file}")
    
    # Try loading the mesh
    vertices, faces = load_vtk_mesh(vtk_file)
    
    if vertices is None:
        print("Trying manual VTK parser...")
        vertices, faces = load_vtk_manual(vtk_file)
    
    if vertices is None:
        print("Failed to load mesh")
        return
    
    print(f"Loaded mesh with {len(vertices)} vertices and {len(faces)} faces")
    
    # Auto-select pinned vertices
    if use_existing:
        pinned_indices = auto_select_pins(vertices)
        print(f"Selected {len(pinned_indices)} pinned vertices: {pinned_indices}")
    else:
        # Manual pin selection for 6 regions
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        center = (bbox_min + bbox_max) / 2
        
        pinned_indices = []
        # Find vertices closest to 6 principal directions
        directions = [
            [1, 0, 0],   # +X
            [-1, 0, 0],  # -X
            [0, 1, 0],   # +Y
            [0, -1, 0],  # -Y
            [0, 0, 1],   # +Z
            [0, 0, -1]   # -Z
        ]
        
        for direction in directions:
            # Find vertex furthest in this direction
            projections = vertices @ direction
            idx = np.argmax(projections) if direction[0] + direction[1] + direction[2] > 0 else np.argmin(projections)
            pinned_indices.append(idx)
        
        print(f"Selected pinned vertices: {pinned_indices}")
    
    # Test with ground-up implementation
    if use_groundup:
        print("\nTesting with ground-up implementation...")
        try:
            # Convert to torch tensors
            v_torch = torch.from_numpy(vertices).float()
            f_torch = torch.from_numpy(faces).long()
            
            # Initialize mesh loss
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using device: {device}")
            
            mesh_loss = MeshLossGroundUp(v_torch, f_torch, device)
            
            # Initialize field
            N = len(vertices)
            f_values = torch.randn(N, 6, device=device) * 0.1
            
            # Pin vertices
            for i, idx in enumerate(pinned_indices[:6]):
                f_values[idx] = 0.0
                f_values[idx, i] = 1.0
            
            # Make it a parameter for optimization
            f_values = torch.nn.Parameter(f_values)
            
            # Run 10 iterations
            optimizer = torch.optim.Adam([f_values], lr=0.01)
            
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
                
                print(f"Iteration {it+1}: Loss={loss.item():.6f} "
                      f"(area={components['area']:.4f}, "
                      f"adj={components['adj']:.4f}, "
                      f"tv={components['tv']:.4f})")
                
                # Backward
                loss.backward()
                
                # Check for NaN
                if torch.isnan(loss):
                    print("ERROR: Loss is NaN!")
                    break
                
                # Step
                optimizer.step()
                
                # Re-pin vertices
                with torch.no_grad():
                    for i, idx in enumerate(pinned_indices[:6]):
                        f_values[idx] = 0.0
                        f_values[idx, i] = 1.0
            
            print("Ground-up test completed successfully!")
            
        except Exception as e:
            print(f"Error in ground-up test: {e}")
            import traceback
            traceback.print_exc()
    
    # Test with existing implementation
    if use_existing and False:  # Disabled for now due to dependencies
        print("\nTesting with existing implementation...")
        try:
            results = optimize_relu_mesh(
                vertices=vertices,
                faces=faces,
                pinned_indices=pinned_indices,
                n_iters=10,
                lr_vertex=0.01,
                lr_offset=0.1,
                print_every=1,
                save_path=None
            )
            print("Existing implementation test completed!")
        except Exception as e:
            print(f"Error in existing implementation test: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()