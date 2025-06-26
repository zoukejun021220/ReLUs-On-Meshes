import numpy as np
import math
import pyvista as pv

def create_icosphere_mesh(target_points=None, subdivisions=2, radius=1.0):
    """
    Creates an icosahedron-based sphere mesh by subdividing faces.
    Vectorized where possible.
    
    Args:
        target_points: Desired number of vertices (approximate) - if provided, 
                      overrides the subdivisions parameter
        subdivisions: Number of times to subdivide the initial icosahedron
        radius: Radius of the final sphere
        
    Returns:
        vertices_np: Array of shape (N, 3) containing vertex positions
        faces_np: Array of shape (T, 3) containing triangle indices
    """
    # If target_points is provided, calculate appropriate subdivision level
    if target_points is not None:
        # Initial icosahedron has 12 vertices
        # Each subdivision approximately quadruples the number of vertices
        # So we can estimate: num_vertices ≈ 12 * 4^subdivisions
        
        # Solve for subdivisions: log_4(target_points/12)
        if target_points < 12:
            print(f"Warning: Target points {target_points} is less than minimum (12). Using 0 subdivisions.")
            subdivisions = 0
        else:
            # Calculate subdivision level needed to reach target_points
            subdivisions = int(np.log(target_points / 12) / np.log(4))
            
            # Calculate actual number of points with this subdivision
            estimated_points = 12 * (4 ** subdivisions)
            print(f"Using {subdivisions} subdivisions to create approximately {estimated_points} points " +
                  f"(target was {target_points}).")
    # Start with icosahedron vertices
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    base_vertices = np.array([
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi, 0, -1],
        [ phi, 0,  1],
        [-phi, 0, -1],
        [-phi, 0,  1],
    ], dtype=np.float32)
    
    # Normalize vertices to lie on a unit sphere (vectorized)
    norms = np.linalg.norm(base_vertices, axis=1, keepdims=True)
    base_vertices /= norms
    
    # Initial faces of the icosahedron
    base_faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ], dtype=np.int32)
    
    # Subdivide the mesh
    vertices = base_vertices
    faces = base_faces
    
    for _ in range(subdivisions):
        vertices, faces = subdivide_mesh(vertices, faces)
    
    # Scale by radius (vectorized)
    vertices *= radius
    
    return vertices, faces

def subdivide_mesh(vertices, faces):
    """
    Subdivide each triangle into four smaller triangles.
    Optimized version with more efficient data structures.
    
    Args:
        vertices: Array of shape (N, 3) containing vertex positions
        faces: Array of shape (T, 3) containing triangle indices
        
    Returns:
        new_vertices: Array of shape (N', 3) containing new vertex positions
        new_faces: Array of shape (4T, 3) containing new triangle indices
    """
    vertices = list(vertices)
    new_faces = []
    edge_to_mid = {}
    
    # Extract all edges from faces for batch processing
    all_edges = []
    for face in faces:
        v1, v2, v3 = face
        all_edges.extend([(min(v1, v2), max(v1, v2)), 
                         (min(v2, v3), max(v2, v3)), 
                         (min(v3, v1), max(v3, v1))])
    
    # Find unique edges
    unique_edges = list(set(all_edges))
    
    # Compute midpoints in one batch
    for edge in unique_edges:
        a, b = edge
        mid = 0.5 * (np.array(vertices[a]) + np.array(vertices[b]))
        mid = mid / np.linalg.norm(mid)
        edge_to_mid[edge] = len(vertices)
        vertices.append(mid)
    
    # Create new faces
    for face in faces:
        v1, v2, v3 = face
        e1 = (min(v1, v2), max(v1, v2))
        e2 = (min(v2, v3), max(v2, v3))
        e3 = (min(v3, v1), max(v3, v1))
        
        a = edge_to_mid[e1]
        b = edge_to_mid[e2]
        c = edge_to_mid[e3]
        
        new_faces.extend([
            [v1, a, c],
            [v2, b, a],
            [v3, c, b],
            [a, b, c]
        ])
    
    return np.array(vertices), np.array(new_faces, dtype=np.int32)



def load_volume_tet_mesh_and_extract_surface(file_path):
    """
    Loads a VTK (or VTU) file containing a volumetric tetrahedral mesh,
    extracts its boundary surface, and returns a (vertices, faces) pair
    with all boundary faces triangulated.

    Args:
        file_path (str): Path to the VTK/VTU file.

    Returns:
        vertices_np (np.ndarray): Array of shape (N, 3) containing surface vertex coordinates.
        faces_np (np.ndarray): Array of shape (F, 3) containing triangulated surface faces (vertex indices).
    """
    # 1) Read mesh from file
    mesh = pv.read(file_path)  # PyVista automatically guesses file type (VTK, VTU, etc.)

    # 2) Extract the boundary surface
    surface_mesh = mesh.extract_surface()

    # 3) Triangulate (ensures only triangular cells)
    surface_mesh = surface_mesh.triangulate()

    # surface_mesh.faces is a "face array" of the form [3, i0, i1, i2, 3, i0, i1, i2, ...]
    # which we can reshape into a matrix of shape (num_faces, 4), and drop the first column (the "3")
    faces_array = surface_mesh.faces.reshape(-1, 4)[:, 1:]  # shape: (num_faces, 3)

    # Extract points
    vertices_np = surface_mesh.points  # shape: (N, 3)

    return vertices_np, faces_array