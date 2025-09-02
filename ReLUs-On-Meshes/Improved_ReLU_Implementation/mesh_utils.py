"""
Mesh utilities for loading and preprocessing meshes.
Includes improved anchor selection methods.
"""
import numpy as np
import torch
import pyvista as pv
import trimesh
from typing import Tuple, Optional, List, Dict
from sklearn.decomposition import PCA


def load_mesh_from_vtk(file_path: str, clean_mesh: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a VTK/VTU file and extract the boundary surface with optional cleaning.
    
    Args:
        file_path: Path to the VTK/VTU file
        clean_mesh: If True, remove degenerate triangles and merge duplicates
        
    Returns:
        vertices_np: Array of shape (N, 3) containing surface vertex coordinates
        faces_np: Array of shape (F, 3) containing triangulated surface faces
    """
    # Read mesh from file
    mesh = pv.read(file_path)
    
    # Extract boundary surface
    surface_mesh = mesh.extract_surface()
    
    # Triangulate (ensures only triangular cells)
    surface_mesh = surface_mesh.triangulate()
    
    if clean_mesh:
        try:
            # Remove degenerate triangles (area < 1e-10 × mean area)
            areas = surface_mesh.compute_cell_sizes().cell_data['Area']
            area_threshold = areas.mean() * 1e-10
            mask = areas > area_threshold
            degenerate_count = (~mask).sum()
            
            if degenerate_count > 0:
                surface_mesh = surface_mesh.extract_cells(mask)
                print(f"Removed {degenerate_count} degenerate faces")
            
            # Merge duplicate vertices and remove zero-length edges
            original_points = surface_mesh.n_points
            surface_mesh = surface_mesh.clean(tolerance=1e-12)
            merged_points = original_points - surface_mesh.n_points
            
            if merged_points > 0:
                print(f"Merged {merged_points} duplicate vertices")
        except Exception as e:
            print(f"Warning: Mesh cleaning failed: {e}")
            print("Proceeding with uncleaned mesh")
        
        print(f"Final mesh: {surface_mesh.n_points} vertices, {surface_mesh.n_cells} faces")
    
    # Extract vertices
    vertices_np = np.array(surface_mesh.points)  # shape: (N, 3)
    
    # Extract faces - handle different PyVista versions and face formats
    try:
        # Try the standard way first
        if hasattr(surface_mesh, 'faces') and surface_mesh.faces is not None:
            # PyVista stores faces as [n_verts, v0, v1, ..., vn, n_verts, ...]
            # For triangulated meshes, this is [3, v0, v1, v2, 3, v0, v1, v2, ...]
            faces = surface_mesh.faces
            
            # Check if it's already in the right format
            if len(faces.shape) == 1:
                # Flatten format - need to parse
                faces_array = faces.reshape(-1, 4)[:, 1:]  # Skip the '3' prefix
            else:
                # Already in 2D format
                faces_array = faces
        else:
            # Fallback: manually extract faces
            faces_list = []
            for i in range(surface_mesh.n_cells):
                cell = surface_mesh.get_cell(i)
                faces_list.append(cell.point_ids)
            faces_array = np.array(faces_list, dtype=np.int32)
            
    except Exception as e:
        print(f"Warning: Face extraction encountered issue: {e}")
        # Last resort: try to extract faces using cell connectivity
        faces_array = np.zeros((surface_mesh.n_cells, 3), dtype=np.int32)
        for i in range(surface_mesh.n_cells):
            try:
                cell_pts = surface_mesh.get_cell(i).point_ids
                faces_array[i] = cell_pts[:3]  # Take first 3 points
            except:
                pass
    
    return vertices_np, faces_array


def create_icosphere_mesh(target_points: int = 5000, radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an icosphere mesh with approximately target_points vertices.
    
    Args:
        target_points: Desired number of vertices (approximate)
        radius: Radius of the sphere
        
    Returns:
        vertices_np: Array of shape (N, 3) containing vertex positions
        faces_np: Array of shape (F, 3) containing triangle indices
    """
    # Calculate subdivision level needed
    subdivisions = int(np.log(target_points / 12) / np.log(4)) if target_points > 12 else 0
    
    # Create base icosahedron
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    base_vertices = np.array([
        [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
        [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
        [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
    ], dtype=np.float32)
    
    # Normalize to unit sphere
    base_vertices /= np.linalg.norm(base_vertices, axis=1, keepdims=True)
    
    # Define icosahedron faces
    base_faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)
    
    # Subdivide mesh
    vertices, faces = base_vertices, base_faces
    for _ in range(subdivisions):
        vertices, faces = subdivide_mesh(vertices, faces)
    
    # Scale by radius
    vertices *= radius
    
    return vertices, faces


def subdivide_mesh(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Subdivide each triangle into 4 smaller triangles."""
    vertices = list(vertices)
    new_faces = []
    edge_to_mid = {}
    
    for face in faces:
        v1, v2, v3 = face
        
        # Get or create midpoints for each edge
        edges = [(min(v1, v2), max(v1, v2)), 
                 (min(v2, v3), max(v2, v3)), 
                 (min(v3, v1), max(v3, v1))]
        
        mids = []
        for edge in edges:
            if edge not in edge_to_mid:
                a, b = edge
                mid = 0.5 * (np.array(vertices[a]) + np.array(vertices[b]))
                mid = mid / np.linalg.norm(mid)  # Project to unit sphere
                edge_to_mid[edge] = len(vertices)
                vertices.append(mid)
            mids.append(edge_to_mid[edge])
        
        a, b, c = mids
        new_faces.extend([
            [v1, a, c], [v2, b, a], [v3, c, b], [a, b, c]
        ])
    
    return np.array(vertices), np.array(new_faces, dtype=np.int32)


def pick_pca_anchors(vertices: np.ndarray) -> List[int]:
    """
    Select 6 anchor vertices using PCA-based approach.
    Quick fix that handles elongated shapes better than axis-aligned bounding box.
    
    Args:
        vertices: Array of shape (N, 3) containing vertex positions
        
    Returns:
        List of 6 vertex indices for anchoring
    """
    # Compute PCA
    pca = PCA(n_components=3)
    pca.fit(vertices)
    
    # Get principal axes
    axes = pca.components_  # (3, 3) matrix, each row is a principal axis
    
    anchors = []
    for axis in axes:
        # Project all vertices onto this axis
        projections = vertices @ axis
        
        # Find vertices with max and min projections
        max_idx = np.argmax(projections)
        min_idx = np.argmin(projections)
        
        anchors.extend([max_idx, min_idx])
    
    return anchors


def pick_raycast_anchors(vertices: np.ndarray, faces: np.ndarray, num_rays: int = 64) -> List[int]:
    """
    Select 6 anchor vertices using raycast approach.
    Robust for cavities - ensures anchors are on outer surface.
    
    Args:
        vertices: Array of shape (N, 3) containing vertex positions
        faces: Array of shape (F, 3) containing triangle indices
        num_rays: Number of rays to cast per direction
        
    Returns:
        List of 6 vertex indices for anchoring
    """
    # Create trimesh object for ray casting
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Define 6 principal directions
    dirs = np.array([[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]], dtype=np.float32)
    
    # Compute mesh centroid
    centroid = vertices.mean(axis=0)
    
    anchors = []
    for d in dirs:
        # Generate jittered rays
        scale = mesh.scale if hasattr(mesh, 'scale') else mesh.extents.max()
        jitter = (np.random.rand(num_rays, 3) - 0.5) * 0.05 * scale
        origins = centroid[None, :] + jitter
        directions = np.tile(d[None, :], (num_rays, 1))
        
        # Cast rays
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=directions,
            multiple_hits=False
        )
        
        if len(locations) == 0:
            # Fallback if no intersections found
            proj = vertices @ d
            best_idx = np.argmax(proj)
        else:
            # Find intersection with largest projection along direction
            locations = np.array(locations)
            projections = locations @ d
            best_loc = locations[np.argmax(projections)]
            
            # Find nearest vertex to best location
            distances = np.linalg.norm(vertices - best_loc[None, :], axis=1)
            best_idx = np.argmin(distances)
        
        anchors.append(best_idx)
    
    return anchors


def pick_axis_aligned_anchors(vertices: np.ndarray) -> List[int]:
    """
    Original axis-aligned bounding box approach for comparison.
    
    Args:
        vertices: Array of shape (N, 3) containing vertex positions
        
    Returns:
        List of 6 vertex indices for anchoring
    """
    # Compute bounding box
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = 0.5 * (vmin + vmax)
    
    # Define anchor points at bbox extremes
    anchor_points = [
        [center[0], center[1], vmax[2]],  # top
        [center[0], center[1], vmin[2]],  # bottom
        [center[0], vmax[1], center[2]],  # front
        [center[0], vmin[1], center[2]],  # back
        [vmax[0], center[1], center[2]],  # right
        [vmin[0], center[1], center[2]],  # left
    ]
    
    # Find nearest vertices
    anchors = []
    for pt in anchor_points:
        distances = np.linalg.norm(vertices - np.array(pt)[None, :], axis=1)
        anchors.append(np.argmin(distances))
    
    return anchors


def compute_mesh_adjacency(faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mesh adjacency information.
    
    Args:
        faces: Array of shape (F, 3) containing triangle indices
        
    Returns:
        edges: Array of shape (E, 2) containing unique edges
        edge2face: Array of shape (E, 2) containing face indices for each edge (-1 for boundary)
        face_adjacency: Array of shape (A, 2) containing pairs of adjacent faces
    """
    from collections import defaultdict
    
    # Build edge to face mapping
    edge_to_faces = defaultdict(list)
    
    for face_idx, face in enumerate(faces):
        v0, v1, v2 = face
        edges = [(min(v0, v1), max(v0, v1)),
                 (min(v1, v2), max(v1, v2)),
                 (min(v2, v0), max(v2, v0))]
        
        for edge in edges:
            edge_to_faces[edge].append(face_idx)
    
    # Extract unique edges and build edge2face mapping
    edges = []
    edge2face = []
    face_adjacency = []
    
    for edge, face_list in edge_to_faces.items():
        edges.append(edge)
        
        if len(face_list) == 1:
            # Boundary edge
            edge2face.append([face_list[0], -1])
        elif len(face_list) == 2:
            # Interior edge
            edge2face.append(face_list)
            face_adjacency.append(sorted(face_list))
        else:
            # Non-manifold edge (shouldn't happen for valid meshes)
            edge2face.append([face_list[0], face_list[1]])
    
    return (np.array(edges, dtype=np.int64),
            np.array(edge2face, dtype=np.int64),
            np.array(face_adjacency, dtype=np.int64))


def compute_mesh_data(vertices: np.ndarray, faces: np.ndarray) -> Dict:
    """
    Compute all mesh data needed for optimization.
    
    Returns a dictionary with:
    - edges: Edge connectivity
    - edge2face: Edge to face adjacency
    - face_areas: Area of each face
    - B: Barycentric gradient matrices
    - face_mask: Valid (non-degenerate) faces
    """
    # Get adjacency
    edges, edge2face, _ = compute_mesh_adjacency(faces)
    
    # Get face areas
    face_areas = compute_face_areas(vertices, faces)
    
    # Get barycentric matrices and face mask
    B, face_mask = compute_barycentric_matrices(vertices, faces)
    
    return {
        'edges': edges,
        'edge2face': edge2face,
        'face_areas': face_areas,
        'B': B,
        'face_mask': face_mask
    }


def compute_face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute area of each triangle face.
    
    Args:
        vertices: Array of shape (N, 3) containing vertex positions
        faces: Array of shape (F, 3) containing triangle indices
        
    Returns:
        areas: Array of shape (F,) containing face areas
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Compute cross product of edge vectors
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    return areas


def compute_barycentric_matrices(vertices: np.ndarray, faces: np.ndarray, 
                                return_mask: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute barycentric coordinate matrices for gradient computation.
    Handles degenerate triangles safely to prevent NaN propagation.
    
    Args:
        vertices: Array of shape (N, 3) containing vertex positions
        faces: Array of shape (F, 3) containing triangle indices
        return_mask: If True, also return mask of valid (non-degenerate) faces
        
    Returns:
        B: Array of shape (F, 3, 3) containing barycentric matrices
        face_mask: Boolean array of shape (F,) indicating valid faces (if return_mask=True)
    """
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]  # (F, 3)
    v2 = vertices[faces[:, 2]]  # (F, 3)
    
    # Edge vectors
    e1 = v1 - v0  # (F, 3)
    e2 = v2 - v0  # (F, 3)
    
    # Compute cross product and area
    cross = np.cross(e1, e2)  # (F, 3)
    area2 = np.linalg.norm(cross, axis=1)  # double area
    
    # Identify valid (non-degenerate) triangles
    eps = 1e-8
    face_mask = area2 > eps
    
    # Initialize normal vectors
    n = np.zeros_like(cross)
    # Only normalize for valid faces to avoid division by zero
    n[face_mask] = cross[face_mask] / area2[face_mask, None]
    
    # Stack to form (F, 3, 3) matrices
    M = np.stack([e1, e2, n], axis=2)  # (F, 3, 3)
    
    # Initialize B matrices
    B = np.zeros_like(M)
    # Only compute inverse for valid faces
    if face_mask.any():
        B[face_mask] = np.linalg.pinv(M[face_mask])
    
    if return_mask:
        return B, face_mask
    else:
        return B