"""
Mesh utilities for loading and preprocessing meshes.
Includes improved anchor selection methods.
"""
import numpy as np
import torch
import pyvista as pv
import trimesh
from typing import Tuple, Optional, List
from sklearn.decomposition import PCA


def load_mesh_from_vtk(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a VTK/VTU file and extract the boundary surface.
    
    Args:
        file_path: Path to the VTK/VTU file
        
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
    
    # Extract faces and vertices
    faces_array = surface_mesh.faces.reshape(-1, 4)[:, 1:]  # shape: (num_faces, 3)
    vertices_np = surface_mesh.points  # shape: (N, 3)
    
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


def compute_barycentric_matrices(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """
    Compute barycentric coordinate matrices for gradient computation.
    
    Args:
        vertices: Array of shape (N, 3) containing vertex positions
        faces: Array of shape (F, 3) containing triangle indices
        
    Returns:
        B: Array of shape (F, 3, 3) containing barycentric matrices
    """
    v0 = vertices[faces[:, 0]]  # (F, 3)
    v1 = vertices[faces[:, 1]]  # (F, 3)
    v2 = vertices[faces[:, 2]]  # (F, 3)
    
    # Edge vectors
    e1 = v1 - v0  # (F, 3)
    e2 = v2 - v0  # (F, 3)
    
    # Build matrices [e1, e2, n] where n is the normal
    n = np.cross(e1, e2)  # (F, 3)
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-10)
    
    # Stack to form (F, 3, 3) matrices
    M = np.stack([e1, e2, n], axis=2)  # (F, 3, 3)
    
    # Compute inverse (actually pseudo-inverse for stability)
    B = np.linalg.pinv(M)  # (F, 3, 3)
    
    return B