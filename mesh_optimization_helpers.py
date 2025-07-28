#!/usr/bin/env python3
"""
Helper functions for mesh processing and initialization.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


def compute_face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area of each triangle face."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Cross product to get area
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    return areas


def build_triangle_adjacency(faces: np.ndarray) -> np.ndarray:
    """Build triangle adjacency list (triangles sharing edges)."""
    from collections import defaultdict
    
    # Build edge to triangle mapping
    edge_to_tris = defaultdict(list)
    
    for t_idx, face in enumerate(faces):
        # Three edges per triangle
        edges = [
            tuple(sorted([face[0], face[1]])),
            tuple(sorted([face[1], face[2]])),
            tuple(sorted([face[2], face[0]]))
        ]
        
        for edge in edges:
            edge_to_tris[edge].append(t_idx)
    
    # Build adjacency pairs
    adjacency = []
    for edge, tris in edge_to_tris.items():
        if len(tris) == 2:
            adjacency.append(tris)
    
    return np.array(adjacency)


def build_vertex_edges(faces: np.ndarray) -> np.ndarray:
    """Build unique vertex edges from faces."""
    edges = set()
    
    for face in faces:
        # Three edges per face
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))
    
    return np.array(list(edges))


def auto_select_pins(vertices: np.ndarray, method: str = 'bbox') -> List[int]:
    """
    Automatically select pinned vertices.
    
    Methods:
    - 'bbox': Select bounding box extremes
    - 'pca': Select along principal directions
    - 'normal_clustering': Cluster by vertex normals
    """
    if method == 'bbox':
        # Find extremes along each axis
        pinned = []
        for axis in range(3):
            pinned.append(np.argmin(vertices[:, axis]))  # Min along axis
            pinned.append(np.argmax(vertices[:, axis]))  # Max along axis
        
        # Remove duplicates while preserving order
        seen = set()
        unique_pinned = []
        for idx in pinned:
            if idx not in seen:
                seen.add(idx)
                unique_pinned.append(idx)
        
        return unique_pinned[:6]  # Return first 6 unique
    
    elif method == 'pca':
        # Center vertices
        centered = vertices - vertices.mean(axis=0)
        
        # Compute PCA
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Select extremes along principal directions
        pinned = []
        for i in range(3):
            projections = centered @ eigvecs[:, i]
            pinned.append(np.argmin(projections))
            pinned.append(np.argmax(projections))
        
        return pinned[:6]
    
    elif method == 'normal_clustering':
        # This requires vertex normals - simplified version
        # For now, fall back to bbox method
        return auto_select_pins(vertices, method='bbox')
    
    else:
        raise ValueError(f"Unknown method: {method}")


def init_6channels_with_pins(
    num_vertices: int,
    pinned_indices: List[int],
    device: torch.device
) -> torch.Tensor:
    """Initialize 6-channel field with pinned vertices."""
    # Random initialization
    f_values = torch.randn(num_vertices, 6, device=device) * 0.1
    
    # Pin vertices to specific channels
    pin_mask = torch.eye(6, device=device) * 2 - 1  # [-1, -1, 1, -1, ...]
    
    for k, idx in enumerate(pinned_indices[:6]):
        f_values[idx] = pin_mask[k]
    
    # Make it a parameter for optimization
    f_values = torch.nn.Parameter(f_values)
    
    return f_values


def estimate_boundary_edges(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    f_values: torch.Tensor,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Estimate boundary edges based on current field values.
    Returns edges where field values change significantly.
    """
    edges = build_vertex_edges(faces.cpu().numpy())
    edges_tensor = torch.from_numpy(edges).to(vertices.device)
    
    # Check field difference across edges
    v1_idx = edges_tensor[:, 0]
    v2_idx = edges_tensor[:, 1]
    
    f1 = f_values[v1_idx]  # (E, 6)
    f2 = f_values[v2_idx]  # (E, 6)
    
    # Compute max channel difference
    max_diff = torch.abs(f1 - f2).max(dim=1)[0]
    
    # Select edges with large differences
    boundary_mask = max_diff > threshold
    boundary_edges = edges_tensor[boundary_mask]
    
    return boundary_edges


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute vertex normals by averaging face normals."""
    # Initialize vertex normals
    vertex_normals = np.zeros_like(vertices)
    
    # Compute face normals
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)
    
    # Accumulate face normals to vertices
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)
    
    # Normalize
    vertex_normals = vertex_normals / (np.linalg.norm(vertex_normals, axis=1, keepdims=True) + 1e-8)
    
    return vertex_normals


def decimate_mesh(vertices: np.ndarray, faces: np.ndarray, target_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple mesh decimation for multi-scale optimization.
    This is a placeholder - for production use, consider using trimesh or other libraries.
    """
    # For now, just return original mesh
    # TODO: Implement proper mesh decimation
    return vertices, faces


def interpolate_field_to_fine_mesh(
    coarse_vertices: np.ndarray,
    coarse_values: np.ndarray,
    fine_vertices: np.ndarray
) -> np.ndarray:
    """
    Interpolate field values from coarse to fine mesh.
    Uses nearest neighbor interpolation as a simple approach.
    """
    from scipy.spatial import cKDTree
    
    # Build KD-tree for coarse vertices
    tree = cKDTree(coarse_vertices)
    
    # Find nearest coarse vertex for each fine vertex
    _, indices = tree.query(fine_vertices)
    
    # Copy field values
    fine_values = coarse_values[indices]
    
    return fine_values