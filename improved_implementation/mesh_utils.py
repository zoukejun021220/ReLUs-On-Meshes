"""
Mesh preprocessing utilities for improved ReLU-on-meshes implementation.
Includes edge/adjacency computation, cotangent weights, and triangle local coordinates.
"""

import torch
import numpy as np
from typing import Tuple

Tensor = torch.Tensor


def build_edges_and_adjacency(faces: Tensor, n_verts: int) -> Tuple[Tensor, Tensor]:
    """
    Build edge list and triangle adjacency information.
    
    Args:
        faces: (T, 3) int64 - triangle vertex indices
        n_verts: number of vertices
        
    Returns:
        edge_idx: (E, 2) int64 - vertex indices (min, max) for each edge
        edge_tris: (E, 2) int64 - adjacent triangle indices (-1 for boundary)
    """
    T = faces.shape[0]
    edge_map = {}  # (vi, vj) -> [t0, t1]
    
    for t in range(T):
        i0, i1, i2 = faces[t].tolist()
        # For each edge in the triangle
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            e = (a, b) if a < b else (b, a)
            if e not in edge_map:
                edge_map[e] = [t, -1]  # First triangle using this edge
            else:
                edge_map[e][1] = t  # Second triangle using this edge
    
    E = len(edge_map)
    edge_idx = torch.empty((E, 2), dtype=faces.dtype)
    edge_tris = torch.full((E, 2), -1, dtype=faces.dtype)
    
    for k, (e, (t0, t1)) in enumerate(edge_map.items()):
        edge_idx[k] = torch.tensor(e, dtype=faces.dtype)
        edge_tris[k] = torch.tensor([t0, t1], dtype=faces.dtype)
    
    return edge_idx, edge_tris


def triangles_to_2d(verts: Tensor, faces: Tensor) -> Tensor:
    """
    Convert each triangle to local 2D coordinates in its tangent plane.
    
    Args:
        verts: (N, 3) - vertex positions
        faces: (T, 3) - triangle vertex indices
        
    Returns:
        tri_xy: (T, 3, 2) - 2D coordinates for each triangle's vertices
    """
    T = faces.shape[0]
    tri_xy = torch.empty((T, 3, 2), dtype=verts.dtype, device=verts.device)
    
    v0 = verts[faces[:, 0]]  # (T, 3)
    v1 = verts[faces[:, 1]]  # (T, 3)
    v2 = verts[faces[:, 2]]  # (T, 3)
    
    # Build local coordinate frame
    e0 = v1 - v0  # First edge
    e1 = v2 - v0  # Second edge
    
    # Normalize first edge to get t0 (tangent vector 1)
    t0 = e0 / (e0.norm(dim=-1, keepdim=True) + 1e-12)
    
    # Get normal vector
    n = torch.cross(e0, e1, dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-12)
    
    # Get orthogonal tangent vector t1
    t1 = torch.cross(n, t0, dim=-1)
    
    # Project vertices to 2D coordinates
    # v0 is at origin
    tri_xy[:, 0, :] = 0
    # v1 along t0 axis
    tri_xy[:, 1, 0] = (e0 * t0).sum(-1)
    tri_xy[:, 1, 1] = (e0 * t1).sum(-1)
    # v2 projected
    tri_xy[:, 2, 0] = (e1 * t0).sum(-1)
    tri_xy[:, 2, 1] = (e1 * t1).sum(-1)
    
    return tri_xy


def mesh_areas(verts: Tensor, faces: Tensor) -> Tensor:
    """
    Compute area of each triangle.
    
    Args:
        verts: (N, 3) - vertex positions
        faces: (T, 3) - triangle vertex indices
        
    Returns:
        areas: (T,) - area of each triangle
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    A = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    return A


def cotan_weights(verts: Tensor, faces: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute cotangent weights for the mesh Laplacian.
    
    Args:
        verts: (N, 3) - vertex positions
        faces: (T, 3) - triangle vertex indices
        
    Returns:
        I: (K,) - row indices for sparse matrix
        J: (K,) - column indices for sparse matrix
        W: (K,) - cotangent weights
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    # Edges opposite to each vertex
    e0 = v1 - v2  # opposite v0
    e1 = v2 - v0  # opposite v1
    e2 = v0 - v1  # opposite v2
    
    # Cotangent computation
    def cot(a, b):
        num = (a * b).sum(-1)
        den = torch.linalg.norm(torch.cross(a, b, dim=-1), dim=-1).clamp_min(1e-12)
        return num / den
    
    cot0 = cot(e1, e2)  # angle at v0
    cot1 = cot(e2, e0)  # angle at v1
    cot2 = cot(e0, e1)  # angle at v2
    
    # Build symmetric entries
    # Edge (v1, v2) gets weight cot0
    # Edge (v2, v0) gets weight cot1
    # Edge (v0, v1) gets weight cot2
    
    I = torch.stack([faces[:, 1], faces[:, 2],
                     faces[:, 2], faces[:, 0],
                     faces[:, 0], faces[:, 1]], dim=1).reshape(-1)
    J = torch.stack([faces[:, 2], faces[:, 1],
                     faces[:, 0], faces[:, 2],
                     faces[:, 1], faces[:, 0]], dim=1).reshape(-1)
    W = 0.5 * torch.cat([cot0, cot0, cot1, cot1, cot2, cot2], dim=0)
    
    return I.long(), J.long(), W


def vertex_masses(verts: Tensor, faces: Tensor, tri_area: Tensor) -> Tensor:
    """
    Compute lumped mass for each vertex (1/3 of incident triangle areas).
    
    Args:
        verts: (N, 3) - vertex positions
        faces: (T, 3) - triangle vertex indices
        tri_area: (T,) - area of each triangle
        
    Returns:
        M: (N,) - lumped mass per vertex
    """
    N = verts.shape[0]
    M = torch.zeros(N, dtype=verts.dtype, device=verts.device)
    
    a_third = (tri_area / 3.0).unsqueeze(-1)  # (T, 1)
    idx = faces.reshape(-1)  # (3T,)
    contrib = a_third.repeat(1, 3).reshape(-1)  # (3T,)
    
    M.index_add_(0, idx, contrib)
    return M


def precompute_mesh_data(verts: Tensor, faces: Tensor) -> dict:
    """
    Precompute all mesh-related data needed for optimization.
    
    Args:
        verts: (N, 3) - vertex positions
        faces: (T, 3) - triangle vertex indices
        
    Returns:
        Dictionary with all precomputed data
    """
    tri_area = mesh_areas(verts, faces)
    tri_xy = triangles_to_2d(verts, faces)
    edge_idx, edge_tris = build_edges_and_adjacency(faces, verts.shape[0])
    I, J, W = cotan_weights(verts, faces)
    M = vertex_masses(verts, faces, tri_area)
    edge_len = (verts[edge_idx[:, 0]] - verts[edge_idx[:, 1]]).norm(dim=-1)
    
    # Compute bounding box diagonal for normalization
    bbox_diag = (verts.max(0).values - verts.min(0).values).norm().item()
    
    return {
        'tri_area': tri_area,
        'tri_xy': tri_xy,
        'edge_idx': edge_idx,
        'edge_tris': edge_tris,
        'cotan_I': I,
        'cotan_J': J,
        'cotan_W': W,
        'vertex_mass': M,
        'edge_len': edge_len,
        'bbox_diag': bbox_diag,
        'total_area': tri_area.sum().item()
    }