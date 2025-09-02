"""
Mesh preprocessing utilities with cotangent weights and intrinsic computations.
Addresses issues from report sections 4.3.3, 4.5.1, and 4.5.2.
"""
import torch
import numpy as np
from typing import Tuple, Dict, Optional

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False


def load_volume_tet_mesh_and_extract_surface(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a VTK/VTU file containing a volumetric tetrahedral mesh,
    extracts its boundary surface.
    
    Args:
        file_path: Path to the VTK/VTU file
        
    Returns:
        vertices_np: Array of shape (N, 3) containing surface vertex coordinates
        faces_np: Array of shape (F, 3) containing triangulated surface faces
    """
    if not HAS_PYVISTA:
        raise ImportError("PyVista is required for loading VTK files. Install with: pip install pyvista")
    
    mesh = pv.read(file_path)
    surface_mesh = mesh.extract_surface()
    surface_mesh = surface_mesh.triangulate()
    
    faces_array = surface_mesh.faces.reshape(-1, 4)[:, 1:]
    vertices_np = surface_mesh.points
    
    return vertices_np, faces_array


def build_edges_and_adjacency(faces: torch.Tensor, n_verts: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge connectivity and triangle adjacency.
    
    Args:
        faces: (T, 3) int64 tensor of face indices
        n_verts: Number of vertices
        
    Returns:
        edge_idx: (E, 2) int64 tensor of vertex indices (min, max)
        edge_tris: (E, 2) int64 tensor of adjacent triangle indices (-1 for boundary)
    """
    T = faces.shape[0]
    edge_map = {}  # (vi, vj) -> [t0, t1]
    
    for t in range(T):
        i0, i1, i2 = faces[t].tolist()
        for a, b in ((i0, i1), (i1, i2), (i2, i0)):
            e = (a, b) if a < b else (b, a)
            if e not in edge_map:
                edge_map[e] = [t, -1]
            else:
                edge_map[e][1] = t
    
    E = len(edge_map)
    edge_idx = torch.empty((E, 2), dtype=faces.dtype)
    edge_tris = torch.full((E, 2), -1, dtype=faces.dtype)
    
    for k, (e, (t0, t1)) in enumerate(edge_map.items()):
        edge_idx[k] = torch.tensor(e, dtype=faces.dtype)
        edge_tris[k] = torch.tensor([t0, t1], dtype=faces.dtype)
        
    return edge_idx, edge_tris


def triangles_to_2d(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute per-triangle 2D coordinates in local tangent frames.
    This addresses the issue of working in intrinsic geometry rather than 3D.
    
    Args:
        verts: (N, 3) vertex positions
        faces: (T, 3) face indices
        
    Returns:
        tri_xy: (T, 3, 2) 2D coordinates for each triangle's vertices
    """
    T = faces.shape[0]
    tri_xy = torch.empty((T, 3, 2), dtype=verts.dtype, device=verts.device)
    
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    e0 = v1 - v0
    e1 = v2 - v0
    
    # Create orthonormal basis in triangle plane
    t0 = e0 / (e0.norm(dim=-1, keepdim=True) + 1e-12)
    n = torch.cross(e0, e1, dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-12)
    t1 = torch.cross(n, t0, dim=-1)
    
    # Project vertices to 2D
    tri_xy[:, 0, :] = 0
    tri_xy[:, 1, 0] = (e0 * t0).sum(-1)
    tri_xy[:, 1, 1] = (e0 * t1).sum(-1)
    tri_xy[:, 2, 0] = (e1 * t0).sum(-1)
    tri_xy[:, 2, 1] = (e1 * t1).sum(-1)
    
    return tri_xy


def mesh_areas(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute triangle areas.
    
    Args:
        verts: (N, 3) vertex positions
        faces: (T, 3) face indices
        
    Returns:
        areas: (T,) triangle areas
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    A = 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=-1), dim=-1)
    return A


def cotan_weights(verts: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute cotangent weights for the discrete Laplacian.
    This addresses the issue of unnormalized edge weights (report section 4.3.3.2).
    
    Args:
        verts: (N, 3) vertex positions
        faces: (T, 3) face indices
        
    Returns:
        I: (K,) source vertex indices
        J: (K,) target vertex indices
        W: (K,) cotangent weights
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    e0 = v1 - v2  # opposite v0
    e1 = v2 - v0  # opposite v1
    e2 = v0 - v1  # opposite v2
    
    def cot(a, b):
        num = (a * b).sum(-1)
        den = torch.linalg.norm(torch.cross(a, b, dim=-1), dim=-1).clamp_min(1e-12)
        return num / den
    
    cot0 = cot(e1, e2)  # opposite v0 -> edge (1, 2)
    cot1 = cot(e2, e0)  # opposite v1 -> edge (2, 0)
    cot2 = cot(e0, e1)  # opposite v2 -> edge (0, 1)
    
    # Build symmetric edge list
    I = torch.stack([faces[:, 1], faces[:, 2],
                     faces[:, 2], faces[:, 0],
                     faces[:, 0], faces[:, 1]], dim=1).reshape(-1)
    J = torch.stack([faces[:, 2], faces[:, 1],
                     faces[:, 0], faces[:, 2],
                     faces[:, 1], faces[:, 0]], dim=1).reshape(-1)
    W = 0.5 * torch.cat([cot0, cot0, cot1, cot1, cot2, cot2], dim=0)
    
    # Zero-clamp: negative cotangents → 0 (don't turn them into attractive springs)
    # This is the standard approach for mesh smoothing energies
    valid = torch.isfinite(W)
    W = torch.where(valid, torch.clamp(W, min=0.0), torch.zeros_like(W))
    
    # Keep only valid weights
    valid = valid & (W > 0)  # Remove zero weights too
    I = I[valid]
    J = J[valid]
    W = W[valid]
    
    return I.long(), J.long(), W


def vertex_masses(verts: torch.Tensor, faces: torch.Tensor, tri_area: torch.Tensor) -> torch.Tensor:
    """
    Compute lumped mass per vertex (1/3 of adjacent triangle areas).
    
    Args:
        verts: (N, 3) vertex positions
        faces: (T, 3) face indices
        tri_area: (T,) triangle areas
        
    Returns:
        M: (N,) vertex masses
    """
    N = verts.shape[0]
    M = torch.zeros(N, dtype=verts.dtype, device=verts.device)
    a_third = (tri_area / 3.0).unsqueeze(-1)
    idx = faces.reshape(-1)
    contrib = a_third.repeat(1, 3).reshape(-1)
    M.index_add_(0, idx, contrib)
    return M


def vertex_normals(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute per-vertex normals by averaging face normals.
    
    Args:
        verts: (N, 3) vertex positions
        faces: (T, 3) face indices
        
    Returns:
        normals: (N, 3) unit vertex normals
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # (T, 3)
    
    # Accumulate face normals to vertices
    vertex_normals = torch.zeros_like(verts)
    for k in range(3):
        vertex_normals.index_add_(0, faces[:, k], face_normals)
    
    # Normalize
    return vertex_normals / (vertex_normals.norm(dim=1, keepdim=True).clamp_min(1e-12))


def unique_preserve_order(tensor: torch.Tensor) -> torch.Tensor:
    """
    Remove duplicates while preserving order (unlike torch.unique which sorts).
    
    Args:
        tensor: Input tensor
        
    Returns:
        tensor with duplicates removed, order preserved
    """
    seen = set()
    result = []
    for x in tensor.tolist():
        if x not in seen:
            result.append(x)
            seen.add(x)
    return torch.tensor(result, device=tensor.device, dtype=tensor.dtype)


def pick_axis_aligned_anchors_by_normals(vertices: torch.Tensor, faces: torch.Tensor) -> Tuple[torch.Tensor, Dict[int, str]]:
    """
    Select 6 anchor vertices by finding vertices whose normals best align with coordinate axes.
    This ensures pins are placed on surfaces that naturally face each axis direction.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (T, 3) face indices
        
    Returns:
        pinned_indices: (6,) indices of anchor vertices in order [+X, -X, +Y, -Y, +Z, -Z]
        channel_labels: Dict mapping channel index to axis name
    """
    device = vertices.device
    
    # Define axis directions and labels
    axis_dirs = torch.tensor([[1, 0, 0], [-1, 0, 0], 
                             [0, 1, 0], [0, -1, 0],
                             [0, 0, 1], [0, 0, -1]], 
                            dtype=vertices.dtype, device=device)
    axis_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
    
    # Compute vertex normals
    normals = vertex_normals(vertices, faces)
    
    # Score each vertex by how well its normal aligns with each axis
    scores = normals @ axis_dirs.T  # (N, 6)
    
    # Greedy selection to ensure distinctness
    used = torch.zeros(vertices.shape[0], dtype=torch.bool, device=device)
    indices = []
    
    for channel in range(6):
        # Sort vertices by alignment score for this axis
        order = torch.argsort(scores[:, channel], descending=True)
        
        # Pick the best unused vertex
        for idx in order:
            if not used[idx]:
                indices.append(idx.item())
                used[idx] = True
                break
    
    pinned_indices = torch.tensor(indices, device=device, dtype=torch.long)
    channel_labels = {i: axis_names[i] for i in range(6)}
    
    return pinned_indices, channel_labels


def pick_axis_aligned_anchors(vertices: torch.Tensor, faces: Optional[torch.Tensor] = None, 
                            method: str = 'normals') -> Tuple[torch.Tensor, Dict[int, str]]:
    """
    Select 6 anchor vertices aligned with principal axes.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (T, 3) face indices (required for 'normals' method)
        method: 'normals' (recommended), 'extremes', or 'bbox_rays'
        
    Returns:
        pinned_indices: (6,) indices of anchor vertices in order [+X, -X, +Y, -Y, +Z, -Z]
        channel_labels: Dict mapping channel index to axis name
    """
    if method == 'normals':
        if faces is None:
            raise ValueError("faces required for normal-based anchor selection")
        return pick_axis_aligned_anchors_by_normals(vertices, faces)
    
    elif method == 'extremes':
        # Original coordinate extremes method, but with preserved order
        indices = []
        for dim in range(3):
            indices.append(torch.argmax(vertices[:, dim]).item())  # +X, +Y, +Z
            indices.append(torch.argmin(vertices[:, dim]).item())  # -X, -Y, -Z
        
        # Preserve order and handle duplicates
        pinned_indices = unique_preserve_order(
            torch.tensor(indices, dtype=torch.long, device=vertices.device)
        )
        
        # If we have duplicates, use farthest point sampling
        if len(pinned_indices) < 6:
            all_indices = torch.arange(vertices.shape[0], device=vertices.device)
            mask = torch.ones(vertices.shape[0], dtype=torch.bool, device=vertices.device)
            mask[pinned_indices] = False
            remaining = all_indices[mask]
            
            while len(pinned_indices) < 6 and len(remaining) > 0:
                anchor_verts = vertices[pinned_indices]
                dists = torch.cdist(vertices[remaining], anchor_verts).min(dim=1)[0]
                farthest_idx = remaining[torch.argmax(dists)]
                pinned_indices = torch.cat([pinned_indices, farthest_idx.unsqueeze(0)])
                mask[farthest_idx] = False
                remaining = all_indices[mask]
        
        axis_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
        channel_labels = {i: axis_names[i] for i in range(6)}
        return pinned_indices[:6], channel_labels
    
    elif method == 'bbox_rays':
        # Cast rays from bbox center along axes
        bbox_center = vertices.mean(dim=0)
        axis_dirs = torch.tensor([[1, 0, 0], [-1, 0, 0],
                                  [0, 1, 0], [0, -1, 0], 
                                  [0, 0, 1], [0, 0, -1]], 
                                 dtype=vertices.dtype, device=vertices.device)
        
        indices = []
        for i, direction in enumerate(axis_dirs):
            # Project vertices onto this axis direction
            projections = (vertices - bbox_center) @ direction
            # Find vertex with maximum projection (furthest along ray)
            indices.append(torch.argmax(projections).item())
        
        pinned_indices = unique_preserve_order(
            torch.tensor(indices, dtype=torch.long, device=vertices.device)
        )
        
        axis_names = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
        channel_labels = {i: axis_names[i] for i in range(6)}
        return pinned_indices, channel_labels
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_mesh_statistics(verts: torch.Tensor, faces: torch.Tensor, 
                          tri_area: torch.Tensor) -> Dict[str, float]:
    """
    Compute mesh statistics for normalization and monitoring.
    
    Args:
        verts: (N, 3) vertex positions
        faces: (T, 3) face indices
        tri_area: (T,) triangle areas
        
    Returns:
        stats: Dictionary of mesh statistics
    """
    bbox_min = verts.min(dim=0).values
    bbox_max = verts.max(dim=0).values
    bbox_diag = (bbox_max - bbox_min).norm().item()
    
    # Edge lengths
    edges = []
    for i in range(3):
        j = (i + 1) % 3
        edges.append((verts[faces[:, j]] - verts[faces[:, i]]).norm(dim=-1))
    edge_lengths = torch.cat(edges)
    
    stats = {
        'bbox_diagonal': bbox_diag,
        'total_area': tri_area.sum().item(),
        'mean_edge_length': edge_lengths.mean().item(),
        'min_edge_length': edge_lengths.min().item(),
        'max_edge_length': edge_lengths.max().item(),
        'num_vertices': verts.shape[0],
        'num_faces': faces.shape[0]
    }
    
    return stats


def preprocess_mesh(vertices_np: np.ndarray, faces_np: np.ndarray, 
                   device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """
    Complete mesh preprocessing pipeline.
    
    Args:
        vertices_np: (N, 3) numpy array of vertices
        faces_np: (T, 3) numpy array of face indices
        device: PyTorch device
        
    Returns:
        Dictionary containing all preprocessed mesh data
    """
    # Convert to torch (copy arrays to avoid warnings)
    verts = torch.from_numpy(vertices_np.copy()).float().to(device)
    faces = torch.from_numpy(faces_np.copy()).long().to(device)
    
    # Compute all preprocessing
    tri_area = mesh_areas(verts, faces)
    tri_xy = triangles_to_2d(verts, faces)
    edge_idx, edge_tris = build_edges_and_adjacency(faces, verts.shape[0])
    I, J, W = cotan_weights(verts, faces)
    M = vertex_masses(verts, faces, tri_area)
    
    # Compute statistics
    stats = compute_mesh_statistics(verts, faces, tri_area)
    
    # Select anchors using normal-based method for better axis alignment
    pinned_indices, channel_labels = pick_axis_aligned_anchors(verts, faces, method='normals')
    
    return {
        'vertices': verts,
        'faces': faces,
        'tri_area': tri_area,
        'tri_xy': tri_xy,
        'edge_idx': edge_idx,
        'edge_tris': edge_tris,
        'cotan_I': I,
        'cotan_J': J,
        'cotan_W': W,
        'vertex_masses': M,
        'pinned_indices': pinned_indices,
        'channel_labels': channel_labels,
        'stats': stats
    }