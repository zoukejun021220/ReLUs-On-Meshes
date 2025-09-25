"""
Soft all-pairs contour alignment loss for stable triple points.
Adapted from the improved training pipeline for the 3D mesh segmentation.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    Safe normalization that avoids NaN gradients when x≈0.
    
    Args:
        x: Input tensor to normalize
        dim: Dimension along which to normalize
        eps: Small value to clamp the norm (default 1e-6)
        
    Returns:
        Normalized tensor with stable gradients
    """
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def grad3d_intrinsic(h_vals: torch.Tensor, v0: torch.Tensor,
                     v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Compute intrinsic gradient on triangles using Gram matrix (batched).
    
    Args:
        h_vals: (B,3) scalar values at triangle vertices
        v0,v1,v2: (B,3) triangle vertex positions
        
    Returns:
        g: (B,3) gradient in R^3 lying in the triangle plane
    """
    e0 = v1 - v0                      # (B,3)
    e1 = v2 - v0                      # (B,3)
    b = torch.stack([h_vals[:,1]-h_vals[:,0],
                     h_vals[:,2]-h_vals[:,0]], dim=1)  # (B,2)

    # Do geometry in float64 for robustness, cast back at the end
    dt = torch.float64
    e0d, e1d, bd = e0.to(dt), e1.to(dt), b.to(dt)

    # Gram matrix G = [[<e0,e0>, <e0,e1>],
    #                  [<e1,e0>, <e1,e1>]]
    G00 = (e0d*e0d).sum(dim=1)          # (B,)
    G01 = (e0d*e1d).sum(dim=1)          # (B,)
    G11 = (e1d*e1d).sum(dim=1)          # (B,)
    det = (G00*G11 - G01*G01)
    
    # Mask degenerate triangles
    mask_degenerate = det <= 1e-10
    det = det.clamp_min(1e-10)
    
    invG00 =  G11 / det
    invG01 = -G01 / det
    invG11 =  G00 / det

    # coefficients a in the (e0,e1) basis: a = G^{-1} b
    a0 = invG00*bd[:,0] + invG01*bd[:,1]    # (B,)
    a1 = invG01*bd[:,0] + invG11*bd[:,1]    # (B,)
    
    # Zero out coefficients for degenerate triangles (no reliable gradient)
    a0 = a0.masked_fill(mask_degenerate, 0.0)
    a1 = a1.masked_fill(mask_degenerate, 0.0)

    # gradient in R^3: g = a0*e0 + a1*e1
    gd = a0[:,None]*e0d + a1[:,None]*e1d     # (B,3)
    g = gd.to(h_vals.dtype)
    
    # Guard: if anything still slipped through, zero it (rare)
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


def contour_alignment_soft_pairs(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    f_values: torch.Tensor,
    pinned_axes: torch.Tensor,
    plane_offsets: torch.Tensor,
    beta_edge: float = 10.0,
    include_triples: bool = False,
    return_weights: bool = False,
) -> torch.Tensor:
    """
    Contour alignment with soft all-pairs mixing for stable triple points.
    
    This version computes weighted averages over all channel pairs instead
    of hard-selecting the top-2, which reduces flicker at junctions.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        f_values: (V, C) multi-channel field values at vertices
        pinned_axes: (C, 3) fixed channel plane normals
        plane_offsets: (C,) plane offset parameters
        beta_edge: Temperature parameter for edge crossing detection
        include_triples: Whether to include triple point terms
        return_weights: Whether to return edge weights
        
    Returns:
        loss: Scalar contour alignment loss (normalized)
        weights: (E,) edge weights if return_weights=True
    """
    device = vertices.device
    dtype = vertices.dtype
    V, C = f_values.shape
    F = faces.shape[0]
    
    # Build edge adjacency using fully vectorized operations
    # Create all edges from faces (3 edges per face)
    edges = torch.stack([
        torch.stack([faces[:, 0], faces[:, 1]], dim=1),  # edge 0-1
        torch.stack([faces[:, 1], faces[:, 2]], dim=1),  # edge 1-2
        torch.stack([faces[:, 2], faces[:, 0]], dim=1),  # edge 2-0
    ], dim=1).reshape(-1, 2)  # (3*F, 2)
    
    # Sort vertices in each edge to ensure consistent ordering
    edges_sorted, _ = torch.sort(edges, dim=1)  # (3*F, 2)
    
    # Create face indices for each edge
    face_indices = torch.arange(F, device=device).repeat_interleave(3)  # (3*F,)
    
    # Use a more efficient approach with sparse tensors
    # Create edge hash
    edge_hash = edges_sorted[:, 0] * V + edges_sorted[:, 1]  # (3*F,)
    
    # Sort by edge hash to group same edges together
    sorted_hash, sort_idx = torch.sort(edge_hash)
    sorted_faces = face_indices[sort_idx]
    sorted_edges = edges_sorted[sort_idx]
    
    # Find where edges change
    edge_changes = torch.cat([
        torch.tensor([True], device=device),
        sorted_hash[1:] != sorted_hash[:-1],
        torch.tensor([True], device=device)
    ])
    change_indices = torch.where(edge_changes)[0]
    
    # Count edges per unique edge
    edge_counts = change_indices[1:] - change_indices[:-1]
    
    # Find internal edges (count == 2)
    internal_mask = edge_counts == 2
    internal_indices = torch.where(internal_mask)[0]
    
    if len(internal_indices) == 0:
        if return_weights:
            return torch.tensor(0., device=device, dtype=dtype), torch.zeros(0, device=device, dtype=dtype)
        return torch.tensor(0., device=device, dtype=dtype)
    
    # Extract internal edges and their triangles using vectorized indexing
    E = len(internal_indices)
    
    # Get start indices for internal edges
    start_indices = change_indices[internal_indices]
    
    # Extract edges
    edge_idx = sorted_edges[start_indices]  # (E, 2)
    
    # Extract triangle pairs
    edge_tris = torch.stack([
        sorted_faces[start_indices],      # First triangle
        sorted_faces[start_indices + 1]   # Second triangle
    ], dim=1)  # (E, 2)
    
    # Edge midpoint field values
    f_mid = 0.5 * (f_values[edge_idx[:, 0]] + f_values[edge_idx[:, 1]])  # (E, C)
    
    # Soft probabilities at midpoints (mild temperature)
    p_mid = torch.softmax(2.0 * f_mid, dim=1)  # (E, C)
    
    # All channel pairs
    ii, jj = torch.triu_indices(C, C, offset=1, device=device)  # (P,) pairs
    P = len(ii)  # number of pairs
    
    # Prior weights for each pair based on midpoint probabilities
    pair_prior = p_mid[:, ii] * p_mid[:, jj]  # (E, P)
    
    # Crossing weights for all pairs
    Fa = f_values[edge_idx[:, 0]][:, ii] - f_values[edge_idx[:, 0]][:, jj]  # (E, P)
    Fb = f_values[edge_idx[:, 1]][:, ii] - f_values[edge_idx[:, 1]][:, jj]  # (E, P)
    w_pairs = torch.sigmoid(-beta_edge * Fa * Fb) * pair_prior  # (E, P)
    
    # Normalize to get pair weights
    pair_w = w_pairs / (w_pairs.sum(dim=1, keepdim=True) + 1e-9)  # (E, P)
    
    # Total edge weight (for gating)
    w = w_pairs.sum(dim=1).clamp_min(1e-3)  # (E,)
    
    # Get triangles
    tL = edge_tris[:, 0]  # (E,)
    tR = edge_tris[:, 1]  # (E,)
    
    # Get triangle vertices
    v0L = vertices[faces[tL, 0]]  # (E, 3)
    v1L = vertices[faces[tL, 1]]
    v2L = vertices[faces[tL, 2]]
    
    v0R = vertices[faces[tR, 0]]
    v1R = vertices[faces[tR, 1]]
    v2R = vertices[faces[tR, 2]]
    
    # Field values at triangle vertices
    faces_L = faces[tL]  # (E, 3)
    faces_R = faces[tR]
    F_L = f_values[faces_L]  # (E, 3, C)
    F_R = f_values[faces_R]
    
    # Height differences for all pairs
    hL_pairs = F_L[..., ii] - F_L[..., jj]  # (E, 3, P)
    hR_pairs = F_R[..., ii] - F_R[..., jj]  # (E, 3, P)
    
    # Weighted combination of height values
    h_L = torch.einsum('ep,evp->ev', pair_w, hL_pairs)  # (E, 3)
    h_R = torch.einsum('ep,evp->ev', pair_w, hR_pairs)  # (E, 3)
    
    # Compute intrinsic 3D gradients
    g_L_3d = grad3d_intrinsic(h_L, v0L, v1L, v2L)  # (E, 3)
    g_R_3d = grad3d_intrinsic(h_R, v0R, v1R, v2R)
    
    # Triangle normals using safe normalization
    e0L = v1L - v0L
    e1L = v2L - v0L
    n_L = safe_normalize(torch.cross(e0L, e1L, dim=1), dim=1, eps=1e-6)
    
    e0R = v1R - v0R
    e1R = v2R - v0R
    n_R = safe_normalize(torch.cross(e0R, e1R, dim=1), dim=1, eps=1e-6)
    
    # Project gradients into the triangle plane first (improves stability)
    g_L_3d = g_L_3d - (g_L_3d * n_L).sum(dim=1, keepdim=True) * n_L
    g_R_3d = g_R_3d - (g_R_3d * n_R).sum(dim=1, keepdim=True) * n_R
    
    # Boundary tangents using safe normalization
    tau_L = safe_normalize(torch.cross(n_L, g_L_3d, dim=1), dim=1, eps=1e-6)
    tau_R = safe_normalize(torch.cross(n_R, g_R_3d, dim=1), dim=1, eps=1e-6)
    
    # Gradient magnitude gating
    mag_L = g_L_3d.norm(dim=1)
    mag_R = g_R_3d.norm(dim=1)
    grad_gate = torch.sqrt(mag_L * mag_R)
    scale = grad_gate.median().detach() + 1e-12
    grad_gate = (grad_gate / scale).clamp(0.0, 3.0)
    
    # Edge length gating
    edge_vec = vertices[edge_idx[:, 1]] - vertices[edge_idx[:, 0]]
    edge_len = edge_vec.norm(dim=1)
    len_gate = (edge_len / (edge_len.mean().detach() + 1e-12)).clamp(0.5, 2.0)
    
    # Apply gates
    w = w * grad_gate * len_gate
    
    # Alignment loss
    cos_angle = (tau_L * tau_R).sum(dim=1).abs().clamp(max=1.0)
    misalignment = 1.0 - cos_angle
    
    # Charbonnier penalty
    epsilon = 1e-6
    loss_contrib = torch.sqrt(misalignment * misalignment + epsilon)
    
    # Weighted mean
    total_loss = (w * loss_contrib).sum() / (w.sum() + 1e-9)
    
    # Triple point loss (optional, fully vectorized)
    if include_triples:
        # Accumulate boundary strength per vertex by adding each edge's weight to both endpoints
        vertex_boundary_count = torch.zeros(V, device=device, dtype=w.dtype)
        vertex_boundary_count.index_add_(0, edge_idx[:, 0], w)
        vertex_boundary_count.index_add_(0, edge_idx[:, 1], w)

        triple_verts = torch.where(vertex_boundary_count >= 2.5)[0]

        if triple_verts.numel() > 0:
            # At triple points, encourage equal mixing
            f_triple = f_values[triple_verts]  # (T, C)
            p_triple = torch.softmax(f_triple, dim=1)

            # Entropy regularization
            entropy = -(p_triple * (p_triple + 1e-9).log()).sum(dim=1)
            max_entropy = torch.log(torch.tensor(3.0, device=device))  # log(3) for 3 channels
            triple_loss = (max_entropy - entropy).mean()

            total_loss = total_loss + 0.1 * triple_loss
    
    if return_weights:
        return total_loss, w
    
    return total_loss
