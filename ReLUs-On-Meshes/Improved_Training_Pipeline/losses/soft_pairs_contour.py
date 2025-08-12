"""
Soft all-pairs contour alignment for stable triple points.
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from .improved_losses import grad3d_intrinsic, safe_normalize


def contour_alignment_soft_pairs(F_field: torch.Tensor,
                                 faces: torch.Tensor,
                                 edge_idx: torch.Tensor,
                                 edge_tris: torch.Tensor,
                                 beta_contour: float,
                                 return_weights: bool = False,
                                 verts: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Contour alignment with soft all-pairs mixing for stable triple points.
    
    This version computes weighted averages over all channel pairs instead
    of hard-selecting the top-2, which reduces flicker at junctions.
    
    Args:
        F_field: (N, C) multi-channel field values
        faces: (T, 3) face indices
        edge_idx: (E, 2) edge vertex indices
        edge_tris: (E, 2) adjacent triangle indices (-1 for boundary)
        beta_contour: Temperature parameter for edge crossing detection
        return_weights: Whether to return edge weights
        verts: (N, 3) vertex positions in 3D (required)
        
    Returns:
        loss: Scalar contour alignment loss (normalized)
        weights: (E,) edge weights if return_weights=True
    """
    device, dtype = F_field.device, F_field.dtype
    C = F_field.shape[1]
    
    # Ensure inputs on same device
    edge_idx = edge_idx.to(device)
    edge_tris = edge_tris.to(device)
    faces = faces.to(device)
    
    if verts is None:
        raise ValueError("verts must be provided for 3D contour alignment")
    verts = verts.to(device)
    
    # Filter out boundary edges
    valid_mask = (edge_tris[:, 0] >= 0) & (edge_tris[:, 1] >= 0)
    valid_edges = edge_idx[valid_mask]  # (E_valid, 2)
    valid_tris = edge_tris[valid_mask]  # (E_valid, 2)
    
    if valid_edges.shape[0] == 0:
        if return_weights:
            return torch.tensor(0., device=device, dtype=dtype), torch.zeros(edge_idx.shape[0], device=device, dtype=dtype)
        return torch.tensor(0., device=device, dtype=dtype)
    
    # Edge midpoint field values
    f_mid = 0.5 * (F_field[valid_edges[:, 0]] + F_field[valid_edges[:, 1]])  # (E_valid, C)
    
    # Soft probabilities at midpoints (mild temperature)
    p_mid = torch.softmax(2.0 * f_mid, dim=1)  # (E_valid, C)
    
    # All channel pairs
    ii, jj = torch.triu_indices(C, C, offset=1, device=device)  # (P,) pairs
    P = len(ii)  # number of pairs
    
    # Prior weights for each pair based on midpoint probabilities
    pair_prior = p_mid[:, ii] * p_mid[:, jj]  # (E_valid, P)
    
    # Crossing weights for all pairs
    Fa = F_field[valid_edges[:, 0]][:, ii] - F_field[valid_edges[:, 0]][:, jj]  # (E_valid, P)
    Fb = F_field[valid_edges[:, 1]][:, ii] - F_field[valid_edges[:, 1]][:, jj]  # (E_valid, P)
    w_pairs = torch.sigmoid(-beta_contour * Fa * Fb) * pair_prior  # (E_valid, P)
    
    # Normalize to get pair weights
    pair_w = w_pairs / (w_pairs.sum(dim=1, keepdim=True) + 1e-9)  # (E_valid, P)
    
    # Total edge weight (for gating)
    w = w_pairs.sum(dim=1).clamp_min(1e-3)  # (E_valid,)
    
    # Get triangles
    tL = valid_tris[:, 0]  # (E_valid,)
    tR = valid_tris[:, 1]  # (E_valid,)
    
    # Get triangle vertices
    v0L = verts[faces[tL, 0]]  # (E_valid, 3)
    v1L = verts[faces[tL, 1]]
    v2L = verts[faces[tL, 2]]
    
    v0R = verts[faces[tR, 0]]
    v1R = verts[faces[tR, 1]]
    v2R = verts[faces[tR, 2]]
    
    # Field values at triangle vertices
    faces_L = faces[tL]  # (E_valid, 3)
    faces_R = faces[tR]
    F_L = F_field[faces_L]  # (E_valid, 3, C)
    F_R = F_field[faces_R]
    
    # Height differences for all pairs
    hL_pairs = F_L[..., ii] - F_L[..., jj]  # (E_valid, 3, P)
    hR_pairs = F_R[..., ii] - F_R[..., jj]  # (E_valid, 3, P)
    
    # Weighted combination of height values
    h_L = torch.einsum('ep,evp->ev', pair_w, hL_pairs)  # (E_valid, 3)
    h_R = torch.einsum('ep,evp->ev', pair_w, hR_pairs)  # (E_valid, 3)
    
    # Compute intrinsic 3D gradients
    g_L_3d = grad3d_intrinsic(h_L, v0L, v1L, v2L)  # (E_valid, 3)
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
    edge_vec = verts[valid_edges[:, 1]] - verts[valid_edges[:, 0]]
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
    
    if return_weights:
        # Return full edge weights
        full_weights = torch.zeros(edge_idx.shape[0], device=device, dtype=dtype)
        valid_indices = torch.where(valid_mask)[0]
        full_weights[valid_indices] = w
        return total_loss, full_weights
    
    return total_loss