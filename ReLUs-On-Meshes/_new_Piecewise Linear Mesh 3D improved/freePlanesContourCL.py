"""
Free planes contour alignment loss with learnable plane parameters.
Uses 6 learnable planes (one per channel) for better stability and global consistency.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def contour_alignment_free_planes(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    f_values: torch.Tensor,
    plane_normals: nn.Parameter,  # (C, 3) learnable normals
    plane_offsets: nn.Parameter,  # (C,) learnable offsets
    pinned_indices: list,         # List of pinned vertex indices
    beta_edge: float = 10.0,
    include_triples: bool = False,
    epsilon: float = 1e-6,
    robust_weight: bool = True,
    pin_weight: float = 10.0,     # Weight for pinning constraint
) -> torch.Tensor:
    """
    Vectorized contour alignment with learnable free planes (not axis-aligned) but with pinned anchor points.
    
    Similar to anchored planes but learns plane orientations instead of using fixed axes:
    - Plane for channel c: n_c^T x + b_c = 0 (learnable n_c and b_c)
    - Maintains pinning: f_values[pinned_indices[c], c] should be maximum for channel c
    - Pairwise decision boundary (i,j): (n_i - n_j)^T x + (b_i - b_j) = 0
    
    This combines the stability of anchor points with the flexibility of free plane orientations.
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        f_values: (V, C) multi-channel field values at vertices
        plane_normals: (C, 3) learnable plane normal vectors (will be normalized)
        plane_offsets: (C,) learnable plane offset parameters
        pinned_indices: List of C vertex indices to pin (one per channel)
        beta_edge: Temperature parameter for edge crossing detection
        include_triples: Whether to include triple point regularization
        epsilon: Small value for numerical stability
        robust_weight: Use Cauchy weights for outlier robustness
        pin_weight: Weight for enforcing pinning constraints
        
    Returns:
        loss: Scalar contour alignment loss + pinning loss
    """
    device = vertices.device
    dtype = vertices.dtype
    V, C = f_values.shape
    num_faces = faces.shape[0]
    
    # Normalize plane normals to unit length
    plane_normals_normalized = F.normalize(plane_normals, p=2, dim=1)
    
    # Build edge adjacency using fully vectorized operations
    # Create all edges from faces (3 edges per face)
    edges = torch.stack([
        torch.stack([faces[:, 0], faces[:, 1]], dim=1),  # edge 0-1
        torch.stack([faces[:, 1], faces[:, 2]], dim=1),  # edge 1-2
        torch.stack([faces[:, 2], faces[:, 0]], dim=1),  # edge 2-0
    ], dim=1).reshape(-1, 2)  # (3*num_faces, 2)
    
    # Sort vertices in each edge
    edges_sorted, _ = torch.sort(edges, dim=1)
    
    # Create face indices for each edge
    face_indices = torch.arange(num_faces, device=device).repeat_interleave(3)
    
    # Find unique edges efficiently
    edge_hash = edges_sorted[:, 0] * V + edges_sorted[:, 1]
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
        return torch.tensor(0., device=device, dtype=dtype)
    
    # Extract internal edges
    E = len(internal_indices)
    start_indices = change_indices[internal_indices]
    edge_idx = sorted_edges[start_indices]  # (E, 2)
    
    # Get edge vertices and field values
    v0 = vertices[edge_idx[:, 0]]  # (E, 3)
    v1 = vertices[edge_idx[:, 1]]  # (E, 3)
    f0 = f_values[edge_idx[:, 0]]  # (E, C)
    f1 = f_values[edge_idx[:, 1]]  # (E, C)
    
    # Create channel pair indices
    i_indices, j_indices = torch.triu_indices(C, C, offset=1, device=device)
    num_pairs = len(i_indices)
    
    # Vectorized computation for all channel pairs
    # Field differences at endpoints for all pairs
    diff0 = f0[:, i_indices] - f0[:, j_indices]  # (E, num_pairs)
    diff1 = f1[:, i_indices] - f1[:, j_indices]  # (E, num_pairs)
    
    # Edge crossing weights for all pairs
    crossing_weights = torch.sigmoid(-beta_edge * diff0 * diff1)  # (E, num_pairs)
    
    # Filter out pairs with negligible crossings
    significant_pairs = crossing_weights.sum(dim=0) > 1e-3
    if not significant_pairs.any():
        return torch.tensor(0., device=device, dtype=dtype)
    
    # Work only with significant pairs
    i_indices = i_indices[significant_pairs]
    j_indices = j_indices[significant_pairs]
    diff0 = diff0[:, significant_pairs]
    diff1 = diff1[:, significant_pairs]
    crossing_weights = crossing_weights[:, significant_pairs]
    num_active_pairs = len(i_indices)
    
    # Pairwise plane parameters (vectorized)
    plane_normals_i = plane_normals_normalized[i_indices]  # (num_active_pairs, 3)
    plane_normals_j = plane_normals_normalized[j_indices]  # (num_active_pairs, 3)
    plane_normals_ij = plane_normals_i - plane_normals_j  # (num_active_pairs, 3)
    plane_normals_ij = F.normalize(plane_normals_ij, p=2, dim=1)
    
    plane_offsets_ij = plane_offsets[i_indices] - plane_offsets[j_indices]  # (num_active_pairs,)
    
    # Linear interpolation parameters for all pairs
    t = diff0 / (diff0 - diff1 + epsilon)  # (E, num_active_pairs)
    t = t.clamp(0.0, 1.0)
    
    # Compute crossing points for all edges and pairs
    # Expand vertices for broadcasting
    v0_expanded = v0.unsqueeze(1)  # (E, 1, 3)
    v1_expanded = v1.unsqueeze(1)  # (E, 1, 3)
    t_expanded = t.unsqueeze(2)  # (E, num_active_pairs, 1)
    
    crossing_points = (1 - t_expanded) * v0_expanded + t_expanded * v1_expanded  # (E, num_active_pairs, 3)
    
    # Point-to-plane distances for all pairs (batch computation)
    # crossing_points: (E, num_active_pairs, 3)
    # plane_normals_ij: (num_active_pairs, 3)
    distances = torch.abs(
        torch.einsum('epd,pd->ep', crossing_points, plane_normals_ij) + 
        plane_offsets_ij.unsqueeze(0)
    )  # (E, num_active_pairs)
    
    # Robust weighting (vectorized)
    if robust_weight:
        # Compute scales per pair
        scales = distances.detach().median(dim=0).values + epsilon  # (num_active_pairs,)
        robust_w = 1.0 / (1.0 + (distances / scales.unsqueeze(0)) ** 2)
        final_weights = crossing_weights * robust_w
    else:
        final_weights = crossing_weights
    
    # Edge length normalization (vectorized)
    edge_lengths = (v1 - v0).norm(dim=1)  # (E,)
    mean_length = edge_lengths.mean()
    length_weights = (edge_lengths / (mean_length + epsilon)).clamp(0.5, 2.0)
    final_weights = final_weights * length_weights.unsqueeze(1)
    
    # Compute total loss
    weighted_distances = final_weights * distances
    total_loss = weighted_distances.sum()
    total_weight = final_weights.sum()
    
    # Normalize by total weight
    if total_weight > epsilon:
        loss = total_loss / total_weight
    else:
        loss = torch.tensor(0., device=device, dtype=dtype)
    
    # Add regularizers (vectorized)
    # L_opp: Opposite pairs should be opposite
    opp_pairs = torch.tensor([(0, 1), (2, 3), (4, 5)], device=device)
    valid_opp = (opp_pairs < C).all(dim=1)
    if valid_opp.any():
        opp_pairs = opp_pairs[valid_opp]
        L_opp = (plane_normals_normalized[opp_pairs[:, 0]] + 
                 plane_normals_normalized[opp_pairs[:, 1]]).pow(2).sum()
        L_opp += (plane_offsets[opp_pairs[:, 0]] + 
                  plane_offsets[opp_pairs[:, 1]]).pow(2).sum()
    else:
        L_opp = torch.tensor(0., device=device, dtype=dtype)
    
    # L_orth: Orthogonal pairs should be orthogonal (vectorized)
    orth_pairs = torch.tensor([
        (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 2), (1, 3), (1, 4), (1, 5),
        (2, 4), (2, 5), (3, 4), (3, 5)
    ], device=device)
    valid_orth = (orth_pairs < C).all(dim=1)
    if valid_orth.any():
        orth_pairs = orth_pairs[valid_orth]
        dot_products = torch.einsum('nd,nd->n',
            plane_normals_normalized[orth_pairs[:, 0]],
            plane_normals_normalized[orth_pairs[:, 1]]
        )
        L_orth = dot_products.pow(2).sum()
    else:
        L_orth = torch.tensor(0., device=device, dtype=dtype)
    
    # Add regularizers with small weights
    loss = loss + 0.01 * L_opp + 0.01 * L_orth
    
    # Pinning constraint loss (vectorized)
    # Vectorized pinning loss (no Python loops)
    pinning_loss = torch.tensor(0., device=device, dtype=dtype)
    pins = torch.as_tensor(pinned_indices, device=device, dtype=torch.long)
    # Valid pins are within [0, V)
    valid_mask = (pins >= 0) & (pins < V)
    if valid_mask.any():
        valid_idx = pins[valid_mask]
        valid_ch = torch.arange(C, device=device, dtype=torch.long)[valid_mask]
        f_pins = f_values[valid_idx]  # (num_valid_pins, C)
        log_softmax = F.log_softmax(f_pins, dim=1)
        log_probs = log_softmax[torch.arange(valid_ch.numel(), device=device), valid_ch]
        pinning_loss = -log_probs.mean()
    
    # Return contour loss and pinning loss separately
    # The caller can decide how to weight them
    contour_loss_only = loss
    
    # Optional triple point regularization (vectorized)
    if include_triples:
        # Vectorized boundary score computation
        # Create masks for all channel pairs
        i_all, j_all = torch.triu_indices(C, C, offset=1, device=device)
        
        # Compute differences for all vertices and channel pairs
        f_expanded = f_values.unsqueeze(2)  # (V, C, 1)
        diffs = f_expanded[:, i_all, 0] - f_expanded[:, j_all, 0]  # (V, num_pairs)
        
        # Boundary probabilities
        boundary_probs = torch.sigmoid(-beta_edge * diffs.abs())  # (V, num_pairs)
        vertex_scores = boundary_probs.sum(dim=1)  # (V,)
        
        # Triple points have high scores
        triple_mask = vertex_scores > 2.5
        
        if triple_mask.any():
            f_triple = f_values[triple_mask]  # (T, C)
            probs = F.softmax(f_triple, dim=1)
            
            # Entropy regularization
            entropy = -(probs * (probs + epsilon).log()).sum(dim=1)
            max_entropy = torch.log(torch.tensor(float(C), device=device))
            
            triple_loss = (max_entropy - entropy).mean()
            contour_loss_only = contour_loss_only + 0.1 * triple_loss
    
    # Return both losses separately to allow proper weighting
    # The pinning loss should not be scaled by lambda_c
    return contour_loss_only, pinning_loss


def contour_alignment_free_planes_combined(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    f_values: torch.Tensor,
    plane_normals: nn.Parameter,
    plane_offsets: nn.Parameter,
    pinned_indices: list,
    beta_edge: float = 10.0,
    include_triples: bool = False,
    epsilon: float = 1e-6,
    robust_weight: bool = True,
    pin_weight: float = 10.0,
) -> torch.Tensor:
    """
    Backward compatible wrapper that returns combined loss.
    """
    contour_loss, pinning_loss = contour_alignment_free_planes(
        vertices, faces, f_values, plane_normals, plane_offsets, pinned_indices,
        beta_edge, include_triples, epsilon, robust_weight, pin_weight
    )
    return contour_loss + pin_weight * pinning_loss


def init_free_plane_normals(n_channels: int, device: torch.device, 
                           init_scale: float = 0.1,
                           pinned_axes: Optional[np.ndarray] = None) -> nn.Parameter:
    """
    Initialize learnable plane normals with small random perturbations.
    
    Args:
        n_channels: Number of channels (typically 6)
        device: Torch device
        init_scale: Scale of random initialization
        pinned_axes: Optional (C, 3) initial axes from PCA or world axes
        
    Returns:
        plane_normals: (C, 3) parameter tensor
    """
    if pinned_axes is not None:
        # Initialize from provided axes (e.g., from PCA)
        init_normals = torch.from_numpy(pinned_axes).float().to(device)
    elif n_channels == 6:
        # Initialize near the 6 main directions
        init_normals = torch.tensor([
            [1.0, 0.0, 0.0],   # +X
            [-1.0, 0.0, 0.0],  # -X
            [0.0, 1.0, 0.0],   # +Y
            [0.0, -1.0, 0.0],  # -Y
            [0.0, 0.0, 1.0],   # +Z
            [0.0, 0.0, -1.0],  # -Z
        ], device=device, dtype=torch.float32)
    else:
        # Random initialization on unit sphere
        init_normals = torch.randn(n_channels, 3, device=device)
        init_normals = F.normalize(init_normals, p=2, dim=1)
    
    # Add small random perturbations
    perturbation = torch.randn_like(init_normals) * init_scale
    init_normals = init_normals + perturbation
    init_normals = F.normalize(init_normals, p=2, dim=1)
    
    return nn.Parameter(init_normals)
