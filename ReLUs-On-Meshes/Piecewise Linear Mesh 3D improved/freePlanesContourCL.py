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
    Contour alignment with learnable free planes (not axis-aligned) but with pinned anchor points.
    
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
    
    # Get edge vertices
    v0 = vertices[edge_idx[:, 0]]  # (E, 3)
    v1 = vertices[edge_idx[:, 1]]  # (E, 3)
    edge_midpoints = 0.5 * (v0 + v1)  # (E, 3)
    
    # Get field values at edge endpoints
    f0 = f_values[edge_idx[:, 0]]  # (E, C)
    f1 = f_values[edge_idx[:, 1]]  # (E, C)
    
    # Compute all pairwise crossings
    total_loss = 0.0
    total_weight = 0.0
    
    # Process all channel pairs
    for i in range(C):
        for j in range(i+1, C):
            # Field difference at endpoints
            diff0 = f0[:, i] - f0[:, j]  # (E,)
            diff1 = f1[:, i] - f1[:, j]  # (E,)
            
            # Edge crossing weight (sigmoid of negative product)
            crossing_weight = torch.sigmoid(-beta_edge * diff0 * diff1)  # (E,)
            
            # Skip if no significant crossings
            if crossing_weight.sum() < 1e-3:
                continue
            
            # Pairwise plane parameters (derived from channel planes)
            # Plane (i,j): (n_i - n_j)^T x + (b_i - b_j) = 0
            plane_normal_ij = plane_normals_normalized[i] - plane_normals_normalized[j]  # (3,)
            plane_offset_ij = plane_offsets[i] - plane_offsets[j]  # scalar
            
            # Normalize the pairwise plane normal
            plane_normal_ij = F.normalize(plane_normal_ij, p=2, dim=0)
            
            # Linear interpolation parameter for crossing point
            # f_i(t) - f_j(t) = 0, where f(t) = (1-t)*f0 + t*f1
            t = diff0 / (diff0 - diff1 + epsilon)  # (E,)
            t = t.clamp(0.0, 1.0)
            
            # Interpolated crossing points
            crossing_points = (1 - t).unsqueeze(1) * v0 + t.unsqueeze(1) * v1  # (E, 3)
            
            # Point-to-plane distances
            distances = torch.abs(
                torch.sum(crossing_points * plane_normal_ij, dim=1) + 
                plane_offset_ij
            )  # (E,)
            
            # Robust weighting (Cauchy/Lorentzian)
            if robust_weight:
                # Cauchy weight: w = 1 / (1 + (d/scale)^2)
                scale = distances.detach().median() + epsilon
                robust_w = 1.0 / (1.0 + (distances / scale) ** 2)
                final_weight = crossing_weight * robust_w
            else:
                final_weight = crossing_weight
            
            # Edge length normalization
            edge_lengths = (v1 - v0).norm(dim=1)
            length_weight = edge_lengths / (edge_lengths.mean() + epsilon)
            final_weight = final_weight * length_weight.clamp(0.5, 2.0)
            
            # Accumulate weighted loss
            pair_loss = (final_weight * distances).sum()
            pair_weight = final_weight.sum()
            
            total_loss += pair_loss
            total_weight += pair_weight
    
    # Normalize by total weight
    if total_weight > epsilon:
        loss = total_loss / total_weight
    else:
        loss = torch.tensor(0., device=device, dtype=dtype)
    
    # Add regularizers to keep plane configuration stable
    # L_opp: Opposite pairs should be opposite
    opp_pairs = [(0, 1), (2, 3), (4, 5)]  # (Top,Bottom), (Front,Back), (Right,Left)
    L_opp = 0.0
    for a, b in opp_pairs:
        if a < C and b < C:  # Check bounds
            L_opp += (plane_normals_normalized[a] + plane_normals_normalized[b]).pow(2).sum()
            L_opp += (plane_offsets[a] + plane_offsets[b]).pow(2)
    
    # L_orth: Orthogonal pairs should be orthogonal
    # Top/Bottom (0,1) should be orthogonal to the other 4 directions
    orth_pairs = [(0, 2), (0, 3), (0, 4), (0, 5), 
                  (1, 2), (1, 3), (1, 4), (1, 5),
                  (2, 4), (2, 5), (3, 4), (3, 5)]  # X vs Y, X vs Z, Y vs Z
    L_orth = 0.0
    for a, b in orth_pairs:
        if a < C and b < C:  # Check bounds
            L_orth += (plane_normals_normalized[a] @ plane_normals_normalized[b]).pow(2)
    
    # Add regularizers with small weights
    loss = loss + 0.01 * L_opp + 0.01 * L_orth
    
    # Add pinning constraint loss
    # Ensure pinned vertices have maximum value for their assigned channel
    pinning_loss = 0.0
    for c, pin_idx in enumerate(pinned_indices):
        if pin_idx < V:  # Valid vertex index
            # Channel c should be maximum at pinned vertex
            f_pin = f_values[pin_idx]  # (C,)
            
            # Soft constraint: channel c should dominate at its pinned vertex
            # Use log-sum-exp for numerical stability
            max_val = f_pin.max()
            log_sum_exp = torch.log(torch.exp(f_pin - max_val).sum()) + max_val
            log_softmax_c = f_pin[c] - log_sum_exp
            
            # Negative log probability (want to maximize probability)
            pinning_loss = pinning_loss - log_softmax_c
    
    # Normalize by number of channels
    pinning_loss = pinning_loss / C
    
    # Combine losses
    total_loss = loss + pin_weight * pinning_loss
    
    # Optional triple point regularization
    if include_triples:
        # Find vertices with high boundary presence
        vertex_scores = torch.zeros(V, device=device)
        
        # Accumulate boundary scores
        for i in range(C):
            for j in range(i+1, C):
                diff0 = f_values[:, i] - f_values[:, j]
                # Soft boundary indicator
                boundary_prob = torch.sigmoid(-beta_edge * diff0.abs())
                vertex_scores += boundary_prob
        
        # Triple points have high scores (3+ boundaries)
        triple_mask = vertex_scores > 2.5
        
        if triple_mask.any():
            # At triple points, encourage equal channel mixing
            f_triple = f_values[triple_mask]  # (T, C)
            probs = F.softmax(f_triple, dim=1)
            
            # Entropy regularization (encourage uniform distribution)
            entropy = -(probs * (probs + epsilon).log()).sum(dim=1)
            max_entropy = torch.log(torch.tensor(float(C), device=device))
            
            # Penalty for low entropy (non-uniform distribution)
            triple_loss = (max_entropy - entropy).mean()
            total_loss = total_loss + 0.1 * triple_loss
    
    return total_loss


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