"""
Free planes contour alignment loss with learnable plane parameters for channel pairs.
Uses learnable planes for each channel pair instead of individual channels.
This is computationally identical to free planes but with direct pairwise parameterization.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def contour_alignment_free_planes_pairwise(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    f_values: torch.Tensor,
    plane_normals: nn.Parameter,  # (num_pairs, 3) learnable normals for channel pairs
    plane_offsets: nn.Parameter,  # (num_pairs,) learnable offsets for channel pairs
    pinned_indices: list,         # List of pinned vertex indices
    beta_edge: float = 10.0,
    include_triples: bool = False,
    epsilon: float = 1e-6,
    robust_weight: bool = True,
    pin_weight: float = 10.0,     # Weight for pinning constraint
) -> torch.Tensor:
    """
    Vectorized contour alignment with learnable free planes for channel pairs.
    
    Instead of one plane per channel, we directly learn one plane per channel pair:
    - For channels (i,j), the decision boundary is defined by plane with normal n_{ij} and offset b_{ij}
    - Decision boundary: n_{ij}^T x + b_{ij} = 0
    - This directly models the pairwise decision boundaries
    
    This implementation is computationally identical to the free planes version,
    just with a different parameterization (15 pairwise planes instead of 6 channel planes).
    
    Args:
        vertices: (V, 3) vertex positions
        faces: (F, 3) face indices
        f_values: (V, C) multi-channel field values at vertices
        plane_normals: (num_pairs, 3) learnable plane normal vectors for channel pairs
        plane_offsets: (num_pairs,) learnable plane offset parameters for channel pairs
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
    
    # Create channel pair indices (excluding opposite pairs)
    from channelPairsConfig import get_valid_channel_pairs
    i_indices, j_indices = get_valid_channel_pairs(C)
    i_indices = i_indices.to(device)
    j_indices = j_indices.to(device)
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
    significant_pair_indices = torch.where(significant_pairs)[0]
    diff0 = diff0[:, significant_pairs]
    diff1 = diff1[:, significant_pairs]
    crossing_weights = crossing_weights[:, significant_pairs]
    num_active_pairs = len(significant_pair_indices)
    
    # Get plane parameters for active pairs directly
    # Note: Unlike free planes version, we directly use the pairwise plane parameters
    # instead of computing differences between individual channel planes
    active_plane_normals = plane_normals_normalized[significant_pair_indices]  # (num_active_pairs, 3)
    active_plane_offsets = plane_offsets[significant_pair_indices]  # (num_active_pairs,)
    
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
    distances = torch.abs(
        torch.einsum('epd,pd->ep', crossing_points, active_plane_normals) + 
        active_plane_offsets.unsqueeze(0)
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
    
    # Pinning constraint loss (vectorized)
    pinning_loss = torch.tensor(0., device=device, dtype=dtype)
    valid_pins = [(c, idx) for c, idx in enumerate(pinned_indices) if idx < V]
    if valid_pins:
        channels, indices = zip(*valid_pins)
        f_pins = f_values[list(indices)]  # (num_valid_pins, C)
        
        # Compute log softmax for all pinned vertices
        log_softmax = F.log_softmax(f_pins, dim=1)
        # Extract the log probabilities for the pinned channels
        log_probs = log_softmax[range(len(channels)), channels]
        pinning_loss = -log_probs.mean()
    
    # Return contour loss and pinning loss separately
    contour_loss_only = loss
    
    # Optional triple point regularization (vectorized)
    if include_triples:
        # Vectorized boundary score computation
        # Compute differences for all vertices and channel pairs
        f_expanded = f_values.unsqueeze(2)  # (V, C, 1)
        diffs = f_expanded[:, i_indices, 0] - f_expanded[:, j_indices, 0]  # (V, num_pairs)
        
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
    return contour_loss_only, pinning_loss


def contour_alignment_free_planes_pairwise_combined(
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
    contour_loss, pinning_loss = contour_alignment_free_planes_pairwise(
        vertices, faces, f_values, plane_normals, plane_offsets, pinned_indices,
        beta_edge, include_triples, epsilon, robust_weight, pin_weight
    )
    return contour_loss + pin_weight * pinning_loss


def init_free_plane_normals_pairwise(n_channels: int, device: torch.device, 
                                    init_scale: float = 0.1) -> nn.Parameter:
    """
    Initialize learnable plane normals for channel pairs.
    
    Args:
        n_channels: Number of channels (typically 6)
        device: Torch device
        init_scale: Scale of random initialization
        
    Returns:
        plane_normals: (num_pairs, 3) parameter tensor where num_pairs = 12 for 6 channels
    """
    from channelPairsConfig import get_num_valid_pairs
    num_pairs = get_num_valid_pairs(n_channels)
    
    if n_channels == 6:
        # Initialize with some reasonable directions for 12 valid channel pairs
        # We'll use a combination of axis-aligned and diagonal directions
        init_normals = []
        
        # Start with random directions on the unit sphere
        for _ in range(num_pairs):
            normal = torch.randn(3, device=device)
            normal = F.normalize(normal, p=2, dim=0)
            init_normals.append(normal)
        
        init_normals = torch.stack(init_normals, dim=0)  # (num_pairs, 3)
    else:
        # Random initialization on unit sphere
        init_normals = torch.randn(num_pairs, 3, device=device)
        init_normals = F.normalize(init_normals, p=2, dim=1)
    
    # Add small random perturbations
    perturbation = torch.randn_like(init_normals) * init_scale
    init_normals = init_normals + perturbation
    init_normals = F.normalize(init_normals, p=2, dim=1)
    
    return nn.Parameter(init_normals)


def init_free_plane_offsets_pairwise(n_channels: int, device: torch.device,
                                    init_scale: float = 0.1) -> nn.Parameter:
    """
    Initialize learnable plane offsets for channel pairs.
    
    Args:
        n_channels: Number of channels
        device: Torch device
        init_scale: Scale of random initialization
        
    Returns:
        plane_offsets: (num_pairs,) parameter tensor
    """
    from channelPairsConfig import get_num_valid_pairs
    num_pairs = get_num_valid_pairs(n_channels)
    init_offsets = torch.randn(num_pairs, device=device) * init_scale
    return nn.Parameter(init_offsets)