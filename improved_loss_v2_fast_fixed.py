#!/usr/bin/env python3
"""
Fixed ultra-fast fully vectorized loss function with corrected boundary weights.
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict


def compute_soft_area_fractions_fast(
    points: torch.Tensor,
    triangles: torch.Tensor,
    f_values: torch.Tensor,
    beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ultra-fast soft area fraction computation."""
    # Get triangle vertices - batch operation
    tri_points = points[triangles]  # (T, 3, 3)
    
    # Vectorized area computation
    e1 = tri_points[:, 1] - tri_points[:, 0]
    e2 = tri_points[:, 2] - tri_points[:, 0]
    areas = 0.5 * torch.norm(torch.cross(e1, e2, dim=1), dim=1)  # (T,)
    
    # Get f values at vertices and compute centers
    tri_f = f_values[triangles]  # (T, 3, C)
    f_centers = tri_f.mean(dim=1)  # (T, C)
    
    # Softmax weights
    weights = torch.softmax(beta * f_centers, dim=1)  # (T, C)
    
    # Area fractions
    weighted_areas = areas.unsqueeze(1) * weights
    p = weighted_areas.sum(dim=0) / areas.sum()
    
    return p, areas


def compute_boundary_weights_fast_fixed(
    f_values: torch.Tensor,
    edges: torch.Tensor,
    beta: float,
    num_channels: int = 6
) -> torch.Tensor:
    """
    FIXED boundary weight computation using correct formula.
    w_e = max over channel pairs of σ(-β * d_ai * d_bi)
    where d_ai = (f_i - f_j) at vertices of edge e for channel a
    """
    # Get f values at edge vertices
    edge_f = f_values[edges]  # (E, 2, C)
    f1, f2 = edge_f[:, 0], edge_f[:, 1]  # (E, C) each
    
    # Compute differences
    d = f1 - f2  # (E, C)
    
    # For efficiency, compute max weight across all channel pairs
    E = edges.shape[0]
    max_weights = torch.zeros(E, device=f_values.device)
    
    # Process each channel pair
    for i in range(num_channels):
        for j in range(i + 1, num_channels):
            # Product of differences for this channel pair
            prod_ij = d[:, i] * d[:, j]  # (E,)
            
            # Weight is high when signs are opposite (negative product)
            w_ij = torch.sigmoid(-beta * prod_ij)  # (E,)
            
            # Keep maximum weight
            max_weights = torch.maximum(max_weights, w_ij)
    
    return max_weights


def compute_triangle_gradients_batch(
    points: torch.Tensor,
    triangles: torch.Tensor,
    f_values: torch.Tensor
) -> torch.Tensor:
    """
    Compute gradients for all triangles and all channels at once.
    Returns: (T, 3, C) gradient vectors
    """
    # Get triangle vertices
    tri_points = points[triangles]  # (T, 3, 3)
    tri_f = f_values[triangles]  # (T, 3, C)
    
    # Edge vectors
    e1 = tri_points[:, 1] - tri_points[:, 0]  # (T, 3)
    e2 = tri_points[:, 2] - tri_points[:, 0]  # (T, 3)
    
    # Face normals and areas
    normals = torch.cross(e1, e2, dim=1)  # (T, 3)
    areas_2 = torch.norm(normals, dim=1, keepdim=True)  # (T, 1)
    normals = normals / (areas_2 + 1e-10)  # (T, 3)
    
    # Function differences for all channels
    df1 = tri_f[:, 1] - tri_f[:, 0]  # (T, C)
    df2 = tri_f[:, 2] - tri_f[:, 0]  # (T, C)
    
    # Compute cross products
    cross_n_e2 = torch.cross(normals.unsqueeze(2), e2.unsqueeze(2).expand(-1, -1, tri_f.shape[2]), dim=1)  # (T, 3, C)
    cross_e1_n = torch.cross(e1.unsqueeze(2).expand(-1, -1, tri_f.shape[2]), normals.unsqueeze(2), dim=1)  # (T, 3, C)
    
    # Gradients for all channels
    gradients = df1.unsqueeze(1) * cross_n_e2 + df2.unsqueeze(1) * cross_e1_n  # (T, 3, C)
    
    return gradients


def build_triangle_edge_adjacency_with_edge_mapping(triangles: torch.Tensor, edges: torch.Tensor, num_vertices: int):
    """
    Build triangle adjacency and map to edge indices.
    Returns: (adj_pairs, edge_indices) where edge_indices[k] is the edge index 
    in the edges array for the k-th adjacent pair.
    """
    device = triangles.device
    T = triangles.shape[0]
    
    # Create edge lookup dictionary
    edge_to_idx = {}
    for idx, (v1, v2) in enumerate(edges.cpu().numpy()):
        edge_to_idx[tuple(sorted([v1, v2]))] = idx
    
    # Create edge-to-triangle mapping
    edge_to_triangles = {}
    
    for t_idx in range(T):
        tri = triangles[t_idx]
        for i in range(3):
            v1, v2 = tri[i].item(), tri[(i+1)%3].item()
            edge = tuple(sorted([v1, v2]))
            
            if edge not in edge_to_triangles:
                edge_to_triangles[edge] = []
            edge_to_triangles[edge].append(t_idx)
    
    # Find adjacent triangle pairs and their shared edges
    adj_pairs = []
    edge_indices = []
    
    for edge, tri_list in edge_to_triangles.items():
        if len(tri_list) == 2 and edge in edge_to_idx:
            adj_pairs.append(tri_list)
            edge_indices.append(edge_to_idx[edge])
    
    if adj_pairs:
        return (torch.tensor(adj_pairs, device=device), 
                torch.tensor(edge_indices, device=device))
    else:
        return (torch.zeros((0, 2), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device))


def improved_loss_function_fast_fixed(
    points: torch.Tensor,
    triangles: torch.Tensor,
    f_values: torch.Tensor,
    edges: torch.Tensor,
    beta: float,
    lambda_area: float = 1.0,
    lambda_adj: float = 5.0,
    lambda_TV: float = 0.05,
    num_channels: int = 6
) -> Tuple[torch.Tensor, Dict]:
    """
    Ultra-fast loss computation with FIXED boundary weights.
    """
    device = points.device
    
    # (A) Area-balance term
    p, areas = compute_soft_area_fractions_fast(points, triangles, f_values, beta)
    target = torch.ones(num_channels, device=device) / num_channels
    area_loss = torch.norm(p - target, p=1)
    
    # (B) Boundary weights - FIXED
    w_e = compute_boundary_weights_fast_fixed(f_values, edges, beta, num_channels)
    
    # (C) Adjacent-direction term - FIXED to use proper edge weights
    adj_loss = torch.zeros(1, device=device)
    
    # Get adjacent triangle pairs WITH their shared edge indices
    num_vertices = points.shape[0]
    adj_pairs, shared_edge_indices = build_triangle_edge_adjacency_with_edge_mapping(
        triangles, edges, num_vertices
    )
    
    if len(adj_pairs) > 0:
        # Compute gradients for all channels at once
        all_adj_loss = 0.0
        count = 0
        
        for c_i in range(num_channels):
            for c_j in range(c_i + 1, num_channels):
                f_diff = f_values[:, c_i] - f_values[:, c_j]
                grads = compute_triangle_gradients_batch(points, triangles, f_diff.unsqueeze(1))[:, :, 0]  # (T, 3)
                
                # Get gradients for adjacent pairs
                t1_idx = adj_pairs[:, 0]  # (num_adj,)
                t2_idx = adj_pairs[:, 1]  # (num_adj,)
                
                g1 = grads[t1_idx]  # (num_adj, 3)
                g2 = grads[t2_idx]  # (num_adj, 3)
                
                # Normalize
                g1_norm = torch.norm(g1, dim=1, keepdim=True)  # (num_adj, 1)
                g2_norm = torch.norm(g2, dim=1, keepdim=True)  # (num_adj, 1)
                
                # Mask out small gradients
                valid_mask = (g1_norm.squeeze() > 1e-8) & (g2_norm.squeeze() > 1e-8)
                
                if valid_mask.any():
                    g1_valid = g1[valid_mask] / g1_norm[valid_mask]
                    g2_valid = g2[valid_mask] / g2_norm[valid_mask]
                    
                    # Batch dot product
                    cos_theta = torch.clamp((g1_valid * g2_valid).sum(dim=1), -1.0, 1.0)
                    
                    # Get the corresponding edge weights
                    edge_weights = w_e[shared_edge_indices[valid_mask]]
                    
                    # Weighted loss
                    all_adj_loss += torch.sum(edge_weights * (1 - cos_theta))
                    count += valid_mask.sum().item()
        
        # Normalize by number of channel pairs and measurements
        num_pairs = num_channels * (num_channels - 1) // 2
        if num_pairs > 0 and count > 0:
            adj_loss = all_adj_loss / (num_pairs * count)
    
    # (D) TV term - already correct
    edge_f = f_values[edges]  # (E, 2, C)
    f_diff = edge_f[:, 0] - edge_f[:, 1]  # (E, C)
    tv_loss = torch.sum((1 - w_e).unsqueeze(1) * f_diff.pow(2).sum(dim=1))
    
    # Combine
    total_loss = lambda_area * area_loss + lambda_adj * adj_loss + lambda_TV * tv_loss
    
    components = {
        'area': area_loss.item(),
        'adjacent': adj_loss.item(),
        'tv': tv_loss.item(),
        'area_fractions': p.detach().cpu().numpy(),
        'mean_boundary_weight': w_e.mean().item()
    }
    
    return total_loss, components