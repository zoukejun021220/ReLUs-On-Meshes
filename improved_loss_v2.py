#!/usr/bin/env python3
"""
Improved loss function implementation for ReLU mesh optimization.
Based on the three-term formulation:
L = λ_area * ||p(β) - 1/C||_1 + λ_adj * Σ w_e (1-cos θ_e) + λ_TV * Σ (1-w_e) ||f_i - f_j||_2^2
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def compute_soft_area_fractions(
    points: torch.Tensor,
    triangles: torch.Tensor,
    f_values: torch.Tensor,
    beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute soft area fractions p(β) for each channel.
    
    Args:
        points: (N, 3) vertex positions
        triangles: (T, 3) triangle indices
        f_values: (N, C) field values at vertices
        beta: Softmax temperature
    
    Returns:
        p: (C,) soft area fractions for each channel
        triangle_areas: (T,) areas of each triangle
    """
    device = points.device
    num_channels = f_values.shape[1]
    
    # Barycentric sampling points (4 samples per triangle)
    bary_points = torch.tensor([
        [1/3, 1/3, 1/3],  # Center
        [0.5, 0.5, 0.0],  # Edge midpoints
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ], device=device)
    
    # Get triangle vertices
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    p0, p1, p2 = points[v0], points[v1], points[v2]
    
    # Compute triangle areas
    e1, e2 = p1 - p0, p2 - p0
    normals = torch.cross(e1, e2, dim=1)
    triangle_areas = 0.5 * torch.norm(normals, dim=1)
    total_area = triangle_areas.sum()
    
    # Get field values at vertices
    f0, f1, f2 = f_values[v0], f_values[v1], f_values[v2]
    
    # Interpolate field at barycentric points
    f_interp = []
    for bary in bary_points:
        f_bary = bary[0] * f0 + bary[1] * f1 + bary[2] * f2
        f_interp.append(f_bary)
    f_interp = torch.stack(f_interp, dim=1)  # (T, S, C)
    
    # Apply softmax with beta
    f_interp_beta = beta * f_interp
    probs = torch.softmax(f_interp_beta, dim=2)  # (T, S, C)
    
    # Average over sample points
    probs_mean = probs.mean(dim=1)  # (T, C)
    
    # Weight by triangle areas and normalize
    weighted_areas = probs_mean * triangle_areas.unsqueeze(1)
    channel_areas = weighted_areas.sum(dim=0)  # (C,)
    p = channel_areas / total_area
    
    return p, triangle_areas


def compute_boundary_weights(
    f_values: torch.Tensor,
    edges: torch.Tensor,
    beta: float
) -> torch.Tensor:
    """
    Compute boundary weights w_e for each edge.
    w_e = σ(-β * d_ai * d_bi) where d = f_i - f_j at edge endpoints
    
    Args:
        f_values: (N, C) field values at vertices
        edges: (E, 2) edge vertex indices
        beta: Temperature parameter
    
    Returns:
        w_e: (E,) boundary weights for each edge
    """
    v1_idx, v2_idx = edges[:, 0], edges[:, 1]
    f1, f2 = f_values[v1_idx], f_values[v2_idx]
    
    # Compute channel-wise differences
    d = f1 - f2  # (E, C)
    
    # For each edge, compute product of differences across channels
    # This gives high values when vertices are on opposite sides of multiple boundaries
    d_prod = torch.prod(d, dim=1)  # (E,)
    
    # Apply sigmoid to get boundary weights
    # Negative sign ensures w_e ≈ 1 when d_prod is negative (opposite sides)
    w_e = torch.sigmoid(-beta * d_prod)
    
    return w_e


def compute_face_gradients(
    points: torch.Tensor,
    triangles: torch.Tensor,
    f_diff: torch.Tensor
) -> torch.Tensor:
    """
    Compute gradients of field differences on triangle faces.
    
    Args:
        points: (N, 3) vertex positions
        triangles: (T, 3) triangle indices
        f_diff: (N,) field difference values at vertices
    
    Returns:
        gradients: (T, 3) gradient vectors on each face
    """
    # Get triangle vertices
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    p0, p1, p2 = points[v0], points[v1], points[v2]
    
    # Edge vectors
    e1, e2 = p1 - p0, p2 - p0
    
    # Field differences at vertices
    f0, f1, f2 = f_diff[v0], f_diff[v1], f_diff[v2]
    df1, df2 = f1 - f0, f2 - f0
    
    # Compute face normals
    normals = torch.cross(e1, e2, dim=1)
    areas = 0.5 * torch.norm(normals, dim=1, keepdim=True)
    normals = normals / (2 * areas + 1e-8)
    
    # Project gradients using linear solve
    # grad = (df1 * cross(n, e2) + df2 * cross(e1, n)) / (2 * area)
    grad = (df1.unsqueeze(1) * torch.cross(normals, e2, dim=1) + 
            df2.unsqueeze(1) * torch.cross(e1, normals, dim=1))
    
    return grad


def improved_loss_function(
    points: torch.Tensor,
    triangles: torch.Tensor,
    f_values: torch.Tensor,
    edges: torch.Tensor,
    triangle_adjacency: torch.Tensor,
    beta: float,
    lambda_area: float = 1.0,
    lambda_adj: float = 5.0,
    lambda_TV: float = 0.05,
    num_channels: int = 6
) -> Tuple[torch.Tensor, dict]:
    """
    Compute the improved three-term loss function.
    
    Args:
        points: (N, 3) vertex positions
        triangles: (T, 3) triangle indices
        f_values: (N, C) field values at vertices
        edges: (E, 2) edge vertex indices
        triangle_adjacency: (T, T) sparse adjacency matrix of triangles
        beta: Softmax temperature
        lambda_area: Weight for area balance term
        lambda_adj: Weight for adjacent direction term
        lambda_TV: Weight for gated total variation term
        num_channels: Number of channels (default 6)
    
    Returns:
        total_loss: Combined loss value
        loss_components: Dictionary with individual loss terms
    """
    device = points.device
    
    # (A) Area-balance term
    p, _ = compute_soft_area_fractions(points, triangles, f_values, beta)
    target = torch.ones(num_channels, device=device) / num_channels
    area_loss = torch.norm(p - target, p=1)
    
    # Compute boundary weights (used in both B and C)
    w_e = compute_boundary_weights(f_values, edges, beta)
    
    # (B) Adjacent-direction term (local planarity surrogate)
    adj_loss = torch.zeros(1, device=device)
    angle_count = 0
    
    # Find edges that connect adjacent triangles
    edge_to_triangles = {}
    for t_idx, tri in enumerate(triangles):
        for i in range(3):
            edge = tuple(sorted([tri[i].item(), tri[(i+1)%3].item()]))
            if edge not in edge_to_triangles:
                edge_to_triangles[edge] = []
            edge_to_triangles[edge].append(t_idx)
    
    # For each channel pair, compute gradient alignment
    for c_i in range(num_channels):
        for c_j in range(c_i + 1, num_channels):
            f_diff = f_values[:, c_i] - f_values[:, c_j]
            gradients = compute_face_gradients(points, triangles, f_diff)
            
            # For each edge between triangles
            for edge_idx, (v1, v2) in enumerate(edges):
                edge = tuple(sorted([v1.item(), v2.item()]))
                if edge in edge_to_triangles and len(edge_to_triangles[edge]) == 2:
                    t1, t2 = edge_to_triangles[edge]
                    
                    # Get gradients on adjacent faces
                    g1, g2 = gradients[t1], gradients[t2]
                    
                    # Normalize gradients
                    g1_norm = torch.norm(g1)
                    g2_norm = torch.norm(g2)
                    
                    if g1_norm > 1e-8 and g2_norm > 1e-8:
                        # Compute angle between gradients
                        cos_theta = torch.clamp(
                            torch.dot(g1, g2) / (g1_norm * g2_norm),
                            -1.0, 1.0
                        )
                        
                        # Add weighted penalty (1 - cos θ)
                        adj_loss += w_e[edge_idx] * (1 - cos_theta)
                        angle_count += 1
    
    # Normalize by number of measurements
    if angle_count > 0:
        adj_loss = adj_loss / angle_count
    
    # (C) Gated total-variation term
    v1_idx, v2_idx = edges[:, 0], edges[:, 1]
    f1, f2 = f_values[v1_idx], f_values[v2_idx]
    f_diff = f1 - f2  # (E, C)
    
    # Apply gate: (1 - w_e) to exclude boundary edges
    tv_loss = torch.sum((1 - w_e).unsqueeze(1) * torch.sum(f_diff**2, dim=1))
    
    # Combine losses
    total_loss = (
        lambda_area * area_loss +
        lambda_adj * adj_loss +
        lambda_TV * tv_loss
    )
    
    # Return detailed components for monitoring
    loss_components = {
        'area': area_loss.item(),
        'adjacent': adj_loss.item(),
        'tv': tv_loss.item(),
        'total': total_loss.item(),
        'area_fractions': p.detach().cpu().numpy(),
        'mean_boundary_weight': w_e.mean().item()
    }
    
    return total_loss, loss_components


def get_beta_schedule(iteration: int, total_iterations: int, 
                     beta_start: float = 2.0, beta_end: float = 25.0,
                     warmup_fraction: float = 0.2) -> float:
    """
    Get beta value according to schedule.
    Linear ramp from beta_start to beta_end over warmup_fraction of iterations.
    """
    t = iteration / total_iterations
    
    if t < warmup_fraction:
        # Linear ramp during warmup
        return beta_start + (beta_end - beta_start) * (t / warmup_fraction)
    else:
        # Hold constant after warmup
        return beta_end


def get_lambda_schedule(iteration: int, total_iterations: int,
                       lambda_adj_start: float = 0.0, lambda_adj_end: float = 5.0,
                       warmup_fraction: float = 0.2) -> float:
    """
    Get lambda_adj value according to schedule.
    Linear ramp from 0 to lambda_adj_end over warmup_fraction of iterations.
    """
    t = iteration / total_iterations
    
    if t < warmup_fraction:
        # Linear ramp during warmup
        return lambda_adj_start + (lambda_adj_end - lambda_adj_start) * (t / warmup_fraction)
    else:
        # Hold constant after warmup
        return lambda_adj_end