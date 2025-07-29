#!/usr/bin/env python3
"""
Improved ReLU mesh optimization loss function with three terms:
(A) Area-balance loss with L1 norm
(B) Adjacent-direction loss (local planarity surrogate)
(C) Gated total-variation loss for smoothness

Based on the fully annotated loss formulation with recommended numeric values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


def compute_soft_region_areas(
    points: torch.Tensor,      # (N, 3) vertex positions
    triangles: torch.Tensor,   # (T, 3) triangle indices
    f_values: torch.Tensor,    # (N, C) field values at vertices
    beta: float                # softmax temperature
) -> torch.Tensor:
    """
    Compute soft region-area fractions p^(β) as in Eq. (4.35)-(4.38).
    
    Returns:
        p: (C,) soft area fractions for each channel
    """
    device = points.device
    num_channels = f_values.shape[1]
    
    # Define barycentric sampling points (4 samples per triangle)
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
    areas = 0.5 * torch.norm(normals, dim=1)  # (T,)
    
    # Get field values at vertices
    f0, f1, f2 = f_values[v0], f_values[v1], f_values[v2]  # (T, C)
    
    # Interpolate field at barycentric points and compute softmax
    channel_areas = torch.zeros(num_channels, device=device)
    
    for bary in bary_points:
        # Interpolate field values
        f_bary = bary[0] * f0 + bary[1] * f1 + bary[2] * f2  # (T, C)
        
        # Apply softmax with temperature beta
        probs = torch.softmax(beta * f_bary, dim=1)  # (T, C)
        
        # Accumulate weighted areas
        channel_areas += (probs * areas.unsqueeze(1)).sum(dim=0)
    
    # Average over sample points and normalize
    channel_areas = channel_areas / len(bary_points)
    total_area = channel_areas.sum()
    
    return channel_areas / total_area


def compute_edge_weights(
    f_values: torch.Tensor,    # (N, C) field values at vertices
    edges: torch.Tensor,       # (E, 2) edge vertex indices
    beta: float                # sigmoid temperature
) -> torch.Tensor:
    """
    Compute boundary weights w_e = σ(-β * d_ai * d_bi) for each edge.
    
    Returns:
        w_e: (E,) weights indicating if edge is on boundary (1) or inside region (0)
    """
    # Get field values at edge endpoints
    f_i = f_values[edges[:, 0]]  # (E, C)
    f_j = f_values[edges[:, 1]]  # (E, C)
    
    # Compute differences for all channels
    d_ij = f_i - f_j  # (E, C)
    
    # For each edge, check if any channel pair has opposite signs
    # This indicates the edge crosses a boundary
    weights = torch.zeros(edges.shape[0], device=f_values.device)
    
    C = f_values.shape[1]
    for c1 in range(C):
        for c2 in range(c1 + 1, C):
            # Product of differences for this channel pair
            prod = d_ij[:, c1] * d_ij[:, c2]
            # Sigmoid activation: negative product means opposite signs
            w_pair = torch.sigmoid(-beta * prod)
            weights = torch.maximum(weights, w_pair)
    
    return weights


def compute_face_gradients(
    points: torch.Tensor,      # (N, 3) vertex positions
    triangles: torch.Tensor,   # (T, 3) triangle indices
    f_values: torch.Tensor     # (N, C) field values at vertices
) -> torch.Tensor:
    """
    Compute per-face gradients of field values using linear finite elements.
    
    Returns:
        gradients: (T, C, 3) gradients for each triangle and channel
    """
    # Get triangle vertices
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    p0, p1, p2 = points[v0], points[v1], points[v2]
    
    # Edge vectors
    e1 = p1 - p0  # (T, 3)
    e2 = p2 - p0  # (T, 3)
    
    # Normal vectors
    normals = torch.cross(e1, e2, dim=1)  # (T, 3)
    areas_2 = torch.norm(normals, dim=1, keepdim=True)  # (T, 1) twice the area
    
    # Get field values at vertices
    f0, f1, f2 = f_values[v0], f_values[v1], f_values[v2]  # (T, C)
    
    # Compute gradients for each channel
    num_channels = f_values.shape[1]
    gradients = torch.zeros(triangles.shape[0], num_channels, 3, device=points.device)
    
    # Use the formula for gradient on a triangle:
    # ∇f = (1/2A) * [(f1-f0)(p2-p0)×n + (f2-f0)(n×(p1-p0))]
    # where n is the normal and A is the area
    
    for c in range(num_channels):
        df1 = f1[:, c] - f0[:, c]  # (T,)
        df2 = f2[:, c] - f0[:, c]  # (T,)
        
        # Cross products
        cross1 = torch.cross(e2, normals, dim=1)  # (T, 3)
        cross2 = torch.cross(normals, e1, dim=1)  # (T, 3)
        
        # Gradient
        grad = (df1.unsqueeze(1) * cross1 + df2.unsqueeze(1) * cross2) / (areas_2.unsqueeze(1) + 1e-8)
        gradients[:, c, :] = grad
    
    return gradients


def area_balance_loss(
    points: torch.Tensor,      # (N, 3) vertex positions
    triangles: torch.Tensor,   # (T, 3) triangle indices
    f_values: torch.Tensor,    # (N, C) field values at vertices
    beta: float,               # softmax temperature
    lambda_area: float = 1.0   # weight for this term
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (A) Area-balance loss: λ_area * ||p^(β) - 1/C||_1
    
    Returns:
        loss: scalar loss value
        p: (C,) soft area fractions for monitoring
    """
    C = f_values.shape[1]
    
    # Compute soft region areas
    p = compute_soft_region_areas(points, triangles, f_values, beta)
    
    # Target: uniform distribution
    target = torch.ones_like(p) / C
    
    # L1 loss
    loss = lambda_area * torch.norm(p - target, p=1)
    
    return loss, p


def adjacent_direction_loss(
    points: torch.Tensor,          # (N, 3) vertex positions
    triangles: torch.Tensor,       # (T, 3) triangle indices
    f_values: torch.Tensor,        # (N, C) field values at vertices
    edges: torch.Tensor,           # (E, 2) edge vertex indices
    triangle_adjacency: torch.Tensor,  # (n_adj, 2) pairs of adjacent triangles
    beta: float,                   # sigmoid temperature for edge weights
    lambda_adj: float = 5.0        # weight for this term
) -> torch.Tensor:
    """
    (B) Adjacent-direction loss: λ_adj * Σ_e w_e * (1 - cos θ_e)
    
    Penalizes misalignment of gradients on adjacent triangles at boundaries.
    """
    device = points.device
    C = f_values.shape[1]
    
    if C < 2 or len(triangle_adjacency) == 0:
        return torch.zeros((), device=device)
    
    # Compute per-face gradients for all channel pairs
    gradients = compute_face_gradients(points, triangles, f_values)  # (T, C, 3)
    
    # Build edge to triangle mapping for weight lookup
    edge_set = set()
    for i in range(len(edges)):
        edge_set.add((edges[i, 0].item(), edges[i, 1].item()))
        edge_set.add((edges[i, 1].item(), edges[i, 0].item()))
    
    total_loss = torch.zeros((), device=device)
    total_weight = 0.0
    
    # Process each channel pair
    for c1 in range(C):
        for c2 in range(c1 + 1, C):
            # Compute difference field gradients
            diff_gradients = gradients[:, c1, :] - gradients[:, c2, :]  # (T, 3)
            
            # For each pair of adjacent triangles
            for t1, t2 in triangle_adjacency:
                t1, t2 = t1.item(), t2.item()
                if t1 < 0 or t2 < 0:
                    continue
                    
                # Get gradients on the two faces
                grad1 = diff_gradients[t1]  # (3,)
                grad2 = diff_gradients[t2]  # (3,)
                
                # Normalize
                norm1 = torch.norm(grad1)
                norm2 = torch.norm(grad2)
                
                if norm1 < 1e-8 or norm2 < 1e-8:
                    continue
                    
                grad1_norm = grad1 / norm1
                grad2_norm = grad2 / norm2
                
                # Compute angle
                cos_theta = torch.clamp(torch.dot(grad1_norm, grad2_norm), -1.0, 1.0)
                
                # Find the shared edge between triangles
                face1 = triangles[t1].cpu().numpy()
                face2 = triangles[t2].cpu().numpy()
                
                # Find common vertices
                common_verts = []
                for v1 in face1:
                    if v1 in face2:
                        common_verts.append(v1)
                
                if len(common_verts) == 2:
                    # This is the shared edge
                    edge = tuple(sorted(common_verts))
                    
                    # Get field values at edge endpoints
                    f_i = f_values[edge[0]]
                    f_j = f_values[edge[1]]
                    
                    # Compute edge weight for this channel pair
                    d_ij = (f_i[c1] - f_j[c1]) * (f_i[c2] - f_j[c2])
                    w_e = torch.sigmoid(-beta * d_ij)
                    
                    # Add to loss
                    total_loss += w_e * (1 - cos_theta)
                    total_weight += w_e
    
    if total_weight > 0:
        return lambda_adj * total_loss / total_weight
    else:
        return torch.zeros((), device=device)


def gated_total_variation_loss(
    f_values: torch.Tensor,    # (N, C) field values at vertices
    edges: torch.Tensor,       # (E, 2) edge vertex indices
    beta: float,               # sigmoid temperature for edge weights
    lambda_tv: float = 0.05    # weight for this term
) -> torch.Tensor:
    """
    (C) Gated total-variation loss: λ_TV * Σ_e (1 - w_e) * ||f_i - f_j||_2^2
    
    Smoothness inside regions, gated by boundary weights.
    """
    # Compute edge weights (boundary indicator)
    w_e = compute_edge_weights(f_values, edges, beta)
    
    # Get field values at edge endpoints
    f_i = f_values[edges[:, 0]]  # (E, C)
    f_j = f_values[edges[:, 1]]  # (E, C)
    
    # Squared L2 differences
    diff_squared = torch.sum((f_i - f_j) ** 2, dim=1)  # (E,)
    
    # Gate by (1 - w_e) to avoid smoothing across boundaries
    gated_diff = (1 - w_e) * diff_squared
    
    return lambda_tv * torch.sum(gated_diff)


def compute_improved_loss(
    points: torch.Tensor,          # (N, 3) vertex positions
    triangles: torch.Tensor,       # (T, 3) triangle indices
    f_values: torch.Tensor,        # (N, C) field values at vertices
    edges: torch.Tensor,           # (E, 2) edge vertex indices
    triangle_adjacency: torch.Tensor,  # (E_faces, 2) pairs of adjacent triangles
    beta: float,                   # softmax/sigmoid temperature
    lambda_area: float = 1.0,      # area balance weight
    lambda_adj: float = 5.0,       # adjacent direction weight
    lambda_tv: float = 0.05,       # total variation weight
) -> Dict[str, torch.Tensor]:
    """
    Compute the complete three-term loss function.
    
    Returns dict with:
        - 'total': total loss
        - 'area': area balance loss
        - 'adj': adjacent direction loss
        - 'tv': gated total variation loss
        - 'area_fractions': (C,) soft area fractions
    """
    # (A) Area-balance loss
    area_loss, area_fractions = area_balance_loss(
        points, triangles, f_values, beta, lambda_area
    )
    
    # (B) Adjacent-direction loss
    adj_loss = adjacent_direction_loss(
        points, triangles, f_values, edges, triangle_adjacency, beta, lambda_adj
    )
    
    # (C) Gated total-variation loss
    tv_loss = gated_total_variation_loss(
        f_values, edges, beta, lambda_tv
    )
    
    # Total loss
    total_loss = area_loss + adj_loss + tv_loss
    
    return {
        'total': total_loss,
        'area': area_loss,
        'adj': adj_loss,
        'tv': tv_loss,
        'area_fractions': area_fractions
    }


class BetaScheduler:
    """Scheduler for beta parameter."""
    
    def __init__(self, start: float = 2.0, end: float = 25.0, warmup_frac: float = 0.2):
        self.start = start
        self.end = end
        self.warmup_frac = warmup_frac
        
    def get_beta(self, progress: float) -> float:
        """Get beta value based on training progress (0 to 1)."""
        if progress < self.warmup_frac:
            # Linear ramp during warmup
            t = progress / self.warmup_frac
            return self.start + (self.end - self.start) * t
        else:
            # Hold at end value
            return self.end


class LambdaScheduler:
    """Scheduler for lambda weights."""
    
    def __init__(
        self,
        lambda_area: float = 1.0,
        lambda_adj_start: float = 0.0,
        lambda_adj_end: float = 5.0,
        lambda_tv: float = 0.05,
        warmup_frac: float = 0.2
    ):
        self.lambda_area = lambda_area
        self.lambda_adj_start = lambda_adj_start
        self.lambda_adj_end = lambda_adj_end
        self.lambda_tv = lambda_tv
        self.warmup_frac = warmup_frac
        
    def get_lambdas(self, progress: float) -> Dict[str, float]:
        """Get lambda values based on training progress (0 to 1)."""
        if progress < self.warmup_frac:
            # Ramp lambda_adj during warmup
            t = progress / self.warmup_frac
            lambda_adj = self.lambda_adj_start + (self.lambda_adj_end - self.lambda_adj_start) * t
        else:
            lambda_adj = self.lambda_adj_end
            
        return {
            'area': self.lambda_area,
            'adj': lambda_adj,
            'tv': self.lambda_tv
        }


def get_recommended_hyperparameters(mesh_name: str = "default") -> Dict:
    """Get recommended hyperparameters for different meshes."""
    
    base_params = {
        'beta_start': 2.0,
        'beta_end': 25.0,
        'lambda_area': 1.0,
        'lambda_adj_start': 0.0,
        'lambda_adj_end': 5.0,
        'lambda_tv': 0.05,
        'warmup_frac': 0.2,
        'lr_base': 2e-3,
        'lr_schedule': 'one_cycle'
    }
    
    # Mesh-specific adjustments
    if mesh_name.lower() == "sphere":
        # Sphere needs only moderate adjacent penalty
        pass
    elif mesh_name.lower() in ["kitty", "rod"]:
        # More complex meshes may need stronger adjacent penalty
        base_params['lambda_adj_end'] = 10.0
        base_params['beta_end'] = 35.0  # Optional stronger beta after 80%
    elif mesh_name.lower() == "angel":
        # Highly detailed mesh needs less TV smoothing
        base_params['lambda_tv'] = 0.01
        
    return base_params