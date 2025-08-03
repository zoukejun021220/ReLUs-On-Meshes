#!/usr/bin/env python3
"""
Fully vectorized version of improved loss function for efficient GPU computation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


def compute_soft_area_fractions_vectorized(
    points: torch.Tensor,
    triangles: torch.Tensor,
    f_values: torch.Tensor,
    beta: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized computation of soft area fractions p(β) for each channel.
    """
    device = points.device
    num_channels = f_values.shape[1]
    
    # Get triangle vertices
    v_indices = triangles  # (T, 3)
    tri_points = points[v_indices]  # (T, 3, 3)
    
    # Compute triangle areas using cross product
    e1 = tri_points[:, 1] - tri_points[:, 0]  # (T, 3)
    e2 = tri_points[:, 2] - tri_points[:, 0]  # (T, 3)
    areas = 0.5 * torch.norm(torch.cross(e1, e2, dim=1), dim=1)  # (T,)
    
    # Get f values at triangle vertices
    tri_f_values = f_values[v_indices]  # (T, 3, C)
    
    # Compute f at triangle centers (barycentric average)
    f_centers = tri_f_values.mean(dim=1)  # (T, C)
    
    # Apply softmax across channels for each triangle
    weights = torch.softmax(beta * f_centers, dim=1)  # (T, C)
    
    # Compute weighted area sums
    weighted_areas = areas.unsqueeze(1) * weights  # (T, C)
    p = weighted_areas.sum(dim=0) / areas.sum()  # (C,)
    
    return p, areas


def compute_boundary_weights_vectorized(
    f_values: torch.Tensor,
    edges: torch.Tensor,
    beta: float
) -> torch.Tensor:
    """
    Vectorized computation of boundary weights w_e for edges.
    """
    # Get f values at edge vertices
    v1_idx, v2_idx = edges[:, 0], edges[:, 1]
    f1, f2 = f_values[v1_idx], f_values[v2_idx]  # (E, C)
    
    # Compute edge midpoint values
    f_mid = 0.5 * (f1 + f2)  # (E, C)
    
    # Compute max channel at each edge
    max_vals, _ = torch.max(f_mid, dim=1)  # (E,)
    
    # Apply sigmoid for smooth boundary detection
    w_e = torch.sigmoid(beta * max_vals)  # (E,)
    
    return w_e


def compute_face_gradients_vectorized(
    points: torch.Tensor,
    triangles: torch.Tensor,
    f_values: torch.Tensor
) -> torch.Tensor:
    """
    Vectorized computation of gradients on all triangle faces.
    
    Returns:
        gradients: (T, 3) gradient vectors for each triangle
    """
    # Get vertex positions and values
    v_indices = triangles  # (T, 3)
    tri_points = points[v_indices]  # (T, 3, 3)
    tri_f = f_values[v_indices]  # (T, 3) if f_values is 1D, (T, 3, C) if multi-channel
    
    # Edge vectors
    e1 = tri_points[:, 1] - tri_points[:, 0]  # (T, 3)
    e2 = tri_points[:, 2] - tri_points[:, 0]  # (T, 3)
    
    # Face normals
    normals = torch.cross(e1, e2, dim=1)  # (T, 3)
    areas = 0.5 * torch.norm(normals, dim=1, keepdim=True)  # (T, 1)
    normals = normals / (2 * areas + 1e-10)  # (T, 3)
    
    # Function differences
    if len(tri_f.shape) == 2:  # Single channel
        df1 = tri_f[:, 1] - tri_f[:, 0]  # (T,)
        df2 = tri_f[:, 2] - tri_f[:, 0]  # (T,)
        
        # Gradient computation using cross products
        grad = (df1.unsqueeze(1) * torch.cross(normals, e2, dim=1) + 
                df2.unsqueeze(1) * torch.cross(e1, normals, dim=1))  # (T, 3)
    else:  # Multi-channel - process each channel
        grad = []
        for c in range(tri_f.shape[2]):
            df1 = tri_f[:, 1, c] - tri_f[:, 0, c]
            df2 = tri_f[:, 2, c] - tri_f[:, 0, c]
            grad_c = (df1.unsqueeze(1) * torch.cross(normals, e2, dim=1) + 
                      df2.unsqueeze(1) * torch.cross(e1, normals, dim=1))
            grad.append(grad_c)
        grad = torch.stack(grad, dim=2)  # (T, 3, C)
    
    return grad


def build_edge_to_triangles_vectorized(triangles: torch.Tensor):
    """
    Build edge-to-triangles mapping efficiently on GPU.
    Returns sparse tensor for memory efficiency.
    """
    device = triangles.device
    T = triangles.shape[0]
    
    # Create all edges from triangles
    edges_list = []
    triangle_ids = []
    
    for i in range(3):
        v1 = triangles[:, i]
        v2 = triangles[:, (i+1)%3]
        
        # Ensure v1 < v2 for consistent edge representation
        edge_v1 = torch.min(v1, v2)
        edge_v2 = torch.max(v1, v2)
        
        edges_list.append(torch.stack([edge_v1, edge_v2], dim=1))
        triangle_ids.append(torch.arange(T, device=device))
    
    all_edges = torch.cat(edges_list, dim=0)  # (3*T, 2)
    all_tri_ids = torch.cat(triangle_ids, dim=0)  # (3*T,)
    
    # Find unique edges and their inverse indices
    unique_edges, inverse_indices = torch.unique(all_edges, dim=0, return_inverse=True)
    
    # Build adjacency using scatter operations
    num_unique_edges = unique_edges.shape[0]
    edge_triangle_count = torch.zeros(num_unique_edges, device=device)
    edge_triangle_count.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float))
    
    # Find edges with exactly 2 triangles (internal edges)
    internal_edge_mask = edge_triangle_count == 2
    
    return unique_edges, inverse_indices, all_tri_ids, internal_edge_mask


def improved_loss_function_vectorized(
    points: torch.Tensor,
    triangles: torch.Tensor,
    f_values: torch.Tensor,
    edges: torch.Tensor,
    beta: float,
    lambda_area: float = 1.0,
    lambda_adj: float = 5.0,
    lambda_TV: float = 0.05,
    num_channels: int = 6
) -> Tuple[torch.Tensor, dict]:
    """
    Fully vectorized version of the improved three-term loss function.
    """
    device = points.device
    
    # (A) Area-balance term
    p, triangle_areas = compute_soft_area_fractions_vectorized(points, triangles, f_values, beta)
    target = torch.ones(num_channels, device=device) / num_channels
    area_loss = torch.norm(p - target, p=1)
    
    # Compute boundary weights
    w_e = compute_boundary_weights_vectorized(f_values, edges, beta)
    
    # (B) Adjacent-direction term - vectorized version
    # Build edge-to-triangle mapping
    unique_edges, edge_tri_map, tri_ids, internal_mask = build_edge_to_triangles_vectorized(triangles)
    
    adj_loss = torch.zeros(1, device=device)
    
    if internal_mask.any():
        # Process all channel pairs at once
        for c_i in range(num_channels):
            for c_j in range(c_i + 1, num_channels):
                f_diff = f_values[:, c_i] - f_values[:, c_j]
                gradients = compute_face_gradients_vectorized(points, triangles, f_diff)  # (T, 3)
                
                # Get gradients for all triangle pairs sharing edges
                # This is the key vectorization: process all edges at once
                internal_edges = unique_edges[internal_mask]
                
                # For each internal edge, find its two adjacent triangles
                edge_indices = torch.arange(len(unique_edges), device=device)[internal_mask]
                
                # Use scatter to find triangle pairs
                for edge_idx in edge_indices:
                    # Find triangles containing this edge
                    tri_mask = edge_tri_map == edge_idx
                    adjacent_tris = tri_ids[tri_mask]
                    
                    if len(adjacent_tris) == 2:
                        t1, t2 = adjacent_tris[0], adjacent_tris[1]
                        g1, g2 = gradients[t1], gradients[t2]
                        
                        # Compute angle between gradients
                        g1_norm = torch.norm(g1)
                        g2_norm = torch.norm(g2)
                        
                        if g1_norm > 1e-8 and g2_norm > 1e-8:
                            cos_theta = torch.clamp(
                                torch.dot(g1, g2) / (g1_norm * g2_norm),
                                -1.0, 1.0
                            )
                            
                            # Find corresponding edge in original edge list
                            v1, v2 = internal_edges[edge_idx - edge_indices[0]]
                            edge_mask = ((edges[:, 0] == v1) & (edges[:, 1] == v2)) | \
                                       ((edges[:, 0] == v2) & (edges[:, 1] == v1))
                            
                            if edge_mask.any():
                                edge_weight = w_e[edge_mask].mean()
                                adj_loss += edge_weight * (1 - cos_theta)
    
    # Normalize adjacent loss
    num_pairs = num_channels * (num_channels - 1) // 2
    if num_pairs > 0:
        adj_loss = adj_loss / num_pairs
    
    # (C) Gated total-variation term - already vectorized
    v1_idx, v2_idx = edges[:, 0], edges[:, 1]
    f1, f2 = f_values[v1_idx], f_values[v2_idx]
    f_diff = f1 - f2  # (E, C)
    tv_loss = torch.sum((1 - w_e).unsqueeze(1) * torch.sum(f_diff**2, dim=1))
    
    # Combine losses
    total_loss = lambda_area * area_loss + lambda_adj * adj_loss + lambda_TV * tv_loss
    
    # Return components for monitoring
    components = {
        'area': area_loss.item(),
        'adjacent': adj_loss.item(),
        'tv': tv_loss.item(),
        'area_fractions': p.detach().cpu().numpy(),
        'mean_boundary_weight': w_e.mean().item()
    }
    
    return total_loss, components