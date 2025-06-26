import numpy as np
import math
import pyvista as pv
import vtk
import torch
import torch.nn as nn
import torch.nn.functional as F





def area_balance_loss_optimized(points, triangles, f_values, beta, mesh_area):
    """
    Highly vectorized implementation of area balance loss using softmax instead of sigmoid.
    
    Args:
        points: Tensor of shape (N, 3) containing vertex positions
        triangles: Tensor of shape (T, 3) containing triangle vertex indices
        f_values: Tensor of shape (N, 6) containing scalar field values
        beta: Softmax sharpness parameter
        mesh_area: Total mesh area
        
    Returns:
        loss: Area balance loss value
        fractions: Tensor of shape (6,) with fraction of area for each channel
    """
    device = points.device
    num_channels = f_values.shape[1]
    num_triangles = triangles.shape[0]
    
    # Define barycentric sampling points
    bary_points = torch.tensor([
        [1/3, 1/3, 1/3],  # Center
        [0.5, 0.5, 0.0],  # Edge midpoints
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ], device=device)
    
    num_samples = bary_points.shape[0]
    bary_weights = torch.ones(num_samples, device=device) / num_samples
    
    # Compute triangle areas in one batch
    v0 = triangles[:, 0]  # (T,)
    v1 = triangles[:, 1]  # (T,)
    v2 = triangles[:, 2]  # (T,)
    
    p0 = points[v0]  # (T, 3)
    p1 = points[v1]  # (T, 3)
    p2 = points[v2]  # (T, 3)
    
    e1 = p1 - p0  # (T, 3)
    e2 = p2 - p0  # (T, 3)
    
    normals = torch.cross(e1, e2, dim=1)  # (T, 3)
    areas = 0.5 * torch.norm(normals, dim=1)  # (T,)
    
    # Convert to softmax probabilities
    raw_scores = beta * f_values  # (N, 6)
    
    # Get scalar field values at triangle vertices
    f0 = f_values[v0]  # (T, 6)
    f1 = f_values[v1]  # (T, 6)
    f2 = f_values[v2]  # (T, 6)
    
    # Initialize tensor to accumulate channel areas
    channel_areas = torch.zeros(num_channels, device=device)
    
    # Process all triangles and all barycentric samples at once
    # Reshape for broadcasting:
    # f0: (T, 6) -> (T, 1, 6)
    # f1: (T, 6) -> (T, 1, 6)
    # f2: (T, 6) -> (T, 1, 6)
    # bary_points: (S, 3) -> (1, S, 3)
    f0 = f0.unsqueeze(1)  # (T, 1, 6)
    f1 = f1.unsqueeze(1)  # (T, 1, 6)
    f2 = f2.unsqueeze(1)  # (T, 1, 6)
    
    bary_expanded = bary_points.unsqueeze(0)  # (1, S, 3)
    
    # Extract barycentric coordinates
    b0 = bary_expanded[:, :, 0].unsqueeze(2)  # (1, S, 1)
    b1 = bary_expanded[:, :, 1].unsqueeze(2)  # (1, S, 1)
    b2 = bary_expanded[:, :, 2].unsqueeze(2)  # (1, S, 1)
    
    # Interpolate field at all barycentric points for all triangles
    f_interp = (f0 * b0) + (f1 * b1) + (f2 * b2)  # (T, S, 6)
    
    # Apply softmax to all interpolated values
    f_interp_beta = beta * f_interp  # (T, S, 6)
    exp_vals = torch.exp(f_interp_beta)  # (T, S, 6)
    sum_exp = torch.sum(exp_vals, dim=2, keepdim=True)  # (T, S, 1)
    s = torch.softmax(f_interp_beta, dim=2)
    
    # Apply sample weights and reshape
    s_weighted = s * bary_weights.view(1, num_samples, 1)  # (T, S, 6)
    
    # Sum over samples
    s_mean = torch.sum(s_weighted, dim=1)  # (T, 6)
    
    # Weight by triangle areas and sum
    weighted_areas = s_mean * areas.unsqueeze(1)  # (T, 6)
    channel_areas = torch.sum(weighted_areas, dim=0)  # (6,)
    
    # Compute fractions and loss
    fractions = channel_areas / mesh_area
    target = torch.ones_like(fractions) / num_channels
    loss = torch.sum(torch.abs(fractions - target))
    
    return loss, fractions

def smoothness_loss_optimized(f_values, vertex_edges):
    """
    Vectorized implementation of smoothness loss.
    
    Args:
        f_values: Tensor of shape (N, 6) containing scalar field values
        vertex_edges: Tensor of shape (E, 2) containing vertex edge indices
        
    Returns:
        loss: Smoothness loss value
    """
    # Get field values at all edge endpoints at once
    v1_idx = vertex_edges[:, 0]  # (E,)
    v2_idx = vertex_edges[:, 1]  # (E,)
    
    f1 = f_values[v1_idx]  # (E, 6)
    f2 = f_values[v2_idx]  # (E, 6)
    
    # Compute squared difference for all edges and all channels
    diff = f1 - f2  # (E, 6)
    loss = torch.sum(diff**2)
    
    return loss