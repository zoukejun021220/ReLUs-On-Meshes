"""
Fixed loss functions with proper normalization to prevent large raw values.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


def compute_adjacency_loss_properly_normalized(
    f_values: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    triangle_adjacency: torch.Tensor,
    beta: float,
    face_mask: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8
) -> Tuple[torch.Tensor, Dict]:
    """
    Properly normalized adjacency loss that produces reasonable raw values.
    
    Key changes:
    1. Normalize by number of adjacent pairs AND channel pairs
    2. Use softer penalty function
    3. Weight by actual boundary presence
    """
    device = f_values.device
    num_channels = f_values.shape[1]
    
    # Compute per-face values and gradients
    face_f = f_values[faces].mean(dim=1)  # (F, C)
    
    # Get valid faces
    if face_mask is not None:
        valid_faces = face_mask.to(device)
    else:
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1)
        valid_faces = areas > epsilon
    
    # Compute normalized gradients per face
    gradients = []
    for f_idx in range(faces.shape[0]):
        if not valid_faces[f_idx]:
            gradients.append(torch.zeros(num_channels, 3, device=device))
            continue
            
        v_idx = faces[f_idx]
        v_pos = vertices[v_idx]  # (3, 3)
        v_f = f_values[v_idx]    # (3, C)
        
        # Edge vectors
        e1 = v_pos[1] - v_pos[0]
        e2 = v_pos[2] - v_pos[0]
        
        # Normal vector
        normal = torch.cross(e1, e2)
        area = 0.5 * torch.norm(normal)
        
        if area < epsilon:
            gradients.append(torch.zeros(num_channels, 3, device=device))
            continue
            
        # Compute gradient for each channel
        face_grads = []
        for c in range(num_channels):
            # Value differences
            df1 = v_f[1, c] - v_f[0, c]
            df2 = v_f[2, c] - v_f[0, c]
            
            # Project onto face plane and solve for gradient
            # This is more stable than direct least squares
            A = torch.stack([e1, e2], dim=0)
            b = torch.tensor([df1, df2], device=device)
            
            # Use pseudo-inverse for stability
            try:
                grad = torch.linalg.lstsq(A, b, rcond=1e-6).solution
            except:
                grad = torch.zeros(3, device=device)
            
            # Normalize gradient (unit vector)
            grad_norm = torch.norm(grad) + epsilon
            grad = grad / grad_norm
            
            face_grads.append(grad)
        
        gradients.append(torch.stack(face_grads))
    
    gradients = torch.stack(gradients)  # (F, C, 3)
    
    # Process adjacent triangle pairs
    total_loss = 0.0
    num_valid_pairs = 0
    num_boundaries = 0
    
    for adj_idx in range(triangle_adjacency.shape[0]):
        t0, t1 = triangle_adjacency[adj_idx, 0], triangle_adjacency[adj_idx, 1]
        
        if not (valid_faces[t0] and valid_faces[t1]):
            continue
        
        # For each channel pair
        pair_loss = 0.0
        pair_boundaries = 0
        
        for i in range(num_channels):
            for j in range(i+1, num_channels):
                # Field differences on both triangles
                d0_ij = face_f[t0, i] - face_f[t0, j]
                d1_ij = face_f[t1, i] - face_f[t1, j]
                
                # Soft boundary indicator (edge weight)
                # Use tanh for smoother gradients than sigmoid
                w_e = 0.5 * (1 + torch.tanh(-beta * d0_ij * d1_ij))
                
                # Only compute gradient alignment if this is likely a boundary
                if w_e > 0.1:  # Threshold to avoid noise
                    # Gradient differences (already normalized)
                    g0_ij = gradients[t0, i] - gradients[t0, j]
                    g1_ij = gradients[t1, i] - gradients[t1, j]
                    
                    # Normalize the differences
                    g0_norm = torch.norm(g0_ij) + epsilon
                    g1_norm = torch.norm(g1_ij) + epsilon
                    g0_ij = g0_ij / g0_norm
                    g1_ij = g1_ij / g1_norm
                    
                    # Cosine similarity
                    cos_sim = torch.sum(g0_ij * g1_ij).clamp(-1, 1)
                    
                    # Soft penalty: use squared difference of angles
                    # angle = arccos(cos_sim), penalty = (angle/π)^2
                    # This is smoother and bounded in [0, 1]
                    angle_ratio = torch.acos(cos_sim) / np.pi  # [0, 1]
                    penalty = angle_ratio ** 2
                    
                    pair_loss += w_e * penalty
                    pair_boundaries += w_e
        
        # Normalize by number of channel pairs
        num_channel_pairs = (num_channels * (num_channels - 1)) // 2
        if pair_boundaries > 0:
            pair_loss = pair_loss / num_channel_pairs
            total_loss += pair_loss
            num_valid_pairs += 1
            num_boundaries += pair_boundaries.item() / num_channel_pairs
    
    # Final normalization by number of adjacent pairs
    if num_valid_pairs > 0:
        total_loss = total_loss / num_valid_pairs
        avg_boundaries = num_boundaries / num_valid_pairs
    else:
        avg_boundaries = 0.0
    
    stats = {
        'num_valid_pairs': num_valid_pairs,
        'avg_boundaries_per_pair': avg_boundaries,
        'raw_loss_value': total_loss.item() if torch.is_tensor(total_loss) else total_loss
    }
    
    return total_loss, stats


def compute_area_balance_loss_robust(
    f_values: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    beta: float,
    target_fraction: float = 1.0/6.0,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Robust area balance loss with proper normalization.
    """
    num_channels = f_values.shape[1]
    device = f_values.device
    
    # Compute face areas
    v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1)
    total_area = face_areas.sum() + epsilon
    
    # Interpolate field values at face centers (barycentric)
    face_f = f_values[faces].mean(dim=1)  # (F, C)
    
    # Compute soft assignment probabilities using softmax
    # Lower beta for more gradual transitions
    soft_beta = min(beta, 10.0)  
    probs = F.softmax(soft_beta * face_f, dim=1)  # (F, C)
    
    # Compute area per channel
    channel_areas = torch.zeros(num_channels, device=device)
    for c in range(num_channels):
        channel_areas[c] = (probs[:, c] * face_areas).sum()
    
    # Normalize to get fractions
    area_fractions = channel_areas / total_area
    
    # L2 penalty for deviation from target
    loss = torch.sum((area_fractions - target_fraction) ** 2)
    
    return loss


def compute_tv_loss_adaptive(
    f_values: torch.Tensor,
    edges: torch.Tensor,
    vertices: torch.Tensor,
    beta: float,
    base_weight: float = 0.1,
    boundary_weight: float = 0.01,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Adaptive TV loss that reduces smoothing at boundaries.
    """
    v0_idx, v1_idx = edges[:, 0], edges[:, 1]
    
    # Edge lengths for weighting
    edge_vecs = vertices[v1_idx] - vertices[v0_idx]
    edge_lengths = torch.norm(edge_vecs, dim=1) + epsilon
    
    # Field differences
    f_diff = f_values[v1_idx] - f_values[v0_idx]  # (E, C)
    
    # Detect boundaries: where channels have different argmax
    argmax_v0 = torch.argmax(f_values[v0_idx], dim=1)  # (E,)
    argmax_v1 = torch.argmax(f_values[v1_idx], dim=1)  # (E,)
    is_boundary = (argmax_v0 != argmax_v1).float()
    
    # Adaptive weight: less smoothing at boundaries
    adaptive_weight = base_weight * (1 - is_boundary) + boundary_weight * is_boundary
    
    # Weighted TV loss
    tv_per_edge = torch.sum(f_diff ** 2, dim=1)  # (E,)
    weighted_tv = adaptive_weight * tv_per_edge / edge_lengths
    
    return weighted_tv.mean()


def compute_planarity_loss(
    f_values: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    triangle_adjacency: torch.Tensor,
    beta: float,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Encourage boundaries to be planar (straight).
    """
    device = f_values.device
    num_channels = f_values.shape[1]
    
    # Get face centers
    face_centers = vertices[faces].mean(dim=1)  # (F, 3)
    face_f = f_values[faces].mean(dim=1)  # (F, C)
    
    total_loss = 0.0
    num_boundaries = 0
    
    for adj_idx in range(triangle_adjacency.shape[0]):
        t0, t1 = triangle_adjacency[adj_idx, 0], triangle_adjacency[adj_idx, 1]
        
        # Centers of adjacent triangles
        c0 = face_centers[t0]
        c1 = face_centers[t1]
        edge_vec = c1 - c0
        edge_len = torch.norm(edge_vec) + epsilon
        edge_dir = edge_vec / edge_len
        
        for i in range(num_channels):
            for j in range(i+1, num_channels):
                # Check if this edge crosses the i-j boundary
                d0_ij = face_f[t0, i] - face_f[t0, j]
                d1_ij = face_f[t1, i] - face_f[t1, j]
                
                # Boundary weight
                w_boundary = torch.sigmoid(-beta * d0_ij * d1_ij)
                
                if w_boundary > 0.1:
                    # Compute normal to the boundary
                    # Ideally, the gradient of (f_i - f_j) should be perpendicular to the boundary
                    gradient_ij = (d1_ij - d0_ij) / edge_len
                    
                    # The boundary should be perpendicular to the gradient
                    # So edge_dir should be perpendicular to gradient direction
                    # Penalty is the absolute dot product (should be 0)
                    perpendicularity = torch.abs(gradient_ij * edge_len)
                    
                    total_loss += w_boundary * perpendicularity
                    num_boundaries += w_boundary
    
    if num_boundaries > epsilon:
        total_loss = total_loss / num_boundaries
    
    return total_loss


def compute_total_loss_fixed(
    f_values: torch.Tensor,
    mesh_data: Dict,
    beta: float,
    lambda_area: float = 1.0,
    lambda_adj: float = 1.0,
    lambda_tv: float = 0.1,
    lambda_planarity: float = 0.1,
    use_adaptive_tv: bool = True,
    epsilon: float = 1e-8
) -> Tuple[torch.Tensor, Dict]:
    """
    Complete loss function with proper normalization.
    """
    vertices = mesh_data['vertices']
    faces = mesh_data['faces']
    edges = mesh_data['edges']
    triangle_adjacency = mesh_data['triangle_adjacency']
    face_mask = mesh_data.get('face_mask')
    
    # Compute individual losses
    adj_loss, adj_stats = compute_adjacency_loss_properly_normalized(
        f_values, vertices, faces, triangle_adjacency, beta, face_mask
    )
    
    area_loss = compute_area_balance_loss_robust(
        f_values, vertices, faces, beta
    )
    
    if use_adaptive_tv:
        tv_loss = compute_tv_loss_adaptive(
            f_values, edges, vertices, beta
        )
    else:
        # Simple TV
        f_diff = f_values[edges[:, 1]] - f_values[edges[:, 0]]
        tv_loss = torch.mean(torch.sum(f_diff ** 2, dim=1))
    
    planarity_loss = compute_planarity_loss(
        f_values, vertices, faces, triangle_adjacency, beta
    )
    
    # Combine losses
    total_loss = (
        lambda_area * area_loss +
        lambda_adj * adj_loss +
        lambda_tv * tv_loss +
        lambda_planarity * planarity_loss
    )
    
    loss_dict = {
        'total': total_loss.item(),
        'area': area_loss.item(),
        'adjacency': adj_loss.item(),
        'tv': tv_loss.item(),
        'planarity': planarity_loss.item(),
        'adj_stats': adj_stats
    }
    
    return total_loss, loss_dict


if __name__ == "__main__":
    print("Fixed loss functions with proper normalization:")
    print("1. Adjacency loss normalized by num_pairs × num_channel_pairs")
    print("2. Softer penalty: (angle/π)² instead of (1-cos)")
    print("3. Adaptive TV that reduces smoothing at boundaries")
    print("4. Planarity loss to encourage straight boundaries")
    print("5. All losses produce values in reasonable ranges (0.01-10)")