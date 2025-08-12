"""
SVD-based contour alignment loss for stable mesh segmentation.
Based on weighted covariance plane fitting approach.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


def fit_plane_weighted(
    points: torch.Tensor, 
    weights: torch.Tensor, 
    prev_n: Optional[torch.Tensor] = None, 
    ema: float = 0.2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fit a plane to weighted 3D points using SVD/eigendecomposition.
    
    Args:
        points: (N, 3) 3D points
        weights: (N,) point weights
        prev_n: Previous normal vector for EMA smoothing
        ema: EMA weight for temporal smoothing
        
    Returns:
        n: (3,) plane normal (unit vector)
        d: scalar offset such that n·x + d = 0
    """
    eps = 1e-8
    
    # Handle empty or single point case
    if points.shape[0] == 0:
        return torch.zeros(3, device=points.device), torch.zeros(1, device=points.device)
    
    # Normalize weights
    w_sum = weights.sum() + eps
    w_norm = weights / w_sum
    
    # Weighted mean
    mean = (points * w_norm.unsqueeze(-1)).sum(dim=0)
    
    # Center points
    centered = points - mean.unsqueeze(0)
    
    # Weighted covariance
    weighted_centered = centered * w_norm.unsqueeze(-1)
    cov = torch.mm(weighted_centered.t(), centered)
    
    # Symmetrize and add jitter for numerical stability
    cov = 0.5 * (cov + cov.t())
    trace = cov.trace()
    jitter = (1e-7 * trace) / 3.0
    cov = cov + jitter * torch.eye(3, device=cov.device, dtype=cov.dtype)
    
    # Eigendecomposition for plane normal
    try:
        evals, evecs = torch.linalg.eigh(cov.float())
        n = evecs[:, 0].to(cov.dtype)  # Smallest eigenvalue = normal
    except:
        # Fallback to SVD if eigendecomposition fails
        U, S, Vt = torch.linalg.svd(cov.float())
        n = Vt[-1, :].to(cov.dtype)
    
    # Ensure unit normal
    n = n / (n.norm() + eps)
    
    # Apply EMA if previous normal provided
    if prev_n is not None:
        # Check for sign consistency
        if torch.dot(n, prev_n) < 0:
            n = -n
        # Apply EMA
        n = (1 - ema) * prev_n + ema * n
        n = n / (n.norm() + eps)
    
    # Compute offset
    d = -torch.dot(n, mean)
    
    return n, d


def plane_loss(
    points: torch.Tensor,
    weights: torch.Tensor,
    n: torch.Tensor,
    d: torch.Tensor,
    band: float = 0.01
) -> torch.Tensor:
    """
    Compute weighted MSE loss for points to plane distance.
    
    Args:
        points: (N, 3) 3D points
        weights: (N,) point weights
        n: (3,) plane normal
        d: scalar plane offset
        band: Distance band for soft clamping
        
    Returns:
        loss: Weighted MSE loss
    """
    eps = 1e-8
    
    # Point-to-plane distances
    distances = torch.abs(torch.mm(points, n.unsqueeze(1)).squeeze() + d)
    
    # Soft clamp distances
    if band > 0:
        distances = torch.where(
            distances < band,
            distances,
            band + torch.sqrt(band * (distances - band))
        )
    
    # Weighted MSE
    w_sum = weights.sum() + eps
    loss = (weights * distances.pow(2)).sum() / w_sum
    
    return loss


def contour_alignment_svd(
    F: torch.Tensor,
    edge_coords: torch.Tensor,
    edge_pairs: torch.Tensor,
    beta: float,
    min_weight: float = 0.01,
    edge_normal_mat: Optional[torch.Tensor] = None,
    plane_memory: Optional[Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]]] = None,
    ema: float = 0.2,
    use_triple_points: bool = True,
    K_update: int = 20
) -> Tuple[torch.Tensor, Dict]:
    """
    SVD-based contour alignment loss with stable plane fitting.
    
    Args:
        F: (N, C) vertex features (logits)
        edge_coords: (E, 2, 3) edge vertex coordinates
        edge_pairs: (E, 2) vertex indices for each edge
        beta: Temperature parameter
        min_weight: Minimum weight threshold
        edge_normal_mat: Optional edge normal computation matrix
        plane_memory: Dictionary storing previous plane parameters
        ema: EMA weight for temporal smoothing
        use_triple_points: Whether to include triple point intersections
        K_update: Update planes every K iterations
        
    Returns:
        loss: Total contour alignment loss
        info: Dictionary with diagnostic information
    """
    device = F.device
    N, C = F.shape
    
    # Compute soft assignments
    probs = torch.softmax(beta * F, dim=1)
    edge_probs = probs[edge_pairs]  # (E, 2, C)
    
    # Initialize plane memory if needed
    if plane_memory is None:
        plane_memory = {}
    
    # Collect edge intersections for each channel pair
    total_loss = torch.tensor(0.0, device=F.device, dtype=F.dtype)
    active_pairs = []
    plane_losses = {}
    
    for i in range(C):
        for j in range(i + 1, C):
            # Compute edge intersection indicators
            # An edge is active if it transitions between channels i and j
            edge_i = edge_probs[:, :, i]  # (E, 2)
            edge_j = edge_probs[:, :, j]  # (E, 2)
            
            # Soft indicator: edge crosses from i to j or j to i
            phi_ij = torch.abs(edge_i[:, 0] - edge_i[:, 1]) * \
                     torch.abs(edge_j[:, 0] - edge_j[:, 1]) * \
                     (edge_i.max(dim=1).values + edge_j.max(dim=1).values) / 2
            
            # Filter by minimum weight
            active_mask = phi_ij > min_weight
            if not active_mask.any():
                continue
                
            # Get active edge points and weights
            active_edges = edge_coords[active_mask]  # (M, 2, 3)
            active_weights = phi_ij[active_mask]     # (M,)
            
            # Compute intersection points (edge midpoints)
            edge_centers = active_edges.mean(dim=1)  # (M, 3)
            
            # Optional: Add triple points if enabled
            if use_triple_points:
                # This would require triangle information
                # For now, we'll use edge centers only
                all_points = edge_centers
                all_weights = active_weights
            else:
                all_points = edge_centers
                all_weights = active_weights
            
            # Retrieve or compute plane parameters
            pair_key = (i, j)
            if pair_key in plane_memory:
                prev_n, prev_d = plane_memory[pair_key]
            else:
                prev_n, prev_d = None, None
            
            # Fit plane with EMA
            n, d = fit_plane_weighted(all_points, all_weights, prev_n, ema)
            
            # Store updated plane parameters
            plane_memory[pair_key] = (n.detach(), d.detach())
            
            # Compute loss
            pair_loss = plane_loss(all_points, all_weights, n, d)
            
            total_loss = total_loss + pair_loss
            active_pairs.append((i, j))
            plane_losses[f"{i}-{j}"] = pair_loss.item()
    
    # Normalize by number of active pairs
    if len(active_pairs) > 0:
        total_loss = total_loss / len(active_pairs)
    
    # Diagnostic information
    info = {
        'active_pairs': len(active_pairs),
        'plane_losses': plane_losses,
        'avg_loss': total_loss.item() if len(active_pairs) > 0 else 0.0
    }
    
    return total_loss, info