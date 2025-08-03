"""
Corrected loss functions based on paper audit feedback.
Main fixes:
1. Adjacency: Use raw gradients, no normalization, linear (1-cos) penalty
2. TV: Use soft (1-w_e) weighting instead of hard assignment
3. Area: Use L1 instead of L2 for better gradient flow
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


def compute_face_gradients(f_values: torch.Tensor, faces: torch.Tensor, 
                          B: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """
    Compute gradients of pairwise field differences on faces.
    CORRECTED: Returns raw gradients without normalization.
    
    Args:
        f_values: Tensor of shape (V, 6) containing field values
        faces: Tensor of shape (F, 3) containing face vertex indices
        B: Tensor of shape (F, 3, 3) containing barycentric matrices
        pairs: Tensor of shape (15, 2) containing channel pair indices
        
    Returns:
        grad15: Tensor of shape (F, 3, 15) containing raw gradients
    """
    # Get field values at face vertices
    Ff = f_values[faces]  # (F, 3, 6)
    
    # Compute gradients for all 6 channels
    grad6 = torch.einsum('fij,fjc->fic', B, Ff)  # (F, 3, 6)
    
    # Compute pairwise gradient differences (RAW, not normalized)
    grad15 = grad6[:, :, pairs[:, 0]] - grad6[:, :, pairs[:, 1]]  # (F, 3, 15)
    
    return grad15


def adjacency_loss_corrected(grad15: torch.Tensor, edge2face: torch.Tensor, 
                            w_e: torch.Tensor, face_mask: Optional[torch.Tensor], 
                            lambda_adj: float, use_squared: bool = False) -> torch.Tensor:
    """
    CORRECTED adjacency loss following the paper exactly.
    
    Key fixes:
    1. Use raw gradients (no per-channel normalization)
    2. Linear (1 - cos) penalty (no squaring)
    3. No threshold on w_e
    4. Normalize by sum of weights
    """
    f1, f2 = edge2face.T
    
    # Only consider interior edges
    interior = (f1 >= 0) & (f2 >= 0)
    
    # Filter out degenerate faces if mask provided
    if face_mask is not None:
        deg1 = (f1 >= 0) & (~face_mask[f1])
        deg2 = (f2 >= 0) & (~face_mask[f2])
        interior = interior & ~(deg1 | deg2)
    
    if interior.sum() == 0:
        return torch.tensor(0.0, device=grad15.device)
    
    # Get gradient pairs for interior edges
    g1 = grad15[f1[interior]]  # (E_interior, 3, 15)
    g2 = grad15[f2[interior]]  # (E_interior, 3, 15)
    
    # Get weights for interior edges
    w_interior = w_e[interior]  # (E_interior, 15)
    
    # VECTORIZED: Compute all pairs at once
    # Compute norms for all pairs
    n1 = g1.norm(dim=1)  # (E_interior, 15)
    n2 = g2.norm(dim=1)  # (E_interior, 15)
    
    # Compute dot products for all pairs
    dot_prod = (g1 * g2).sum(dim=1)  # (E_interior, 15)
    
    # Compute cosine similarity for all pairs
    cos_sim = dot_prod / (n1 * n2 + 1e-10)
    
    # CRITICAL: Clamp cosine to prevent -inf
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
    
    # Penalty: (1 - cos) with optional squaring for better convergence
    penalty = torch.relu(1.0 - cos_sim)  # (E_interior, 15)
    
    if use_squared:
        # Squared penalty for stronger gradients near convergence
        # Divide by 4 to keep same scale as linear version
        penalty = penalty.pow(2) / 4.0
    
    # Additional safety: replace any inf/nan with max penalty
    max_penalty = 0.5 if use_squared else 2.0
    penalty = torch.where(torch.isfinite(penalty), penalty, penalty.new_tensor(max_penalty))
    
    # VECTORIZED: Normalize each pair independently
    # Compute weighted penalties
    weighted_penalties = w_interior * penalty  # (E_interior, 15)
    
    # Sum over edges for each pair, then normalize by weight sum per pair
    pair_losses = weighted_penalties.sum(dim=0)  # (15,)
    weight_sums = w_interior.sum(dim=0).clamp_min(1e-8)  # (15,)
    normalized_losses = pair_losses / weight_sums  # (15,)
    
    # Total adjacency loss is AVERAGE over all pairs (divide by 15)
    L_adj_raw = normalized_losses.mean()  # Changed from sum() to mean()
    
    # Return both weighted and raw for monitoring
    if lambda_adj == 0:
        # Still return the raw value for monitoring
        return torch.tensor(0.0, device=grad15.device), L_adj_raw
    else:
        return lambda_adj * L_adj_raw, L_adj_raw


def gated_tv_loss_corrected(d_v: torch.Tensor, edges: torch.Tensor, 
                           w_e: torch.Tensor, lambda_tv: float) -> torch.Tensor:
    """
    CORRECTED TV loss using soft (1 - w_e) weighting.
    
    Key fix: Use soft complement instead of hard boundary mask.
    """
    va, vb = edges.T
    d_i = d_v[va]  # (E, 15)
    d_j = d_v[vb]  # (E, 15)
    
    # Squared difference in field values
    diff_squared = (d_i - d_j).pow(2)
    
    # CORRECTED: Use w_e*(1-w_e) for better gradient flow
    # This peaks at boundaries (w_e=0.5) and vanishes both inside (w_e=0) and at converged boundaries (w_e=1)
    # Divide by number of channel pairs (15) to match adjacency scale
    gating = w_e * (1 - w_e)  # Maximum at boundaries, zero when converged
    L_tv = (gating * diff_squared).sum() / d_v.shape[1]
    
    return lambda_tv * L_tv


def area_balance_loss_corrected(f_values: torch.Tensor, faces: torch.Tensor, 
                               face_areas: torch.Tensor, face_mask: Optional[torch.Tensor],
                               beta: float, lambda_area: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CORRECTED area balance loss using L1 norm.
    
    Key fixes:
    1. Use same beta as other terms (no capping)
    2. L1 deviation for better gradient flow
    """
    # Compute softmax probabilities at vertices
    Pv = torch.softmax(beta * f_values, dim=1)  # (V, 6)
    
    # Average probabilities over face vertices
    Pf = Pv[faces].mean(dim=1)  # (F, 6)
    
    # Filter out degenerate faces if mask is provided
    if face_mask is not None:
        Pf = Pf[face_mask]
        face_areas_valid = face_areas[face_mask]
    else:
        face_areas_valid = face_areas
    
    # Compute area-weighted fractions
    total_area = face_areas_valid.sum()
    frac = (Pf.T * face_areas_valid).sum(dim=1) / total_area  # (6,)
    
    # CORRECTED: L1 deviation (not L2)
    L_area = lambda_area * torch.abs(frac - 1/6).sum()
    
    return L_area, frac


def compute_edge_weights(d_v: torch.Tensor, edges: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Compute edge weights using sigmoid (NO THRESHOLD).
    
    CORRECTED: No filtering of small weights - they contain important gradients!
    """
    va, vb = edges.T
    d_i = d_v[va]  # (E, 15)
    d_j = d_v[vb]  # (E, 15)
    
    # Element-wise product for each pair
    prod = d_i * d_j  # (E, 15)
    
    # Sigmoid weight (NO THRESHOLD!)
    w_e = torch.sigmoid(-beta * prod)  # (E, 15)
    
    return w_e


def compute_total_loss_corrected(f_values: torch.Tensor,
                                vertices: torch.Tensor,
                                faces: torch.Tensor,
                                edges: torch.Tensor,
                                edge2face: torch.Tensor,
                                face_areas: torch.Tensor,
                                B: torch.Tensor,
                                face_mask: Optional[torch.Tensor] = None,
                                beta: float = 10.0,
                                lambda_area: float = 1.0,
                                lambda_adj: float = 5.0,
                                lambda_tv: float = 0.1,
                                return_components: bool = False) -> Dict[str, torch.Tensor]:
    """
    Compute total loss with all CORRECTED components.
    """
    # Get channel pairs
    import itertools
    pairs = torch.tensor(list(itertools.combinations(range(6), 2)))
    
    # 1. Compute pairwise differences (VECTORIZED)
    # Create index pairs for all combinations
    num_channels = f_values.shape[1]
    idx_i = []
    idx_j = []
    for i in range(num_channels):
        for j in range(i+1, num_channels):
            idx_i.append(i)
            idx_j.append(j)
    
    # Vectorized computation of all pairwise differences
    d_v = f_values[:, idx_i] - f_values[:, idx_j]  # (V, 15)
    
    # 2. Compute edge weights (NO THRESHOLD)
    w_e = compute_edge_weights(d_v, edges, beta)
    
    # 3. Compute face gradients (RAW, not normalized)
    grad15 = compute_face_gradients(f_values, faces, B, pairs)
    
    # 4. Compute individual losses (all corrected)
    L_area, area_frac = area_balance_loss_corrected(f_values, faces, face_areas, face_mask, beta, lambda_area)
    
    # Use squared penalty when beta is large enough (boundaries are well-defined)
    use_squared = beta > 10.0
    L_adj_result = adjacency_loss_corrected(grad15, edge2face, w_e, face_mask, lambda_adj, use_squared=use_squared)
    
    # Handle the tuple return from adjacency_loss_corrected
    if isinstance(L_adj_result, tuple):
        L_adj, L_adj_raw = L_adj_result
    else:
        L_adj = L_adj_result
        L_adj_raw = L_adj / lambda_adj if lambda_adj > 0 else torch.tensor(0.0)
    
    L_tv = gated_tv_loss_corrected(d_v, edges, w_e, lambda_tv)
    
    # 5. Total loss
    total = L_area + L_adj + L_tv
    
    result = {'total': total}
    
    if return_components:
        # For monitoring
        weight_sum = w_e.sum()
        
        result.update({
            'area': L_area,
            'adjacency': L_adj,
            'tv': L_tv,
            'area_fractions': area_frac,
            'weight_sum': weight_sum,  # Monitor this - should drop by 2 orders of magnitude
            'raw_adj_normalized': L_adj_raw  # This is the actual raw adjacency value
        })
    
    return result