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
                            lambda_adj: float) -> torch.Tensor:
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
    
    # Compute loss for each pair
    L_adj = 0.0
    for pair in range(15):
        # Get gradients for this pair
        g1_pair = g1[:, :, pair]  # (E_interior, 3)
        g2_pair = g2[:, :, pair]  # (E_interior, 3)
        
        # Compute norms
        n1 = g1_pair.norm(dim=1)  # (E_interior,)
        n2 = g2_pair.norm(dim=1)  # (E_interior,)
        
        # Compute cosine similarity (with safety for zero gradients)
        dot_prod = (g1_pair * g2_pair).sum(dim=1)  # (E_interior,)
        cos_sim = dot_prod / (n1 * n2 + 1e-10)
        
        # Linear penalty: (1 - cos)
        penalty = 1.0 - cos_sim
        
        # Weight and accumulate
        weighted_penalty = w_interior[:, pair] * penalty
        L_adj += weighted_penalty.sum()
    
    # Normalize by sum of weights (critical!)
    total_weight = w_e.sum().clamp_min(1e-8)
    L_adj = L_adj / total_weight
    
    return lambda_adj * L_adj


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
    
    # CORRECTED: Use soft complement (1 - w_e)
    # This gives smooth transition from 1 inside regions to 0 at boundaries
    L_tv = ((1 - w_e) * diff_squared).sum()
    
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
    
    # 1. Compute pairwise differences
    from loss_functions import compute_pairwise_differences
    d_v, _ = compute_pairwise_differences(f_values)
    
    # 2. Compute edge weights (NO THRESHOLD)
    w_e = compute_edge_weights(d_v, edges, beta)
    
    # 3. Compute face gradients (RAW, not normalized)
    grad15 = compute_face_gradients(f_values, faces, B, pairs)
    
    # 4. Compute individual losses (all corrected)
    L_area, area_frac = area_balance_loss_corrected(f_values, faces, face_areas, face_mask, beta, lambda_area)
    L_adj = adjacency_loss_corrected(grad15, edge2face, w_e, face_mask, lambda_adj)
    L_tv = gated_tv_loss_corrected(d_v, edges, w_e, lambda_tv)
    
    # 5. Total loss
    total = L_area + L_adj + L_tv
    
    result = {'total': total}
    
    if return_components:
        # For monitoring
        weight_sum = w_e.sum()
        raw_adj = L_adj / lambda_adj if lambda_adj > 0 else torch.tensor(0.0)
        
        result.update({
            'area': L_area,
            'adjacency': L_adj,
            'tv': L_tv,
            'area_fractions': area_frac,
            'weight_sum': weight_sum,  # Monitor this - should drop by 2 orders of magnitude
            'raw_adj_normalized': raw_adj
        })
    
    return result