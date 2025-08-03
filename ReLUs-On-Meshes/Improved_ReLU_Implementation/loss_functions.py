"""
Revised loss functions for ReLU mesh segmentation.
Implements the improved formulation from the report.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def compute_pairwise_differences(f_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute all pairwise channel differences.
    
    Args:
        f_values: Tensor of shape (V, 6) containing 6-channel field values
        
    Returns:
        d_v: Tensor of shape (V, 15) containing all pairwise differences
        pairs: Tensor of shape (15, 2) containing channel index pairs
    """
    # Generate all pairs (i, j) where i < j
    pairs = torch.combinations(torch.arange(6, device=f_values.device), r=2)  # (15, 2)
    
    # Compute pairwise differences
    Fi = f_values[:, pairs[:, 0]]  # (V, 15)
    Fj = f_values[:, pairs[:, 1]]  # (V, 15)
    d_v = Fi - Fj  # (V, 15)
    
    return d_v, pairs


def compute_edge_weights(d_v: torch.Tensor, edges: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Compute edge weights for boundary detection.
    
    Args:
        d_v: Tensor of shape (V, 15) containing pairwise differences
        edges: Tensor of shape (E, 2) containing edge vertex indices
        beta: Temperature parameter for sigmoid
        
    Returns:
        w_e: Tensor of shape (E, 15) containing edge weights
    """
    va, vb = edges.T
    w_e = torch.sigmoid(-beta * d_v[va] * d_v[vb])  # (E, 15)
    return w_e


def compute_face_gradients(f_values: torch.Tensor, faces: torch.Tensor, B: torch.Tensor,
                          pairs: torch.Tensor) -> torch.Tensor:
    """
    Compute gradients of pairwise differences on each face.
    
    Args:
        f_values: Tensor of shape (V, 6) containing field values
        faces: Tensor of shape (F, 3) containing face vertex indices
        B: Tensor of shape (F, 3, 3) containing barycentric matrices
        pairs: Tensor of shape (15, 2) containing channel pairs
        
    Returns:
        grad15: Tensor of shape (F, 3, 15) containing gradients
    """
    # Get field values at face vertices
    Ff = f_values[faces]  # (F, 3, 6)
    
    # Compute gradients for all 6 channels
    grad6 = torch.einsum('fij,fjc->fic', B, Ff)  # (F, 3, 6)
    
    # Compute pairwise gradient differences
    grad15 = grad6[:, :, pairs[:, 0]] - grad6[:, :, pairs[:, 1]]  # (F, 3, 15)
    
    return grad15


def adjacency_loss(grad15: torch.Tensor, edge2face: torch.Tensor, w_e: torch.Tensor,
                   face_mask: Optional[torch.Tensor], lambda_adj: float) -> torch.Tensor:
    """
    Compute adjacency loss (revised formulation using local cosine).
    Numerically stable version with proper masking and clamping.
    
    Args:
        grad15: Tensor of shape (F, 3, 15) containing face gradients
        edge2face: Tensor of shape (E, 2) containing face indices per edge
        w_e: Tensor of shape (E, 15) containing edge weights
        lambda_adj: Weight for adjacency loss
        
    Returns:
        L_adj: Adjacency loss value
    """
    eps = 1e-8
    clamp_val = 0.999999
    
    f1, f2 = edge2face.T
    
    # Only consider interior edges (both faces valid)
    interior = (f1 >= 0) & (f2 >= 0)
    
    # Also filter out edges adjacent to degenerate faces
    if face_mask is not None:
        # Check if either face is degenerate
        deg1 = (f1 >= 0) & (~face_mask[f1])
        deg2 = (f2 >= 0) & (~face_mask[f2])
        interior = interior & ~(deg1 | deg2)
    
    if interior.sum() == 0:
        return torch.tensor(0.0, device=grad15.device)
    
    # Get gradient pairs for interior edges
    g1 = grad15[f1[interior]]  # (E_interior, 3, 15)
    g2 = grad15[f2[interior]]  # (E_interior, 3, 15)
    
    # Compute norms
    n1 = g1.norm(dim=1)  # (E_interior, 15)
    n2 = g2.norm(dim=1)  # (E_interior, 15)
    
    # Mask out edges whose gradients are both ~0
    valid = (n1 > eps) & (n2 > eps)  # (E_interior, 15)
    
    if valid.sum() == 0:
        return torch.tensor(0.0, device=grad15.device)
    
    # Compute dot product and cosine similarity for all edges
    dot_prod = (g1 * g2).sum(dim=1)  # (E_interior, 15)
    
    # Compute cosine similarity in fp32 for stability
    cos_theta = (dot_prod / (n1 * n2 + 1e-10)).float()
    cos_theta = cos_theta.clamp(-clamp_val, clamp_val)  # Clamp to (-1, 1)
    
    # Get corresponding weights
    w_interior = w_e[interior]  # (E_interior, 15)
    
    # Apply valid mask and compute loss
    valid_contribution = valid.float() * w_interior * (1.0 - cos_theta)
    
    # Sum all contributions (this is the unnormalized raw sum)
    raw_sum = valid_contribution.sum()
    
    # CRITICAL FIX: Normalize by number of edges and channel pairs
    # This prevents the raw loss from being in the hundreds of thousands
    num_interior_edges = interior.sum().item()
    num_channel_pairs = 15  # 6 choose 2
    normalizer = max(num_interior_edges * num_channel_pairs, 1)
    
    L_adj_normalized = raw_sum / normalizer
    
    # Debug output occasionally
    if torch.rand(1).item() < 0.001:  # 0.1% chance
        print(f"  [ADJ DEBUG] raw_sum={raw_sum:.1f}, normalizer={normalizer}, normalized={L_adj_normalized:.4f}, λ={lambda_adj:.2f}")
    
    L_adj = lambda_adj * L_adj_normalized
    
    return L_adj


def gated_tv_loss(d_v: torch.Tensor, edges: torch.Tensor, w_e: torch.Tensor,
                  edge2face: torch.Tensor, face_mask: Optional[torch.Tensor],
                  lambda_tv: float) -> torch.Tensor:
    """
    Compute gated total variation loss.
    Numerically stable version with clamping to prevent explosions.
    Ignores edges adjacent to degenerate faces.
    
    Args:
        d_v: Tensor of shape (V, 15) containing pairwise differences
        edges: Tensor of shape (E, 2) containing edge vertex indices
        w_e: Tensor of shape (E, 15) containing edge weights
        edge2face: Tensor of shape (E, 2) containing face indices per edge
        face_mask: Boolean tensor of shape (F,) indicating valid faces
        lambda_tv: Weight for TV loss
        
    Returns:
        L_tv: Gated TV loss value
    """
    tv_clip = 2e2  # Reduced from 1e3 for better stability
    
    # Filter edges - ignore if either adjacent face is degenerate
    if face_mask is not None:
        f1, f2 = edge2face.T
        # Check if faces are degenerate
        deg1 = (f1 >= 0) & (~face_mask[f1])
        deg2 = (f2 >= 0) & (~face_mask[f2])
        good_edge = ~(deg1 | deg2)
        
        # Filter edges
        edges = edges[good_edge]
        w_e = w_e[good_edge]
    
    if len(edges) == 0:
        return torch.tensor(0.0, device=d_v.device)
    
    va, vb = edges.T
    d_i = d_v[va]  # (E_good, 15)
    d_j = d_v[vb]  # (E_good, 15)
    
    # Compute squared differences with clamping
    diff_squared = (d_i - d_j).pow(2).clamp(max=tv_clip)
    
    # Gated TV: (1 - w_e) * clamped_diff^2
    L_tv = lambda_tv * ((1 - w_e) * diff_squared).sum()
    
    return L_tv


def area_balance_loss(f_values: torch.Tensor, faces: torch.Tensor, face_areas: torch.Tensor,
                     face_mask: Optional[torch.Tensor], beta: float, lambda_area: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute area balance loss (L1 deviation from 1/6).
    
    Args:
        f_values: Tensor of shape (V, 6) containing field values
        faces: Tensor of shape (F, 3) containing face vertex indices
        face_areas: Tensor of shape (F,) containing face areas
        beta: Temperature parameter for softmax
        lambda_area: Weight for area loss
        
    Returns:
        L_area: Area balance loss value
        frac: Tensor of shape (6,) containing area fractions
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
    
    # L1 deviation from uniform distribution
    L_area = lambda_area * torch.abs(frac - 1/6).sum()
    
    return L_area, frac


def compute_total_loss(f_values: torch.Tensor,
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
    Compute the total loss with all components.
    
    Args:
        f_values: Tensor of shape (V, 6) containing field values
        vertices: Tensor of shape (V, 3) containing vertex positions
        faces: Tensor of shape (F, 3) containing face indices
        edges: Tensor of shape (E, 2) containing edge indices
        edge2face: Tensor of shape (E, 2) containing face adjacency
        face_areas: Tensor of shape (F,) containing face areas
        B: Tensor of shape (F, 3, 3) containing barycentric matrices
        beta: Temperature parameter
        lambda_area: Weight for area balance
        lambda_adj: Weight for adjacency
        lambda_tv: Weight for TV regularization
        return_components: If True, return individual loss components
        
    Returns:
        Dictionary containing 'total' loss and optionally individual components
    """
    # 1. Compute pairwise differences
    d_v, pairs = compute_pairwise_differences(f_values)
    
    # 2. Compute edge weights
    w_e = compute_edge_weights(d_v, edges, beta)
    
    # 3. Compute face gradients
    grad15 = compute_face_gradients(f_values, faces, B, pairs)
    
    # 4. Compute individual losses
    L_area, area_frac = area_balance_loss(f_values, faces, face_areas, face_mask, beta, lambda_area)
    L_adj = adjacency_loss(grad15, edge2face, w_e, face_mask, lambda_adj)
    L_tv = gated_tv_loss(d_v, edges, w_e, edge2face, face_mask, lambda_tv)
    
    # 5. Total loss
    total = L_area + L_adj + L_tv
    
    result = {'total': total}
    
    if return_components:
        # For debugging: compute raw normalized adjacency (before lambda multiplication)
        # This is the actual normalized value that should be in [0, 2] range
        raw_adj_normalized = L_adj / lambda_adj if lambda_adj > 0 else torch.tensor(0.0)
        
        result.update({
            'area': L_area,
            'adjacency': L_adj,
            'tv': L_tv,
            'area_fractions': area_frac,
            'raw_adj_normalized': raw_adj_normalized  # Add this for debugging
        })
    
    return result


class GradNorm:
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing.
    Based on "GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks"
    """
    def __init__(self, num_tasks: int = 3, alpha: float = 1.5):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.weights = torch.ones(num_tasks) / num_tasks
        self.initial_losses = None
        
    def update_weights(self, losses: Dict[str, torch.Tensor], shared_params: torch.nn.Parameter):
        """
        Update task weights based on gradient magnitudes.
        
        Args:
            losses: Dictionary of individual loss components
            shared_params: Shared parameters (e.g., f_values)
        """
        if self.initial_losses is None:
            self.initial_losses = {k: v.item() for k, v in losses.items() 
                                 if k not in ['total', 'area_fractions']}
            return
        
        # Compute gradients for each task
        grads = []
        loss_ratios = []
        
        for i, (key, loss) in enumerate(losses.items()):
            if key in ['total', 'area_fractions']:
                continue
                
            # Compute gradient magnitude
            grad = torch.autograd.grad(loss, shared_params, retain_graph=True)[0]
            grad_norm = grad.norm()
            grads.append(grad_norm)
            
            # Compute loss ratio
            ratio = loss.item() / (self.initial_losses[key] + 1e-8)
            loss_ratios.append(ratio)
        
        grads = torch.stack(grads)
        loss_ratios = torch.tensor(loss_ratios)
        
        # Safety check - skip update if any gradient is NaN
        if torch.isnan(grads).any() or torch.isinf(grads).any():
            return
        
        # Compute mean gradient norm
        mean_grad = grads.mean()
        
        # Compute relative training rates
        relative_rates = loss_ratios / loss_ratios.mean()
        
        # Update weights with safe division
        for i in range(self.num_tasks):
            target = mean_grad * (relative_rates[i] ** self.alpha)
            # Avoid division by zero with safe scaling
            safe = grads[i] > 1e-12
            if safe:
                scale = (target / grads[i].clamp(min=1e-12)).item()
            else:
                scale = 1.0
            self.weights[i] *= scale
        
        # Handle NaN/inf and normalize weights
        self.weights = torch.nan_to_num(self.weights, nan=1.0/self.num_tasks, posinf=1.0, neginf=1.0)
        self.weights = self.weights / self.weights.sum()
        
    def get_weighted_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply current weights to losses."""
        weighted_loss = 0
        i = 0
        for key, loss in losses.items():
            if key != 'total' and key != 'area_fractions':
                weighted_loss += self.weights[i] * loss
                i += 1
        return weighted_loss