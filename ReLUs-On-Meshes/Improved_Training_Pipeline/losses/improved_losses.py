"""
Improved loss functions addressing convergence issues from the report.
Key improvements:
- Intrinsic contour alignment (replaces 3D SVD-based plane fitting)
- Cotangent Laplacian smoothness (replaces unnormalized edge differences)  
- KL divergence area balance (replaces L1 loss)
- Soft pin penalty (replaces hard projection)
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """
    Safe normalization that avoids NaN gradients when x≈0.
    
    Args:
        x: Input tensor to normalize
        dim: Dimension along which to normalize
        eps: Small value to clamp the norm (default 1e-6)
        
    Returns:
        Normalized tensor with stable gradients
    """
    # Clamp the denominator so backward never sees a divide-by-zero
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def grad3d_intrinsic(h_vals: torch.Tensor, v0: torch.Tensor,
                     v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Compute intrinsic gradient on triangles using Gram matrix (batched).
    
    Args:
        h_vals: (B,3) scalar values at triangle vertices
        v0,v1,v2: (B,3) triangle vertex positions
        
    Returns:
        g: (B,3) gradient in R^3 lying in the triangle plane
    """
    e0 = v1 - v0                      # (B,3)
    e1 = v2 - v0                      # (B,3)
    b = torch.stack([h_vals[:,1]-h_vals[:,0],
                     h_vals[:,2]-h_vals[:,0]], dim=1)  # (B,2)

    # Do geometry in float64 for robustness, cast back at the end
    dt = torch.float64
    e0d, e1d, bd = e0.to(dt), e1.to(dt), b.to(dt)

    # Gram matrix G = [[<e0,e0>, <e0,e1>],
    #                  [<e1,e0>, <e1,e1>]]
    G00 = (e0d*e0d).sum(dim=1)          # (B,)
    G01 = (e0d*e1d).sum(dim=1)          # (B,)
    G11 = (e1d*e1d).sum(dim=1)          # (B,)
    det = (G00*G11 - G01*G01)
    
    # Mask degenerate triangles
    mask_degenerate = det <= 1e-10
    det = det.clamp_min(1e-10)
    
    invG00 =  G11 / det
    invG01 = -G01 / det
    invG11 =  G00 / det

    # coefficients a in the (e0,e1) basis: a = G^{-1} b
    a0 = invG00*bd[:,0] + invG01*bd[:,1]    # (B,)
    a1 = invG01*bd[:,0] + invG11*bd[:,1]    # (B,)
    
    # Zero out coefficients for degenerate triangles (no reliable gradient)
    a0 = a0.masked_fill(mask_degenerate, 0.0)
    a1 = a1.masked_fill(mask_degenerate, 0.0)

    # gradient in R^3: g = a0*e0 + a1*e1
    gd = a0[:,None]*e0d + a1[:,None]*e1d     # (B,3)
    g = gd.to(h_vals.dtype)
    
    # Guard: if anything still slipped through, zero it (rare)
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


def contour_alignment_intrinsic(F: torch.Tensor,
                               faces: torch.Tensor,
                               edge_idx: torch.Tensor,
                               edge_tris: torch.Tensor,
                               beta_contour: float,
                               top_k: int = 2,
                               return_weights: bool = False,
                               verts: Optional[torch.Tensor] = None,
                               tri_xy: Optional[torch.Tensor] = None,
                               debug_log: bool = False) -> torch.Tensor:
    """
    Contour alignment loss with proper 3D alignment.
    
    Key improvements:
    - Computes boundary tangents in 3D space for proper alignment
    - No SVD or global plane fitting
    - Pair gating to avoid O(C^2) noise
    - Robust Charbonnier penalty
    
    Args:
        F: (N, C) multi-channel field values
        faces: (T, 3) face indices
        edge_idx: (E, 2) edge vertex indices
        edge_tris: (E, 2) adjacent triangle indices (-1 for boundary)
        beta_contour: Temperature parameter for edge crossing detection
        top_k: Number of top channels to consider (pair gating)
        return_weights: Whether to return edge weights
        verts: (N, 3) vertex positions in 3D (required for 3D alignment)
        tri_xy: (T, 3, 2) DEPRECATED - kept for compatibility, not used
        
    Returns:
        loss: Scalar contour alignment loss (normalized)
        weights: (E,) edge weights if return_weights=True
    """
    device, dtype = F.device, F.dtype
    C = F.shape[1]
    
    # Ensure input tensors are on the same device as F
    edge_idx = edge_idx.to(device)
    edge_tris = edge_tris.to(device)
    faces = faces.to(device)
    
    # Need 3D vertices for proper alignment
    if verts is None:
        raise ValueError("verts must be provided for 3D contour alignment")
    verts = verts.to(device)
    
    # No need to precompute frames - we'll compute normals per active triangle
    
    # Filter out boundary edges
    valid_mask = (edge_tris[:, 0] >= 0) & (edge_tris[:, 1] >= 0)
    valid_edges = edge_idx[valid_mask]  # (E_valid, 2)
    valid_tris = edge_tris[valid_mask]  # (E_valid, 2)
    
    if valid_edges.shape[0] == 0:
        return torch.tensor(0., device=device, dtype=dtype)
    
    # Edge midpoint field values
    f_mid = 0.5 * (F[valid_edges[:, 0]] + F[valid_edges[:, 1]])  # (E_valid, C)
    
    # Pair gating: select top 2 channels per edge
    if top_k < C and C > 1:
        top_vals, top_idx = torch.topk(f_mid, k=min(top_k, C), dim=1)  # (E_valid, top_k)
        # Use first two channels from top-k
        chan_i = top_idx[:, 0]  # (E_valid,)
        chan_j = top_idx[:, 1] if top_idx.shape[1] > 1 else top_idx[:, 0]
    else:
        # Use fixed channel pair for all edges
        chan_i = torch.zeros(valid_edges.shape[0], dtype=torch.long, device=device)
        chan_j = torch.ones(valid_edges.shape[0], dtype=torch.long, device=device)
    
    # Edge crossing weights
    da = F[valid_edges[:, 0], chan_i] - F[valid_edges[:, 0], chan_j]  # (E_valid,)
    db = F[valid_edges[:, 1], chan_i] - F[valid_edges[:, 1], chan_j]  # (E_valid,)
    w = torch.sigmoid(-beta_contour * da * db)  # (E_valid,)
    
    # NEW: Confidence gate - only treat high-confidence crossings as boundaries
    conf = 0.5 * (da.abs() + db.abs())
    k = 5.0  # sharpness
    m = 0.3  # margin threshold (could be scheduled from 0.1 -> 0.4)
    w = w * torch.sigmoid(k * (conf - m))
    
    # Soft gating to maintain gradient flow
    w = w.clamp_min(1e-3)  # Keep gradients flowing
    
    # Use all edges with non-zero weights for soft gating
    active_edges = valid_edges
    active_tris = valid_tris
    active_chan_i = chan_i
    active_chan_j = chan_j
    active_w = w
    
    # Additional robustness: downweight edges where a third channel is close
    if C > 2:
        # Get the 3rd highest value at edge midpoints
        # Note: we're using all edges now (soft gating)
        top3_vals, _ = torch.topk(f_mid, k=min(3, C), dim=1)
        if top3_vals.shape[1] == 3:
            # Difference between 2nd and 3rd channels
            gap = top3_vals[:, 1] - top3_vals[:, 2]
            # Downweight when gap is small (triple point)
            triple_weight = torch.sigmoid(10.0 * (gap - 0.1))
            active_w = active_w * triple_weight
    
    # Store original edges for gradient/length gating
    original_active_edges = active_edges
    
    # Get triangles
    tL = active_tris[:, 0]  # (E_active,)
    tR = active_tris[:, 1]  # (E_active,)
    
    # Vectorized height computation for left triangles
    faces_L = faces[tL]  # (E_active, 3)
    F_L = F[faces_L]  # (E_active, 3, C)
    h_L = F_L[torch.arange(len(tL)), :, active_chan_i] - \
          F_L[torch.arange(len(tL)), :, active_chan_j]  # (E_active, 3)
    
    # Vectorized height computation for right triangles  
    faces_R = faces[tR]  # (E_active, 3)
    F_R = F[faces_R]  # (E_active, 3, C)
    h_R = F_R[torch.arange(len(tR)), :, active_chan_i] - \
          F_R[torch.arange(len(tR)), :, active_chan_j]  # (E_active, 3)
    
    # Get triangle vertices for intrinsic gradient computation
    v0L = verts[faces[tL, 0]]  # (E_active, 3)
    v1L = verts[faces[tL, 1]]  # (E_active, 3)
    v2L = verts[faces[tL, 2]]  # (E_active, 3)
    
    v0R = verts[faces[tR, 0]]  # (E_active, 3)
    v1R = verts[faces[tR, 1]]  # (E_active, 3)
    v2R = verts[faces[tR, 2]]  # (E_active, 3)
    
    # Compute intrinsic 3D gradients using Gram matrix
    g_L_3d = grad3d_intrinsic(h_L, v0L, v1L, v2L)  # (E_active, 3)
    g_R_3d = grad3d_intrinsic(h_R, v0R, v1R, v2R)  # (E_active, 3)
    
    # Compute triangle normals using safe normalization
    e0L = v1L - v0L
    e1L = v2L - v0L
    n_L = safe_normalize(torch.cross(e0L, e1L, dim=1), dim=1, eps=1e-6)
    
    e0R = v1R - v0R
    e1R = v2R - v0R
    n_R = safe_normalize(torch.cross(e0R, e1R, dim=1), dim=1, eps=1e-6)
    
    # Project gradients into the triangle plane first (improves stability)
    g_L_3d = g_L_3d - (g_L_3d * n_L).sum(dim=1, keepdim=True) * n_L
    g_R_3d = g_R_3d - (g_R_3d * n_R).sum(dim=1, keepdim=True) * n_R
    
    # Boundary tangent in 3D: tau = normalize(n × g)
    tau_L = torch.cross(n_L, g_L_3d, dim=1)  # (E_active, 3)
    tau_R = torch.cross(n_R, g_R_3d, dim=1)  # (E_active, 3)
    
    # Normalize tangents using safe normalization
    tau_L = safe_normalize(tau_L, dim=1, eps=1e-6)
    tau_R = safe_normalize(tau_R, dim=1, eps=1e-6)
    
    # Guard: if any NaN snuck in
    tau_L = torch.nan_to_num(tau_L)
    tau_R = torch.nan_to_num(tau_R)
    active_w = torch.nan_to_num(active_w, nan=1e-3, posinf=1.0, neginf=1e-3)
    
    # Gradient magnitude gating: stronger signals = more reliable tangents
    mag_L = g_L_3d.norm(dim=1)  # (E_active,)
    mag_R = g_R_3d.norm(dim=1)
    
    # Down-weight edges if either adjacent triangle was degenerate
    tiny = 1e-12
    deg_mask = (mag_L < tiny) | (mag_R < tiny)
    active_w = active_w * (~deg_mask).float()
    
    grad_gate = torch.sqrt(mag_L * mag_R)  # geometric mean
    # Normalize but keep relative strength (ignore NaNs)
    scale = torch.nanmedian(grad_gate).detach().clamp_min(1e-6)  # ignore NaNs
    grad_gate = (grad_gate / scale).clamp(0.0, 2.0)  # Cap at 2.0 for high β stability
    grad_gate = torch.nan_to_num(grad_gate, nan=0.0, posinf=2.0, neginf=0.0)
    
    # Edge length gating: longer edges matter more
    edge_vec = verts[original_active_edges[:, 1]] - verts[original_active_edges[:, 0]]
    edge_len = edge_vec.norm(dim=1)  # (E_active,)
    len_gate = (edge_len / (edge_len.mean().detach() + 1e-12)).clamp(0.5, 2.0)
    
    # Apply both gates to weights
    active_w = active_w * grad_gate * len_gate
    
    # Debug logging if requested
    if debug_log:
        print(f"[DEBUG contour] grad_gate: min={grad_gate.min():.6f}, max={grad_gate.max():.6f}, "
              f"median={torch.nanmedian(grad_gate):.6f}, %nan={torch.isnan(grad_gate).float().mean():.1%}")
        print(f"[DEBUG contour] active_w: min={active_w.min():.6f}, max={active_w.max():.6f}, "
              f"sum={active_w.sum():.3f}")
    
    # Alignment in 3D: 1 - |cos θ|
    cos_angle = (tau_L * tau_R).sum(dim=1).abs().clamp(max=1.0)  # (E_active,)
    misalignment = 1.0 - cos_angle
    
    # Charbonnier penalty
    epsilon = 1e-6
    loss_contrib = torch.sqrt(misalignment * misalignment + epsilon)
    
    # Weighted mean
    total_loss = (active_w * loss_contrib).sum()
    total_weight = active_w.sum() + 1e-9
    
    if return_weights:
        # Return full edge weights (including inactive edges)
        full_weights = torch.zeros(edge_idx.shape[0], device=device, dtype=dtype)
        # Map valid edges back
        valid_indices = torch.where(valid_mask)[0]
        full_weights[valid_indices] = active_w
        return total_loss / total_weight, full_weights
    
    return total_loss / total_weight


def smoothness_cotan(F: torch.Tensor, 
                    I: torch.Tensor, 
                    J: torch.Tensor, 
                    W: torch.Tensor) -> torch.Tensor:
    """
    Cotangent Laplacian smoothness loss.
    This replaces the unnormalized edge-based smoothness (report section 4.3.3.2).
    
    Args:
        F: (N, C) multi-channel field values
        I: (K,) source vertex indices
        J: (K,) target vertex indices  
        W: (K,) cotangent weights
        
    Returns:
        loss: Normalized smoothness loss
    """
    diff = F[I] - F[J]  # (K, C)
    squared_diff = (diff * diff).sum(dim=-1)  # (K,)
    
    # Weighted sum normalized by total weight
    numerator = (W * squared_diff).sum()
    denominator = W.sum().clamp_min(1e-12)
    
    return numerator / denominator


def area_fractions_and_kl(F: torch.Tensor, 
                         faces: torch.Tensor, 
                         tri_area: torch.Tensor, 
                         beta_area: float,
                         use_entropy_regularization: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Area balance loss using KL divergence to uniform distribution.
    This replaces L1 loss which has vanishing gradients (report section 4.3.2).
    
    Args:
        F: (N, C) multi-channel field values
        faces: (T, 3) face indices
        tri_area: (T,) triangle areas
        beta_area: Temperature parameter for softmax
        
    Returns:
        loss: KL divergence loss
        frac: (C,) area fractions per channel
    """
    C = F.shape[1]
    
    # Barycentric sampling points (center + edge midpoints)
    bary = torch.tensor([[1/3, 1/3, 1/3],
                        [1/2, 1/2, 0.0],
                        [1/2, 0.0, 1/2],
                        [0.0, 1/2, 1/2]], device=F.device, dtype=F.dtype)  # (4, 3)
    
    # Sample field at barycentric points
    Ft = F[faces]  # (T, 3, C)
    Ft_sampled = torch.einsum('sb,tbc->tsc', bary, Ft)  # (T, 4, C)
    
    # Softmax probabilities
    P = torch.softmax(beta_area * Ft_sampled, dim=-1)  # (T, 4, C)
    P_mean = P.mean(dim=1)  # (T, C) average over sample points
    
    # Area-weighted channel probabilities
    area_per_channel = (tri_area[:, None] * P_mean).sum(dim=0)  # (C,)
    total_area = tri_area.sum().clamp_min(1e-12)
    frac = area_per_channel / total_area
    
    # KL(frac || uniform) = sum_c frac_c * log(frac_c / uniform)
    uniform = 1.0 / C
    # Clamp frac to avoid log(0)
    frac = frac.clamp(min=1e-8, max=1-1e-8)
    kl_loss = (frac * torch.log(frac / uniform)).sum()
    
    # During warmup, use variance loss for stronger gradient signal
    if use_entropy_regularization:
        # Variance of area fractions from uniform
        variance_loss = ((frac - uniform) ** 2).sum() * C * 10.0  # Scale up for stronger signal
        
        # Also add entropy regularization to prevent winner-take-all
        P_mean_clamped = P_mean.clamp(min=1e-8, max=1-1e-8)
        entropy_per_tri = -(P_mean_clamped * torch.log(P_mean_clamped)).sum(dim=1)  # (T,)
        mean_entropy = (tri_area * entropy_per_tri).sum() / total_area
        max_entropy = -torch.log(torch.tensor(1.0/C, device=F.device))
        entropy_loss = (max_entropy - mean_entropy) * 0.1
        
        # Return variance + entropy loss instead of KL during warmup
        return variance_loss + entropy_loss, frac
    
    return kl_loss, frac


def pin_loss(F: torch.Tensor, 
            pin_idx: torch.Tensor, 
            pin_target: torch.Tensor,
            use_huber: bool = True,
            delta: float = 1.0) -> torch.Tensor:
    """
    Soft pinning penalty (annealed to hard constraint).
    This replaces hard projection after each step (report section 4.4.7).
    
    Args:
        F: (N, C) multi-channel field values
        pin_idx: (P,) indices of pinned vertices
        pin_target: (P, C) target values for pinned vertices
        use_huber: Whether to use Huber loss for robustness
        delta: Huber loss threshold
        
    Returns:
        loss: Pin constraint loss
    """
    if pin_idx.numel() == 0:
        return torch.tensor(0., device=F.device, dtype=F.dtype)
    
    diff = F[pin_idx] - pin_target  # (P, C)
    
    if use_huber:
        # Huber loss for robustness
        abs_diff = diff.abs()
        delta_t = torch.full_like(abs_diff, delta)
        quadratic = torch.minimum(abs_diff, delta_t)
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic**2 + delta * linear
        return loss.mean()
    else:
        # Standard L2 loss
        return (diff * diff).mean()


def compute_boundary_stats(F: torch.Tensor,
                          edge_idx: torch.Tensor,
                          verts: torch.Tensor,
                          beta_contour: float,
                          top_k: int = 2) -> Tuple[float, float]:
    """
    Compute boundary length and active edge fraction for monitoring.
    
    Args:
        F: (N, C) multi-channel field values
        edge_idx: (E, 2) edge vertex indices
        verts: (N, 3) vertex positions
        beta_contour: Temperature parameter
        top_k: Number of top channels to consider
        
    Returns:
        length: Estimated boundary length
        active_fraction: Fraction of edges with w > 0.5
    """
    device = F.device
    C = F.shape[1]
    
    # Ensure edge_idx is on same device
    edge_idx = edge_idx.to(device)
    
    # Edge midpoint values
    f_mid = 0.5 * (F[edge_idx[:, 0]] + F[edge_idx[:, 1]])  # (E, C)
    
    # Get top 2 channels per edge
    if top_k < C and C > 1:
        top_vals, top_idx = torch.topk(f_mid, k=min(top_k, C), dim=1)
        chan_i = top_idx[:, 0]
        chan_j = top_idx[:, 1] if top_idx.shape[1] > 1 else top_idx[:, 0]
    else:
        chan_i = torch.zeros(edge_idx.shape[0], dtype=torch.long, device=device)
        chan_j = torch.ones(edge_idx.shape[0], dtype=torch.long, device=device)
    
    # Edge crossing weights
    da = F[edge_idx[:, 0], chan_i] - F[edge_idx[:, 0], chan_j]
    db = F[edge_idx[:, 1], chan_i] - F[edge_idx[:, 1], chan_j]
    w = torch.sigmoid(-beta_contour * da * db)
    
    # Active edges (w > 0.5)
    active_mask = w > 0.5
    active_fraction = active_mask.float().mean().item()
    
    # Boundary length (sum of active edge lengths)
    if active_mask.any():
        active_edges = edge_idx[active_mask]
        edge_lengths = (verts[active_edges[:, 0]] - verts[active_edges[:, 1]]).norm(dim=1)
        length = edge_lengths.sum().item()
    else:
        length = 0.0
    
    return length, active_fraction


def compute_boundary_length_estimate(F: torch.Tensor,
                                   edge_idx: torch.Tensor,
                                   verts: torch.Tensor,
                                   beta_contour: float,
                                   top_k: int = 2) -> float:
    """
    Backward compatibility wrapper.
    """
    length, _ = compute_boundary_stats(F, edge_idx, verts, beta_contour, top_k)
    return length


def non_boundary_margin_loss(F: torch.Tensor, 
                            edge_idx: torch.Tensor, 
                            edge_weights: torch.Tensor,
                            tau: float = 0.3) -> torch.Tensor:
    """
    Encourage vertices not on boundaries to have clear winner channels.
    
    Args:
        F: (N, C) field values
        edge_idx: (E, 2) edge indices
        edge_weights: (E,) boundary weights from contour loss
        tau: Minimum margin to enforce
        
    Returns:
        loss: Margin sharpening loss
    """
    # Ensure tensors are on same device
    device = F.device
    edge_idx = edge_idx.to(device)
    edge_weights = edge_weights.to(device)
    
    # Edges that are clearly not on boundaries
    nb_mask = (edge_weights.detach() < 0.2)
    if not nb_mask.any():
        return F.new_tensor(0.0)
    
    va = edge_idx[nb_mask, 0]
    vb = edge_idx[nb_mask, 1]
    
    # Get margins (top1 - top2) for vertices
    def margin(v_idx):
        top2_vals, _ = torch.topk(F[v_idx], k=2, dim=1)
        return top2_vals[:, 0] - top2_vals[:, 1]
    
    # Take minimum margin across edge endpoints
    m = torch.minimum(margin(va), margin(vb))
    
    # Penalize margins below tau
    return torch.nn.functional.relu(tau - m).mean()


def total_variation_loss(F: torch.Tensor,
                        edge_idx: torch.Tensor,
                        edge_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Total variation regularization for additional smoothness.
    
    Args:
        F: (N, C) multi-channel field values
        edge_idx: (E, 2) edge vertex indices
        edge_lengths: (E,) optional edge lengths for weighting
        
    Returns:
        loss: Total variation loss
    """
    va, vb = edge_idx.T
    diff = F[va] - F[vb]  # (E, C)
    
    # L1 norm per edge
    tv_per_edge = diff.abs().sum(dim=-1)  # (E,)
    
    if edge_lengths is not None:
        # Weight by inverse edge length
        weights = 1.0 / (edge_lengths + 1e-12)
        tv_per_edge = tv_per_edge * weights
        normalizer = weights.sum()
    else:
        normalizer = edge_idx.shape[0]
    
    return tv_per_edge.sum() / normalizer


def potts_smoothness_loss(F: torch.Tensor,
                         edge_idx: torch.Tensor,
                         edge_weights: torch.Tensor,
                         beta_area: float,
                         gamma: float = 2.0) -> torch.Tensor:
    """
    Potts/TV-style smoothness on soft probabilities to reduce speckles.
    
    Args:
        F: (N, C) field values
        edge_idx: (E, 2) edge indices
        edge_weights: (E,) boundary weights from contour loss (activity)
        beta_area: Temperature for softmax
        gamma: Exponent for gating away from boundaries (default 2.0 for softer gating)
        
    Returns:
        loss: Potts smoothness loss
    """
    device = F.device
    edge_idx = edge_idx.to(device)
    edge_weights = edge_weights.to(device)
    
    # Compute soft probabilities
    p = torch.softmax(beta_area * F, dim=1)  # (N, C)
    
    # Get edge endpoint probabilities
    va, vb = edge_idx.T
    p_a = p[va]  # (E, C)
    p_b = p[vb]  # (E, C)
    
    # Potts loss: 1 - p_i^T p_j = 0.5 * ||p_i - p_j||^2
    potts_per_edge = 1.0 - (p_a * p_b).sum(dim=1)  # (E,)
    
    # Gate away from boundaries
    gate = (1.0 - edge_weights.detach()).pow(gamma)
    
    # Weighted mean
    return (gate * potts_per_edge).sum() / (gate.sum() + 1e-9)


def boundary_length_regularizer(edge_idx: torch.Tensor,
                               edge_weights: torch.Tensor,
                               verts: torch.Tensor) -> torch.Tensor:
    """
    Regularize total boundary length to reduce ragged seams (scale-invariant).
    
    Args:
        edge_idx: (E, 2) edge indices
        edge_weights: (E,) boundary weights (activity)
        verts: (N, 3) vertex positions
        
    Returns:
        loss: Normalized boundary length penalty
    """
    device = verts.device
    edge_idx = edge_idx.to(device)
    edge_weights = edge_weights.to(device)
    
    # Edge lengths
    va, vb = edge_idx.T
    edge_lengths = (verts[va] - verts[vb]).norm(dim=1)  # (E,)
    
    # Normalize by total edge length for scale invariance
    denom = edge_lengths.sum().clamp_min(1e-9)
    
    # Weighted sum of active edge lengths, normalized
    return (edge_weights * edge_lengths).sum() / denom


def normal_axis_losses(verts: torch.Tensor,
                      faces: torch.Tensor,
                      tri_area: torch.Tensor,
                      F_field: torch.Tensor,
                      beta_area: float,
                      axis_per_channel: torch.Tensor,
                      eps: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Normal-based losses to encourage axis-aligned patches.
    
    Args:
        verts: (N, 3) vertex positions
        faces: (T, 3) face indices
        tri_area: (T,) triangle areas
        F_field: (N, C) multi-channel field values
        beta_area: Temperature for softmax
        axis_per_channel: (C, 3) unit vectors for each channel's target axis
        eps: Small value for numerical stability
        
    Returns:
        loss_align: Mean axis misalignment across channels
        loss_disp: Mean within-patch normal dispersion across channels
    """
    device = verts.device
    
    # Face normals (unit)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = torch.cross(v1 - v0, v2 - v0, dim=1)
    n = safe_normalize(n, dim=1, eps=1e-6)  # (T, 3)
    
    # Soft probabilities per vertex and then per face (average over the 3 vertices)
    p_v = torch.softmax(beta_area * F_field, dim=1)  # (N, C)
    p_f = (p_v[faces[:, 0]] + p_v[faces[:, 1]] + p_v[faces[:, 2]]) / 3.0  # (T, C)
    
    # Area-weighted per-channel mean normals
    w = tri_area[:, None] * p_f  # (T, C)
    m = (w[:, :, None] * n[:, None, :]).sum(dim=0)  # (C, 3)
    m = safe_normalize(m, dim=1, eps=1e-6)  # (C, 3)
    
    # (a) Axis alignment: 1 - |dot(mean_normal, axis)|
    axes = safe_normalize(axis_per_channel, dim=1, eps=1e-6)
    misalign = 1.0 - (m * axes).sum(dim=1).abs()  # (C,)
    loss_align = misalign.mean()
    
    # (b) Dispersion: 1 - cos^2 between each face normal and its patch mean
    cos = (n[:, None, :] * m[None, :, :]).sum(dim=2).clamp(-1, 1)  # (T, C)
    disp_per_face = (1.0 - cos**2)  # (T, C)
    loss_disp = (w * disp_per_face).sum(dim=0) / (w.sum(dim=0) + eps)  # (C,)
    loss_disp = loss_disp.mean()
    
    return loss_align, loss_disp


def area_kl_to_prior(frac: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """
    KL divergence from area fractions to a non-uniform prior.
    Useful when you want 5 patches instead of 6.
    
    Args:
        frac: (C,) actual area fractions that sum to 1
        prior: (C,) target area fractions that sum to 1
        
    Returns:
        KL divergence loss
    """
    eps = 1e-8
    f = frac.clamp(eps, 1.0)
    q = prior.clamp(eps, 1.0)
    return (f * (f / q).log()).sum()


def contour_alignment_intrinsic_v2(
    F: torch.Tensor,
    faces: torch.Tensor,
    edge_idx: torch.Tensor,
    edge_tris: torch.Tensor,
    verts: torch.Tensor,
    beta_contour: float = 6.0,
    eps: float = 1e-6,
    return_weights: bool = False,
    debug_log: bool = False,
) -> torch.Tensor:
    """
    Robust contour alignment:
      - all-pairs per-edge with per-edge softmax over pairs
      - soft-OR coverage weighting per edge
      - intrinsic 3D gradient/tangent as before
    """
    device, dtype = F.device, F.dtype
    N, C = F.shape
    edge_idx   = edge_idx.to(device)
    edge_tris  = edge_tris.to(device)
    faces      = faces.to(device)
    verts      = verts.to(device)

    # only interior edges
    valid = (edge_tris[:,0] >= 0) & (edge_tris[:,1] >= 0)
    if not torch.any(valid):
        z = F.new_tensor(0.0)
        return (z, torch.zeros_like(valid, dtype=F.dtype)) if return_weights else z

    va = edge_idx[valid,0]
    vb = edge_idx[valid,1]
    tL = edge_tris[valid,0]
    tR = edge_tris[valid,1]

    # ----- pairwise crossing weights on edges -----
    # all channel pairs
    ii, jj = torch.triu_indices(C, C, offset=1, device=device)  # P = C*(C-1)//2
    # values at endpoints for all pairs
    da = F[va][:, ii] - F[va][:, jj]      # (E,P)
    db = F[vb][:, ii] - F[vb][:, jj]      # (E,P)

    # crossing score: sign change & confidence
    w_pairs = torch.sigmoid(-beta_contour * da * db)           # (E,P)
    conf    = 0.5*(da.abs() + db.abs())
    w_pairs = w_pairs * torch.sigmoid(5.0*(conf - 0.2))        # soften but keep grads
    w_pairs = w_pairs.clamp_min(1e-6)

    # per-edge mixing over pairs (prevents cherry-picking)
    pair_mix = w_pairs / (w_pairs.sum(dim=1, keepdim=True) + eps)  # (E,P)

    # soft-OR edge activity (coverage)
    phi = 1.0 - torch.prod(1.0 - w_pairs, dim=1)  # (E,)

    # ----- per-pair 3D tangents on each side -----
    faces_L = faces[tL]                     # (E,3)
    faces_R = faces[tR]
    F_L     = F[faces_L]                    # (E,3,C)
    F_R     = F[faces_R]

    # build h values for all pairs: (E,3,P)
    hL = F_L[:,:,ii] - F_L[:,:,jj]
    hR = F_R[:,:,ii] - F_R[:,:,jj]

    # triangle vertices
    v0L, v1L, v2L = [verts[faces_L[:,k]] for k in (0,1,2)]
    v0R, v1R, v2R = [verts[faces_R[:,k]] for k in (0,1,2)]

    # reuse your intrinsic gradient in 3D
    def grad_many(h, v0, v1, v2):
        # h: (E,3,P) -> (E,P,3)
        E, _, P = h.shape
        h_flat  = h.permute(0,2,1).reshape(E*P, 3)
        g = grad3d_intrinsic(h_flat, v0.repeat_interleave(P,0),
                                       v1.repeat_interleave(P,0),
                                       v2.repeat_interleave(P,0))
        return g.reshape(E, P, 3)

    gL = grad_many(hL, v0L, v1L, v2L)     # (E,P,3)
    gR = grad_many(hR, v0R, v1R, v2R)

    # normals per triangle side
    def normals(v0, v1, v2):
        e0 = v1 - v0
        e1 = v2 - v0
        n  = torch.cross(e0, e1, dim=1)
        return safe_normalize(n, dim=1, eps=1e-6)

    nL = normals(v0L, v1L, v2L)           # (E,3)
    nR = normals(v0R, v1R, v2R)

    # project gradients into the plane (stability)
    def proj_in_plane(g, n):
        # g: (E,P,3), n: (E,3)
        dot = (g * n[:,None,:]).sum(dim=2, keepdim=True)
        return g - dot * n[:,None,:]

    gL = proj_in_plane(gL, nL)
    gR = proj_in_plane(gR, nR)

    # tangents tau = n x g
    tauL = safe_normalize(torch.cross(nL[:,None,:], gL, dim=2), dim=2, eps=1e-6)  # (E,P,3)
    tauR = safe_normalize(torch.cross(nR[:,None,:], gR, dim=2), dim=2, eps=1e-6)

    cosang = (tauL * tauR).sum(dim=2).abs().clamp_max(1.0)  # (E,P)
    mis    = 1.0 - cosang                                   # (E,P)

    # per-edge expected misalignment over pairs
    mis_edge = (pair_mix * mis).sum(dim=1)                  # (E,)

    # gradient gate per edge: geometric mean of magnitudes across pairs
    gLm = gL.norm(dim=2).mean(dim=1) + eps                  # (E,)
    gRm = gR.norm(dim=2).mean(dim=1) + eps
    grad_gate = torch.sqrt(gLm * gRm)
    scale = torch.nanmedian(grad_gate).detach().clamp_min(1e-6)
    grad_gate = (grad_gate/scale).clamp(0.0, 2.0)
    grad_gate = torch.nan_to_num(grad_gate, nan=0.0, posinf=2.0, neginf=0.0)

    # length gate
    edge_vec = verts[vb] - verts[va]
    edge_len = edge_vec.norm(dim=1)
    len_gate = (edge_len / (edge_len.mean().detach() + 1e-12)).clamp(0.5, 2.0)

    # final per-edge weight
    w_edge = (phi.clamp_min(1e-4) * grad_gate * len_gate)   # (E,)

    # Charbonnier
    loss_edge = torch.sqrt(mis_edge*mis_edge + 1e-6)

    num = (w_edge * loss_edge).sum()
    den = (w_edge.sum() + 1e-9)
    loss = num / den

    if debug_log:
        afrac = (phi > 0.5).float().mean().item()
        print(f"[contour_v2] loss={loss.item():.4g}  act_frac={afrac:.3f}  "
              f"w_sum={w_edge.sum().item():.2f}")

    if return_weights:
        full = torch.zeros(edge_idx.shape[0], dtype=dtype, device=device)
        full[valid] = phi   # use φ as "boundary-ness" for other terms
        return loss, full

    return loss


def triple_point_barrier(F: torch.Tensor, faces: torch.Tensor, tri_area: torch.Tensor, beta_area: float = 10.0, margin: float = 0.10) -> torch.Tensor:
    """
    Discourage three equal channels in one triangle (speckles/"Y" junctions everywhere).
    Looks at the softmaxed per-triangle distribution and pushes the (2nd−3rd) gap above a margin.
    
    Args:
        F: (N, C) multi-channel field values
        faces: (T, 3) face indices
        tri_area: (T,) triangle areas
        beta_area: Temperature for softmax
        margin: Minimum gap between 2nd and 3rd channel
        
    Returns:
        loss: Triple point barrier loss
    """
    # soft probabilities per triangle (avg over vertices)
    p_v = torch.softmax(beta_area * F, dim=1)                      # (N,C)
    p_t = (p_v[faces[:,0]] + p_v[faces[:,1]] + p_v[faces[:,2]])/3  # (T,C)
    top3, _ = torch.topk(p_t, k=min(3, F.shape[1]), dim=1)         # (T,3)
    if top3.shape[1] < 3:
        return F.new_tensor(0.0)
    gap = top3[:,1] - top3[:,2]                                    # (T,)
    # penalize when 3rd channel too close to 2nd
    pen = torch.nn.functional.relu(margin - gap)
    w   = tri_area / (tri_area.sum() + 1e-9)
    return (w * pen).sum()


def area_balance_loss(
    F: torch.Tensor,                # (N,C)
    faces: torch.Tensor,            # (T,3)
    tri_area: torch.Tensor,         # (T,)
    beta_area: float,
    use_straight_through: bool = True,
    method: str = "rev_kl",         # "rev_kl" | "js" | "l2"
    min_frac: Optional[float] = None,
    max_frac: Optional[float] = None,
    barrier_w: float = 0.05,
    eps: float = 1e-8,
    entropy_weight: float = 0.0     # Add entropy regularization
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Area balance that actually resists collapse.
    - rev_kl  : cross-entropy to uniform = KL(uniform || frac) + const
    - js      : symmetric Jensen–Shannon (safer than either KL)
    - l2      : simple squared error to uniform
    Optionally uses straight-through hard counting to align with argmax.
    
    Args:
        F: (N, C) multi-channel field values
        faces: (T, 3) face indices
        tri_area: (T,) triangle areas
        beta_area: Temperature for softmax
        use_straight_through: Use hard counting in forward, soft in backward
        method: Loss method - "rev_kl", "js", or "l2"
        min_frac: Minimum allowed fraction per channel (default 0.5/C)
        max_frac: Maximum allowed fraction per channel (default 2.0/C)
        barrier_w: Weight for soft barrier penalty
        eps: Small value for numerical stability
        
    Returns:
        loss: Area balance loss
        frac: (C,) area fractions per channel
    """
    device, dtype = F.device, F.dtype
    N, C = F.shape
    uniform = F.new_full((C,), 1.0 / C)

    # Vertex probabilities
    p = torch.softmax(beta_area * F, dim=1)  # (N,C)

    if use_straight_through:
        # forward = hard one-hot; backward = soft p
        hard = torch.zeros_like(p).scatter_(1, p.argmax(dim=1, keepdim=True), 1.0)
        p = hard - p.detach() + p

    # Face probs (average of vertices)
    p_f = (p[faces[:, 0]] + p[faces[:, 1]] + p[faces[:, 2]]) / 3.0  # (T,C)

    # Area per channel and fractions
    A_c = (tri_area[:, None] * p_f).sum(dim=0)                      # (C,)
    total = tri_area.sum().clamp_min(eps)
    frac = (A_c / total).clamp(eps, 1.0 - eps)                      # (C,)

    # Losses
    if method == "rev_kl":
        # = - sum_c (1/C) log frac_c  (punishes tiny frac_c strongly)
        loss = -(uniform * frac.log()).sum()
    elif method == "js":
        m = 0.5 * (frac + uniform)
        loss = 0.5 * ((frac * (frac.add(eps).log() - m.add(eps).log())).sum()
                      + (uniform * (uniform.add(eps).log() - m.add(eps).log())).sum())
    elif method == "l2":
        loss = ((frac - uniform) ** 2).sum() * C  # Scale by C for consistency
    else:
        raise ValueError("method must be 'rev_kl', 'js', or 'l2'")

    # Soft barrier box around 1/C to keep every channel alive
    if min_frac is None: min_frac = 0.5 / C     # e.g. 50% of target
    if max_frac is None: max_frac = 2.0 / C     # e.g. 200% of target
    barrier = (torch.relu(min_frac - frac).pow(2) + torch.relu(frac - max_frac).pow(2)).sum()
    loss = loss + barrier_w * barrier
    
    # Add entropy regularization to encourage decisive assignments
    if entropy_weight > 0 and use_straight_through:
        # Compute entropy at vertex level (before ST)
        p_soft = torch.softmax(beta_area * F, dim=1)
        vertex_entropy = -(p_soft * (p_soft + eps).log()).sum(dim=1).mean()
        # Lower entropy = more decisive assignments
        loss = loss + entropy_weight * vertex_entropy

    return loss, frac


def compute_boundary_stats_v2(F: torch.Tensor,
                             edge_idx: torch.Tensor,
                             verts: torch.Tensor,
                             beta_contour: float) -> Tuple[float, float, float]:
    """
    Compute boundary length and active edge fraction for monitoring (v2 version).
    Uses soft-OR coverage from all pairs instead of top-k gating.
    
    Args:
        F: (N, C) multi-channel field values
        edge_idx: (E, 2) edge vertex indices
        verts: (N, 3) vertex positions
        beta_contour: Temperature parameter
        
    Returns:
        length: Estimated boundary length
        active_fraction: Fraction of edges with φ > 0.5
        median_phi: Median coverage value
    """
    device = F.device
    C = F.shape[1]
    
    # Ensure edge_idx is on same device
    edge_idx = edge_idx.to(device)
    
    # all channel pairs
    ii, jj = torch.triu_indices(C, C, offset=1, device=device)
    
    # values at endpoints for all pairs
    va, vb = edge_idx.T
    da = F[va][:, ii] - F[va][:, jj]      # (E,P)
    db = F[vb][:, ii] - F[vb][:, jj]      # (E,P)
    
    # crossing weights for all pairs
    w_pairs = torch.sigmoid(-beta_contour * da * db)
    conf = 0.5*(da.abs() + db.abs())
    w_pairs = w_pairs * torch.sigmoid(5.0*(conf - 0.2))
    
    # soft-OR coverage per edge
    phi = 1.0 - torch.prod(1.0 - w_pairs, dim=1)  # (E,)
    
    # Active edges (φ > 0.5)
    active_mask = phi > 0.5
    active_fraction = active_mask.float().mean().item()
    median_phi = phi.median().item()
    
    # Boundary length (sum of active edge lengths)
    if active_mask.any():
        active_edges = edge_idx[active_mask]
        edge_lengths = (verts[active_edges[:, 0]] - verts[active_edges[:, 1]]).norm(dim=1)
        length = edge_lengths.sum().item()
    else:
        length = 0.0
    
    return length, active_fraction, median_phi


def compute_hard_area_fractions(
    F: torch.Tensor,
    faces: torch.Tensor,
    tri_area: torch.Tensor
) -> torch.Tensor:
    """
    Compute area fractions based on hard argmax assignments.
    This shows what's actually happening in the visualization.
    
    Args:
        F: (N, C) multi-channel field values
        faces: (T, 3) face indices
        tri_area: (T,) triangle areas
        
    Returns:
        frac_hard: (C,) hard area fractions
    """
    # Hard assignments at vertices
    labels = F.argmax(dim=1)  # (N,)
    hard_v = torch.zeros_like(F).scatter_(1, labels[:, None], 1.0)  # (N, C)
    
    # Face assignments (majority vote or average)
    hard_f = (hard_v[faces[:, 0]] + hard_v[faces[:, 1]] + hard_v[faces[:, 2]]) / 3.0  # (T, C)
    
    # Area per channel
    A_hard = (tri_area[:, None] * hard_f).sum(0)  # (C,)
    total = A_hard.sum().clamp_min(1e-12)
    frac_hard = A_hard / total
    
    return frac_hard


def margin_separation_loss(
    F: torch.Tensor,
    tau: float = 0.5
) -> torch.Tensor:
    """
    Encourage separation between top channels at each vertex.
    This helps break symmetry and form distinct regions.
    
    Args:
        F: (N, C) field values
        tau: Minimum margin between top-2 channels
        
    Returns:
        loss: Margin separation loss
    """
    # Get top 2 values at each vertex
    top2, _ = torch.topk(F, k=min(2, F.shape[1]), dim=1)
    
    if top2.shape[1] < 2:
        return F.new_tensor(0.0)
    
    # Margin between top 2
    margin = top2[:, 0] - top2[:, 1]
    
    # Penalize small margins
    loss = torch.relu(tau - margin).mean()
    
    return loss