"""
Improved loss functions for ReLU-on-meshes optimization.
Includes intrinsic contour alignment, cotangent Laplacian smoothness, and KL-based area balance.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

Tensor = torch.Tensor


def tri_grad_2d(vals_3: Tensor, tri_xy_3x2: Tensor) -> Tensor:
    """
    Compute gradient of a scalar field within a triangle in local 2D coordinates.
    
    Args:
        vals_3: (3,) - scalar values at three vertices
        tri_xy_3x2: (3, 2) - 2D coordinates of vertices in triangle plane
        
    Returns:
        g: (2,) - gradient vector in 2D
    """
    p1, p2, p3 = tri_xy_3x2[0], tri_xy_3x2[1], tri_xy_3x2[2]
    
    # Build matrix M = [p2-p1, p3-p1]
    M = torch.stack([p2 - p1, p3 - p1], dim=1)  # (2, 2)
    
    # Build RHS b = [f2-f1, f3-f1]
    b = torch.stack([vals_3[1] - vals_3[0], vals_3[2] - vals_3[0]], dim=0)  # (2,)
    
    # Solve for gradient
    g = torch.linalg.solve(M, b)
    return g


def contour_alignment_intrinsic(
    F: Tensor,
    faces: Tensor,
    tri_xy: Tensor,
    edge_idx: Tensor,
    edge_tris: Tensor,
    beta_contour: float
) -> Tensor:
    """
    Intrinsic contour alignment loss that aligns boundary tangents across adjacent triangles.
    This replaces the unstable 3D plane fitting approach.
    
    Args:
        F: (N, C) - multi-channel field values at vertices
        faces: (T, 3) - triangle vertex indices
        tri_xy: (T, 3, 2) - 2D coordinates for each triangle
        edge_idx: (E, 2) - edge vertex indices
        edge_tris: (E, 2) - adjacent triangles for each edge (-1 for boundary)
        beta_contour: temperature parameter for edge crossing detection
        
    Returns:
        scalar loss value (normalized)
    """
    device, dtype = F.device, F.dtype
    C = F.shape[1]
    
    # 90-degree rotation matrix
    J90 = torch.tensor([[0., -1.], [1., 0.]], device=device, dtype=dtype)
    
    total = F.new_zeros(())
    wsum = F.new_zeros(())
    
    for e in range(edge_idx.shape[0]):
        a, b = edge_idx[e].tolist()
        tL, tR = edge_tris[e].tolist()
        
        # Skip boundary edges (only one adjacent triangle)
        if tL < 0 or tR < 0:
            continue
        
        # Pair gating: only consider top-2 channels at edge midpoint
        f_mid = 0.5 * (F[a] + F[b])  # (C,)
        top2 = torch.topk(f_mid, k=min(2, C)).indices
        
        if len(top2) < 2:
            continue
            
        i, j = int(top2[0]), int(top2[1])
        
        # Edge crossing weight (soft detection)
        da = F[a, i] - F[a, j]
        db = F[b, i] - F[b, j]
        w = torch.sigmoid(-beta_contour * da * db)
        
        # Skip if weight is too small
        if w.item() < 1e-6:
            continue
        
        # Compute tangent directions in both adjacent triangles
        t_dirs = []
        
        for t in (tL, tR):
            if t < 0:
                continue
                
            tri = faces[t]  # (3,)
            h = F[tri, i] - F[tri, j]  # (3,) - height difference for channel pair
            
            # Gradient in 2D triangle coordinates
            g = tri_grad_2d(h, tri_xy[t])  # (2,)
            
            # Rotate gradient by 90° to get tangent (level set direction)
            tvec = J90 @ g
            nrm = tvec.norm() + 1e-12
            t_dirs.append(tvec / nrm)
        
        if len(t_dirs) == 2:
            # Orientation-invariant misalignment: 1 - |cos θ|
            cosang = (t_dirs[0] * t_dirs[1]).sum().abs().clamp(max=1.0)
            mis = 1.0 - cosang
            
            # Robust Charbonnier penalty to reduce outlier influence
            total = total + w * torch.sqrt(mis * mis + 1e-6)
            wsum = wsum + w
    
    return total / (wsum + 1e-9)


def smoothness_cotan(
    F: Tensor,
    I: Tensor,
    J: Tensor,
    W: Tensor
) -> Tensor:
    """
    Cotangent Laplacian smoothness loss.
    This properly accounts for mesh geometry unlike simple edge differences.
    
    Args:
        F: (N, C) - multi-channel field values
        I: (K,) - row indices for cotangent weights
        J: (K,) - column indices for cotangent weights
        W: (K,) - cotangent weights
        
    Returns:
        scalar smoothness loss (normalized)
    """
    diff = F[I] - F[J]  # (K, C)
    e2 = (diff * diff).sum(dim=-1)  # (K,) - squared differences per edge
    
    num = (W * e2).sum()
    den = W.sum().clamp_min(1e-12)
    
    return num / den


def area_fractions_and_kl(
    F: Tensor,
    faces: Tensor,
    tri_area: Tensor,
    beta_area: float
) -> Tuple[Tensor, Tensor]:
    """
    Area balance loss using KL divergence to uniform distribution.
    This maintains gradients better than L1 loss when regions are small.
    
    Args:
        F: (N, C) - multi-channel field values
        faces: (T, 3) - triangle vertex indices
        tri_area: (T,) - area of each triangle
        beta_area: temperature for softmax
        
    Returns:
        L_area: scalar KL divergence loss
        frac: (C,) - area fraction for each channel
    """
    C = F.shape[1]
    
    # Barycentric sample points for quadrature
    bary = torch.tensor([
        [1/3, 1/3, 1/3],  # centroid
        [1/2, 1/2, 0.0],  # edge midpoint 1
        [1/2, 0.0, 1/2],  # edge midpoint 2
        [0.0, 1/2, 1/2],  # edge midpoint 3
    ], device=F.device, dtype=F.dtype)  # (4, 3)
    
    # Get field values at triangle vertices
    Ft = F[faces]  # (T, 3, C)
    
    # Interpolate to sample points
    Ft_s = (bary[:, :, None] * Ft[:, None, :, :]).sum(dim=2)  # (T, 4, C)
    
    # Apply softmax to get probabilities
    P = torch.softmax(beta_area * Ft_s, dim=-1)  # (T, 4, C)
    
    # Average over sample points
    Pmean = P.mean(dim=1)  # (T, C)
    
    # Compute area-weighted fractions
    Ac = (tri_area[:, None] * Pmean).sum(dim=0)  # (C,)
    At = tri_area.sum().clamp_min(1e-12)
    frac = Ac / At
    
    # KL divergence from uniform: KL(frac || uniform) = sum_c frac_c * log(C * frac_c)
    L = (frac * torch.log(C * frac + 1e-9)).sum()
    
    return L, frac


def pin_loss(
    F: Tensor,
    pin_idx: Tensor,
    pin_target: Tensor
) -> Tensor:
    """
    Soft pinning loss for anchor vertices.
    
    Args:
        F: (N, C) - field values
        pin_idx: (P,) - indices of pinned vertices
        pin_target: (P, C) - target values for pinned vertices
        
    Returns:
        scalar loss
    """
    if pin_idx.numel() == 0:
        return F.new_zeros(())
    
    err = F[pin_idx] - pin_target
    return (err * err).mean()


def compute_total_loss(
    F: Tensor,
    mesh_data: dict,
    faces: Tensor,
    pin_idx: Optional[Tensor],
    pin_target: Optional[Tensor],
    weights: dict,
    beta_contour: float,
    beta_area: float
) -> Tuple[dict, Tensor]:
    """
    Compute all losses and weighted total.
    
    Args:
        F: (N, C) - field values
        mesh_data: precomputed mesh data
        faces: (T, 3) - triangle indices
        pin_idx: optional pinned vertex indices
        pin_target: optional pinning targets
        weights: dictionary of loss weights
        beta_contour: temperature for contour detection
        beta_area: temperature for area softmax
        
    Returns:
        losses: dictionary of individual losses
        total: weighted sum of losses
    """
    losses = {}
    
    # Smoothness loss
    losses['smooth'] = smoothness_cotan(
        F, mesh_data['cotan_I'], mesh_data['cotan_J'], mesh_data['cotan_W']
    )
    
    # Contour alignment loss
    losses['contour'] = contour_alignment_intrinsic(
        F, faces, mesh_data['tri_xy'], mesh_data['edge_idx'], 
        mesh_data['edge_tris'], beta_contour
    )
    
    # Area balance loss
    losses['area'], frac = area_fractions_and_kl(
        F, faces, mesh_data['tri_area'], beta_area
    )
    
    # Pin loss
    if pin_idx is not None and pin_target is not None:
        losses['pin'] = pin_loss(F, pin_idx, pin_target)
    else:
        losses['pin'] = F.new_zeros(())
    
    # Weighted total
    total = sum(weights.get(k, 0.0) * v for k, v in losses.items())
    
    # Store area fractions for monitoring
    losses['_frac'] = frac
    
    return losses, total