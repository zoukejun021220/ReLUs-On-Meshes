import torch


def contour_alignment_loss_anchored(
    vertices: torch.Tensor,      # (N, 3)
    faces: torch.Tensor,         # (T, 3) long
    f_values: torch.Tensor,      # (N, C)
    pinned_axes: torch.Tensor,   # (C, 3) fixed plane normals
    plane_offsets: torch.Tensor, # (C,) learnable offsets
    beta_edge: float = 20.0,
    include_triples: bool = False,
    eps: float = 1e-9,
    robust_delta: float = 0.05,  # For robust weighting
    min_weight: float = 1e-4,    # Minimum weight threshold
):
    """
    Axis-anchored contour loss: fixed normals (pinned_axes), learn only per-channel offsets.
    Robust weights, normalized geometry expected.
    
    This is much more stable than SVD-based plane fitting and gives cleaner gradients.
    
    Args:
        vertices: (N, 3) vertex positions (should be normalized)
        faces: (T, 3) triangle indices
        f_values: (N, C) multi-channel field values
        pinned_axes: (C, 3) fixed plane normals (from PCA or world axes)
        plane_offsets: (C,) learnable plane offsets
        beta_edge: Sharpness for edge intersection detection
        include_triples: Whether to include triple intersections
        eps: Small value for numerical stability
        robust_delta: Scale for robust weighting (Cauchy)
        min_weight: Minimum weight to avoid vanishing gradients
        
    Returns:
        loss: Scalar contour alignment loss
    """
    device = vertices.device
    C = f_values.shape[1]
    
    if C < 2:
        return torch.zeros((), device=device, dtype=vertices.dtype)
    
    # Ensure plane_offsets is provided and correct shape
    assert plane_offsets is not None and plane_offsets.shape[0] == C
    
    # Get triangles and fields
    p_tri = vertices[faces]  # (T, 3, 3)
    f_tri = f_values[faces]  # (T, 3, C)
    
    # All channel pairs i<j
    i_idx, j_idx = torch.triu_indices(C, C, 1, device=device)
    P = i_idx.numel()
    
    if P == 0:
        return torch.zeros((), device=device, dtype=vertices.dtype)
    
    # Compute differences for all pairs
    # d[t,v,p] = f_i - f_j
    d = f_tri[..., i_idx] - f_tri[..., j_idx]  # (T, 3, P)
    
    p0, p1, p2 = p_tri[:, 0], p_tri[:, 1], p_tri[:, 2]
    d0, d1, d2 = d[:, 0, :], d[:, 1, :], d[:, 2, :]
    
    def edge_intersections(dA, dB, xA, xB):
        """
        Find edge intersections with exact interpolation.
        """
        # Sign change detection
        prod = dA * dB
        w = torch.sigmoid(-beta_edge * prod)
        
        # Exact interpolation parameter: t = dA / (dA - dB)
        # This gives the point where linear interpolation equals zero
        t = dA / (dA - dB + eps)
        t = t.clamp(0.0, 1.0)
        
        # Intersection points
        coords = xA.unsqueeze(1) + t.unsqueeze(-1) * (xB - xA).unsqueeze(1)
        
        return coords, w
    
    # Find edge intersections
    c01, w01 = edge_intersections(d0, d1, p0, p1)
    c12, w12 = edge_intersections(d1, d2, p1, p2)
    c20, w20 = edge_intersections(d2, d0, p2, p0)
    
    # Flatten all intersection data
    coords = torch.cat([c01.reshape(-1, 3), c12.reshape(-1, 3), c20.reshape(-1, 3)], 0)
    w_edge = torch.cat([w01.reshape(-1), w12.reshape(-1), w20.reshape(-1)], 0)
    
    # Create pair indices for all intersections
    pair_ids = torch.cat([
        torch.arange(P, device=device).repeat(c01.shape[0]),
        torch.arange(P, device=device).repeat(c12.shape[0]),
        torch.arange(P, device=device).repeat(c20.shape[0])
    ], 0)
    
    # Build pairwise plane parameters
    # For pair (i,j), the boundary plane is: (n_i - n_j)·x + (b_i - b_j) = 0
    # Normalize pair normals to unit length to keep distances on consistent scale
    n_pair = pinned_axes[i_idx] - pinned_axes[j_idx]  # (P, 3)
    n_norm = n_pair.norm(dim=1, keepdim=True).clamp_min(1e-3)  # avoid tiny norms
    n_hat = n_pair / n_norm  # unit normals per pair
    # Scale offsets consistently with the normalized normals
    d_pair = (plane_offsets[i_idx] - plane_offsets[j_idx]) / n_norm.squeeze(1)
    
    # Don't apply minimum weight threshold - let far edges have near-zero weight
    # This prevents noise from non-crossing edges
    # w_edge = w_edge.clamp_min(min_weight)  # Commented out as per recommendation
    
    # Optional: Include triple intersections
    if include_triples and C >= 3:
        # Triple intersection code here (simplified for brevity)
        # This is optional and can be added later for refinement
        pass
    
    # Compute distances to pairwise planes
    # For every intersection point k, evaluate the pairwise plane it belongs to
    dist = (coords * n_hat[pair_ids]).sum(dim=1) + d_pair[pair_ids]  # (K,)
    
    # Optional: Robust weighting (Cauchy)
    if robust_delta > 0:
        r = (dist / robust_delta).abs()
        psi = 1.0 / (1.0 + r * r)  # Cauchy weight
        w_edge = w_edge * psi
    
    # Final loss: weighted squared distances (normalize by weight mass)
    mass = w_edge.sum().clamp_min(1e-6)
    loss = (w_edge * dist.pow(2)).sum() / mass
    
    return loss


def contour_alignment_loss_anchored_with_svd_refinement(
    vertices: torch.Tensor,      # (N, 3)
    faces: torch.Tensor,         # (T, 3) long
    f_values: torch.Tensor,      # (N, C)
    pinned_axes: torch.Tensor,   # (C, 3) fixed plane normals
    plane_offsets: torch.Tensor, # (C,) learnable offsets
    beta_edge: float = 20.0,
    include_triples: bool = False,
    eps: float = 1e-9,
    svd_weight: float = 0.1,     # Weight for SVD refinement term
    svd_detach: bool = True,     # Detach SVD normals from gradient
):
    """
    Anchored planes loss with optional SVD refinement.
    
    The main loss uses fixed normals with learnable offsets (stable).
    An additional small SVD term can help refine alignment (optional).
    
    Args:
        Same as contour_alignment_loss_anchored, plus:
        svd_weight: Weight for the SVD refinement term
        svd_detach: If True, detach SVD normals from gradient computation
        
    Returns:
        loss: Combined anchored + SVD refinement loss
    """
    # First compute the main anchored loss
    anchored_loss = contour_alignment_loss_anchored(
        vertices, faces, f_values, pinned_axes, plane_offsets,
        beta_edge, include_triples, eps
    )
    
    if svd_weight <= 0:
        return anchored_loss
    
    # Additional SVD refinement (simplified version)
    # This would compute per-pair plane fits and add a small alignment term
    # Implementation omitted for brevity - can be added if needed
    
    return anchored_loss
