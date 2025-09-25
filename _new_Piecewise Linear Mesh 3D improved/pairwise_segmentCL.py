import torch


def contour_alignment_loss(
    vertices:   torch.Tensor,   # (N, 3)
    faces:      torch.Tensor,   # (T, 3) long
    f_values:   torch.Tensor,   # (N, C)
    pinned_axes:torch.Tensor,   # (C, 3) float => the axis normal for each channel
    beta_edge:  float = 20.0,
    beta_triple:float = 20.0,
    include_triples: bool = False,
    adajancy: torch.Tensor = None,
    eps: float = 1e-9
) -> torch.Tensor:
    """
    Fully vectorized / differentiable version of the contour alignment loss,
    with no Python loops.
    """
    beta=beta_edge
    device = vertices.device
    T = faces.shape[0]
    C = f_values.shape[1]

    # -----------------------------------------------------
    # 1) Build all channel pairs (i < j) via triu_indices
    #    pairs_ij: shape (2, P), where P = C*(C-1)/2
    # -----------------------------------------------------
    pairs_ij = torch.triu_indices(C, C, offset=1, device=device)  # (2, P)
    i_idx, j_idx = pairs_ij[0], pairs_ij[1]  # each (P,)
    P = i_idx.shape[0]                      # number of (i<j) pairs

    # -----------------------------------------------------
    # 2) Gather per-triangle data
    #    p_tri: (T, 3, 3),   f_tri: (T, 3, C)
    # -----------------------------------------------------
    p_tri = vertices[faces]        # shape (T, 3, 3)
    f_tri = f_values[faces]        # shape (T, 3, C)

    # -----------------------------------------------------
    # 3) Compute differences for all pairs in one shot
    #    f_i, f_j => shape (T, 3, P), then d = f_i - f_j
    # -----------------------------------------------------
    f_i = f_tri[:, :, i_idx]       # (T, 3, P)
    f_j = f_tri[:, :, j_idx]       # (T, 3, P)
    d   = f_i - f_j                # (T, 3, P)

    # -----------------------------------------------------
    # 4) Compute edge intersections (v0->v1, v1->v2, v2->v0)
    #    Helper:  coords, weight = edge_intersection(d0, d1, p0, p1)
    # -----------------------------------------------------
    p0 = p_tri[:, 0]  # (T, 3)
    p1 = p_tri[:, 1]
    p2 = p_tri[:, 2]

    d0 = d[:, 0, :]   # (T, P)
    d1 = d[:, 1, :]
    d2 = d[:, 2, :]

    def edge_intersection(dA, dB, xA, xB):
        """
        dA,dB: (T,P)
        xA,xB: (T,3)
        returns coords: (T,P,3), weight: (T,P)
        """
        # product ~ sign(dA*dB): large negative => crossing
        prod  = dA * dB                  # (T,P)
        w_    = torch.sigmoid(-beta*prod)  
        # alpha: how far along edge
        denom = torch.abs(dA) + torch.abs(dB) + eps
        alpha = torch.abs(dA) / denom
        # coords
        edge_vec = (xB - xA).unsqueeze(1)    # (T,1,3)
        coords   = xA.unsqueeze(1) + alpha.unsqueeze(-1) * edge_vec
        return coords, w_

    coords_01, w_01 = edge_intersection(d0, d1, p0, p1)  # (T,P,3), (T,P)
    coords_12, w_12 = edge_intersection(d1, d2, p1, p2)
    coords_20, w_20 = edge_intersection(d2, d0, p2, p0)

    # -----------------------------------------------------
    # 5) Flatten intersections -> shape (K,3), (K,) + pair_idx
    #    K ~ 3*T*P (plus optional triples)
    # -----------------------------------------------------
    def flatten_edge(coords_tp3, w_tp):
        # coords_tp3: (T,P,3), w_tp: (T,P) => flatten to (T*P,3), (T*P)
        coords_flat = coords_tp3.reshape(-1, 3)
        w_flat      = w_tp.reshape(-1)
        # pidx => [0..P-1] repeated T times
        pidx_arange = torch.arange(P, device=device).view(1, P).expand(T, P)
        pidx_flat   = pidx_arange.reshape(-1)
        return coords_flat, w_flat, pidx_flat

    coords_01f, w_01f, pidx_01f = flatten_edge(coords_01, w_01)
    coords_12f, w_12f, pidx_12f = flatten_edge(coords_12, w_12)
    coords_20f, w_20f, pidx_20f = flatten_edge(coords_20, w_20)

    all_coords = torch.cat([coords_01f, coords_12f, coords_20f], dim=0)  # (K, 3)
    all_w      = torch.cat([w_01f,      w_12f,      w_20f],      dim=0)  # (K,)
    all_pidx   = torch.cat([pidx_01f,   pidx_12f,   pidx_20f],   dim=0)  # (K,)

    # Optionally include triple-points similarly (skipped).
    if include_triples:
        pass

    # -----------------------------------------------------
    # 6) Compute weighted covariance for each pair via index_add
    #
    #    sum_w[p], sum_x[p], sum_xx[p]
    #      sum_xx is 9D flattened => reshape to (P,3,3) later
    # -----------------------------------------------------
    sum_w   = torch.zeros((P,),      device=device, dtype=all_w.dtype)
    sum_x   = torch.zeros((P, 3),    device=device, dtype=all_coords.dtype)
    sum_xx  = torch.zeros((P, 9),    device=device, dtype=all_coords.dtype)  # will reshape to (P,3,3)

    # -- accumulate sum_w
    sum_w.index_add_(0, all_pidx, all_w)

    # -- accumulate sum_x
    #    x = coords * w (broadcast over last dim)
    x = all_coords * all_w.unsqueeze(1)  # (K,3)
    sum_x.index_add_(0, all_pidx, x)

    # -- accumulate sum_xx
    #    Each row => outer(x, x) => (3,3) => flatten (9,)
    #    Then index_add
    x_outer = (x.unsqueeze(2) * x.unsqueeze(1)).reshape(-1, 9)  # (K,9)
    sum_xx.index_add_(0, all_pidx, x_outer)

    # Reshape sum_xx to (P,3,3)
    sum_xx = sum_xx.reshape(P, 3, 3)

    # -----------------------------------------------------
    # 7) Weighted covariance & plane via SVD
    # -----------------------------------------------------
    sum_w_clamped = sum_w.clamp_min(eps)  # (P,)
    mean_ = sum_x / sum_w_clamped.unsqueeze(1)   # (P, 3)

    # Cov = E[xx] - mu mu^T
    mean_outer = mean_.unsqueeze(2) * mean_.unsqueeze(1)  # (P,3,3)
    cov = sum_xx / sum_w_clamped.view(-1, 1, 1) - mean_outer

    # SVD => normal is last singular vector => plane_n
    # (convert to float32 for stability, then back)
    cov_f32 = cov.float()
    U, S, Vt = torch.linalg.svd(cov_f32, full_matrices=False)
    plane_n = Vt[:, -1, :].to(cov.dtype)  # (P, 3)
    plane_n = plane_n / (plane_n.norm(dim=1, keepdim=True) + eps)

    plane_d = -(plane_n * mean_).sum(dim=1)  # (P,)

    # -----------------------------------------------------
    # 8) Second pass: accumulate MSE in distance to plane
    # -----------------------------------------------------
    sum_sq = torch.zeros((P,), device=device, dtype=all_w.dtype)

    # dist_k = dot(plane_n[p], x_k) + plane_d[p]
    # => we gather plane_n, plane_d by pidx => no loops
    # => accum dist^2 * w in sum_sq
    n0 = plane_n[:, 0]
    n1 = plane_n[:, 1]
    n2 = plane_n[:, 2]
    d_ = plane_d

    x0 = all_coords[:, 0]
    x1 = all_coords[:, 1]
    x2 = all_coords[:, 2]

    dist = x0 * n0[all_pidx] + x1 * n1[all_pidx] + x2 * n2[all_pidx] + d_[all_pidx]
    dist_sq = dist.square() * all_w

    sum_sq.index_add_(0, all_pidx, dist_sq)

    mse_pairs  = sum_sq / (sum_w_clamped + eps)
    total_loss = mse_pairs.sum()

    return total_loss

