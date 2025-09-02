import torch

def contour_alignment_loss(
   vertices:   torch.Tensor,   # (N, 3)
    faces:      torch.Tensor,   # (T, 3) long
    f_values:   torch.Tensor,   # (N, C)
    pinned_axes:torch.Tensor, 
    beta: float =20.0,  # (C, 3) float => the axis normal for each channel
    beta_edge:  float = 20.0,
    beta_triple: float = 20.0,
    include_triples: bool = False,
    adajancy: torch.Tensor = None,
    eps: float = 1e-9,
    lambda_plane: float = 1.0,
    lambda_contour: float = 1.0,
) -> torch.Tensor:
    """
    A fully vectorized, "vigorous" contour-alignment loss that:
      1) Collects edge intersections for every channel pair (i<j) using a logistic weight
         for sign-changes in (f_i - f_j).
      2) Optionally finds "soft triple intersections" for (c0,c1,c2) if C>=3:
         - We do a closed-form barycentric solve for (f_c0-f_c1)=0, (f_c0-f_c2)=0
         - We multiply by each corner's softmax-prob of channels c0,c1,c2,
           plus a smooth clamp to keep the barycentric coords in [0,1].
      3) Builds a weighted covariance for each pair in one pass. Does an SVD => plane normal.
      4) In a second pass, accumulates the MSE distances of all intersection points
         to that pair's plane. Summation => final scalar loss.

    *No loops* over triangles or pairs in Python. Everything is broadcast/batch in PyTorch.
    *Fully differentiable*—no discrete argmax or masks. If C<2 => returns 0.0 (no pairs).

    Args:
      vertices: (N,3) float
      faces: (T,3) long
      f_values: (N,C) float   # multi-channel field
      beta: float  # logistic sharpness for edge intersection
      eps: float
      soft_inside: float  # how sharply to clamp barycentric coords for triple intersection
      include_triples: bool

    Returns:
      total_loss: scalar float
    """
    soft_inside = 0.0  # TODO: add this to the function signature
    device = vertices.device
    N = vertices.shape[0]
    T = faces.shape[0]
    C = f_values.shape[1]

    # 0) If <2 channels => no pairs => return 0.0
    if C < 2:
        return torch.zeros((), device=device, dtype=vertices.dtype)

    # ------------------------------------------------------------------------
    # 1) Build all channel pairs i<j => shape (P,)
    # ------------------------------------------------------------------------
    i2, j2 = torch.triu_indices(C, C, offset=1, device=device)  # each => (P,)
    if i2.numel() == 0:
        # Edge case if somehow no pairs. Return 0.
        return torch.zeros((), device=device, dtype=vertices.dtype)

    P = i2.shape[0]  # number of (i<j) pairs

    # Build a map (cA,cB) => pair index
    pair_idx_mat = torch.full((C,C), -1, device=device, dtype=torch.long)
    p_arange     = torch.arange(P, device=device)
    pair_idx_mat[i2, j2] = p_arange
    pair_idx_mat[j2, i2] = p_arange

    # ------------------------------------------------------------------------
    # 2) Edge Intersections => logistic weighting
    # ------------------------------------------------------------------------
    # faces => (T,3), so gather coords => (T,3,3)
    p_tri = vertices[faces]
    # gather field => (T,3,C)
    f_tri = f_values[faces]

    # d => shape (T,3,P).  d[t,v,p] = f_tri[t,v, i2[p]] - f_tri[t,v, j2[p]]
    d = f_tri[..., i2] - f_tri[..., j2]  # (T,3,P)

    p0 = p_tri[:,0]
    p1 = p_tri[:,1]
    p2 = p_tri[:,2]

    def edge_intersection(dA, dB, vA, vB):
        """
        dA,dB: (T,P)
        vA,vB: (T,3)
        return coords: (T,P,3), w: (T,P)
        """
        prod  = dA*dB
        w_    = torch.sigmoid(-beta*prod)  # logistic weight
        alpha = torch.abs(dA)/(torch.abs(dA)+torch.abs(dB)+eps)
        coords = vA.unsqueeze(1) + alpha.unsqueeze(-1)*(vB - vA).unsqueeze(1)
        return coords, w_

    d0, d1, d2 = d[:,0,:], d[:,1,:], d[:,2,:]
    coords_01, w_01 = edge_intersection(d0, d1, p0, p1)
    coords_12, w_12 = edge_intersection(d1, d2, p1, p2)
    coords_20, w_20 = edge_intersection(d2, d0, p2, p0)

    def flatten_edge(coords_tp3, w_tp):
        # coords_tp3 => (T,P,3), w_tp => (T,P)
        coords_flat = coords_tp3.reshape(-1,3)
        w_flat      = w_tp.reshape(-1)
        pair_idx_arange = torch.arange(P, device=device).view(1,P).expand(coords_tp3.shape[0],-1).reshape(-1)
        return coords_flat, w_flat, pair_idx_arange

    coords_01f, w_01f, pidx_01f = flatten_edge(coords_01, w_01)
    coords_12f, w_12f, pidx_12f = flatten_edge(coords_12, w_12)
    coords_20f, w_20f, pidx_20f = flatten_edge(coords_20, w_20)

    all_coords = torch.cat([coords_01f, coords_12f, coords_20f], dim=0)
    all_w      = torch.cat([w_01f,      w_12f,      w_20f],      dim=0)
    all_pidx   = torch.cat([pidx_01f,   pidx_12f,   pidx_20f],   dim=0)

    # ------------------------------------------------------------------------
    # 3) Soft triple intersection => only if C>=3 and include_triples=True
    # ------------------------------------------------------------------------
    if include_triples and C >= 3:
        f0 = f_tri[:,0,:]  # (T,C)
        f1 = f_tri[:,1,:]
        f2 = f_tri[:,2,:]
        # softmax => each corner has distribution over channels
        pi0 = torch.softmax(f0, dim=1)
        pi1 = torch.softmax(f1, dim=1)
        pi2 = torch.softmax(f2, dim=1)

        # All combos c0< c1< c2
        combs = torch.combinations(torch.arange(C, device=device), r=3)
        # e.g. shape => (#comb,3)
        if combs.numel() > 0:  # ensure there's something
            ncomb = combs.shape[0]
            c0_idx = combs[:,0].view(1,ncomb)
            c1_idx = combs[:,1].view(1,ncomb)
            c2_idx = combs[:,2].view(1,ncomb)

            expand_t = (T, ncomb)
            f0_c0 = torch.gather(f0, 1, c0_idx.expand(expand_t))
            f0_c1 = torch.gather(f0, 1, c1_idx.expand(expand_t))
            f0_c2 = torch.gather(f0, 1, c2_idx.expand(expand_t))
            f1_c0 = torch.gather(f1, 1, c0_idx.expand(expand_t))
            f1_c1 = torch.gather(f1, 1, c1_idx.expand(expand_t))
            f1_c2 = torch.gather(f1, 1, c2_idx.expand(expand_t))
            f2_c0 = torch.gather(f2, 1, c0_idx.expand(expand_t))
            f2_c1 = torch.gather(f2, 1, c1_idx.expand(expand_t))
            f2_c2 = torch.gather(f2, 1, c2_idx.expand(expand_t))

            # differences
            rg0 = f0_c0 - f0_c1
            rg1 = f1_c0 - f1_c1
            rg2 = f2_c0 - f2_c1
            rb0 = f0_c0 - f0_c2
            rb1 = f1_c0 - f1_c2
            rb2 = f2_c0 - f2_c2

            A_xy = rg0 - rg2
            B_xy = rg1 - rg2
            X_xy = rg2
            A_xz = rb0 - rb2
            B_xz = rb1 - rb2
            X_xz = rb2

            det = A_xy*B_xz - B_xy*A_xz
            alpha = (X_xy*B_xz - B_xy*X_xz)/(det+eps)
            beta_  = (A_xy*X_xz - X_xy*A_xz)/(det+eps)
            gamma_ = 1.0 - alpha - beta_

            # "soft inside" factor
            if soft_inside>0.0:
                def smoothstep(x):
                    return torch.sigmoid(soft_inside*x)
                insideFactor = smoothstep(alpha)*smoothstep(beta_)*smoothstep(gamma_)
            else:
                insideFactor = torch.ones_like(alpha)

            # Probability corner0 picks c0, corner1 picks c1, corner2 picks c2
            pi0_c0 = torch.gather(pi0, 1, c0_idx.expand(expand_t))
            pi1_c1 = torch.gather(pi1, 1, c1_idx.expand(expand_t))
            pi2_c2 = torch.gather(pi2, 1, c2_idx.expand(expand_t))
            triple_prob = pi0_c0 * pi1_c1 * pi2_c2  # (T,ncomb)

            triple_w = triple_prob*insideFactor

            # Barycentric => 3D
            p0_3 = p_tri[:,0,:].unsqueeze(1)
            p1_3 = p_tri[:,1,:].unsqueeze(1)
            p2_3 = p_tri[:,2,:].unsqueeze(1)

            alpha_e = alpha.unsqueeze(-1)
            beta_e  = beta_.unsqueeze(-1)
            gamma_e = gamma_.unsqueeze(-1)
            triple_coords = alpha_e*p0_3 + beta_e*p1_3 + gamma_e*p2_3

            # replicate each triple for the 3 pairs => (c0,c1),(c0,c2),(c1,c2)
            pair0 = pair_idx_mat[combs[:,0], combs[:,1]]
            pair1 = pair_idx_mat[combs[:,0], combs[:,2]]
            pair2 = pair_idx_mat[combs[:,1], combs[:,2]]
            triple_pairs = torch.stack([pair0, pair1, pair2], dim=1)

            coords_flat = triple_coords.view(-1,3)
            w_flat      = triple_w.view(-1)

            coords_rep = coords_flat.repeat_interleave(3, dim=0)
            w_rep      = w_flat.repeat_interleave(3, dim=0)
            triple_pairs_flat = triple_pairs.view(-1)
            triple_pairs_full = triple_pairs_flat.unsqueeze(0).expand(T, -1).reshape(-1)

            # for difference=0 => product=0 => logistic(0)=0.5, or pick any factor
            w_rep_final = w_rep*0.5

            all_coords = torch.cat([all_coords, coords_rep], dim=0)
            all_w      = torch.cat([all_w,      w_rep_final], dim=0)
            all_pidx   = torch.cat([all_pidx,   triple_pairs_full], dim=0)

    # ------------------------------------------------------------------------
    # 4) Weighted covariance => plane fits
    # ------------------------------------------------------------------------
    sum_w   = torch.zeros((P,), device=device, dtype=all_coords.dtype)
    sum_x   = torch.zeros((P,3), device=device, dtype=all_coords.dtype)
    sum_xx  = torch.zeros((P,3,3), device=device, dtype=all_coords.dtype)

    weighted_coords = all_coords*all_w.unsqueeze(-1)
    sum_w.index_add_(0, all_pidx, all_w)
    sum_x.index_add_(0, all_pidx, weighted_coords)

    outer_ = weighted_coords.unsqueeze(2)*all_coords.unsqueeze(1)
    sum_xx_flat = sum_xx.view(P,9)
    outer_flat  = outer_.reshape(-1,9)
    sum_xx_flat.index_add_(0, all_pidx, outer_flat)
    sum_xx = sum_xx_flat.view(P,3,3)

    sum_w_clamped = sum_w.clamp_min(eps)
    mean_ = sum_x/sum_w_clamped.unsqueeze(-1)
    mean_outer = mean_.unsqueeze(2)*mean_.unsqueeze(1)
    cov = sum_xx/sum_w_clamped.view(-1,1,1) - mean_outer

    # ------------------------------------------------------------------------
    # 5) Plane from SVD
    # ------------------------------------------------------------------------
    cov_f32 = cov.float()
    U,S,Vt = torch.linalg.svd(cov_f32, full_matrices=False)  # => (P,3,3)
    plane_n = Vt[:, -1, :].to(cov.dtype)
    plane_n = plane_n/(plane_n.norm(dim=1,keepdim=True)+eps)
    plane_d = -(plane_n*mean_).sum(dim=1)

    # ------------------------------------------------------------------------
    # 6) Second pass => MSE
    # ------------------------------------------------------------------------
    sum_sq = torch.zeros((P,), device=device, dtype=all_coords.dtype)
    n_idx = plane_n[all_pidx]
    d_idx = plane_d[all_pidx]
    dist  = (n_idx*all_coords).sum(dim=1)+d_idx
    dist_sq = dist.square()*all_w
    sum_sq.index_add_(0, all_pidx, dist_sq)

    mse_pairs  = sum_sq/(sum_w_clamped+eps)
    total_loss = mse_pairs.sum()
    return total_loss