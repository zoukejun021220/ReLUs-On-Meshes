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
    r"""
    A fully vectorized version of the "contour_alignment_loss" with:
      1) No explicit Python loops over channels or adjacency.
      2) Fully differentiable.

    Returns a scalar loss = (adjacency direction misalignment) + (plane penalty)
    accumulated over all channels.
    """
    device = vertices.device
    dtype  = vertices.dtype

    # -------------------------------------------------------------------------
    # Basic sizes
    # -------------------------------------------------------------------------
    T = faces.shape[0]
    C = f_values.shape[1]
    E = adajancy.shape[0]

    # -------------------------------------------------------------------------
    # 1) Precompute geometry for triangles
    #    p_tri:  (T, 3, 3)   => triangle xyz coords
    #    f_tri:  (T, 3, C)   => f_values at triangle corners
    # -------------------------------------------------------------------------
    p_tri = vertices[faces]      # (T,3,3)
    f_tri = f_values[faces]      # (T,3,C)

    p0 = p_tri[:,0,:]  # (T,3)
    p1 = p_tri[:,1,:]
    p2 = p_tri[:,2,:]
    E1 = p1 - p0       # (T,3)
    E2 = p2 - p0       # (T,3)

    # -------------------------------------------------------------------------
    # 2) Compute per-channel gradients: grad_all[c,t,:] = ∇ f_c in triangle t
    #    Here we vectorize it so we get shape (C,T,3).
    # -------------------------------------------------------------------------
    # Cross & norms
    n      = torch.cross(E1, E2, dim=1)                      # (T,3)
    norm_n = n.norm(dim=1, keepdim=True).clamp_min(eps)       # (T,1)
    cross_e2n = torch.cross(E2, n, dim=1)                     # (T,3)
    cross_nE1 = torch.cross(n, E1, dim=1)                     # (T,3)

    # f-values at each corner, shape (T,C)
    f0 = f_tri[:,0,:]  # (T,C)
    f1 = f_tri[:,1,:]  # (T,C)
    f2 = f_tri[:,2,:]  # (T,C)

    # Differences
    df01 = (f1 - f0).unsqueeze(2)   # (T,C,1)
    df02 = (f2 - f0).unsqueeze(2)   # (T,C,1)

    # Combine
    # -> shape (T,C,3)
    grad_t_c_3 = (df01 * cross_e2n.unsqueeze(1) +
                  df02 * cross_nE1.unsqueeze(1)) / norm_n.unsqueeze(1)

    # Reorder to (C,T,3)
    grad_all = grad_t_c_3.permute(1,0,2).contiguous()  # (C,T,3)

    # -------------------------------------------------------------------------
    # 3) Build all pairs (c, i) with c != i, in a single tensor of shape (K, 2).
    # -------------------------------------------------------------------------
    c_idx = torch.arange(C, device=device)[:, None].expand(C, C)  # (C,C)
    i_idx = torch.arange(C, device=device)[None, :].expand(C, C)  # (C,C)
    mask  = (c_idx != i_idx)
    pair_idx = torch.stack([c_idx[mask], i_idx[mask]], dim=1)  # (K,2)
    # K = C*(C-1)

    # We'll need these split out
    c_of_pair = pair_idx[:, 0]  # shape (K,)
    i_of_pair = pair_idx[:, 1]  # shape (K,)

    K = pair_idx.shape[0]  # number of channel pairs

    # -------------------------------------------------------------------------
    # 4) Compute "soft boundary" intersection weights w01, w12, w20
    #
    #    We'll do this for all T triangles and all K channel pairs at once.
    #
    #    f_tri shape => (T,3,C). We'll reorder to (C,T,3) for easier indexing:
    # -------------------------------------------------------------------------
    f_tri_ct3 = f_tri.permute(2,0,1)  # (C,T,3)

    # d_ = f_tri(c) - f_tri(i) => shape (T,3,K)
    #  step 1: gather f-values for c: shape => (K,T,3)
    fc_t3 = f_tri_ct3[c_of_pair]  # (K,T,3)
    fi_t3 = f_tri_ct3[i_of_pair]  # (K,T,3)
    d_kt3 = fc_t3 - fi_t3         # (K,T,3)
    # reorder to (T,3,K)
    d_t3k = d_kt3.permute(1,2,0).contiguous()  # (T,3,K)

    # Each corner => d0, d1, d2 => shape (T,K)
    d0 = d_t3k[:,0,:]  # (T,K)
    d1 = d_t3k[:,1,:]
    d2 = d_t3k[:,2,:]

    # Intersection weight functions
    w01 = torch.sigmoid(-beta * (d0 * d1))   # (T,K)
    w12 = torch.sigmoid(-beta * (d1 * d2))
    w20 = torch.sigmoid(-beta * (d2 * d0))

    # Barycentric factor for intersection points
    abs_d0 = d0.abs()
    abs_d1 = d1.abs()
    abs_d2 = d2.abs()

    alpha01 = abs_d0 / (abs_d0 + abs_d1 + eps)  # (T,K)
    alpha12 = abs_d1 / (abs_d1 + abs_d2 + eps)
    alpha20 = abs_d2 / (abs_d2 + abs_d0 + eps)

    # Intersection coords for edges
    # coords_01 = p0 + alpha01*(p1-p0), etc.
    # We'll broadcast over the extra dim K.
    E01_ = (p1 - p0).unsqueeze(1)  # (T,1,3)
    E12_ = (p2 - p1).unsqueeze(1)  # (T,1,3)
    E20_ = (p0 - p2).unsqueeze(1)  # (T,1,3)

    coords_01 = p0.unsqueeze(1) + alpha01.unsqueeze(2)*E01_  # (T,K,3)
    coords_12 = p1.unsqueeze(1) + alpha12.unsqueeze(2)*E12_
    coords_20 = p2.unsqueeze(1) + alpha20.unsqueeze(2)*E20_

    # Seg intersection "likelihood" if both corners are on the boundary
    segW0 = w01 * w12  # (T,K)
    segW1 = w12 * w20
    segW2 = w20 * w01

    # -------------------------------------------------------------------------
    # 5) Adjacency direction misalignment
    #
    #    For each pair (c,i), we have grad_diff = grad_c - grad_i => (K,T,3).
    #    For adjacency edges (t1, t2), we want the difference in directions
    #    to be weighted by the intersection-likelihood from segW_ of each tri.
    # -------------------------------------------------------------------------
    # grad_diff_ => shape (K,T,3)
    gc_t3 = grad_all[c_of_pair]  # (K,T,3)
    gi_t3 = grad_all[i_of_pair]  # (K,T,3)
    grad_diff_kt3 = gc_t3 - gi_t3

    # Normalize each difference so we compare directions only
    norm_kt1 = grad_diff_kt3.norm(dim=2, keepdim=True).clamp_min(eps)
    grad_diff_kt3 = grad_diff_kt3 / norm_kt1  # direction only

    # Now gather for adjacency: adjacency is (E,2) of triangle indices
    t1_ = adajancy[:,0]  # (E,)
    t2_ = adajancy[:,1]  # (E,)

    # shape => (K,E,3)
    grad_t1 = grad_diff_kt3[:, t1_, :]
    grad_t2 = grad_diff_kt3[:, t2_, :]
    dir_diff = grad_t1 - grad_t2
    dir_diff_sq = (dir_diff * dir_diff).sum(dim=2) 
    dir_diff_sq_e_k = dir_diff_sq.permute(1, 0) # (K,E)

    # We also need the intersection-likelihood for t1_, t2_
    # segW_ shape => (T,K,3). We'll combine segW0, segW1, segW2 => (T,K,3).
    segW_stack = torch.stack([segW0, segW1, segW2], dim=2)  # (T,K,3)

    # shape => (E,K,3)
    segW_t1 = segW_stack[t1_, ...]
    segW_t2 = segW_stack[t2_, ...]

    # Outer product in the last dimension => sum
    # segW_t1_exp => (E,K,3,1)
    # segW_t2_exp => (E,K,1,3)
    segW_t1_exp = segW_t1.unsqueeze(3)
    segW_t2_exp = segW_t2.unsqueeze(2)
    w_outer = segW_t1_exp * segW_t2_exp  # (E,K,3,3)
    sum_w_e_k = w_outer.sum(dim=(2,3))   # (E,K)

    # Weighted adjacency cost
    cost_e_k = sum_w_e_k * dir_diff_sq_e_k  # (E,K)
    adjacency_loss = cost_e_k.sum()     # scalar

    # -------------------------------------------------------------------------
    # 6) Plane-fitting cost: for each channel c, gather intersection points
    #    from pairs (c, i) for i != c. Then encourage them to be coplanar.
    #
    #    We'll do this by computing (per channel c) the weighted sums:
    #      sum_w[c], sum_w_x[c], sum_w_xx[c], and from that we get a 3x3 cov.
    #    Then do a batched SVD or eigen across c, pick the smallest singular
    #    value as out-of-plane measure, sum up.
    # -------------------------------------------------------------------------
    # We have intersection coords_01, coords_12, coords_20 => shape (T,K,3).
    # Their weights => w01, w12, w20 => shape (T,K).
    #
    # But each pair (k) belongs to channel c_of_pair[k].
    #
    # We'll flatten T*K points to one dimension => N = T*K. Then replicate 3 segments => 3*N.
    # We'll also replicate the "owner channel" => plane_idx for each of those points.
    #
    # Let's do it step by step:
    #
    # Flatten intersection coords into shape => (3*N, 3),
    # Flatten intersection weights => (3*N,),
    # Build channel indices => (3*N,) in [0..C-1].
    # Then scatter-add into per-channel sums.

    # coords_01f => (T*K, 3)
    coords_01f = coords_01.view(-1, 3)
    coords_12f = coords_12.view(-1, 3)
    coords_20f = coords_20.view(-1, 3)

    w01f = w01.view(-1)  # (T*K,)
    w12f = w12.view(-1)
    w20f = w20.view(-1)

    # We stack them up => shape => (3*T*K, 3)
    coords_all = torch.cat([coords_01f, coords_12f, coords_20f], dim=0)
    w_all      = torch.cat([w01f,       w12f,       w20f      ], dim=0)

    # The channel "owner" of these intersection lines is c_of_pair[k],
    # repeated T times for each k, and then repeated once more for each segment.
    # c_of_pair => (K,). We replicate across T => shape => (T*K,).
    # Then replicate x3 for the 3 segments => shape => (3*T*K,).
    c_of_pair_T = c_of_pair.unsqueeze(0).expand(T, K).reshape(-1)  # (T*K,)
    c_all       = torch.cat([c_of_pair_T, c_of_pair_T, c_of_pair_T], dim=0)  # (3*T*K,)

    # We will scatter into arrays of shape (C, ...).
    # sum_w[c], sum_w_x[c], sum_w_xx[c]. Then do SVD per c.

    # 6a) sum_w[c]
    sum_w_c = torch.zeros(C, dtype=dtype, device=device)
    sum_w_c = sum_w_c.scatter_add(0, c_all, w_all)

    # 6b) sum_w_x[c,:], shape => (C,3)
    sum_w_x_c = torch.zeros(C, 3, dtype=dtype, device=device)
    sum_w_x_c = sum_w_x_c.scatter_add(0, c_all.unsqueeze(1).expand(-1,3), coords_all * w_all.unsqueeze(1))

    # 6c) sum_w_xx[c,:,:], shape => (C,3,3)
    # We'll do outer products x x^T for each point, weighted by w.
    x_outer = coords_all.unsqueeze(2) * coords_all.unsqueeze(1)  # (3*T*K,3,3)
    w_x_outer = x_outer * w_all.view(-1,1,1)

    sum_w_xx_c = torch.zeros(C,3,3, dtype=dtype, device=device)
    # We need an index for scatter into 2D. We'll do:
    #   c_all => shape (3*T*K,)
    #   We want to scatter each (3,3) sub-block.  We'll flatten the last two dims:
    w_x_outer_2d = w_x_outer.view(-1, 9)     # (3*T*K, 9)
    sum_w_xx_c_2d = sum_w_xx_c.view(C, 9)    # (C,9)

    sum_w_xx_c_2d = sum_w_xx_c_2d.scatter_add(0, c_all.unsqueeze(1).expand(-1,9), w_x_outer_2d)
    sum_w_xx_c = sum_w_xx_c_2d.view(C,3,3)

    # -------------------------------------------------------------------------
    # Now compute the "plane penalty" for each c by analyzing the 3x3 covariance.
    # Weighted mean => mean_c = sum_w_x_c[c]/sum_w_c[c]
    # Weighted cov => sum_w_xx_c[c] / sum_w_c[c] - outer(mean_c, mean_c)
    # The out-of-plane error ~ the smallest singular value of that 3x3.
    # We'll do a batched approach across c => shape (C,3,3), then SVD => (C,3), pick S[-1].
    # -------------------------------------------------------------------------
    # Avoid channels that have sum_w_c[c] < small threshold:
    plane_loss_per_c = torch.zeros(C, dtype=dtype, device=device)

    valid = (sum_w_c > 1e-12)
    if valid.any():
        mean_c = torch.zeros(C, 3, dtype=dtype, device=device)
        mean_c[valid] = sum_w_x_c[valid] / sum_w_c[valid].unsqueeze(-1)

        # Covariance (unscaled):
        #   cov_c = sum_w_xx_c[c]/sum_w_c[c] - outer(mean_c, mean_c)
        # We'll do it in two steps:
        cov_c = sum_w_xx_c.clone()
        cov_c[valid] = cov_c[valid] / sum_w_c[valid].unsqueeze(-1).unsqueeze(-1)

        # subtract outer(mean_c, mean_c):
        # We'll expand mean_c => (C,3,1) * (C,1,3) => (C,3,3).
        mc = mean_c.view(C,3,1)
        outer_mm = mc @ mc.transpose(1,2)  # (C,3,3)
        cov_c[valid] = cov_c[valid] - outer_mm[valid]

        # SVD => shape => U[SVD], S[SVD], V[SVD] with shape (C,3,3) => S => (C,3)
        # if not valid => keep it zero
        # For numerical stability, cast to float if needed.
        cov_f32 = cov_c.float()
        U,S,Vt = torch.linalg.svd(cov_f32, full_matrices=False)  # S => (C,3)
        # plane_loss = smallest singular value for each c
        # (We typically take S[-1], but S is sorted descending, so S[-1] is largest.
        #  Actually torch.linalg.svd doesn't guarantee sorted. We'll just do S.min(dim=1).)
        s_min, _ = S.min(dim=1)  # shape => (C,)
        plane_loss_per_c[valid] = s_min.to(dtype)

    # Weighted plane cost => sum of plane_loss_per_c * lambda_plane
    plane_loss = lambda_plane * plane_loss_per_c.sum()

    # -------------------------------------------------------------------------
    # Final total
    # -------------------------------------------------------------------------
    total_loss = adjacency_loss + plane_loss
    return total_loss