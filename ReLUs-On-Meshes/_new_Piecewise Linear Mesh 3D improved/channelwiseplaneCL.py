import torch

#######################################################################
# A GLOBAL OR MODULE-LEVEL PARAMETER: plane_offsets
#   must define this as an nn.Parameter of shape (C,)
#   in the code, and set it before calling contour_alignment_loss.
# e.g.:
# plane_offsets = nn.Parameter(torch.zeros(C))  
#######################################################################
plane_offsets = None   # Placeholder to be set externally


def contour_alignment_loss(
    vertices:   torch.Tensor,   # (N, 3)
    faces:      torch.Tensor,   # (T, 3) long
    f_values:   torch.Tensor,   # (N, C)
    pinned_axes:torch.Tensor,   # (C,3) pinned plane normals
    beta: float =20.0,          # logistic edge weight param
    beta_edge:  float = 20.0,
    beta_triple: float = 20.0,
    include_triples: bool = False,
    adajancy: torch.Tensor = None,
    eps: float = 1e-9,
    lambda_plane: float = 1.0,
    lambda_contour: float = 1.0,
    plane_offsets: torch.Tensor = None,  # learnable offsets for each channel
) -> torch.Tensor:
    r"""
    Fully vectorized function that enforces one axis-aligned plane per channel.
    *Now uses a learnable offset* (plane_offsets[c]) for each channel c
    instead of solving d_i in closed form.

    Steps:
      1) Gather "soft" intersection points for (f_i=f_j) on edges & triple corners (if include_triples=True).
      2) For each intersection point, we compute dotvals[m,c] = pinned_axes[c] · x_m
      3) We add plane_offsets[c], forming dist[m,c] = dotvals[m,c] + plane_offsets[c].
      4) Weighted MSE => sum_{m,c} w_{m,c} * dist[m,c]^2

    pinned_axes: (C,3) => e.g. for 6 channels:
      +z => (0,0,1), -z => (0,0,-1), +y => (0,1,0), -y => (0,-1,0),
      +x => (1,0,0), -x => (-1,0,0)

 
    """
    device = vertices.device
    (N, D) = vertices.shape
    C = f_values.shape[1]
    T = faces.shape[0]

    # --------------------------------------------------------
    #  A) Build geometry arrays
    # --------------------------------------------------------
    tri_verts = vertices[faces]       # (T,3,3)
    tri_fvals = f_values[faces]       # (T,3,C)

    # For each triangle, define edges v0->v1, v1->v2, v2->v0
    edge_starts = torch.tensor([0,1,2], device=device)
    edge_ends   = torch.tensor([1,2,0], device=device)
    p0 = tri_verts[:, edge_starts, :]  # (T,3,3)
    p1 = tri_verts[:, edge_ends,   :]  # (T,3,3)

    # d[t,v,i,j] = tri_fvals[t,v,i] - tri_fvals[t,v,j]
    # => shape (T,3,C,C)
    tvf = tri_fvals.unsqueeze(3)   # (T,3,C,1)
    d_full = tvf - tvf.transpose(2,3)  # (T,3,C,C)

    d0 = d_full[:, edge_starts, :, :]   # (T,3,C,C)
    d1 = d_full[:, edge_ends,   :, :]   # (T,3,C,C)

    # --------------------------------------------------------
    #  B) Soft edge intersections for i<j
    # --------------------------------------------------------
    prod = d0*d1                       # (T,3,C,C)
    w_edge = torch.sigmoid(-beta_edge*prod)
    alpha = torch.abs(d0)/(torch.abs(d0)+torch.abs(d1)+eps)

    # coords => p0 + alpha*(p1-p0), shape => (T,3,C,C,3)
    edge_coords = p0.unsqueeze(2).unsqueeze(3) + alpha.unsqueeze(-1)*(p1 - p0).unsqueeze(2).unsqueeze(3)

    # keep only i<j
    i_arange = torch.arange(C, device=device)
    I, J = torch.meshgrid(i_arange, i_arange, indexing='ij')  # (C,C)
    pair_mask = (I < J)
    i_idx = I[pair_mask]  # ~C*(C-1)/2
    j_idx = J[pair_mask]
    Pp = i_idx.shape[0]

    # Flatten (T*3, C, C,3) -> (T*3, C,C, 3)
    ec_flat = edge_coords.view(T*3, C, C, 3)
    w_flat  = w_edge.view(T*3, C, C)

    coords_ij = ec_flat[:, i_idx, j_idx, :]   # (T*3, P, 3)
    w_ij      = w_flat[:, i_idx, j_idx]       # (T*3, P)

    # membership => channels i_idx, j_idx
    pair_membership = torch.zeros(Pp, C, device=device)
    row_ = torch.arange(Pp, device=device)
    pair_membership[row_, i_idx] = 1.0
    pair_membership[row_, j_idx] = 1.0

    coords_ij_flat = coords_ij.reshape(-1,3)   # (T*3*P, 3)
    w_ij_flat      = w_ij.reshape(-1)         # (T*3*P,)
    memb_ij_flat   = pair_membership.unsqueeze(0)\
                                     .expand(coords_ij.shape[0], -1, -1)\
                                     .reshape(-1,C)

    # --------------------------------------------------------
    #  C) Triple intersections 
    # --------------------------------------------------------
    coords_triple_flat = None
    w_triple_flat      = None
    memb_triple_flat   = None
    if include_triples:
        I3, J3, K3 = torch.meshgrid(i_arange, i_arange, i_arange, indexing='ij')
        triple_mask = (I3 < J3) & (J3 < K3)
        i_idx3 = I3[triple_mask]
        j_idx3 = J3[triple_mask]
        k_idx3 = K3[triple_mask]
        P3 = i_idx3.shape[0]

        # centroid approach
        centroid = tri_verts.mean(dim=1)  # (T,3)
        centroid_expanded = centroid.unsqueeze(1) # (T,1,3)
        centroid_3d = centroid_expanded.expand(-1, P3, -1)  # (T,P3,3)

        triple_memb = torch.zeros(P3, C, device=device)
        rr = torch.arange(P3, device=device)
        triple_memb[rr, i_idx3] = 1.0
        triple_memb[rr, j_idx3] = 1.0
        triple_memb[rr, k_idx3] = 1.0

        f0 = tri_fvals[:,0,:]   # (T,C)
        f1 = tri_fvals[:,1,:]
        f2 = tri_fvals[:,2,:]

        f0_sel = f0[:, i_idx3]   # (T,P3)
        f1_sel = f1[:, j_idx3]
        f2_sel = f2[:, k_idx3]

        max0 = f0.max(dim=1,keepdim=True)[0]
        max1 = f1.max(dim=1,keepdim=True)[0]
        max2 = f2.max(dim=1,keepdim=True)[0]

        w0 = torch.sigmoid( beta_triple*(f0_sel - max0) )
        w1 = torch.sigmoid( beta_triple*(f1_sel - max1) )
        w2 = torch.sigmoid( beta_triple*(f2_sel - max2) )
        w_triple = w0 * w1 * w2  # (T,P3)

        coords_triple_flat = centroid_3d.reshape(-1,3)
        w_triple_flat      = w_triple.reshape(-1)
        memb_triple_flat   = triple_memb.unsqueeze(0)\
                                          .expand(T,-1,-1)\
                                          .reshape(-1,C)

    # --------------------------------------------------------
    #  D) Combine pairwise + triple
    # --------------------------------------------------------
    if include_triples and coords_triple_flat is not None:
        all_coords  = torch.cat([coords_ij_flat, coords_triple_flat], dim=0)
        all_weights = torch.cat([w_ij_flat,      w_triple_flat],      dim=0)
        all_memb    = torch.cat([memb_ij_flat,   memb_triple_flat],   dim=0)
    else:
        all_coords  = coords_ij_flat
        all_weights = w_ij_flat
        all_memb    = memb_ij_flat

    # --------------------------------------------------------
    #  E) Instead of closed-form offset, we use a LEARNABLE
    #     offset in plane_offsets (global nn.Parameter).
    # --------------------------------------------------------
    # pinned_axes[c] => normal for channel c
    # dotvals => (M,C)
    dotvals = all_coords @ pinned_axes.transpose(0,1)  # => (M, C)

    # final_w => shape (M,C)
    final_w = all_memb * all_weights.unsqueeze(-1)

    # distance => dist[m,c] = dotvals[m,c] + plane_offsets[c]
    # (plane_offsets is a global param that must be set externally.)
    
    if plane_offsets is None:
        raise RuntimeError(
            "Global plane_offsets is not defined. Please set "
            "plane_offsets = nn.Parameter(...) shape=(C,)."
        )

    dist = dotvals + plane_offsets.unsqueeze(0)  # => (M,C)
    dist_sq = dist**2
    w_dist_sq = dist_sq * final_w
    total_loss = w_dist_sq.sum()

    return total_loss
