import torch
from typing import Tuple


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _grad3d_intrinsic_pairs(e0: torch.Tensor,
                            e1: torch.Tensor,
                            h: torch.Tensor,
                            eps: float = 1e-10) -> torch.Tensor:
    """
    Compute intrinsic gradients for many channel pairs on identical triangle geometry.

    Args:
        e0, e1: (E,3) triangle edge vectors (v1-v0, v2-v0)
        h: (E,3,P) scalar values per triangle vertex and per pair
        eps: numerical clamp for Gram determinant

    Returns:
        g: (E,P,3) gradient vectors in R^3 per pair
    """
    # Gram components depend only on geometry (E,)
    G00 = (e0 * e0).sum(dim=1)
    G01 = (e0 * e1).sum(dim=1)
    G11 = (e1 * e1).sum(dim=1)
    det = (G00 * G11 - G01 * G01).clamp_min(eps)
    invG00 = G11 / det
    invG01 = -G01 / det
    invG11 = G00 / det

    # b = [h1-h0, h2-h0] with broadcasting over P
    b0 = h[:, 1, :] - h[:, 0, :]  # (E,P)
    b1 = h[:, 2, :] - h[:, 0, :]  # (E,P)

    # Solve a = G^{-1} b (broadcast invG** over P)
    a0 = invG00.unsqueeze(1) * b0 + invG01.unsqueeze(1) * b1  # (E,P)
    a1 = invG01.unsqueeze(1) * b0 + invG11.unsqueeze(1) * b1  # (E,P)

    # g = a0*e0 + a1*e1 (broadcast over P)
    g = a0.unsqueeze(2) * e0.unsqueeze(1) + a1.unsqueeze(2) * e1.unsqueeze(1)  # (E,P,3)
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


# Simple cache keyed by faces storage pointer and shape
_EDGE_CACHE = {}


def _build_edges_and_adjacency(faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build undirected edges and adjacent triangle indices from face list using
    fully vectorized tensor operations (no Python loops).

    Returns:
        edge_idx: (E,2) vertex indices (min, max)
        edge_tris: (E,2) adjacent triangle indices (-1 for boundary)
    """
    # Cache lookup by storage pointer and shape
    key = (int(faces.storage().data_ptr()), tuple(faces.shape), faces.device.type)
    cached = _EDGE_CACHE.get(key)
    if cached is not None:
        return cached[0], cached[1]

    device = faces.device
    T = faces.shape[0]

    # All directed edges from faces: (T,3,2) -> (3T,2)
    edges = torch.stack(
        [
            torch.stack([faces[:, 0], faces[:, 1]], dim=1),
            torch.stack([faces[:, 1], faces[:, 2]], dim=1),
            torch.stack([faces[:, 2], faces[:, 0]], dim=1),
        ],
        dim=1,
    ).reshape(-1, 2)

    # Sort endpoints to get undirected edge keys
    edges_sorted, _ = torch.sort(edges, dim=1)  # (3T,2)

    # Track which face each edge came from
    face_indices = torch.arange(T, device=device, dtype=faces.dtype).repeat_interleave(3)

    # Unique edge hashing and grouping
    # Compute a GPU-resident scalar for hashing to avoid CPU syncs
    Vmax = faces.max().to(torch.int64) + 1  # tensor scalar on device
    edge_hash = edges_sorted[:, 0].to(torch.int64) * Vmax + edges_sorted[:, 1].to(torch.int64)
    sort_hash, sort_idx = torch.sort(edge_hash)
    sorted_edges = edges_sorted[sort_idx]
    sorted_faces = face_indices[sort_idx]

    # Find run boundaries where the edge changes
    first = torch.ones(sort_hash.shape[0], dtype=torch.bool, device=device)
    first[1:] = sort_hash[1:] != sort_hash[:-1]
    change_idx = torch.nonzero(first, as_tuple=False).squeeze(1)
    # Append sentinel end
    change_idx = torch.cat([change_idx, sort_hash.new_tensor([sort_hash.shape[0]])])

    # Counts per unique edge
    counts = change_idx[1:] - change_idx[:-1]

    # Build edge index array for all unique edges
    edge_idx_unique = sorted_edges[change_idx[:-1]]  # (U,2)

    # For adjacency, collect up to two triangles per edge
    # Prepare output filled with -1
    U = edge_idx_unique.shape[0]
    edge_tris = torch.full((U, 2), -1, dtype=faces.dtype, device=device)

    # Indices for the first occurrence of each run
    first_pos = change_idx[:-1]
    edge_tris[:, 0] = sorted_faces[first_pos]

    # For edges that appear at least twice, take the second occurrence as the other triangle
    has_second = counts >= 2
    if has_second.any():
        second_pos = first_pos[has_second] + 1
        edge_tris[has_second, 1] = sorted_faces[second_pos]

    # Save cache and return
    _EDGE_CACHE[key] = (edge_idx_unique, edge_tris)
    return edge_idx_unique, edge_tris


def contour_alignment_codex(
    vertices: torch.Tensor,      # (N,3)
    faces: torch.Tensor,         # (T,3)
    f_values: torch.Tensor,      # (N,C)
    *,
    beta_edge: float = 6.0,
    include_triples: bool = False,
    edge_idx: torch.Tensor = None,
    edge_tris: torch.Tensor = None,
) -> torch.Tensor:
    """
    Intrinsic 3D gradient-alignment contour loss (Codex version).

    - Works on the surface: uses per-face intrinsic gradients (Gram solve)
    - Detects active boundaries per edge via soft pairwise sign-change
    - Aligns boundary tangents across adjacent faces
    - Uses soft-OR coverage and gating by gradient magnitude and edge length

    Returns:
        Scalar loss tensor
    """
    device, dtype = f_values.device, f_values.dtype
    N, C = f_values.shape
    if C < 2:
        return torch.zeros((), device=device, dtype=dtype)

    # Build topology once
    if edge_idx is None or edge_tris is None:
        edge_idx, edge_tris = _build_edges_and_adjacency(faces)
    valid = (edge_tris[:, 0] >= 0) & (edge_tris[:, 1] >= 0)
    if not torch.any(valid):
        return torch.zeros((), device=device, dtype=dtype)

    va = edge_idx[valid, 0]
    vb = edge_idx[valid, 1]
    tL = edge_tris[valid, 0]
    tR = edge_tris[valid, 1]

    # All channel pairs (i<j)
    ii, jj = torch.triu_indices(C, C, offset=1, device=device)
    P = ii.numel()

    # Pairwise edge crossing weights for all pairs
    Fa = f_values[va][:, ii] - f_values[va][:, jj]  # (E,P)
    Fb = f_values[vb][:, ii] - f_values[vb][:, jj]  # (E,P)
    w_pairs = torch.sigmoid(-beta_edge * Fa * Fb)   # (E,P)
    conf = 0.5 * (Fa.abs() + Fb.abs())
    w_pairs = w_pairs * torch.sigmoid(5.0 * (conf - 0.2))
    w_pairs = w_pairs.clamp_min(1e-6)

    # Per-edge mixing over pairs and soft-OR activity
    pair_mix = w_pairs / (w_pairs.sum(dim=1, keepdim=True) + 1e-9)  # (E,P)
    phi = 1.0 - torch.prod(1.0 - w_pairs, dim=1)                    # (E,)

    # Per-pair per-side height values on faces
    faces_L = faces[tL]
    faces_R = faces[tR]
    F_L = f_values[faces_L]  # (E,3,C)
    F_R = f_values[faces_R]
    hL = F_L[:, :, ii] - F_L[:, :, jj]  # (E,3,P)
    hR = F_R[:, :, ii] - F_R[:, :, jj]

    # Triangle geometry
    v0L = vertices[faces_L[:, 0]]
    v1L = vertices[faces_L[:, 1]]
    v2L = vertices[faces_L[:, 2]]
    v0R = vertices[faces_R[:, 0]]
    v1R = vertices[faces_R[:, 1]]
    v2R = vertices[faces_R[:, 2]]

    # Compute intrinsic gradients for all pairs using shared geometry (no replication)
    e0L = v1L - v0L
    e1L = v2L - v0L
    e0R = v1R - v0R
    e1R = v2R - v0R

    gL = _grad3d_intrinsic_pairs(e0L, e1L, hL)  # (E,P,3)
    gR = _grad3d_intrinsic_pairs(e0R, e1R, hR)  # (E,P,3)

    # Face normals and in-plane projection
    def normals(v0, v1, v2):
        e0 = v1 - v0
        e1 = v2 - v0
        n = torch.cross(e0, e1, dim=1)
        return _safe_normalize(n, dim=1)

    nL = normals(v0L, v1L, v2L)  # (E,3)
    nR = normals(v0R, v1R, v2R)

    def proj_in_plane(g, n):
        dot = (g * n[:, None, :]).sum(dim=2, keepdim=True)
        return g - dot * n[:, None, :]

    gL = proj_in_plane(gL, nL)
    gR = proj_in_plane(gR, nR)

    # Boundary tangents per pair and side
    tauL = _safe_normalize(torch.cross(nL[:, None, :], gL, dim=2), dim=2)
    tauR = _safe_normalize(torch.cross(nR[:, None, :], gR, dim=2), dim=2)

    # Misalignment per pair
    cosang = (tauL * tauR).sum(dim=2).abs().clamp_max(1.0)  # (E,P)
    mis = 1.0 - cosang

    # Expected misalignment per edge over pairs
    mis_edge = (pair_mix * mis).sum(dim=1)  # (E,)

    # Gradient magnitude gating and edge-length gating
    gLm = gL.norm(dim=2).mean(dim=1) + 1e-8
    gRm = gR.norm(dim=2).mean(dim=1) + 1e-8
    grad_gate = torch.sqrt(gLm * gRm)
    scale = torch.nanmedian(grad_gate).detach().clamp_min(1e-6)
    grad_gate = (grad_gate / scale).clamp(0.0, 2.0)
    grad_gate = torch.nan_to_num(grad_gate, nan=0.0, posinf=2.0, neginf=0.0)

    edge_vec = vertices[vb] - vertices[va]
    edge_len = edge_vec.norm(dim=1)
    len_gate = (edge_len / (edge_len.mean().detach() + 1e-12)).clamp(0.5, 2.0)

    # Final per-edge weight
    w_edge = (phi.clamp_min(1e-4) * grad_gate * len_gate)

    # Charbonnier aggregation
    loss_edge = torch.sqrt(mis_edge * mis_edge + 1e-6)
    num = (w_edge * loss_edge).sum()
    den = (w_edge.sum() + 1e-9)
    loss = num / den

    # Optional: triple point barrier on faces (lightweight)
    if include_triples and C >= 3:
        # Face probabilities (avg over vertices)
        p_v = torch.softmax(2.0 * f_values, dim=1)
        p_t = (p_v[faces[:, 0]] + p_v[faces[:, 1]] + p_v[faces[:, 2]]) / 3.0  # (T,C)
        top3, _ = torch.topk(p_t, k=min(3, C), dim=1)
        if top3.shape[1] == 3:
            gap = top3[:, 1] - top3[:, 2]
            tri_area = 0.5 * torch.linalg.norm(
                torch.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                            vertices[faces[:, 2]] - vertices[faces[:, 0]], dim=1), dim=-1)
            w = tri_area / (tri_area.sum() + 1e-9)
            triple = (w * torch.relu(0.10 - gap)).sum()
            loss = loss + 0.1 * triple  # small weight

    return loss
