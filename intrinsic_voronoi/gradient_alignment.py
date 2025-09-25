"""Gradient-alignment loss adapted from codex_grad_alignment."""

from __future__ import annotations

from typing import Tuple

import torch


def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def _grad3d_intrinsic_pairs(
    e0: torch.Tensor,
    e1: torch.Tensor,
    h: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Intrinsic gradients for multiple channel pairs on shared triangle geometry."""

    G00 = (e0 * e0).sum(dim=1)
    G01 = (e0 * e1).sum(dim=1)
    G11 = (e1 * e1).sum(dim=1)
    det = (G00 * G11 - G01 * G01).clamp_min(eps)
    invG00 = G11 / det
    invG01 = -G01 / det
    invG11 = G00 / det

    b0 = h[:, 1, :] - h[:, 0, :]
    b1 = h[:, 2, :] - h[:, 0, :]

    a0 = invG00.unsqueeze(1) * b0 + invG01.unsqueeze(1) * b1
    a1 = invG01.unsqueeze(1) * b0 + invG11.unsqueeze(1) * b1

    g = a0.unsqueeze(2) * e0.unsqueeze(1) + a1.unsqueeze(2) * e1.unsqueeze(1)
    return torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)


_EDGE_CACHE: dict[Tuple[int, Tuple[int, ...], str], Tuple[torch.Tensor, torch.Tensor]] = {}


def _build_edges_and_adjacency(faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return undirected edges and adjacent triangle indices from faces."""

    key = (int(faces.storage().data_ptr()), tuple(faces.shape), faces.device.type)
    cached = _EDGE_CACHE.get(key)
    if cached is not None:
        return cached

    device = faces.device
    T = faces.shape[0]

    edges = torch.stack(
        [
            torch.stack([faces[:, 0], faces[:, 1]], dim=1),
            torch.stack([faces[:, 1], faces[:, 2]], dim=1),
            torch.stack([faces[:, 2], faces[:, 0]], dim=1),
        ],
        dim=1,
    ).reshape(-1, 2)

    edges_sorted, _ = torch.sort(edges, dim=1)
    face_indices = torch.arange(T, device=device, dtype=faces.dtype).repeat_interleave(3)

    vmax = faces.max().to(torch.int64) + 1
    edge_hash = edges_sorted[:, 0].to(torch.int64) * vmax + edges_sorted[:, 1].to(torch.int64)
    sort_hash, sort_idx = torch.sort(edge_hash)
    sorted_edges = edges_sorted[sort_idx]
    sorted_faces = face_indices[sort_idx]

    first = torch.ones(sort_hash.shape[0], dtype=torch.bool, device=device)
    first[1:] = sort_hash[1:] != sort_hash[:-1]
    change_idx = torch.nonzero(first, as_tuple=False).squeeze(1)
    change_idx = torch.cat([change_idx, sort_hash.new_tensor([sort_hash.shape[0]])])

    counts = change_idx[1:] - change_idx[:-1]
    edge_idx_unique = sorted_edges[change_idx[:-1]]

    U = edge_idx_unique.shape[0]
    edge_tris = torch.full((U, 2), -1, dtype=faces.dtype, device=device)
    first_pos = change_idx[:-1]
    edge_tris[:, 0] = sorted_faces[first_pos]

    has_second = counts >= 2
    if has_second.any():
        second_pos = first_pos[has_second] + 1
        edge_tris[has_second, 1] = sorted_faces[second_pos]

    _EDGE_CACHE[key] = (edge_idx_unique, edge_tris)
    return edge_idx_unique, edge_tris


def contour_alignment_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
    field: torch.Tensor,
    *,
    beta_edge: float = 6.0,
    include_triples: bool = False,
    edge_indices: torch.Tensor | None = None,
    edge_faces: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gradient-alignment contour loss operating on intrinsic surface geometry."""

    device, dtype = field.device, field.dtype
    num_channels = field.shape[1]
    if num_channels < 2:
        return torch.zeros((), device=device, dtype=dtype)

    if edge_indices is None or edge_faces is None:
        edge_indices, edge_faces = _build_edges_and_adjacency(faces)

    valid = (edge_faces[:, 0] >= 0) & (edge_faces[:, 1] >= 0)
    if not torch.any(valid):
        return torch.zeros((), device=device, dtype=dtype)

    edge_indices = edge_indices[valid]
    edge_faces = edge_faces[valid]

    va = edge_indices[:, 0]
    vb = edge_indices[:, 1]
    left_tris = edge_faces[:, 0]
    right_tris = edge_faces[:, 1]

    ii, jj = torch.triu_indices(num_channels, num_channels, offset=1, device=device)
    num_pairs = ii.numel()

    Fa = field.index_select(0, va)[:, ii] - field.index_select(0, va)[:, jj]
    Fb = field.index_select(0, vb)[:, ii] - field.index_select(0, vb)[:, jj]
    w_pairs = torch.sigmoid(-beta_edge * Fa * Fb)
    conf = 0.5 * (Fa.abs() + Fb.abs())
    w_pairs = w_pairs * torch.sigmoid(5.0 * (conf - 0.2))
    w_pairs = w_pairs.clamp_min(1e-6)

    pair_mix = w_pairs / (w_pairs.sum(dim=1, keepdim=True) + 1e-9)
    phi = 1.0 - torch.prod(1.0 - w_pairs, dim=1)

    faces_left = faces.index_select(0, left_tris)
    faces_right = faces.index_select(0, right_tris)
    field_left = field.index_select(0, faces_left.reshape(-1)).reshape(-1, 3, num_channels)
    field_right = field.index_select(0, faces_right.reshape(-1)).reshape(-1, 3, num_channels)

    h_left = field_left[:, :, ii] - field_left[:, :, jj]
    h_right = field_right[:, :, ii] - field_right[:, :, jj]

    v0L = vertices.index_select(0, faces_left[:, 0])
    v1L = vertices.index_select(0, faces_left[:, 1])
    v2L = vertices.index_select(0, faces_left[:, 2])
    v0R = vertices.index_select(0, faces_right[:, 0])
    v1R = vertices.index_select(0, faces_right[:, 1])
    v2R = vertices.index_select(0, faces_right[:, 2])

    e0L = v1L - v0L
    e1L = v2L - v0L
    e0R = v1R - v0R
    e1R = v2R - v0R

    gL = _grad3d_intrinsic_pairs(e0L, e1L, h_left)
    gR = _grad3d_intrinsic_pairs(e0R, e1R, h_right)

    def normals(v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        cross = torch.cross(v1 - v0, v2 - v0, dim=1)
        return _safe_normalize(cross, dim=1)

    nL = normals(v0L, v1L, v2L)
    nR = normals(v0R, v1R, v2R)

    def project(g: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        dot = (g * n[:, None, :]).sum(dim=2, keepdim=True)
        return g - dot * n[:, None, :]

    gL = project(gL, nL)
    gR = project(gR, nR)

    tauL = _safe_normalize(torch.cross(nL[:, None, :], gL, dim=2), dim=2)
    tauR = _safe_normalize(torch.cross(nR[:, None, :], gR, dim=2), dim=2)

    cosang = (tauL * tauR).sum(dim=2).abs().clamp_max(1.0)
    mis = 1.0 - cosang
    mis_edge = (pair_mix * mis).sum(dim=1)

    gLm = gL.norm(dim=2).mean(dim=1) + 1e-8
    gRm = gR.norm(dim=2).mean(dim=1) + 1e-8
    grad_gate = torch.sqrt(gLm * gRm)
    scale = torch.nanmedian(grad_gate).detach().clamp_min(1e-6)
    grad_gate = (grad_gate / scale).clamp(0.0, 2.0)
    grad_gate = torch.nan_to_num(grad_gate, nan=0.0, posinf=2.0, neginf=0.0)

    edge_vec = vertices.index_select(0, vb) - vertices.index_select(0, va)
    edge_len = edge_vec.norm(dim=1)
    len_gate = (edge_len / (edge_len.mean().detach() + 1e-12)).clamp(0.5, 2.0)

    w_edge = phi.clamp_min(1e-4) * grad_gate * len_gate
    loss_edge = torch.sqrt(mis_edge * mis_edge + 1e-6)
    numerator = (w_edge * loss_edge).sum()
    denominator = w_edge.sum() + 1e-9
    loss = numerator / denominator

    if include_triples and num_channels >= 3:
        probs = torch.softmax(2.0 * field, dim=1)
        faces_probs = probs.index_select(0, faces.reshape(-1)).reshape(faces.shape[0], 3, num_channels)
        p_face = faces_probs.mean(dim=1)
        topk = torch.topk(p_face, k=min(3, num_channels), dim=1)
        if topk.values.shape[1] == 3:
            gap = topk.values[:, 1] - topk.values[:, 2]
            tri_area = 0.5 * torch.linalg.norm(
                torch.cross(
                    vertices.index_select(0, faces[:, 1]) - vertices.index_select(0, faces[:, 0]),
                    vertices.index_select(0, faces[:, 2]) - vertices.index_select(0, faces[:, 0]),
                    dim=1,
                ),
                dim=-1,
            )
            weight = tri_area / (tri_area.sum() + 1e-9)
            triple = (weight * torch.relu(0.10 - gap)).sum()
            loss = loss + 0.1 * triple

    return loss

