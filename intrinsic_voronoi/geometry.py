"""Differential geometry helpers for the Voronoi pipeline."""

from __future__ import annotations

from typing import Tuple

import torch

from .mesh import PrecomputedGeometry


def face_gradients(field: torch.Tensor, precomp: PrecomputedGeometry) -> torch.Tensor:
    """Return projected per-face gradients for each channel.

    Args:
        field: Tensor of shape (num_vertices, num_channels).
        precomp: Precomputed geometry information.

    Returns:
        Tensor of shape (num_faces, num_channels, 3) in float32 matching
        the dtype of ``field``.
    """

    faces = precomp.faces
    device = field.device
    channels = field.shape[1]

    field64 = field.to(torch.float64)
    face_values = field64.index_select(0, faces.reshape(-1)).reshape(faces.shape[0], 3, channels)
    v0 = face_values[:, 0, :]
    v1 = face_values[:, 1, :]
    v2 = face_values[:, 2, :]

    b0 = v1 - v0
    b1 = v2 - v0
    stacked = torch.stack([b0, b1], dim=-1)  # (faces, channels, 2)

    gram_inv = precomp.gram_inv.unsqueeze(1)
    coeffs = torch.matmul(gram_inv, stacked.unsqueeze(-1)).squeeze(-1)

    e0 = precomp.edge_vectors[:, 0].unsqueeze(1)
    e1 = precomp.edge_vectors[:, 1].unsqueeze(1)
    grad = coeffs[..., 0].unsqueeze(-1) * e0 + coeffs[..., 1].unsqueeze(-1) * e1

    normals = precomp.face_normals.unsqueeze(1)
    proj = grad - (grad * normals).sum(dim=-1, keepdim=True) * normals
    return proj.to(field.dtype)


def pairwise_gradients(face_grads: torch.Tensor, pair_indices: torch.Tensor) -> torch.Tensor:
    """Compute gradients of pairwise differences for each face."""

    grad_i = face_grads.index_select(1, pair_indices[:, 0])
    grad_j = face_grads.index_select(1, pair_indices[:, 1])
    return grad_i - grad_j


def edge_mean_pair_norms(
    pair_grads: torch.Tensor,
    edge_faces: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return mean pairwise gradient norms for the two faces of each edge."""

    num_edges = edge_faces.shape[0]
    device = pair_grads.device
    norms = pair_grads.norm(dim=-1)

    left_idx = edge_faces[:, 0]
    right_idx = edge_faces[:, 1]
    mask_left = left_idx >= 0
    mask_right = right_idx >= 0

    zeros = torch.zeros((num_edges,), device=device, dtype=pair_grads.dtype)
    g_left = zeros.clone()
    g_right = zeros.clone()

    if mask_left.any():
        g_left[mask_left] = norms[left_idx[mask_left]].mean(dim=1)
    if mask_right.any():
        g_right[mask_right] = norms[right_idx[mask_right]].mean(dim=1)

    return g_left, g_right


def bisector_mask_from_phi(phi: torch.Tensor, face_edges: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return per-face bisector mask using edge activity values."""

    face_phi = phi.index_select(0, face_edges.reshape(-1)).reshape(face_edges.shape[0], 3)
    return (face_phi.max(dim=1).values >= threshold)
