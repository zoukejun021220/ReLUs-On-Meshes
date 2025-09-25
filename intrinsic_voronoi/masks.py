"""Mask computations for losses."""

from __future__ import annotations

from typing import List

import torch


def seed_face_mask(
    faces: torch.Tensor,
    seed_rings: List[torch.Tensor],
    num_vertices: int,
) -> torch.Tensor:
    """Return boolean mask of shape (num_faces, num_channels).

    Each channel mask is true if any vertex of the face belongs to the
    precomputed seed ring for that channel.
    """

    device = faces.device
    num_faces = faces.shape[0]
    num_channels = len(seed_rings)
    mask = torch.zeros((num_faces, num_channels), dtype=torch.bool, device=device)

    for channel, ring in enumerate(seed_rings):
        if ring.numel() == 0:
            continue
        vertex_mask = torch.zeros(num_vertices, dtype=torch.bool, device=device)
        vertex_mask[ring] = True
        face_mask = vertex_mask[faces[:, 0]] | vertex_mask[faces[:, 1]] | vertex_mask[faces[:, 2]]
        mask[:, channel] = face_mask

    return mask


def eikonal_mask(seed_mask: torch.Tensor, bisector_mask: torch.Tensor) -> torch.Tensor:
    """Return mask according to specification."""
    return (1 - seed_mask.to(torch.float32)) * (1 - bisector_mask.unsqueeze(-1).to(torch.float32))
