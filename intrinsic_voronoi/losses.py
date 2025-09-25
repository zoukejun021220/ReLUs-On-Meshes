"""Loss terms for the intrinsic Voronoi pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F

from .gating import EdgeGates
from .gradient_alignment import contour_alignment_loss
from .utils import charbonnier


@dataclass
class LossWeights:
    seed: float = 1.0
    eikonal: float = 4.0
    lipschitz: float = 2.0
    interface: float = 1.0


@dataclass
class LossTerms:
    seed: torch.Tensor
    eikonal: torch.Tensor
    lipschitz: torch.Tensor
    interface: torch.Tensor
    tv: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.seed + self.eikonal + self.lipschitz + self.interface + self.tv


def seed_anchor_loss(
    field: torch.Tensor,
    seed_rings: List[torch.Tensor],
    margin: float,
    dominance: float,
) -> torch.Tensor:
    """Fully vectorized seed anchoring with dominance margin per specification."""

    device, dtype = field.device, field.dtype
    num_channels = field.shape[1]

    if not seed_rings:
        return field.new_tensor(0.0)

    # Concatenate all seed ring indices with their corresponding channel ids
    valid_rings = [ring for ring in seed_rings if ring is not None and ring.numel() > 0]
    if not valid_rings:
        return field.new_tensor(0.0)

    channel_ids = []
    vertex_ids = []
    for channel, ring in enumerate(seed_rings):
        if ring is None or ring.numel() == 0:
            continue
        channel_ids.append(torch.full((ring.numel(),), channel, device=device, dtype=torch.long))
        vertex_ids.append(ring.to(device=device, dtype=torch.long))

    channel_idx = torch.cat(channel_ids, dim=0)
    vertex_idx = torch.cat(vertex_ids, dim=0)

    values = field.index_select(0, vertex_idx)  # (N_total, C)
    fk = values[torch.arange(values.shape[0], device=device), channel_idx]  # (N_total,)

    anchor_per_sample = fk.pow(2)

    if num_channels > 1:
        margin_t = torch.as_tensor(margin, device=device, dtype=dtype)
        diff = margin_t - (values - fk.unsqueeze(1))  # (N_total, C)
        mask = 1.0 - F.one_hot(channel_idx, num_classes=num_channels).to(dtype)
        hinge = F.relu(diff) * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        suppression_per_sample = (hinge.pow(2).sum(dim=1) / denom)
    else:
        suppression_per_sample = anchor_per_sample.new_zeros(anchor_per_sample.shape)

    # Aggregate per channel via scatter-add
    anchor_sum = torch.zeros(num_channels, device=device, dtype=dtype)
    anchor_sum.scatter_add_(0, channel_idx, anchor_per_sample)
    suppression_sum = torch.zeros(num_channels, device=device, dtype=dtype)
    suppression_sum.scatter_add_(0, channel_idx, suppression_per_sample)
    counts = torch.zeros(num_channels, device=device, dtype=dtype)
    counts.scatter_add_(0, channel_idx, torch.ones_like(anchor_per_sample, dtype=dtype))

    valid_mask = counts > 0
    counts_clamped = counts.clamp_min(1.0)

    anchor_mean = anchor_sum / counts_clamped
    suppression_mean = suppression_sum / counts_clamped

    combined = anchor_mean + dominance * suppression_mean

    if valid_mask.any():
        return combined[valid_mask].mean()
    return combined.mean()


def eikonal_loss(
    gradients: torch.Tensor,
    mask: torch.Tensor,
    face_areas: torch.Tensor,
    delta: float,
    *,
    upper_weight: float = 1.0,
    lower_weight: float = 0.25,
) -> torch.Tensor:
    """Face-isotropic hinge enforcing unit-speed with mild under-speed encouragement."""

    dtype = gradients.dtype
    mask_f = mask.to(dtype)

    area_mask = face_areas > 1e-16
    if area_mask.sum() < face_areas.numel():
        gradients = gradients[area_mask]
        mask_f = mask_f[area_mask]
        face_areas = face_areas[area_mask]

    norms = gradients.norm(dim=-1)
    upper = torch.clamp(norms - 1.0, min=0.0).pow(2)
    lower = torch.clamp(1.0 - norms, min=0.0).pow(2)
    residual = upper_weight * upper + lower_weight * lower
    weighted = residual * mask_f * face_areas.unsqueeze(1)
    denom = face_areas.sum() * gradients.shape[1] + 1e-12
    return weighted.sum() / denom


def lipschitz_hinge_loss(
    field: torch.Tensor,
    edges: torch.Tensor,
    edge_lengths: torch.Tensor,
) -> torch.Tensor:
    """Squared hinge enforcing 1-Lipschitz constraint per edge and channel."""

    va = edges[:, 0]
    vb = edges[:, 1]
    lengths = torch.clamp(edge_lengths.to(field.dtype), min=1e-12).unsqueeze(1)
    diffs = field.index_select(0, va) - field.index_select(0, vb)
    hinge = F.relu(diffs.abs() - lengths)
    return hinge.pow(2).mean()


def hamilton_jacobi_loss(
    field_gradients: torch.Tensor,
    pair_gradients: torch.Tensor,
    face_normals: torch.Tensor,
    edge_faces: torch.Tensor,
    edge_gates: EdgeGates,
    pair_indices: torch.Tensor,
    edge_unit_vectors: torch.Tensor,
    edge_dihedral_cos: torch.Tensor,
    edge_dihedral_sin: torch.Tensor,
    *,
    delta: float,
    epsilon_g: float,
) -> torch.Tensor:
    device = field_gradients.device
    dtype = field_gradients.dtype
    num_edges = edge_faces.shape[0]

    left_idx = edge_faces[:, 0]
    right_idx = edge_faces[:, 1]
    mask_left = left_idx >= 0
    mask_right = right_idx >= 0

    zeros_channels = torch.zeros((num_edges, field_gradients.shape[1], 3), device=device, dtype=dtype)
    zeros_pairs = torch.zeros((num_edges, pair_gradients.shape[1], 3), device=device, dtype=dtype)

    grad_left = zeros_channels.clone()
    grad_right = zeros_channels.clone()
    pair_left = zeros_pairs.clone()
    pair_right = zeros_pairs.clone()
    normal_left = torch.zeros((num_edges, 3), device=device, dtype=dtype)
    normal_right = torch.zeros((num_edges, 3), device=device, dtype=dtype)

    if mask_left.any():
        grad_left[mask_left] = field_gradients[left_idx[mask_left]]
        pair_left[mask_left] = pair_gradients[left_idx[mask_left]]
        normal_left[mask_left] = face_normals[left_idx[mask_left]]
    if mask_right.any():
        grad_right[mask_right] = field_gradients[right_idx[mask_right]]
        pair_right[mask_right] = pair_gradients[right_idx[mask_right]]
        normal_right[mask_right] = face_normals[right_idx[mask_right]]

    axis = F.normalize(edge_unit_vectors.to(device=device, dtype=dtype), dim=-1)
    axis = torch.nan_to_num(axis, nan=0.0, posinf=0.0, neginf=0.0)
    cos_t = edge_dihedral_cos.to(device=device, dtype=dtype)
    sin_t = edge_dihedral_sin.to(device=device, dtype=dtype)

    def _rotate_nd(vec: torch.Tensor) -> torch.Tensor:
        expand_shape = vec.shape[1:-1]
        axis_view = axis.view(axis.shape[0], *([1] * len(expand_shape)), 3)
        axis_exp = axis_view.expand(-1, *expand_shape, 3)
        cos_view = cos_t.view(-1, *([1] * len(expand_shape)), 1)
        sin_view = sin_t.view(-1, *([1] * len(expand_shape)), 1)
        cross = torch.cross(axis_exp, vec, dim=-1)
        dot = (vec * axis_exp).sum(dim=-1, keepdim=True)
        return vec * cos_view + cross * sin_view + axis_exp * dot * (1.0 - cos_view)

    def rotate(vec: torch.Tensor) -> torch.Tensor:
        if vec.dim() == 2:
            return _rotate_nd(vec.unsqueeze(1)).squeeze(1)
        return _rotate_nd(vec)

    grad_right = rotate(grad_right)
    pair_right = rotate(pair_right)
    normal_right = rotate(normal_right)

    counts = mask_left.float().view(-1, 1, 1) + mask_right.float().view(-1, 1, 1)
    counts = torch.clamp(counts, min=1.0)
    bar_grad = (grad_left + grad_right) / counts

    def normalize(vec: torch.Tensor) -> torch.Tensor:
        return vec / (vec.norm(dim=-1, keepdim=True) + epsilon_g)

    n_left = normalize(pair_left)
    n_right = normalize(pair_right)
    sum_normals = n_left + n_right
    both = (mask_left & mask_right).view(-1, 1, 1)
    n_hat = torch.where(both, normalize(sum_normals), torch.where(mask_left.view(-1, 1, 1), n_left, n_right))

    def tangent(normals_face: torch.Tensor, face_norm: torch.Tensor) -> torch.Tensor:
        cross = torch.cross(face_norm.unsqueeze(1).expand_as(normals_face), normals_face)
        return normalize(cross)

    tau_left = tangent(n_left, normal_left)
    tau_right = tangent(n_right, normal_right)
    sum_tau = tau_left + tau_right
    tau_hat = torch.where(both, normalize(sum_tau), torch.where(mask_left.view(-1, 1, 1), tau_left, tau_right))

    grad_i = bar_grad.index_select(1, pair_indices[:, 0])
    grad_j = bar_grad.index_select(1, pair_indices[:, 1])

    t_norm = ((grad_i + grad_j) * n_hat).sum(dim=-1)
    t_tan = ((grad_i - grad_j) * tau_hat).sum(dim=-1)

    penalties = charbonnier(t_norm, delta) + charbonnier(t_tan, delta)
    weighted = edge_gates.weight.unsqueeze(1) * edge_gates.mixture * penalties
    denom = edge_gates.weight.sum() + 1e-12
    return weighted.sum() / denom


def gradient_alignment_loss(
    field: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    edge_indices: torch.Tensor,
    edge_faces: torch.Tensor,
    *,
    beta_edge: float,
    include_triples: bool,
) -> torch.Tensor:
    vertices_t = vertices.to(device=field.device, dtype=field.dtype)
    faces_t = faces.to(device=field.device)
    edge_idx_t = edge_indices.to(device=field.device)
    edge_faces_t = edge_faces.to(device=field.device)
    return contour_alignment_loss(
        vertices_t,
        faces_t,
        field,
        beta_edge=beta_edge,
        include_triples=include_triples,
        edge_indices=edge_idx_t,
        edge_faces=edge_faces_t,
    )


def tv_along_isolines(
    gradients: torch.Tensor,
    face_areas: torch.Tensor,
    *,
    weight: float,
) -> torch.Tensor:
    if weight <= 0.0:
        return gradients.new_tensor(0.0)

    dtype = gradients.dtype
    normals = F.normalize(gradients, dim=-1)
    normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
    eye = torch.eye(3, device=gradients.device, dtype=dtype)
    proj = eye - normals.unsqueeze(-1) @ normals.unsqueeze(-2)
    tangential = (proj @ gradients.unsqueeze(-1)).squeeze(-1)
    magnitudes = tangential.norm(dim=-1)
    weighted = magnitudes * face_areas.unsqueeze(1)
    denom = face_areas.sum() * gradients.shape[1] + 1e-12
    return weight * weighted.sum() / denom
