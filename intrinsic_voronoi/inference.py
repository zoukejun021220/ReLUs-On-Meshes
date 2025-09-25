"""Inference utilities for trained intrinsic Voronoi fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .gating import EdgeGateConfig, compute_edge_gates
from .geometry import bisector_mask_from_phi, edge_mean_pair_norms, face_gradients, pairwise_gradients
from .utils import pair_indices


@dataclass
class InferenceResult:
    vertex_labels: torch.Tensor
    vertex_distances: torch.Tensor
    edge_boundaries: torch.Tensor


class VoronoiInference:
    def __init__(self, precomp: PrecomputedGeometry, tau_bis: float = 0.3, epsilon: float = 0.05) -> None:
        self.precomp = precomp
        self.tau_bis = tau_bis
        self.epsilon = epsilon

    def run(self, field: torch.Tensor) -> InferenceResult:
        device = field.device
        labels = torch.argmin(field, dim=1)
        distances = torch.min(field, dim=1).values / self.precomp.scale_factor

        pair_idx = pair_indices(field.shape[1], device)
        grads = face_gradients(field, self.precomp)
        pair_grads = pairwise_gradients(grads, pair_idx)
        g_left, g_right = edge_mean_pair_norms(pair_grads, self.precomp.edge_faces)
        gates = compute_edge_gates(
            field,
            self.precomp.edge_indices,
            self.precomp.edge_lengths.to(device, dtype=field.dtype),
            g_left,
            g_right,
            config=EdgeGateConfig(),
        )

        edges = self.precomp.edge_indices
        labels_a = labels.index_select(0, edges[:, 0])
        labels_b = labels.index_select(0, edges[:, 1])
        boundary = labels_a != labels_b

        phi_mask = gates.phi >= self.tau_bis

        values_a = field.index_select(0, edges[:, 0])
        values_b = field.index_select(0, edges[:, 1])
        ar = torch.arange(edges.shape[0], device=device)
        diff_a = (values_a[ar, labels_a] - values_a[ar, labels_b]).abs()
        diff_b = (values_b[ar, labels_a] - values_b[ar, labels_b]).abs()
        value_mask = (diff_a <= self.epsilon) & (diff_b <= self.epsilon)

        boundary_mask = boundary & phi_mask & value_mask

        return InferenceResult(
            vertex_labels=labels,
            vertex_distances=distances,
            edge_boundaries=boundary_mask,
        )
