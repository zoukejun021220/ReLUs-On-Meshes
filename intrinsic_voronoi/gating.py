"""Edge gating computations for Voronoi constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .utils import clip_tensor, pair_indices


@dataclass
class EdgeGateConfig:
    beta_edge: float = 6.0
    gamma: float = 5.0
    tau0: float = 0.2
    tau_bis: float = 0.3
    epsilon_pi: float = 1e-12
    max_pairs: Optional[int] = None


@dataclass
class EdgeGates:
    tilde_w: torch.Tensor
    mixture: torch.Tensor
    phi: torch.Tensor
    weight: torch.Tensor


def compute_edge_gates(
    field: torch.Tensor,
    edge_indices: torch.Tensor,
    edge_lengths: torch.Tensor,
    grad_diff_norms_left: torch.Tensor,
    grad_diff_norms_right: torch.Tensor,
    *,
    config: EdgeGateConfig,
) -> EdgeGates:
    """Compute detached edge gates per unordered pair of channels.

    Args:
        field: Tensor of shape (num_vertices, num_channels).
        edge_indices: Tensor of shape (num_edges, 2) with vertex indices.
        edge_lengths: Tensor of shape (num_edges,) with scalar lengths.
        grad_diff_norms_left/right: Mean of pairwise gradient norms per incident
            face (shape: num_edges,). Use zero for missing faces.
        config: Parameters controlling the gating.

    Returns:
        EdgeGates with detached tensors for tilde weights, mixture weights,
        activity probability, and final edge weights.
    """

    if field.ndim != 2:
        raise ValueError("field must be (num_vertices, num_channels)")

    num_channels = field.shape[1]
    device = field.device
    dtype = field.dtype

    with torch.no_grad():
        pairs = pair_indices(num_channels, device)
        num_pairs = pairs.shape[0]

        va = edge_indices[:, 0]
        vb = edge_indices[:, 1]
        f_a = field[va]
        f_b = field[vb]

        fa_i = f_a.index_select(1, pairs[:, 0])
        fa_j = f_a.index_select(1, pairs[:, 1])
        fb_i = f_b.index_select(1, pairs[:, 0])
        fb_j = f_b.index_select(1, pairs[:, 1])

        diff_a = fa_i - fa_j
        diff_b = fb_i - fb_j

        prod = diff_a * diff_b
        sigma = torch.sigmoid
        gate_sign = sigma(-config.beta_edge * prod)
        mag = 0.5 * (diff_a.abs() + diff_b.abs())
        gate_conf = sigma(config.gamma * (mag - config.tau0))
        tilde_w = gate_sign * gate_conf

        sum_w = tilde_w.sum(dim=1, keepdim=True)
        mixture = tilde_w / (sum_w + config.epsilon_pi)
        phi = 1.0 - torch.prod(1.0 - tilde_w, dim=1)
        labels = torch.argmin(field, dim=1)
        label_flip = (labels[va] != labels[vb]).to(phi.dtype)
        phi = 1.0 - (1.0 - phi) * (1.0 - label_flip)
        phi = phi.clamp(min=0.0, max=1.0)

        grad_left = grad_diff_norms_left.to(dtype)
        grad_right = grad_diff_norms_right.to(dtype)
        geom = torch.sqrt(torch.clamp(grad_left, min=0.0) * torch.clamp(grad_right, min=0.0))
        median = torch.median(geom[geom > 0.0]) if torch.any(geom > 0.0) else torch.tensor(1.0, device=device, dtype=dtype)
        gamma_hat = geom / (median + 1e-12)
        gamma_hat = clip_tensor(gamma_hat, 0.0, 2.0)

        mean_len = edge_lengths.to(dtype).mean()
        lambda_len = clip_tensor(edge_lengths.to(dtype) / (mean_len + 1e-12), 0.5, 2.0)
        phi_safe = torch.maximum(phi, torch.full_like(phi, 1e-4))
        weight = phi_safe * gamma_hat * lambda_len

    return EdgeGates(
        tilde_w=tilde_w.detach(),
        mixture=mixture.detach(),
        phi=phi.detach(),
        weight=weight.detach(),
    )
