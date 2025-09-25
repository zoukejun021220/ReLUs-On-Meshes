"""Utility helpers for the intrinsic Voronoi pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch


def charbonnier(value: torch.Tensor, delta: float) -> torch.Tensor:
    """Charbonnier penalty: sqrt(x^2 + delta^2)."""
    return torch.sqrt(value * value + delta * delta)


def pair_indices(num_channels: int, device: torch.device) -> torch.Tensor:
    """Return all unordered channel pair indices as a (P, 2) tensor."""
    if num_channels < 2:
        raise ValueError("Need at least two channels for pairwise operations")
    return torch.combinations(torch.arange(num_channels, device=device), r=2)


def safe_normalize(vec: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize vectors with epsilon regularization."""
    return vec / (vec.norm(dim=-1, keepdim=True) + eps)


def clip_tensor(t: torch.Tensor, min_value: float, max_value: float) -> torch.Tensor:
    """Clamp tensor without modifying in-place."""
    return torch.clamp(t, min_value, max_value)
