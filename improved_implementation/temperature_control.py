"""
Progress-gated temperature control for stable optimization.
Increases beta based on convergence metrics rather than fixed schedules.
"""

import torch
from dataclasses import dataclass
from typing import Optional

Tensor = torch.Tensor


@dataclass
class TempController:
    """Progress-gated temperature controller."""
    # Current temperatures
    beta_contour: float = 5.0
    beta_area: float = 2.0
    
    # Maximum temperatures
    beta_contour_max: float = 40.0
    beta_area_max: float = 20.0
    
    # Step sizes for increases
    step_up_contour: float = 1.0
    step_up_area: float = 0.5
    
    # Convergence thresholds
    tau_area: float = 0.08  # Max deviation from uniform area distribution
    min_boundary_len: float = 0.05  # Minimum boundary length as fraction of bbox diagonal
    
    # History for monitoring
    history: dict = None
    
    def __post_init__(self):
        self.history = {
            'beta_contour': [],
            'beta_area': [],
            'area_deviation': [],
            'boundary_length': []
        }


def approx_boundary_length(
    F: Tensor,
    edge_idx: Tensor,
    beta_contour: float,
    verts: Tensor,
    edge_len: Optional[Tensor] = None
) -> float:
    """
    Approximate total boundary length using edge crossing detection.
    
    Args:
        F: (N, C) - field values
        edge_idx: (E, 2) - edge vertex indices
        beta_contour: current temperature
        verts: (N, 3) - vertex positions
        edge_len: (E,) - precomputed edge lengths (optional)
        
    Returns:
        Total estimated boundary length
    """
    E = edge_idx.shape[0]
    C = F.shape[1]
    length = 0.0
    
    # Precompute edge lengths if not provided
    if edge_len is None:
        edge_len = (verts[edge_idx[:, 0]] - verts[edge_idx[:, 1]]).norm(dim=-1)
    
    for e in range(E):
        a, b = edge_idx[e].tolist()
        
        # Get field values at edge midpoint
        f_mid = 0.5 * (F[a] + F[b])
        
        # Find top-2 channels
        if C >= 2:
            top2 = torch.topk(f_mid, k=2).indices.tolist()
            i, j = top2[0], top2[1]
            
            # Compute crossing weight
            da = F[a, i] - F[a, j]
            db = F[b, i] - F[b, j]
            w = torch.sigmoid(-beta_contour * da * db).item()
            
            # Only count confident crossings
            if w > 0.5:
                length += edge_len[e].item()
    
    return length


def check_convergence_metrics(
    frac: Tensor,
    boundary_len: float,
    bbox_diag: float,
    controller: TempController
) -> dict:
    """
    Check if convergence criteria are met for temperature increase.
    
    Args:
        frac: (C,) - area fractions for each channel
        boundary_len: estimated boundary length
        bbox_diag: mesh bounding box diagonal
        controller: temperature controller
        
    Returns:
        Dictionary with convergence status
    """
    # Area deviation from uniform
    uniform = 1.0 / frac.shape[0]
    area_dev = (frac - uniform).abs().max().item()
    
    # Normalized boundary length
    norm_boundary = boundary_len / bbox_diag
    
    # Check criteria
    area_converged = area_dev < controller.tau_area
    boundary_formed = norm_boundary > controller.min_boundary_len
    
    return {
        'area_deviation': area_dev,
        'norm_boundary_length': norm_boundary,
        'area_converged': area_converged,
        'boundary_formed': boundary_formed,
        'can_increase_beta': area_converged and boundary_formed
    }


def maybe_raise_betas(
    controller: TempController,
    frac: Tensor,
    boundary_len: float,
    bbox_diag: float
) -> bool:
    """
    Conditionally increase temperatures based on convergence metrics.
    
    Args:
        controller: temperature controller to update
        frac: (C,) - area fractions
        boundary_len: estimated boundary length
        bbox_diag: mesh bounding box diagonal
        
    Returns:
        True if temperatures were increased
    """
    metrics = check_convergence_metrics(frac, boundary_len, bbox_diag, controller)
    
    # Update history
    controller.history['area_deviation'].append(metrics['area_deviation'])
    controller.history['boundary_length'].append(metrics['norm_boundary_length'])
    controller.history['beta_contour'].append(controller.beta_contour)
    controller.history['beta_area'].append(controller.beta_area)
    
    if metrics['can_increase_beta']:
        # Increase contour beta
        old_beta_contour = controller.beta_contour
        controller.beta_contour = min(
            controller.beta_contour + controller.step_up_contour,
            controller.beta_contour_max
        )
        
        # Increase area beta
        old_beta_area = controller.beta_area
        controller.beta_area = min(
            controller.beta_area + controller.step_up_area,
            controller.beta_area_max
        )
        
        return (old_beta_contour != controller.beta_contour or 
                old_beta_area != controller.beta_area)
    
    return False


def get_adaptive_weights(
    step: int,
    total_steps: int,
    stage_transition: float = 0.6
) -> dict:
    """
    Get adaptive loss weights based on training progress.
    
    Args:
        step: current step
        total_steps: total number of steps
        stage_transition: fraction of steps for Stage A
        
    Returns:
        Dictionary of loss weights
    """
    progress = step / total_steps
    
    if progress < stage_transition:
        # Stage A: Focus on smoothness and basic segmentation
        return {
            'smooth': 1.0,
            'contour': 0.05,  # Light contour alignment
            'area': 0.05,     # Light area balance
            'pin': 0.01       # Soft pinning
        }
    else:
        # Stage B: Refine boundaries
        # Gradually increase contour weight
        contour_ramp = min((progress - stage_transition) / (0.2), 1.0)
        return {
            'smooth': 0.5,
            'contour': 0.1 + 0.9 * contour_ramp,  # Ramp up to 1.0
            'area': 0.2,
            'pin': 0.02
        }


def get_learning_rate(
    step: int,
    total_steps: int,
    initial_lr: float = 1e-3,
    min_lr: float = 1e-5,
    stage_transition: float = 0.6
) -> float:
    """
    Get adaptive learning rate based on training progress.
    
    Args:
        step: current step
        total_steps: total number of steps
        initial_lr: starting learning rate
        min_lr: minimum learning rate
        stage_transition: fraction of steps for Stage A
        
    Returns:
        Current learning rate
    """
    progress = step / total_steps
    
    if progress < stage_transition:
        # Stage A: Constant learning rate
        return initial_lr
    else:
        # Stage B: Exponential decay
        decay_progress = (progress - stage_transition) / (1.0 - stage_transition)
        decay_factor = 0.3  # Final LR = initial_lr * decay_factor
        return initial_lr * (decay_factor ** decay_progress)