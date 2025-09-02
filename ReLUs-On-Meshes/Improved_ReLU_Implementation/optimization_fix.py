"""
Fixed optimization with adaptive lambda scheduling and better loss formulation.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class AdaptiveScheduleConfig:
    """Configuration for adaptive parameter scheduling."""
    # Beta schedule
    beta_warmup_steps: int = 500  # Reduced from 1500
    beta_start: float = 0.0
    beta_end: float = 15.0  # Reduced from 25
    
    # Lambda adjacency schedule
    lambda_adj_start: float = 0.0
    lambda_adj_end: float = 1.0  # Significantly reduced from 4-8
    lambda_adj_plateau_threshold: float = 0.01  # Stop increasing if raw loss plateaus
    
    # Lambda TV schedule  
    lambda_tv: float = 0.05  # Reduced to prevent over-smoothing
    tv_clip: float = 50.0  # Reduced from 200
    
    # Lambda area
    lambda_area: float = 1.0
    
    # Gradient norm
    use_grad_norm_beta_threshold: float = 0.5  # Start much earlier
    
    # Plateau detection
    plateau_window: int = 1000
    plateau_tolerance: float = 0.01


class AdaptiveLossScheduler:
    """Manages adaptive scheduling of loss weights based on convergence."""
    
    def __init__(self, config: AdaptiveScheduleConfig):
        self.config = config
        self.raw_adj_history = deque(maxlen=config.plateau_window)
        self.lambda_adj_frozen = False
        self.frozen_lambda_adj = config.lambda_adj_start
        
    def get_beta(self, step: int, total_steps: int) -> float:
        """Get beta value with smooth warmup."""
        if step < self.config.beta_warmup_steps:
            # Smooth warmup
            progress = step / self.config.beta_warmup_steps
            return self.config.beta_start + (self.config.beta_end - self.config.beta_start) * progress
        else:
            # Stay at beta_end
            return self.config.beta_end
            
    def get_lambda_adj(self, step: int, total_steps: int, raw_adj_loss: float) -> float:
        """Get lambda_adj with plateau detection."""
        # Track raw adjacency loss
        self.raw_adj_history.append(raw_adj_loss)
        
        # Check for plateau if we have enough history
        if len(self.raw_adj_history) >= self.config.plateau_window and not self.lambda_adj_frozen:
            # Calculate variance in recent history
            recent_values = list(self.raw_adj_history)
            mean_val = np.mean(recent_values)
            std_val = np.std(recent_values)
            
            # If variance is low relative to mean, we've plateaued
            if mean_val > 0 and std_val / mean_val < self.config.plateau_tolerance:
                print(f"Adjacency loss plateaued at {mean_val:.2e}, freezing lambda_adj")
                self.lambda_adj_frozen = True
                self.frozen_lambda_adj = self._linear_schedule(
                    step, total_steps, 
                    self.config.lambda_adj_start, 
                    self.config.lambda_adj_end
                )
        
        if self.lambda_adj_frozen:
            return self.frozen_lambda_adj
        else:
            return self._linear_schedule(
                step, total_steps,
                self.config.lambda_adj_start,
                self.config.lambda_adj_end
            )
    
    def _linear_schedule(self, step: int, total_steps: int, start: float, end: float) -> float:
        """Linear interpolation."""
        progress = min(step / total_steps, 1.0)
        return start + (end - start) * progress
        
    def should_use_grad_norm(self, beta: float) -> bool:
        """Determine if GradNorm should be active."""
        return beta >= self.config.use_grad_norm_beta_threshold


def compute_boundary_length_regularization(
    f_values: torch.Tensor,
    vertices: torch.Tensor, 
    edges: torch.Tensor,
    beta: float,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    Regularize total boundary length to encourage simpler boundaries.
    
    Args:
        f_values: (V, 6) vertex field values
        vertices: (V, 3) vertex positions
        edges: (E, 2) edge connectivity
        beta: sharpness parameter
        epsilon: numerical stability
        
    Returns:
        Scalar loss encouraging shorter boundaries
    """
    # Get edge vertices
    v0_idx, v1_idx = edges[:, 0], edges[:, 1]
    f0 = f_values[v0_idx]  # (E, 6)
    f1 = f_values[v1_idx]  # (E, 6)
    
    # Edge lengths
    edge_vecs = vertices[v1_idx] - vertices[v0_idx]
    edge_lengths = torch.norm(edge_vecs, dim=1) + epsilon  # (E,)
    
    # For each channel pair, compute boundary indicator
    total_length = 0.0
    num_pairs = 0
    
    for i in range(6):
        for j in range(i+1, 6):
            # Difference in channel values across edge
            d0_ij = f0[:, i] - f0[:, j]  # (E,)
            d1_ij = f1[:, i] - f1[:, j]  # (E,)
            
            # Boundary indicator: sigmoid of sign change
            # If signs differ, this edge crosses the boundary
            boundary_weight = torch.sigmoid(-beta * d0_ij * d1_ij)
            
            # Weight by edge length
            weighted_length = (boundary_weight * edge_lengths).sum()
            total_length += weighted_length
            num_pairs += 1
    
    return total_length / (num_pairs + epsilon)


def compute_adjacency_loss_normalized(
    f_values: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    triangle_adjacency: torch.Tensor,
    beta: float,
    face_mask: Optional[torch.Tensor] = None,
    epsilon: float = 1e-8
) -> Tuple[torch.Tensor, Dict]:
    """
    Improved adjacency loss that normalizes gradients and avoids saturation.
    
    Returns:
        loss: scalar adjacency loss
        stats: dict with debugging info
    """
    # Compute face barycenters and interpolated values
    face_centers = vertices[faces].mean(dim=1)  # (F, 3)
    face_f = f_values[faces].mean(dim=1)  # (F, 6)
    
    # Filter out degenerate faces
    if face_mask is not None:
        valid_faces = face_mask
    else:
        # Compute face areas
        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0), dim=1)
        valid_faces = areas > epsilon
    
    # Compute gradients per face per channel (normalized)
    gradients = []
    for face_idx in range(faces.shape[0]):
        if not valid_faces[face_idx]:
            gradients.append(torch.zeros(6, 3, device=faces.device))
            continue
            
        v_idx = faces[face_idx]
        v_pos = vertices[v_idx]  # (3, 3)
        v_f = f_values[v_idx]  # (3, 6)
        
        # Compute gradient using barycentric coordinates
        # Set up linear system for each channel
        e1 = v_pos[1] - v_pos[0]
        e2 = v_pos[2] - v_pos[0]
        
        grad_per_channel = []
        for c in range(6):
            df1 = v_f[1, c] - v_f[0, c]
            df2 = v_f[2, c] - v_f[0, c]
            
            # Solve for gradient: [e1, e2]^T @ grad = [df1, df2]^T
            # Using least squares for stability
            A = torch.stack([e1, e2], dim=0)  # (2, 3)
            b = torch.tensor([df1, df2], device=A.device)  # (2,)
            
            # Solve using pseudo-inverse
            grad = torch.linalg.lstsq(A, b).solution
            
            # Normalize gradient
            grad_norm = torch.norm(grad) + epsilon
            grad_normalized = grad / grad_norm
            
            grad_per_channel.append(grad_normalized)
        
        gradients.append(torch.stack(grad_per_channel))  # (6, 3)
    
    gradients = torch.stack(gradients)  # (F, 6, 3)
    
    # Process adjacency pairs
    total_loss = 0.0
    num_pairs = 0
    edge_weights = []
    
    for adj_pair in triangle_adjacency:
        t0, t1 = adj_pair[0], adj_pair[1]
        
        if not (valid_faces[t0] and valid_faces[t1]):
            continue
            
        for i in range(6):
            for j in range(i+1, 6):
                # Channel differences
                d0_i = face_f[t0, i] - face_f[t0, j]
                d1_i = face_f[t1, i] - face_f[t1, j]
                
                # Edge weight (soft indicator of boundary)
                w_e = torch.sigmoid(-beta * d0_i * d1_i)
                edge_weights.append(w_e.item())
                
                # Normalized gradient difference
                g0 = gradients[t0, i] - gradients[t0, j]  # Already normalized
                g1 = gradients[t1, i] - gradients[t1, j]
                
                # Cosine similarity (avoid saturation by using smooth approximation)
                cos_sim = torch.sum(g0 * g1) / (torch.norm(g0) * torch.norm(g1) + epsilon)
                
                # Use smoother penalty: (1 - cos)^2 instead of (1 - cos)
                # This gives gradients even when cos = -1
                penalty = ((1 - cos_sim) / 2) ** 2  # Range [0, 1]
                
                total_loss += w_e * penalty
                num_pairs += 1
    
    if num_pairs > 0:
        total_loss = total_loss / num_pairs
    
    stats = {
        'num_valid_faces': valid_faces.sum().item(),
        'num_adj_pairs': num_pairs,
        'mean_edge_weight': np.mean(edge_weights) if edge_weights else 0.0,
        'std_edge_weight': np.std(edge_weights) if edge_weights else 0.0
    }
    
    return total_loss, stats


def train_with_adaptive_schedule(
    f_values: torch.Tensor,
    mesh_data: Dict,
    config: AdaptiveScheduleConfig,
    total_steps: int = 100000,
    device: str = 'cuda',
    print_every: int = 1000
) -> Tuple[torch.Tensor, Dict]:
    """
    Train with adaptive scheduling and improved loss formulation.
    """
    import torch.optim as optim
    from loss_functions import compute_area_balance_loss, compute_tv_loss
    
    # Initialize
    f_values = f_values.to(device)
    f_values.requires_grad = True
    
    vertices = mesh_data['vertices'].to(device)
    faces = mesh_data['faces'].to(device)
    edges = mesh_data['edges'].to(device)
    triangle_adjacency = mesh_data['triangle_adjacency'].to(device)
    face_mask = mesh_data.get('face_mask')
    if face_mask is not None:
        face_mask = face_mask.to(device)
    
    # Optimizer
    optimizer = optim.AdamW([f_values], lr=1e-3, weight_decay=1e-4)
    
    # Scheduler
    scheduler = AdaptiveLossScheduler(config)
    
    # Training loop
    history = []
    
    for step in range(total_steps):
        # Get current parameters
        beta = scheduler.get_beta(step, total_steps)
        
        # Compute raw losses first
        with torch.no_grad():
            # Raw adjacency for plateau detection
            raw_adj, _ = compute_adjacency_loss_normalized(
                f_values, vertices, faces, triangle_adjacency, 
                beta, face_mask
            )
            raw_adj_val = raw_adj.item()
        
        # Get adaptive lambda_adj
        lambda_adj = scheduler.get_lambda_adj(step, total_steps, raw_adj_val)
        
        # Forward pass with gradients
        adj_loss, adj_stats = compute_adjacency_loss_normalized(
            f_values, vertices, faces, triangle_adjacency,
            beta, face_mask
        )
        
        area_loss = compute_area_balance_loss(f_values, vertices, faces, beta)
        
        tv_loss = compute_tv_loss(
            f_values, edges, 
            clip_value=config.tv_clip
        )
        
        boundary_length = compute_boundary_length_regularization(
            f_values, vertices, edges, beta
        )
        
        # Total loss with adaptive weights
        total_loss = (
            config.lambda_area * area_loss +
            lambda_adj * adj_loss +
            config.lambda_tv * tv_loss +
            0.01 * boundary_length  # Small weight for boundary regularization
        )
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([f_values], max_norm=1.0)
        
        optimizer.step()
        
        # Logging
        if step % print_every == 0:
            print(f"Step {step}/{total_steps}: Total={total_loss:.4f}, "
                  f"Area={area_loss:.4f}, Adj={adj_loss:.4f} (λ={lambda_adj:.3f}), "
                  f"TV={tv_loss:.4f}, Boundary={boundary_length:.4f}, "
                  f"β={beta:.2f}, EdgeW={adj_stats['mean_edge_weight']:.3f}±{adj_stats['std_edge_weight']:.3f}")
            
            history.append({
                'step': step,
                'total_loss': total_loss.item(),
                'area_loss': area_loss.item(),
                'adj_loss': adj_loss.item(),
                'tv_loss': tv_loss.item(),
                'boundary_length': boundary_length.item(),
                'beta': beta,
                'lambda_adj': lambda_adj,
                'edge_weight_mean': adj_stats['mean_edge_weight'],
                'edge_weight_std': adj_stats['std_edge_weight']
            })
    
    return f_values.detach(), {'history': history, 'final_stats': adj_stats}


# Example usage
if __name__ == "__main__":
    print("Optimization fix module loaded.")
    print("\nKey improvements:")
    print("1. Adaptive lambda_adj that freezes when loss plateaus")
    print("2. Normalized gradients in adjacency loss")
    print("3. Smoother cosine penalty: (1-cos)^2/4 instead of (1-cos)")
    print("4. Boundary length regularization")
    print("5. Earlier GradNorm activation (β >= 0.5)")
    print("6. Reduced beta_end (15 instead of 25)")
    print("7. Reduced lambda_adj_end (1.0 instead of 4-8)")