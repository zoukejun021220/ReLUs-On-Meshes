#!/usr/bin/env python3
"""
ReLU Mesh Optimization using the ground-up loss implementation.
Integrates the clean loss function into the existing optimization pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, List, Tuple
import time
from collections import defaultdict

# Import the ground-up loss implementation
from loss_groundup import MeshLossGroundUp

# Import helper modules
from mesh_optimization_helpers import (
    compute_face_areas, build_triangle_adjacency, build_vertex_edges,
    auto_select_pins, init_6channels_with_pins, estimate_boundary_edges,
    compute_vertex_normals, decimate_mesh, interpolate_field_to_fine_mesh
)


class GradientMonitor:
    """Monitors gradient statistics for each loss component."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.grad_history = defaultdict(lambda: [])
        
    def update(self, grad_dict: Dict[str, float]):
        for key, value in grad_dict.items():
            self.grad_history[key].append(value)
            if len(self.grad_history[key]) > self.window_size:
                self.grad_history[key].pop(0)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for key, values in self.grad_history.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        return stats


def optimize_relu_mesh_groundup(
    vertices: np.ndarray,
    faces: np.ndarray,
    pinned_indices: List[int],
    # Basic parameters
    n_iters: int = 100000,
    lr: float = 2e-3,
    # Beta schedule
    beta_start: float = 2.0,
    beta_end: float = 25.0,
    beta_warmup_fraction: float = 0.2,
    # Lambda schedule  
    lambda_area: float = 1.0,
    lambda_adj_start: float = 0.0,
    lambda_adj_end: float = 5.0,
    lambda_TV: float = 0.05,
    # Optimization settings
    gradient_clip: float = 5.0,
    weight_decay: float = 1e-4,
    # Learning rate schedule
    lr_warmup_fraction: float = 0.2,
    lr_decay_points: List[float] = [0.2, 0.8],
    lr_decay_factors: List[float] = [1.0, 0.5, 0.1],
    # Special options
    beta_freeze_threshold: float = 0.8,  # Freeze beta after this fraction
    lambda_adj_boost: float = 2.0,  # Boost lambda_adj in final phase
    lambda_tv_complex: float = 0.01,  # Lower TV for complex meshes
    # Monitoring
    print_every: int = 100,
    save_path: str = "optimized_relu_mesh_groundup.npz"
) -> Dict:
    """
    Main optimization function using ground-up loss implementation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert to torch tensors
    v_torch = torch.from_numpy(vertices).float()
    f_torch = torch.from_numpy(faces).long()
    
    # Initialize mesh loss computer
    print("Initializing mesh topology...")
    mesh_loss = MeshLossGroundUp(v_torch, f_torch, device)
    
    # Determine if mesh is complex (>1000 faces)
    is_complex = len(faces) > 1000
    if is_complex:
        print(f"Complex mesh detected ({len(faces)} faces), using lambda_TV={lambda_tv_complex}")
        lambda_TV = lambda_tv_complex
    
    # Initialize field values
    N = len(vertices)
    f_values = torch.randn(N, 6, device=device) * 0.1
    f_values.requires_grad = True
    
    # Pin vertices
    pin_mask = torch.eye(6, device=device) * 2 - 1
    with torch.no_grad():
        for k, idx in enumerate(pinned_indices[:6]):
            f_values[idx] = pin_mask[k]
    
    # Set up optimizer
    optimizer = optim.AdamW([f_values], lr=lr, weight_decay=weight_decay)
    
    # Set up learning rate schedule
    def get_lr_multiplier(t: float) -> float:
        """Get learning rate multiplier based on schedule."""
        if t < lr_warmup_fraction:
            # Warmup phase
            return t / lr_warmup_fraction
        elif t < lr_decay_points[1]:
            # First decay phase
            return lr_decay_factors[1]
        else:
            # Final decay phase
            return lr_decay_factors[2]
    
    # Initialize monitoring
    grad_monitor = GradientMonitor()
    history = []
    best_loss = float('inf')
    best_iter = 0
    
    # Main optimization loop
    print("Starting optimization...")
    t0 = time.time()
    
    for it in range(1, n_iters + 1):
        # Progress fraction
        t = it / n_iters
        
        # Update schedules
        warmup_frac = min(1.0, it / (beta_warmup_fraction * n_iters))
        
        # Beta schedule with optional freezing
        if t < beta_freeze_threshold:
            beta = beta_start + (beta_end - beta_start) * warmup_frac
        else:
            beta = beta_end
            
        # Lambda_adj schedule with optional boost
        lambda_adj = lambda_adj_start + (lambda_adj_end - lambda_adj_start) * warmup_frac
        if t > 0.8:  # Boost in final 20%
            lambda_adj *= lambda_adj_boost
        
        # Update learning rate
        lr_mult = get_lr_multiplier(t)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * lr_mult
        
        # Forward pass
        optimizer.zero_grad()
        
        # Compute loss using ground-up implementation
        total_loss, loss_components = mesh_loss.compute_loss(
            f_values,
            beta=beta,
            lambda_adj=lambda_adj,
            lambda_tv=lambda_TV,
            lambda_area=lambda_area
        )
        
        # Backward pass
        total_loss.backward()
        
        # Monitor gradients
        grad_norms = {
            'total': f_values.grad.norm().item() if f_values.grad is not None else 0
        }
        grad_monitor.update(grad_norms)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([f_values], gradient_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Re-pin vertices
        with torch.no_grad():
            for k, idx in enumerate(pinned_indices[:6]):
                f_values[idx] = pin_mask[k]
        
        # Track best loss
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_iter = it
        
        # Logging
        if it % print_every == 0 or it == 1:
            grad_stats = grad_monitor.get_stats()
            elapsed_time = time.time() - t0
            iter_per_sec = it / elapsed_time
            eta_seconds = (n_iters - it) / iter_per_sec if iter_per_sec > 0 else 0
            
            # Compute area fractions
            with torch.no_grad():
                Pv = torch.softmax(beta * f_values, dim=1)
                Pf = Pv[mesh_loss.faces].mean(dim=1)
                area_fractions = (Pf.T * mesh_loss.face_areas).sum(dim=1) / mesh_loss.face_areas.sum()
                area_fractions = area_fractions.cpu().numpy()
            
            print(f"\n[Iter {it:6d}/{n_iters}] Time: {elapsed_time/60:.1f}min, Speed: {iter_per_sec:.1f} it/s, ETA: {eta_seconds/60:.1f}min")
            print(f"  Loss: {total_loss.item():.3e} (best: {best_loss:.3e} @ iter {best_iter})")
            print(f"  - Area: {loss_components['area']:.3e} | Adj: {loss_components['adj']:.3e} | TV: {loss_components['tv']:.3e}")
            print(f"  Params: β={beta:.1f}, λ_adj={lambda_adj:.2f}, LR={lr * lr_mult:.4f}")
            print(f"  Grads: {grad_norms['total']:.2e}")
            print(f"  Area fractions: {area_fractions}")
            
            # Store history
            history.append({
                'iter': it,
                'total_loss': total_loss.item(),
                'area_loss': loss_components['area'],
                'adj_loss': loss_components['adj'],
                'tv_loss': loss_components['tv'],
                'beta': beta,
                'lambdas': {'area': lambda_area, 'adj': lambda_adj, 'TV': lambda_TV},
                'grad_norms': grad_norms,
                'area_fractions': area_fractions.copy()
            })
        
        # Early stopping check
        if it - best_iter > 10000:
            print(f"\nEarly stopping at iteration {it} (no improvement for 10000 steps)")
            break
    
    # Save results
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETED")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"Total iterations: {it}")
    print(f"Average speed: {it/total_time:.1f} iterations/second")
    print(f"Best loss: {best_loss:.3e} (achieved at iteration {best_iter})")
    print(f"Final loss: {total_loss.item():.3e}")
    print(f"Improvement: {(history[0]['total_loss'] - best_loss) / history[0]['total_loss'] * 100:.1f}%")
    print(f"{'='*60}")
    
    results = {
        'vertices': vertices,
        'faces': faces,
        'f_values': f_values.detach().cpu().numpy(),
        'history': history,
        'best_loss': best_loss,
        'best_iter': best_iter,
        'total_time': total_time
    }
    
    if save_path:
        np.savez_compressed(save_path, **results)
        print(f"Results saved to {save_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ReLU Mesh Segmentation with Ground-up Loss")
    parser.add_argument("--mesh", type=str, required=True, help="Path to input mesh (.npz)")
    parser.add_argument("--output", type=str, default="optimized_mesh_groundup.npz", help="Output path")
    parser.add_argument("--iters", type=int, default=100000, help="Number of iterations")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--lambda-tv", type=float, default=0.05, help="TV regularization weight")
    parser.add_argument("--print-every", type=int, default=100, help="Print frequency")
    
    args = parser.parse_args()
    
    # Load mesh
    data = np.load(args.mesh)
    vertices = data['vertices']
    faces = data['faces']
    
    # Auto-select pinned vertices
    pinned_indices = auto_select_pins(vertices)
    print(f"Auto-selected {len(pinned_indices)} pinned vertices")
    
    # Run optimization
    results = optimize_relu_mesh_groundup(
        vertices, faces, pinned_indices,
        n_iters=args.iters,
        lr=args.lr,
        lambda_TV=args.lambda_tv,
        print_every=args.print_every,
        save_path=args.output
    )
    
    print("Optimization complete!")