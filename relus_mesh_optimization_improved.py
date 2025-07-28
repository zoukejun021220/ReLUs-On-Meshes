#!/usr/bin/env python3
"""
Improved ReLUs on Meshes Optimization Script
Integrates all optimization improvements for robust convergence on complex shapes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, List, Tuple
import time
import math
from collections import defaultdict

# Import helper modules
from mesh_optimization_helpers import (
    compute_face_areas, build_triangle_adjacency, build_vertex_edges,
    auto_select_pins, init_6channels_with_pins, estimate_boundary_edges,
    compute_vertex_normals, decimate_mesh, interpolate_field_to_fine_mesh
)
from contour_alignment_improved import (
    contour_alignment_v1_fixed_normals,
    contour_alignment_v2_gradient_based, 
    contour_alignment_v3_fully_vectorized,
    compute_contour_loss
)

# ============================================================================================
# IMPROVED LOSS FUNCTIONS WITH NUMERICAL STABILITY
# ============================================================================================

def area_balance_loss_improved(
    points: torch.Tensor, 
    triangles: torch.Tensor, 
    f_values: torch.Tensor, 
    beta: float, 
    mesh_area: float,
    start_beta: float = 4.0  # Higher starting beta to avoid vanishing gradients
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Improved area balance loss with better numerical stability.
    - Uses cross-entropy formulation for non-zero gradients at initialization
    - Starts with higher beta to avoid gradient vanishing
    """
    device = points.device
    num_channels = f_values.shape[1]
    
    # Use higher initial beta to avoid gradient vanishing
    effective_beta = max(beta, start_beta)
    
    # Define barycentric sampling points
    bary_points = torch.tensor([
        [1/3, 1/3, 1/3],  # Center
        [0.5, 0.5, 0.0],  # Edge midpoints
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ], device=device)
    
    # Get triangle vertices
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    p0, p1, p2 = points[v0], points[v1], points[v2]
    
    # Compute triangle areas
    e1, e2 = p1 - p0, p2 - p0
    normals = torch.cross(e1, e2, dim=1)
    areas = 0.5 * torch.norm(normals, dim=1)
    
    # Get field values at vertices
    f0, f1, f2 = f_values[v0], f_values[v1], f_values[v2]
    
    # Interpolate field at barycentric points
    f_interp = []
    for bary in bary_points:
        f_bary = bary[0] * f0 + bary[1] * f1 + bary[2] * f2
        f_interp.append(f_bary)
    f_interp = torch.stack(f_interp, dim=1)  # (T, S, C)
    
    # Apply softmax with effective beta
    f_interp_beta = effective_beta * f_interp
    probs = torch.softmax(f_interp_beta, dim=2)  # (T, S, C)
    
    # Average over sample points
    probs_mean = probs.mean(dim=1)  # (T, C)
    
    # Weight by triangle areas
    weighted_areas = probs_mean * areas.unsqueeze(1)
    channel_areas = weighted_areas.sum(dim=0)  # (C,)
    
    # Cross-entropy formulation for better gradients
    fractions = channel_areas / mesh_area
    target = torch.ones_like(fractions) / num_channels
    
    # Use cross-entropy instead of L1/L2 for better gradient flow
    loss = -torch.sum(target * torch.log(fractions + 1e-8))
    
    return loss, fractions


def smoothness_loss_improved(
    f_values: torch.Tensor, 
    vertex_edges: torch.Tensor,
    boundary_edges: Optional[torch.Tensor] = None,
    lambda_boundary: float = 0.1
) -> torch.Tensor:
    """
    Improved smoothness loss that optionally excludes boundary edges.
    """
    v1_idx, v2_idx = vertex_edges[:, 0], vertex_edges[:, 1]
    f1, f2 = f_values[v1_idx], f_values[v2_idx]
    diff = f1 - f2
    
    if boundary_edges is not None:
        # Create mask for non-boundary edges
        edge_set = set(map(tuple, vertex_edges.cpu().numpy()))
        boundary_set = set(map(tuple, boundary_edges.cpu().numpy()))
        
        # Apply reduced weight to boundary edges
        weights = torch.ones(len(vertex_edges), device=f_values.device)
        for i, edge in enumerate(vertex_edges.cpu().numpy()):
            if tuple(edge) in boundary_set or tuple(edge[::-1]) in boundary_set:
                weights[i] = lambda_boundary
        
        loss = torch.sum(weights.unsqueeze(1) * diff**2)
    else:
        loss = torch.sum(diff**2)
    
    return loss




# ============================================================================================
# GRADIENT MONITORING AND DYNAMIC LOSS REWEIGHTING
# ============================================================================================

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


class DynamicLossReweighter:
    """Dynamically adjusts loss weights to balance gradient contributions."""
    
    def __init__(self, target_ratio: float = 1.0, alpha: float = 0.1):
        self.target_ratio = target_ratio
        self.alpha = alpha  # Learning rate for weight updates
        self.weights = {'contour': 1.0, 'smooth': 1.0, 'area': 1.0}
        
    def update_weights(self, grad_norms: Dict[str, float]):
        # Compute average gradient norm
        avg_norm = np.mean(list(grad_norms.values()))
        
        # Update weights to balance gradient contributions
        for key in self.weights:
            if key in grad_norms and grad_norms[key] > 0:
                ratio = avg_norm / grad_norms[key]
                self.weights[key] = (1 - self.alpha) * self.weights[key] + self.alpha * ratio
        
        # Normalize weights
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total
        
        return self.weights


# ============================================================================================
# IMPROVED INITIALIZATION
# ============================================================================================

def initialize_field_and_planes(
    vertices: np.ndarray,
    pinned_indices: List[int],
    use_pca_alignment: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize field values and plane parameters with improved strategy.
    """
    N = len(vertices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize 6-channel field
    f_values = torch.randn(N, 6, device=device) * 0.1
    
    # Set up plane normals (axis-aligned for V1)
    plane_normals = torch.tensor([
        [1, 0, 0], [-1, 0, 0],  # +X, -X
        [0, 1, 0], [0, -1, 0],  # +Y, -Y
        [0, 0, 1], [0, 0, -1]   # +Z, -Z
    ], dtype=torch.float32, device=device)
    
    # Initialize plane offsets using median slices
    v_tensor = torch.from_numpy(vertices).float().to(device)
    plane_offsets = torch.zeros(6, device=device)
    
    for c in range(6):
        # Project vertices onto plane normal
        proj = torch.matmul(v_tensor, plane_normals[c])
        # Set offset to median projection (ensures 50/50 split)
        plane_offsets[c] = -torch.median(proj)
    
    # Optionally add PCA-aligned planes
    if use_pca_alignment:
        # Compute PCA of vertex positions
        centered = vertices - vertices.mean(axis=0)
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Add 3 more planes aligned with PCA directions
        pca_normals = torch.from_numpy(eigvecs.T).float().to(device)
        plane_normals = torch.cat([plane_normals, pca_normals, -pca_normals])
        
        # Initialize offsets for PCA planes
        pca_offsets = torch.zeros(6, device=device)
        for i in range(3):
            proj = torch.matmul(v_tensor, pca_normals[i])
            pca_offsets[i] = -torch.median(proj)
            pca_offsets[i+3] = -pca_offsets[i]
        
        plane_offsets = torch.cat([plane_offsets, pca_offsets])
    
    # Pin vertices
    pin_mask = torch.eye(6, device=device) * 2 - 1  # [-1, -1, 1, -1, ...]
    for k, idx in enumerate(pinned_indices[:6]):
        f_values[idx] = pin_mask[k]
    
    return f_values, plane_normals, plane_offsets


# ============================================================================================
# MAIN OPTIMIZATION FUNCTION
# ============================================================================================

def optimize_relu_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    pinned_indices: List[int],
    # Basic parameters
    n_iters: int = 100000,
    lr_vertex: float = 2e-3,
    lr_offset: float = 2e-2,  # 10x larger for plane offsets
    # Beta schedule
    beta_start: float = 4.0,  # Start higher to avoid vanishing gradients
    beta_end: float = 20.0,   # Don't go too high
    beta_schedule: str = "sigmoid",  # "linear", "sigmoid", or "log"
    # Lambda schedule  
    lambda_contour: Tuple[float, float] = (1.0, 4.0),  # Start high
    lambda_smooth: float = 0.1,
    lambda_area: Tuple[float, float] = (0.0, 100.0),
    reverse_schedule: bool = True,  # Start with contour loss
    # Optimization settings
    use_dynamic_reweighting: bool = True,
    gradient_clip: float = 5.0,
    weight_decay: float = 1e-4,
    # Multi-scale settings
    use_multiscale: bool = False,
    scale_factors: List[float] = [0.25, 0.5, 1.0],
    # Monitoring
    print_every: int = 100,
    save_path: str = "optimized_relu_mesh.npz"
) -> Dict:
    """
    Main optimization function with all improvements integrated.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert to torch tensors
    v = torch.from_numpy(vertices).float().to(device)
    f = torch.from_numpy(faces).long().to(device)
    
    # Build mesh connectivity
    tri_adj = torch.from_numpy(build_triangle_adjacency(faces)).long().to(device)
    vert_edges = torch.from_numpy(build_vertex_edges(faces)).long().to(device)
    mesh_area = compute_face_areas(vertices, faces).sum()
    
    # Initialize field and planes with improved strategy
    f_values, plane_normals, plane_offsets = initialize_field_and_planes(
        vertices, pinned_indices, use_pca_alignment=True
    )
    
    # Make plane offsets learnable
    plane_offsets = nn.Parameter(plane_offsets)
    
    # Set up optimizer with separate learning rates
    optimizer = optim.AdamW([
        {'params': [f_values], 'lr': lr_vertex},
        {'params': [plane_offsets], 'lr': lr_offset}
    ], weight_decay=weight_decay)
    
    # Set up schedules
    def get_schedule_value(schedule_type: str, t: float, start: float, end: float) -> float:
        """Get scheduled value based on schedule type."""
        if schedule_type == "linear":
            return start + (end - start) * t
        elif schedule_type == "sigmoid":
            # Sigmoid ramp: slow start, fast middle, slow end
            s = 1 / (1 + np.exp(-10 * (t - 0.5)))
            return start + (end - start) * s
        elif schedule_type == "log":
            # Logarithmic ramp: fast start, slow end
            s = np.log(1 + 9 * t) / np.log(10)
            return start + (end - start) * s
        else:
            return start
    
    # Initialize monitoring tools
    grad_monitor = GradientMonitor()
    loss_reweighter = DynamicLossReweighter() if use_dynamic_reweighting else None
    
    # Training history
    history = []
    best_loss = float('inf')
    best_iter = 0
    
    # Main optimization loop
    print("Starting optimization...")
    t0 = time.time()
    
    for it in range(1, n_iters + 1):
        # Progress fraction
        t = it / n_iters
        
        # Update beta
        beta = get_schedule_value(beta_schedule, t, beta_start, beta_end)
        
        # Update lambdas
        if reverse_schedule and t < 0.3:
            # First 30%: focus on contour alignment
            lambda_c = lambda_contour[1]
            lambda_a = 0.0
            lambda_s = lambda_smooth * 0.1
        else:
            # After 30%: gradually introduce other losses
            t_adj = (t - 0.3) / 0.7 if reverse_schedule else t
            lambda_c = get_schedule_value("linear", t_adj, lambda_contour[0], lambda_contour[1])
            lambda_a = get_schedule_value("linear", t_adj, lambda_area[0], lambda_area[1])
            lambda_s = lambda_smooth
        
        # Get current loss weights from dynamic reweighter
        if loss_reweighter:
            weights = loss_reweighter.weights
        else:
            weights = {'contour': 1.0, 'smooth': 1.0, 'area': 1.0}
        
        # Forward pass
        optimizer.zero_grad()
        
        # Compute losses
        contour_loss = contour_alignment_v1_fixed_normals(
            v, f, f_values, plane_normals, plane_offsets, 
            beta_edge=beta, min_intersections=20
        )
        
        smooth_loss = smoothness_loss_improved(f_values, vert_edges)
        
        area_loss, area_fracs = area_balance_loss_improved(
            v, f, f_values, beta, mesh_area, start_beta=beta_start
        )
        
        # Combined loss with dynamic weights
        total_loss = (
            lambda_c * weights['contour'] * contour_loss +
            lambda_s * weights['smooth'] * smooth_loss +
            lambda_a * weights['area'] * area_loss
        )
        
        # Backward pass
        total_loss.backward()
        
        # Monitor gradients before clipping
        grad_norms = {
            'contour': f_values.grad.norm().item() if lambda_c > 0 else 0,
            'smooth': f_values.grad.norm().item() if lambda_s > 0 else 0,
            'area': f_values.grad.norm().item() if lambda_a > 0 else 0,
            'offsets': plane_offsets.grad.norm().item()
        }
        grad_monitor.update(grad_norms)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([f_values, plane_offsets], gradient_clip)
        
        # Optimizer step
        optimizer.step()
        
        # Re-pin vertices
        with torch.no_grad():
            pin_mask = torch.eye(6, device=device) * 2 - 1
            for k, idx in enumerate(pinned_indices[:6]):
                f_values[idx] = pin_mask[k]
        
        # Update dynamic loss weights
        if loss_reweighter and it % 50 == 0:
            loss_reweighter.update_weights(grad_norms)
        
        # Track best loss
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_iter = it
        
        # Logging
        if it % print_every == 0 or it == 1:
            grad_stats = grad_monitor.get_stats()
            
            print(f"\nIter {it:6d}/{n_iters}")
            print(f"  Total loss: {total_loss.item():.3e} (best: {best_loss:.3e} @ iter {best_iter})")
            print(f"  Components: contour={contour_loss.item():.3e}, smooth={smooth_loss.item():.3e}, area={area_loss.item():.3e}")
            print(f"  Schedules: β={beta:.1f}, λ_c={lambda_c:.2f}, λ_s={lambda_s:.2f}, λ_a={lambda_a:.1f}")
            print(f"  Grad norms: {grad_norms}")
            if loss_reweighter:
                print(f"  Dynamic weights: {weights}")
            print(f"  Area fractions: {area_fracs.cpu().numpy()}")
            
            # Store history
            history.append({
                'iter': it,
                'total_loss': total_loss.item(),
                'contour_loss': contour_loss.item(),
                'smooth_loss': smooth_loss.item(),
                'area_loss': area_loss.item(),
                'beta': beta,
                'lambdas': {'contour': lambda_c, 'smooth': lambda_s, 'area': lambda_a},
                'grad_norms': grad_norms,
                'weights': weights.copy() if loss_reweighter else None,
                'area_fractions': area_fracs.cpu().numpy().copy()
            })
        
        # Early stopping check
        if it - best_iter > 5000:
            print(f"\nEarly stopping at iteration {it} (no improvement for 5000 steps)")
            break
    
    # Save results
    print(f"\nOptimization completed in {(time.time() - t0) / 60:.1f} minutes")
    
    results = {
        'vertices': vertices,
        'faces': faces,
        'f_values': f_values.detach().cpu().numpy(),
        'plane_normals': plane_normals.cpu().numpy(),
        'plane_offsets': plane_offsets.detach().cpu().numpy(),
        'history': history,
        'best_loss': best_loss,
        'best_iter': best_iter
    }
    
    np.savez_compressed(save_path, **results)
    print(f"Results saved to {save_path}")
    
    return results


# ============================================================================================
# MULTI-SCALE OPTIMIZATION
# ============================================================================================

def optimize_multiscale(
    vertices: np.ndarray,
    faces: np.ndarray,
    pinned_indices: List[int],
    scale_factors: List[float] = [0.25, 0.5, 1.0],
    iters_per_scale: List[int] = [10000, 20000, 70000],
    **kwargs
) -> Dict:
    """
    Multi-scale optimization: start with coarse mesh, refine progressively.
    """
    print("Starting multi-scale optimization...")
    
    # TODO: Implement mesh decimation and upsampling
    # For now, just run single scale
    return optimize_relu_mesh(vertices, faces, pinned_indices, **kwargs)


# ============================================================================================
# MAIN ENTRY POINT
# ============================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized ReLU Mesh Segmentation")
    parser.add_argument("--mesh", type=str, required=True, help="Path to input mesh (.npz)")
    parser.add_argument("--output", type=str, default="optimized_mesh.npz", help="Output path")
    parser.add_argument("--iters", type=int, default=100000, help="Number of iterations")
    parser.add_argument("--multiscale", action="store_true", help="Use multi-scale optimization")
    parser.add_argument("--dynamic-reweight", action="store_true", help="Use dynamic loss reweighting")
    
    args = parser.parse_args()
    
    # Load mesh
    data = np.load(args.mesh)
    vertices = data['vertices']
    faces = data['faces']
    
    # Auto-select pinned vertices (bounding box extremes)
    from MeshParamCalculation import auto_select_pins
    pinned_indices = auto_select_pins(vertices)
    
    # Run optimization
    if args.multiscale:
        results = optimize_multiscale(
            vertices, faces, pinned_indices,
            n_iters=args.iters,
            use_dynamic_reweighting=args.dynamic_reweight,
            save_path=args.output
        )
    else:
        results = optimize_relu_mesh(
            vertices, faces, pinned_indices,
            n_iters=args.iters,
            use_dynamic_reweighting=args.dynamic_reweight,
            save_path=args.output
        )
    
    print("Optimization complete!")