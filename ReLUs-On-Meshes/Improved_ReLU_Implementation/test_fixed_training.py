#!/usr/bin/env python3
"""
Test the fixed optimization with adaptive scheduling.
This addresses the issue where losses don't drop due to saturated adjacency loss.
"""
import torch
import numpy as np
from mesh_utils import load_mesh_from_file, compute_mesh_data
from optimization_fix import (
    AdaptiveScheduleConfig, 
    train_with_adaptive_schedule,
    compute_adjacency_loss_normalized,
    compute_boundary_length_regularization
)
from loss_functions import compute_area_balance_loss, compute_tv_loss
import json


def test_fixed_optimization():
    """Test the fixed optimization pipeline."""
    
    # Load mesh
    print("Loading mesh...")
    mesh_path = "../../Piecewise Linear Mesh 3D/l1-poly-dat/hex/kitty/orig.tet.vtk"
    vertices, faces = load_mesh_from_file(mesh_path)
    
    print(f"Mesh loaded: {vertices.shape[0]} vertices, {faces.shape[0]} faces")
    
    # Compute mesh data
    print("Computing mesh data...")
    mesh_data = compute_mesh_data(vertices, faces)
    
    # Initialize field with 6 channels
    num_vertices = vertices.shape[0]
    f_values = torch.randn(num_vertices, 6) * 0.1
    
    # Pin some vertices (example: corners of bounding box)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    
    # Find vertices near corners
    corners = [
        bbox_min,  # -x, -y, -z
        [bbox_max[0], bbox_min[1], bbox_min[2]],  # +x, -y, -z
        [bbox_min[0], bbox_max[1], bbox_min[2]],  # -x, +y, -z
        [bbox_min[0], bbox_min[1], bbox_max[2]],  # -x, -y, +z
        [bbox_max[0], bbox_max[1], bbox_min[2]],  # +x, +y, -z
        [bbox_min[0], bbox_max[1], bbox_max[2]],  # -x, +y, +z
    ]
    
    pinned_indices = []
    for i, corner in enumerate(corners[:6]):  # Use first 6 corners
        distances = np.linalg.norm(vertices - corner, axis=1)
        nearest_idx = np.argmin(distances)
        pinned_indices.append(nearest_idx)
        
        # Set pinned values
        f_values[nearest_idx] = -1.0
        f_values[nearest_idx, i] = 1.0
    
    print(f"Pinned {len(pinned_indices)} vertices")
    
    # Configure adaptive training
    config = AdaptiveScheduleConfig(
        beta_warmup_steps=500,
        beta_start=0.0,
        beta_end=10.0,  # Lower than before
        lambda_adj_start=0.0,
        lambda_adj_end=0.5,  # Much lower than before
        lambda_tv=0.05,
        tv_clip=30.0,
        lambda_area=1.0,
        use_grad_norm_beta_threshold=0.5,
        plateau_window=1000,
        plateau_tolerance=0.02
    )
    
    # Train with fixed optimization
    print("\nStarting fixed optimization...")
    print("Key improvements:")
    print("- Adaptive λ_adj that freezes when loss plateaus")
    print("- Normalized gradients in adjacency loss")
    print("- Boundary length regularization")
    print("- Lower β_end and λ_adj_end values")
    print("-" * 50)
    
    trained_f, results = train_with_adaptive_schedule(
        f_values=f_values,
        mesh_data=mesh_data,
        config=config,
        total_steps=10000,  # Shorter test
        device='cuda' if torch.cuda.is_available() else 'cpu',
        print_every=500
    )
    
    # Analyze results
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    history = results['history']
    if history:
        final_stats = history[-1]
        initial_stats = history[0]
        
        print(f"\nLoss reduction:")
        print(f"  Total: {initial_stats['total_loss']:.4f} → {final_stats['total_loss']:.4f}")
        print(f"  Area:  {initial_stats['area_loss']:.4f} → {final_stats['area_loss']:.4f}")
        print(f"  Adj:   {initial_stats['adj_loss']:.4f} → {final_stats['adj_loss']:.4f}")
        print(f"  TV:    {initial_stats['tv_loss']:.4f} → {final_stats['tv_loss']:.4f}")
        
        print(f"\nFinal parameters:")
        print(f"  β: {final_stats['beta']:.2f}")
        print(f"  λ_adj: {final_stats['lambda_adj']:.3f}")
        print(f"  Edge weight: {final_stats['edge_weight_mean']:.3f} ± {final_stats['edge_weight_std']:.3f}")
    
    # Save results
    save_path = "results/fixed_optimization_test.json"
    with open(save_path, 'w') as f:
        json.dump({
            'config': config.__dict__,
            'history': history,
            'final_stats': results['final_stats']
        }, f, indent=2)
    print(f"\nResults saved to {save_path}")
    
    # Save trained field
    np.savez("results/fixed_field_values.npz",
             f_values=trained_f.cpu().numpy(),
             vertices=vertices,
             faces=faces,
             pinned_indices=pinned_indices)
    print("Field values saved to results/fixed_field_values.npz")
    
    return trained_f, results


def analyze_convergence(history):
    """Analyze convergence behavior."""
    import matplotlib.pyplot as plt
    
    if not history:
        return
    
    steps = [h['step'] for h in history]
    total_losses = [h['total_loss'] for h in history]
    adj_losses = [h['adj_loss'] for h in history]
    lambda_adjs = [h['lambda_adj'] for h in history]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    # Total loss
    axes[0].plot(steps, total_losses)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss Convergence')
    axes[0].grid(True)
    
    # Raw adjacency loss
    axes[1].plot(steps, adj_losses, label='Raw Adj Loss')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Raw Adjacency Loss')
    axes[1].set_title('Raw Adjacency Loss (should plateau)')
    axes[1].grid(True)
    axes[1].legend()
    
    # Lambda_adj schedule
    axes[2].plot(steps, lambda_adjs)
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('λ_adj')
    axes[2].set_title('λ_adj Schedule (should freeze when loss plateaus)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/fixed_convergence_analysis.png', dpi=150)
    print("Convergence plots saved to results/fixed_convergence_analysis.png")
    plt.show()


if __name__ == "__main__":
    print("="*60)
    print("TESTING FIXED OPTIMIZATION FOR RELU MESH SEGMENTATION")
    print("="*60)
    
    trained_f, results = test_fixed_optimization()
    
    # Analyze convergence
    if 'history' in results:
        print("\nAnalyzing convergence...")
        analyze_convergence(results['history'])
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)