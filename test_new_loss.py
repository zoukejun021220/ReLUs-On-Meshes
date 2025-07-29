#!/usr/bin/env python3
"""
Test script for the new improved loss function.
"""

import numpy as np
import torch
from improved_loss_v2 import improved_loss_function, get_beta_schedule, get_lambda_schedule

def create_test_mesh():
    """Create a simple test mesh (tetrahedron)."""
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ], dtype=np.int64)
    
    # All edges in the mesh
    edges = np.array([
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3], [2, 3]
    ], dtype=np.int64)
    
    return vertices, faces, edges

def test_loss_function():
    """Test the improved loss function."""
    print("Testing improved loss function...")
    
    # Create test mesh
    vertices, faces, edges = create_test_mesh()
    
    # Convert to torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points = torch.from_numpy(vertices).float().to(device)
    triangles = torch.from_numpy(faces).long().to(device)
    edges_tensor = torch.from_numpy(edges).long().to(device)
    
    # Initialize random field values
    num_vertices = len(vertices)
    num_channels = 6
    f_values = torch.randn(num_vertices, num_channels, device=device, requires_grad=True)
    
    # Create dummy triangle adjacency
    triangle_adjacency = torch.eye(len(faces), device=device)
    
    # Test with different beta values
    for beta in [2.0, 10.0, 25.0]:
        print(f"\n--- Testing with beta={beta} ---")
        
        loss, components = improved_loss_function(
            points=points,
            triangles=triangles,
            f_values=f_values,
            edges=edges_tensor,
            triangle_adjacency=triangle_adjacency,
            beta=beta,
            lambda_area=1.0,
            lambda_adj=5.0,
            lambda_TV=0.05
        )
        
        print(f"Total loss: {loss.item():.4f}")
        print(f"Area loss: {components['area']:.4f}")
        print(f"Adjacent loss: {components['adjacent']:.4f}")
        print(f"TV loss: {components['tv']:.4f}")
        print(f"Area fractions: {components['area_fractions']}")
        print(f"Mean boundary weight: {components['mean_boundary_weight']:.4f}")
        
        # Check gradients
        loss.backward()
        grad_norm = f_values.grad.norm().item()
        print(f"Gradient norm: {grad_norm:.4f}")
        f_values.grad.zero_()

def test_schedules():
    """Test beta and lambda schedules."""
    print("\n\nTesting schedules...")
    
    total_iters = 1000
    checkpoints = [0, 100, 200, 300, 500, 800, 1000]
    
    print("\nIteration | Beta | Lambda_adj")
    print("-" * 30)
    
    for it in checkpoints:
        beta = get_beta_schedule(it, total_iters)
        lambda_adj = get_lambda_schedule(it, total_iters)
        print(f"{it:9d} | {beta:4.1f} | {lambda_adj:10.2f}")

if __name__ == "__main__":
    print("Starting tests...")
    test_loss_function()
    test_schedules()
    print("\nAll tests completed!")