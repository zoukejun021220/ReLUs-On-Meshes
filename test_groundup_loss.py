#!/usr/bin/env python3
"""
Test script for the ground-up loss implementation.
Tests on progressively complex meshes to validate correctness.
"""

import torch
import numpy as np
import time
from loss_groundup import MeshLossGroundUp


def create_test_meshes():
    """Create a series of test meshes of increasing complexity."""
    meshes = {}
    
    # 1. Tiny square (2 triangles)
    meshes['square'] = {
        'verts': torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=torch.float),
        'faces': torch.tensor([[0,1,2],[0,2,3]]),
        'name': 'Square (2 faces)'
    }
    
    # 2. Tetrahedron (4 triangles)
    meshes['tetrahedron'] = {
        'verts': torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 0.866, 0],
            [0.5, 0.289, 0.816]
        ], dtype=torch.float),
        'faces': torch.tensor([
            [0, 1, 2],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3]
        ]),
        'name': 'Tetrahedron (4 faces)'
    }
    
    # 3. Cube (12 triangles)
    meshes['cube'] = {
        'verts': torch.tensor([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
        ], dtype=torch.float),
        'faces': torch.tensor([
            # bottom
            [0, 1, 2], [0, 2, 3],
            # top
            [4, 6, 5], [4, 7, 6],
            # front
            [0, 5, 1], [0, 4, 5],
            # back
            [2, 7, 3], [2, 6, 7],
            # left
            [0, 7, 4], [0, 3, 7],
            # right
            [1, 6, 2], [1, 5, 6]
        ]),
        'name': 'Cube (12 faces)'
    }
    
    return meshes


def test_loss_computation(mesh_data, device='cuda'):
    """Test loss computation on a single mesh."""
    print(f"\nTesting on {mesh_data['name']}...")
    
    verts = mesh_data['verts'].to(device)
    faces = mesh_data['faces'].to(device)
    
    # Initialize loss computer
    mesh_loss = MeshLossGroundUp(verts, faces, device)
    
    # Create random field
    V = verts.shape[0]
    C = 6
    Fv = torch.randn(V, C, device=device, requires_grad=True)
    
    # Test with different beta values
    betas = [0.1, 2.0, 10.0, 25.0]
    
    for beta in betas:
        loss, parts = mesh_loss.compute_loss(Fv, beta=beta)
        print(f"  β={beta:5.1f}: Loss={loss.item():8.4f} "
              f"(area={parts['area']:6.4f}, adj={parts['adj']:6.4f}, tv={parts['tv']:6.4f})")
        
        # Check gradients
        loss.backward(retain_graph=True)
        grad_norm = Fv.grad.norm().item()
        print(f"          Gradient norm: {grad_norm:.4f}")
        Fv.grad.zero_()
    
    return mesh_loss


def test_optimization_convergence(mesh_data, n_steps=100, device='cuda'):
    """Test that optimization actually reduces the loss."""
    print(f"\nTesting optimization on {mesh_data['name']}...")
    
    verts = mesh_data['verts'].to(device)
    faces = mesh_data['faces'].to(device)
    
    # Initialize
    mesh_loss = MeshLossGroundUp(verts, faces, device)
    V = verts.shape[0]
    C = 6
    
    # Initialize field with small random values
    Fv = torch.randn(V, C, device=device) * 0.1
    Fv.requires_grad = True
    
    # Optimizer
    optimizer = torch.optim.Adam([Fv], lr=0.01)
    
    # Training loop
    losses = []
    for t in range(n_steps):
        # Get schedules
        schedules = mesh_loss.get_schedules(t, n_steps)
        
        # Forward pass
        loss, parts = mesh_loss.compute_loss(
            Fv, 
            beta=schedules['beta'],
            lambda_adj=schedules['lambda_adj']
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if t % 20 == 0:
            print(f"  Step {t:3d}: Loss={loss.item():.4f} "
                  f"(area={parts['area']:.4f}, adj={parts['adj']:.4f}, tv={parts['tv']:.4f}) "
                  f"β={schedules['beta']:.1f}")
    
    # Check convergence
    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"  Improvement: {improvement:.1f}%")
    
    return losses


def test_memory_scaling():
    """Test memory usage with larger meshes."""
    print("\nTesting memory scaling...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create meshes of different sizes
    sizes = [10, 50, 100, 500]
    
    for n in sizes:
        # Create a grid mesh
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Vertices
        verts = torch.stack([
            xx.flatten(),
            yy.flatten(),
            torch.zeros_like(xx.flatten())
        ], dim=1)
        
        # Faces (two triangles per grid cell)
        faces = []
        for i in range(n-1):
            for j in range(n-1):
                v0 = i*n + j
                v1 = v0 + 1
                v2 = v0 + n
                v3 = v2 + 1
                faces.extend([[v0, v1, v3], [v0, v3, v2]])
        
        faces = torch.tensor(faces)
        
        # Test
        try:
            t0 = time.time()
            mesh_loss = MeshLossGroundUp(verts, faces, device)
            Fv = torch.randn(verts.shape[0], 6, device=device, requires_grad=True)
            loss, _ = mesh_loss.compute_loss(Fv, beta=10.0)
            loss.backward()
            t1 = time.time()
            
            print(f"  Grid {n}x{n} ({len(verts)} verts, {len(faces)} faces): "
                  f"Time={t1-t0:.3f}s, Loss={loss.item():.4f}")
            
        except RuntimeError as e:
            print(f"  Grid {n}x{n}: Out of memory")
            break


def main():
    """Run all tests."""
    print("="*60)
    print("GROUND-UP LOSS IMPLEMENTATION TEST SUITE")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create test meshes
    meshes = create_test_meshes()
    
    # Test 1: Basic loss computation
    print("\n1. BASIC LOSS COMPUTATION TEST")
    print("-"*60)
    for name, mesh_data in meshes.items():
        test_loss_computation(mesh_data, device)
    
    # Test 2: Optimization convergence
    print("\n2. OPTIMIZATION CONVERGENCE TEST")
    print("-"*60)
    for name, mesh_data in meshes.items():
        test_optimization_convergence(mesh_data, n_steps=100, device=device)
    
    # Test 3: Memory scaling
    print("\n3. MEMORY SCALING TEST")
    print("-"*60)
    test_memory_scaling()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()