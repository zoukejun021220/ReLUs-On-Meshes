#!/usr/bin/env python3
"""Test script to verify the 3D contour alignment fix"""
import torch
import numpy as np
from losses.improved_losses import contour_alignment_intrinsic


def create_test_mesh():
    """Create a simple test mesh with known geometry"""
    # Simple square mesh with 2 triangles
    verts = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0]
    ], dtype=torch.float32)
    
    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.long)
    
    # 2D coordinates for triangles
    tri_xy = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
        [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ], dtype=torch.float32)
    
    # Edge information
    edge_idx = torch.tensor([
        [0, 1], [1, 2], [2, 3], [3, 0], [0, 2]  # diagonal edge
    ], dtype=torch.long)
    
    edge_tris = torch.tensor([
        [0, -1], [0, -1], [1, -1], [1, -1], [0, 1]  # only diagonal is interior
    ], dtype=torch.long)
    
    return verts, faces, tri_xy, edge_idx, edge_tris


def test_straight_boundary():
    """Test that a straight boundary has low loss"""
    verts, faces, tri_xy, edge_idx, edge_tris = create_test_mesh()
    
    # Create field with straight diagonal boundary
    # Channel 0 dominates bottom-left, Channel 1 dominates top-right
    F = torch.zeros(4, 2)
    F[0, 0] = 1.0  # bottom-left
    F[1, :] = 0.5  # middle
    F[2, 1] = 1.0  # top-right
    F[3, 0] = 1.0  # top-left
    
    loss, weights = contour_alignment_intrinsic(
        F, faces, tri_xy, edge_idx, edge_tris,
        beta_contour=10.0,
        return_weights=True,
        verts=verts
    )
    
    print(f"Straight boundary loss: {loss.item():.6f}")
    print(f"Edge weights: {weights}")
    print(f"Active edges: {(weights > 0.1).sum().item()}/{len(weights)}")
    
    return loss.item()


def test_crooked_boundary():
    """Test that a crooked boundary has high loss"""
    verts, faces, tri_xy, edge_idx, edge_tris = create_test_mesh()
    
    # Create field with crooked boundary
    F = torch.zeros(4, 2)
    F[0, 0] = 1.0  # bottom-left
    F[1, 1] = 0.8  # bottom-right (different from straight)
    F[2, 1] = 1.0  # top-right
    F[3, 0] = 0.8  # top-left (different from straight)
    
    loss, weights = contour_alignment_intrinsic(
        F, faces, tri_xy, edge_idx, edge_tris,
        beta_contour=10.0,
        return_weights=True,
        verts=verts
    )
    
    print(f"\nCrooked boundary loss: {loss.item():.6f}")
    print(f"Edge weights: {weights}")
    print(f"Active edges: {(weights > 0.1).sum().item()}/{len(weights)}")
    
    return loss.item()


def test_3d_mesh():
    """Test on a non-planar 3D mesh"""
    # Create a bent mesh (two triangles at an angle)
    verts = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.5],  # raised in z
        [0.0, 1.0, 0.0]
    ], dtype=torch.float32)
    
    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.long)
    
    tri_xy = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]],
        [[0.0, 0.0], [0.5, 1.0], [0.0, 1.0]]
    ], dtype=torch.float32)
    
    edge_idx = torch.tensor([
        [0, 1], [1, 2], [2, 3], [3, 0], [0, 2]
    ], dtype=torch.long)
    
    edge_tris = torch.tensor([
        [0, -1], [0, -1], [1, -1], [1, -1], [0, 1]
    ], dtype=torch.long)
    
    # Straight boundary in 3D
    F = torch.zeros(4, 2)
    F[0, 0] = 1.0
    F[1, 1] = 1.0
    F[2, :] = 0.5  # boundary point
    F[3, 0] = 1.0
    
    loss = contour_alignment_intrinsic(
        F, faces, tri_xy, edge_idx, edge_tris,
        beta_contour=10.0,
        verts=verts
    )
    
    print(f"\n3D mesh boundary loss: {loss.item():.6f}")
    
    return loss.item()


if __name__ == "__main__":
    print("Testing 3D contour alignment fix...")
    print("=" * 50)
    
    straight_loss = test_straight_boundary()
    crooked_loss = test_crooked_boundary()
    loss_3d = test_3d_mesh()
    
    print("\n" + "=" * 50)
    print("Summary:")
    print(f"Straight boundary loss: {straight_loss:.6f} (should be low)")
    print(f"Crooked boundary loss: {crooked_loss:.6f} (should be higher)")
    print(f"3D mesh loss: {loss_3d:.6f}")
    
    # Verify the fix is working
    if crooked_loss > straight_loss:
        print("\n✓ Success: Crooked boundaries have higher loss than straight ones!")
    else:
        print("\n✗ Warning: Something might be wrong with the implementation")