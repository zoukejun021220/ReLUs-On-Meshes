"""
Quick test script to verify the improved pipeline works correctly.
"""
import torch
import numpy as np
from pathlib import Path

# Import our modules
from utils.mesh_preprocessing import preprocess_mesh
from losses.improved_losses import (
    contour_alignment_intrinsic,
    smoothness_cotan,
    area_fractions_and_kl,
    pin_loss
)
from optimization.temperature_control import TempController


def create_test_mesh():
    """Create a simple test mesh (tetrahedron)."""
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3)/2, 0],
        [0.5, np.sqrt(3)/6, np.sqrt(6)/3]
    ], dtype=np.float32)
    
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3]
    ], dtype=np.int32)
    
    return vertices, faces


def test_preprocessing():
    """Test mesh preprocessing."""
    print("Testing mesh preprocessing...")
    
    vertices, faces = create_test_mesh()
    mesh_data = preprocess_mesh(vertices, faces, device='cpu')
    
    assert 'vertices' in mesh_data
    assert 'faces' in mesh_data
    assert 'tri_area' in mesh_data
    assert 'cotan_W' in mesh_data
    
    print(f"  ✓ Preprocessed mesh: {mesh_data['vertices'].shape[0]} vertices")
    print(f"  ✓ Cotangent weights: {mesh_data['cotan_W'].shape[0]} entries")
    print(f"  ✓ Triangle areas sum: {mesh_data['tri_area'].sum():.4f}")


def test_losses():
    """Test loss functions."""
    print("\nTesting loss functions...")
    
    vertices, faces = create_test_mesh()
    mesh_data = preprocess_mesh(vertices, faces, device='cpu')
    
    # Create random field
    n_verts = vertices.shape[0]
    n_channels = 6
    F = torch.randn(n_verts, n_channels) * 0.1
    
    # Test smoothness loss
    loss_smooth = smoothness_cotan(
        F, 
        mesh_data['cotan_I'],
        mesh_data['cotan_J'],
        mesh_data['cotan_W']
    )
    assert loss_smooth.item() >= 0
    print(f"  ✓ Smoothness loss: {loss_smooth.item():.6f}")
    
    # Test area balance loss
    loss_area, frac = area_fractions_and_kl(
        F,
        mesh_data['faces'],
        mesh_data['tri_area'],
        beta_area=2.0
    )
    assert loss_area.item() >= 0
    assert frac.sum().item() > 0.99
    print(f"  ✓ Area balance loss: {loss_area.item():.6f}")
    print(f"    Area fractions: {frac.numpy()}")
    
    # Test contour alignment loss
    loss_contour = contour_alignment_intrinsic(
        F,
        mesh_data['faces'],
        mesh_data['edge_idx'],
        mesh_data['edge_tris'],
        beta_contour=5.0,
        verts=mesh_data['vertices']
    )
    assert loss_contour.item() >= 0
    print(f"  ✓ Contour alignment loss: {loss_contour.item():.6f}")
    
    # Test pin loss
    pin_idx = torch.tensor([0, 1], dtype=torch.long)
    pin_target = torch.zeros(2, n_channels)
    pin_target[0, 0] = 1.0
    pin_target[1, 1] = 1.0
    
    loss_pin = pin_loss(F, pin_idx, pin_target)
    assert loss_pin.item() >= 0
    print(f"  ✓ Pin constraint loss: {loss_pin.item():.6f}")


def test_temperature_control():
    """Test temperature controller."""
    print("\nTesting temperature control...")
    
    ctrl = TempController()
    
    # Test with good progress
    area_frac = torch.tensor([0.165, 0.168, 0.167, 0.166, 0.167, 0.167])
    boundary_len = 2.0
    bbox_diag = 10.0
    
    initial_beta = ctrl.beta_contour
    updated = ctrl.check_and_update(area_frac, boundary_len, bbox_diag, step=0)
    
    print(f"  ✓ Initial β_contour: {initial_beta}")
    print(f"  ✓ Updated: {updated}")
    print(f"  ✓ New β_contour: {ctrl.beta_contour}")
    
    # Test with poor progress
    area_frac_bad = torch.tensor([0.3, 0.3, 0.1, 0.1, 0.1, 0.1])
    updated_bad = ctrl.check_and_update(area_frac_bad, 0.1, bbox_diag, step=1)
    
    print(f"  ✓ Bad progress - Updated: {updated_bad}")


def test_gradient_flow():
    """Test gradient flow through losses."""
    print("\nTesting gradient flow...")
    
    vertices, faces = create_test_mesh()
    mesh_data = preprocess_mesh(vertices, faces, device='cpu')
    
    # Create field with gradients
    F = torch.randn(vertices.shape[0], 6, requires_grad=True)
    
    # Compute total loss
    loss_smooth = smoothness_cotan(F, mesh_data['cotan_I'], 
                                  mesh_data['cotan_J'], mesh_data['cotan_W'])
    loss_area, _ = area_fractions_and_kl(F, mesh_data['faces'], 
                                       mesh_data['tri_area'], beta_area=2.0)
    
    total_loss = loss_smooth + loss_area
    total_loss.backward()
    
    assert F.grad is not None
    grad_norm = F.grad.norm().item()
    print(f"  ✓ Gradient norm: {grad_norm:.6f}")
    print(f"  ✓ Gradient shape: {F.grad.shape}")


def main():
    """Run all tests."""
    print("=== Testing Improved ReLU Mesh Segmentation Pipeline ===\n")
    
    test_preprocessing()
    test_losses()
    test_temperature_control()
    test_gradient_flow()
    
    print("\n✅ All tests passed!")
    print("\nThe improved pipeline is ready to use.")
    print("Run: python train_improved.py --mesh /path/to/mesh.vtk")


if __name__ == '__main__':
    main()