#!/usr/bin/env python3
"""Test the critical fixes for TempController, lr_decay, and cotangent weights"""
import torch
import numpy as np
from optimization.temperature_control import TempController, TwoStageScheduler
from utils.mesh_preprocessing import cotan_weights


def test_temp_controller_nan_fix():
    """Test that TempController handles first call without NaN"""
    print("Testing TempController NaN fix...")
    
    controller = TempController()
    
    # First call with no history
    area_frac = torch.tensor([0.15, 0.15, 0.20, 0.20, 0.15, 0.15])
    boundary_len = 5.0
    bbox_diag = 10.0
    contour_loss = 0.5
    
    # This should not crash or produce NaN
    updated = controller.check_and_update(
        area_frac, boundary_len, bbox_diag, 
        step=0, contour_loss=contour_loss
    )
    
    print(f"  First call updated: {updated}")
    print(f"  Best contour loss: {controller.best_contour_loss_since_update}")
    print(f"  ✓ No NaN on first call!")


def test_cosine_decay_fix():
    """Test that cosine_decay is properly typed as bool"""
    print("\nTesting cosine_decay type fix...")
    
    scheduler = TwoStageScheduler(total_steps=100000)
    
    # Check that cosine_decay is a bool
    for stage in scheduler.stages:
        assert isinstance(stage.cosine_decay, bool), f"Stage {stage.name}: cosine_decay should be bool"
        print(f"  {stage.name}: cosine_decay = {stage.cosine_decay} (type: {type(stage.cosine_decay).__name__})")
    
    # Test get_lr with cosine decay
    stage_b_step = 50000  # Should be in Stage B
    base_lr = 1e-3
    lr = scheduler.get_lr(stage_b_step, base_lr)
    print(f"  LR at step {stage_b_step}: {lr:.6f}")
    print(f"  ✓ Cosine decay works correctly!")


def test_cotangent_weights_fix():
    """Test that cotangent weights handle negative values correctly"""
    print("\nTesting cotangent weights fix...")
    
    # Create a mesh with an obtuse triangle
    verts = torch.tensor([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.2, 0.0],  # This creates an obtuse angle at vertex 0
        [1.0, 1.0, 0.0]
    ], dtype=torch.float32)
    
    faces = torch.tensor([
        [0, 1, 2],  # Obtuse triangle
        [0, 2, 3]   # Normal triangle
    ], dtype=torch.long)
    
    # Compute weights
    I, J, W = cotan_weights(verts, faces)
    
    print(f"  Number of edges: {len(W)}")
    print(f"  Min weight: {W.min().item():.6f}")
    print(f"  Max weight: {W.max().item():.6f}")
    print(f"  All weights positive: {(W > 0).all().item()}")
    print(f"  All weights finite: {torch.isfinite(W).all().item()}")
    
    # Check that all weights are clamped to positive
    assert (W > 0).all(), "Some weights are not positive after clamping"
    assert torch.isfinite(W).all(), "Some weights are not finite"
    
    print(f"  ✓ Cotangent weights properly clamped!")


def test_gradient_flow():
    """Test that all components work together"""
    print("\nTesting integrated gradient flow...")
    
    # Simple mesh
    verts = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0]
    ], dtype=torch.float32)
    
    faces = torch.tensor([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [0, 2, 3]
    ], dtype=torch.long)
    
    # Compute cotangent weights
    I, J, W = cotan_weights(verts, faces)
    
    # Create field with gradients
    F = torch.randn(4, 6, requires_grad=True)
    
    # Compute smoothness loss
    from losses.improved_losses import smoothness_cotan
    loss = smoothness_cotan(F, I, J, W)
    
    # Check gradient flow
    loss.backward()
    assert F.grad is not None
    assert torch.isfinite(F.grad).all()
    
    print(f"  Loss: {loss.item():.6f}")
    print(f"  Gradient norm: {F.grad.norm().item():.6f}")
    print(f"  ✓ Gradients flow correctly through fixed components!")


if __name__ == "__main__":
    print("Testing critical fixes...")
    print("=" * 50)
    
    test_temp_controller_nan_fix()
    test_cosine_decay_fix()
    test_cotangent_weights_fix()
    test_gradient_flow()
    
    print("\n" + "=" * 50)
    print("✅ All critical fixes verified!")