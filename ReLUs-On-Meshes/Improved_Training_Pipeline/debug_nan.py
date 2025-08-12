#!/usr/bin/env python3
"""Debug script to identify source of NaN values"""
import torch
import numpy as np

def check_nan_checkpoint(checkpoint_path):
    """Check a checkpoint for NaN values"""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    ckpt = np.load(checkpoint_path)
    
    # Check field values
    F = ckpt['field_values']
    print(f"\nField values shape: {F.shape}")
    print(f"Contains NaN: {np.any(np.isnan(F))}")
    print(f"Contains Inf: {np.any(np.isinf(F))}")
    
    if np.any(np.isnan(F)):
        nan_mask = np.isnan(F)
        print(f"NaN locations: {np.where(nan_mask)}")
        print(f"Number of NaN values: {np.sum(nan_mask)}")
    
    # Check field statistics
    if not np.all(np.isnan(F)):
        valid_F = F[~np.isnan(F)]
        print(f"\nValid field statistics:")
        print(f"  Min: {np.min(valid_F):.6f}")
        print(f"  Max: {np.max(valid_F):.6f}")
        print(f"  Mean: {np.mean(valid_F):.6f}")
        print(f"  Std: {np.std(valid_F):.6f}")
    
    # Check losses
    print(f"\nLoss values:")
    print(f"  Total loss: {ckpt['total_loss']}")
    print(f"  Smooth loss: {ckpt['loss_smooth']}")
    print(f"  Contour loss: {ckpt['loss_contour']}")
    print(f"  Area loss: {ckpt['loss_area']}")
    
    # Check temperatures
    print(f"\nTemperatures:")
    print(f"  β_contour: {ckpt['beta_contour']}")
    print(f"  β_area: {ckpt['beta_area']}")
    
    return ckpt


def find_nan_source(F_tensor, mesh_data):
    """Find which component is causing NaN"""
    F = F_tensor.detach()
    
    print("Checking field values:")
    print(f"  Contains NaN: {torch.isnan(F).any()}")
    print(f"  Contains Inf: {torch.isinf(F).any()}")
    
    if torch.isnan(F).any():
        nan_verts = torch.where(torch.isnan(F).any(dim=1))[0]
        print(f"  Vertices with NaN: {nan_verts}")
        print(f"  Number of NaN vertices: {len(nan_verts)}")
        
        # Check if NaN vertices are pinned
        if 'pinned_indices' in mesh_data:
            pinned = mesh_data['pinned_indices']
            nan_pinned = [v.item() for v in nan_verts if v in pinned]
            if nan_pinned:
                print(f"  NaN in pinned vertices: {nan_pinned}")
    
    # Check field range
    if not torch.isnan(F).all():
        valid_mask = ~torch.isnan(F)
        valid_F = F[valid_mask]
        print(f"\nField range (non-NaN values):")
        print(f"  Min: {valid_F.min():.6f}")
        print(f"  Max: {valid_F.max():.6f}")
        print(f"  Mean: {valid_F.mean():.6f}")
    
    # Check gradients if available
    if F_tensor.grad is not None:
        grad = F_tensor.grad
        print(f"\nGradient statistics:")
        print(f"  Contains NaN: {torch.isnan(grad).any()}")
        print(f"  Contains Inf: {torch.isinf(grad).any()}")
        if not torch.isnan(grad).all():
            print(f"  Grad norm: {grad.norm():.6f}")
            print(f"  Max grad: {grad.abs().max():.6f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        check_nan_checkpoint(checkpoint_path)
    else:
        print("Usage: python debug_nan.py <checkpoint_path>")
        print("\nChecking for recent checkpoints...")
        
        import glob
        checkpoints = sorted(glob.glob("checkpoints/checkpoint_*.npz"))
        if checkpoints:
            print(f"Found {len(checkpoints)} checkpoints")
            print(f"Latest: {checkpoints[-1]}")
            check_nan_checkpoint(checkpoints[-1])