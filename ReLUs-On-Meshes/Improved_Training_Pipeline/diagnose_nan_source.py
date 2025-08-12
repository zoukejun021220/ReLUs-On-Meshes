#!/usr/bin/env python3
"""Diagnostic script to identify which loss component causes NaN/Inf gradients"""
import torch
import numpy as np
from pathlib import Path

def diagnose_nan_checkpoint(checkpoint_path, mesh_data_path=None):
    """
    Load a checkpoint and compute per-loss gradients to identify NaN source.
    
    Args:
        checkpoint_path: Path to checkpoint npz file
        mesh_data_path: Optional path to saved mesh data
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = np.load(checkpoint_path)
    
    # Load field values
    F_np = ckpt['field_values']
    vertices_np = ckpt['vertices']
    faces_np = ckpt['faces']
    edge_idx_np = ckpt['edge_idx']
    tri_area_np = ckpt['tri_area']
    pinned_indices_np = ckpt['pinned_indices']
    
    # Current state
    beta_contour = float(ckpt['beta_contour'])
    beta_area = float(ckpt['beta_area'])
    
    print(f"\nCheckpoint state:")
    print(f"  Step: {ckpt['step']}")
    print(f"  β_contour: {beta_contour}")
    print(f"  β_area: {beta_area}")
    print(f"  Field shape: {F_np.shape}")
    
    # Convert to torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    F = torch.tensor(F_np, device=device, dtype=torch.float32, requires_grad=True)
    verts = torch.tensor(vertices_np, device=device, dtype=torch.float32)
    faces = torch.tensor(faces_np, device=device, dtype=torch.long)
    edge_idx = torch.tensor(edge_idx_np, device=device, dtype=torch.long)
    tri_area = torch.tensor(tri_area_np, device=device, dtype=torch.float32)
    pinned_indices = torch.tensor(pinned_indices_np, device=device, dtype=torch.long)
    
    # Import loss functions
    from losses.improved_losses import (
        contour_alignment_intrinsic,
        smoothness_cotan,
        area_fractions_and_kl,
        pin_loss,
        potts_smoothness_loss,
        boundary_length_regularizer,
        non_boundary_margin_loss
    )
    from utils.mesh_preprocessing import cotan_weights
    
    # Compute cotangent weights
    I, J, W = cotan_weights(verts, faces)
    
    print("\nComputing individual losses...")
    losses = {}
    
    # Compute each loss
    try:
        losses['smooth'] = smoothness_cotan(F, I, J, W)
        print(f"  Smooth loss: {losses['smooth'].item():.6f}")
    except Exception as e:
        print(f"  Smooth loss: ERROR - {e}")
        losses['smooth'] = None
    
    try:
        # Need edge_tris for contour loss - reconstruct if not saved
        if 'edge_tris' in ckpt:
            edge_tris = torch.tensor(ckpt['edge_tris'], device=device, dtype=torch.long)
        else:
            # Simple reconstruction - won't be perfect but good enough for diagnosis
            from utils.mesh_preprocessing import build_edges_and_adjacency
            _, edge_tris = build_edges_and_adjacency(faces, verts.shape[0])
            edge_tris = edge_tris.to(device)
        
        losses['contour'], edge_weights = contour_alignment_intrinsic(
            F, faces, edge_idx, edge_tris,
            beta_contour=beta_contour,
            return_weights=True,
            verts=verts
        )
        print(f"  Contour loss: {losses['contour'].item():.6f}")
    except Exception as e:
        print(f"  Contour loss: ERROR - {e}")
        losses['contour'] = None
        edge_weights = torch.zeros(edge_idx.shape[0], device=device)
    
    try:
        losses['area'], area_frac = area_fractions_and_kl(
            F, faces, tri_area, beta_area=beta_area
        )
        print(f"  Area loss: {losses['area'].item():.6f}")
    except Exception as e:
        print(f"  Area loss: ERROR - {e}")
        losses['area'] = None
    
    try:
        # Compute pin targets
        n_channels = F.shape[1]
        pin_targets = torch.full((len(pinned_indices), n_channels), -1.0, device=device)
        for i in range(min(len(pinned_indices), n_channels)):
            pin_targets[i, i] = 1.0
        
        losses['pin'] = pin_loss(F, pinned_indices, pin_targets)
        print(f"  Pin loss: {losses['pin'].item():.6f}")
    except Exception as e:
        print(f"  Pin loss: ERROR - {e}")
        losses['pin'] = None
    
    try:
        losses['potts'] = potts_smoothness_loss(F, edge_idx, edge_weights, beta_area)
        print(f"  Potts loss: {losses['potts'].item():.6f}")
    except Exception as e:
        print(f"  Potts loss: ERROR - {e}")
        losses['potts'] = None
    
    try:
        losses['boundary_length'] = boundary_length_regularizer(edge_idx, edge_weights, verts)
        print(f"  Boundary length: {losses['boundary_length'].item():.6f}")
    except Exception as e:
        print(f"  Boundary length: ERROR - {e}")
        losses['boundary_length'] = None
    
    # Check gradients for each loss
    print("\nChecking gradients for each loss component...")
    bad_grads = []
    
    for name, loss in losses.items():
        if loss is None:
            continue
            
        try:
            # Compute gradient without retaining graph
            F.grad = None
            loss.backward(retain_graph=True)
            
            if F.grad is None:
                print(f"  {name}: No gradient computed")
            else:
                nan_count = torch.isnan(F.grad).sum().item()
                inf_count = torch.isinf(F.grad).sum().item()
                grad_norm = F.grad.norm().item()
                
                if nan_count > 0 or inf_count > 0:
                    print(f"  {name}: BAD - NaN: {nan_count}, Inf: {inf_count}, norm: {grad_norm}")
                    bad_grads.append(name)
                else:
                    print(f"  {name}: OK - norm: {grad_norm:.6f}")
        except Exception as e:
            print(f"  {name}: ERROR computing gradient - {e}")
    
    if bad_grads:
        print(f"\n⚠️  Bad gradients detected in: {', '.join(bad_grads)}")
    else:
        print("\n✓ All gradients are finite")
    
    # Additional diagnostics
    print("\nField statistics:")
    print(f"  Min: {F.min().item():.6f}")
    print(f"  Max: {F.max().item():.6f}")
    print(f"  Mean: {F.mean().item():.6f}")
    print(f"  Std: {F.std().item():.6f}")
    
    # Check for any extreme values
    extreme_mask = (F.abs() > 10.0)
    if extreme_mask.any():
        print(f"  Extreme values (|F| > 10): {extreme_mask.sum().item()} entries")
    
    return bad_grads


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose NaN source in checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint npz file')
    parser.add_argument('--mesh-data', type=str, help='Optional path to mesh data')
    
    args = parser.parse_args()
    
    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)
    
    bad_grads = diagnose_nan_checkpoint(args.checkpoint, args.mesh_data)
    
    if bad_grads:
        print(f"\nDiagnosis: The following losses produce NaN/Inf gradients: {bad_grads}")
        print("Most likely culprit is 'contour' due to normalization of near-zero vectors at high β.")
    else:
        print("\nDiagnosis: No NaN/Inf gradients detected in this checkpoint.")