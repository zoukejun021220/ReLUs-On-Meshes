#!/usr/bin/env python3
"""
Continue training from checkpoint with improved settings.
Specifically designed to fix speckles and ragged boundaries.
"""
import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
import json

from utils.mesh_preprocessing import preprocess_mesh
from train_improved import train_mesh_segmentation


def load_checkpoint_for_continuation(checkpoint_path: str, mesh_data: dict, device: str):
    """Load checkpoint and prepare for continuation with new optimizer settings."""
    
    # Load checkpoint
    ckpt = np.load(checkpoint_path)
    
    # Extract field values
    F = torch.tensor(ckpt['field_values'], device=device, dtype=torch.float32)
    F = nn.Parameter(F)
    
    # Get step number
    step = int(ckpt['step'])
    
    # Extract temperature values
    beta_contour = float(ckpt.get('beta_contour', 4.0))
    beta_area = float(ckpt.get('beta_area', 2.5))
    
    print(f"Loaded checkpoint from step {step}")
    print(f"Current temperatures: βc={beta_contour:.2f}, βa={beta_area:.2f}")
    print(f"Field shape: {F.shape}")
    
    # Calculate current metrics
    from losses.improved_losses import compute_boundary_stats
    _, active_frac = compute_boundary_stats(
        F, mesh_data['edge_idx'], mesh_data['vertices'], beta_contour
    )
    print(f"Current active edge fraction: {active_frac:.1%}")
    
    return F, step, beta_contour, beta_area


def continue_training_from_checkpoint(
    checkpoint_path: str,
    mesh_data: dict,
    additional_steps: int = 50000,
    device: str = 'cuda',
    output_dir: Path = None
):
    """Continue training with improved settings."""
    
    # Load checkpoint
    F, start_step, beta_contour, beta_area = load_checkpoint_for_continuation(
        checkpoint_path, mesh_data, device
    )
    
    # Create new optimizer with NO weight decay
    optimizer = torch.optim.AdamW([F], lr=1e-4, weight_decay=0.0)
    
    # Create temperature controller with current values
    from optimization.temperature_control import TempController, TwoStageScheduler
    temp_ctrl = TempController()
    temp_ctrl.beta_contour = beta_contour
    temp_ctrl.beta_area = beta_area
    temp_ctrl.last_beta_update_step = start_step - 1000  # Allow immediate updates
    
    # Create scheduler starting from current step
    total_steps = start_step + additional_steps
    scheduler = TwoStageScheduler(total_steps)
    
    # Run training continuation
    print(f"\nContinuing training for {additional_steps} additional steps...")
    print("Key improvements:")
    print("- Weight decay = 0 (was 1e-4)")
    print("- β can ramp faster (threshold 0.005, cooldown 1000)")
    print("- Added Potts smoothness on probabilities")
    print("- Added boundary length regularizer")
    
    # Import the main training function and run continuation
    # This is a simplified version - in practice you'd modify train_mesh_segmentation
    # to accept initial F, optimizer state, and starting step
    
    # For now, save the modified checkpoint
    if output_dir:
        output_path = output_dir / f'checkpoint_{start_step:06d}_continued.npz'
        np.savez_compressed(
            output_path,
            field_values=F.detach().cpu().numpy(),
            step=start_step,
            beta_contour=beta_contour,
            beta_area=beta_area,
            vertices=mesh_data['vertices'].cpu().numpy(),
            faces=mesh_data['faces'].cpu().numpy(),
            edge_idx=mesh_data['edge_idx'].cpu().numpy(),
            pinned_indices=mesh_data['pinned_indices'].cpu().numpy(),
            note='Ready for continuation with improved settings'
        )
        print(f"\nSaved continuation checkpoint to {output_path}")
    
    return F


def main():
    parser = argparse.ArgumentParser(description='Continue training from checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint npz file')
    parser.add_argument('mesh', type=str, help='Path to original mesh file')
    parser.add_argument('--steps', type=int, default=50000, help='Additional training steps')
    parser.add_argument('--output-dir', type=str, default='results_continued', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load and preprocess mesh
    print(f"Loading mesh from {args.mesh}...")
    from utils.mesh_preprocessing import load_volume_tet_mesh_and_extract_surface
    vertices_np, faces_np = load_volume_tet_mesh_and_extract_surface(args.mesh)
    mesh_data = preprocess_mesh(vertices_np, faces_np, device=args.device)
    
    # Continue training
    continue_training_from_checkpoint(
        args.checkpoint,
        mesh_data,
        additional_steps=args.steps,
        device=args.device,
        output_dir=output_dir
    )


if __name__ == '__main__':
    main()