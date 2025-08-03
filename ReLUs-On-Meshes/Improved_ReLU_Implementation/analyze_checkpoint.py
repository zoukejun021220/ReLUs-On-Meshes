"""
Analyze training checkpoints to diagnose convergence issues.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def analyze_checkpoint(checkpoint_path):
    """Analyze a single checkpoint file."""
    print(f"\nAnalyzing checkpoint: {checkpoint_path.name}")
    print("="*60)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract metrics
    metrics = ckpt['metrics']
    step = ckpt['step']
    stage = ckpt.get('stage', 'unknown')
    
    print(f"Stage: {stage}, Step: {step}")
    print(f"\nLoss components:")
    print(f"  Total loss: {metrics['loss']:.4f}")
    print(f"  Area loss: {metrics['area']:.4f}")
    print(f"  Adjacency loss: {metrics['adjacency']:.4f}")
    print(f"  TV loss: {metrics['tv']:.4f}")
    
    print(f"\nKey metrics:")
    print(f"  Raw adjacency (normalized): {metrics['raw_adj']:.6f}")
    print(f"  Weight sum: {metrics['weight_sum']:.1f}")
    print(f"  Beta: {metrics['beta']:.2f}")
    print(f"  Lambda_adj: {metrics['lambda_adj']:.2f}")
    
    # Analyze area fractions
    area_fractions = metrics['area_fractions']
    print(f"\nArea fractions: {area_fractions}")
    print(f"  Deviation from 1/6: {np.abs(area_fractions - 1/6).mean():.4f}")
    print(f"  Max fraction: {area_fractions.max():.4f}")
    print(f"  Min fraction: {area_fractions.min():.4f}")
    
    # Analyze field values
    f_values = ckpt['f_values']
    print(f"\nField value statistics:")
    print(f"  Shape: {f_values.shape}")
    print(f"  Mean: {f_values.mean().item():.4f}")
    print(f"  Std: {f_values.std().item():.4f}")
    print(f"  Min: {f_values.min().item():.4f}")
    print(f"  Max: {f_values.max().item():.4f}")
    
    # Compute hardmax statistics
    hardmax = f_values.argmax(dim=1)
    unique, counts = torch.unique(hardmax, return_counts=True)
    print(f"\nHardmax distribution:")
    for i, (label, count) in enumerate(zip(unique, counts)):
        print(f"  Region {label}: {count} vertices ({count/len(hardmax)*100:.1f}%)")
    
    # Check confidence
    beta = metrics['beta']
    if beta > 0:
        probs = torch.softmax(beta * f_values, dim=1)
        max_probs = probs.max(dim=1)[0]
        print(f"\nSoftmax confidence (with β={beta:.1f}):")
        print(f"  Mean max prob: {max_probs.mean().item():.4f}")
        print(f"  Min max prob: {max_probs.min().item():.4f}")
        print(f"  % confident (>0.9): {(max_probs > 0.9).float().mean().item() * 100:.1f}%")
        print(f"  % confident (>0.99): {(max_probs > 0.99).float().mean().item() * 100:.1f}%")
    
    return ckpt


def compare_checkpoints(checkpoint_dir):
    """Compare all checkpoints in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Collect metrics over time
    steps = []
    raw_adj_values = []
    weight_sums = []
    total_losses = []
    area_deviations = []
    
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        metrics = ckpt['metrics']
        
        steps.append(ckpt['step'])
        raw_adj_values.append(metrics['raw_adj'])
        weight_sums.append(metrics['weight_sum'])
        total_losses.append(metrics['loss'])
        area_deviations.append(np.abs(metrics['area_fractions'] - 1/6).mean())
    
    # Plot evolution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Raw adjacency
    ax = axes[0, 0]
    ax.semilogy(steps, raw_adj_values, 'o-')
    ax.axhline(y=0.05, color='g', linestyle='--', label='Target (<0.05)')
    ax.axhline(y=2.0, color='r', linestyle='--', label='Max theoretical (2.0)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Raw Adjacency (normalized)')
    ax.set_title('Raw Adjacency Evolution')
    ax.legend()
    ax.grid(True)
    
    # Weight sum
    ax = axes[0, 1]
    ax.semilogy(steps, weight_sums, 'o-')
    ax.set_xlabel('Step')
    ax.set_ylabel('Weight Sum')
    ax.set_title('Weight Sum Evolution (should drop 10-100x)')
    ax.grid(True)
    
    # Total loss
    ax = axes[1, 0]
    ax.semilogy(steps, total_losses, 'o-')
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss Evolution')
    ax.grid(True)
    
    # Area deviation
    ax = axes[1, 1]
    ax.semilogy(steps, area_deviations, 'o-')
    ax.axhline(y=0.01, color='g', linestyle='--', label='Good (<0.01)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean |fraction - 1/6|')
    ax.set_title('Area Balance Deviation')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(checkpoint_dir.parent / 'checkpoint_analysis.png')
    print(f"\nSaved analysis plot to {checkpoint_dir.parent / 'checkpoint_analysis.png'}")
    
    # Print summary
    print("\n" + "="*60)
    print("CHECKPOINT ANALYSIS SUMMARY")
    print("="*60)
    
    if raw_adj_values[-1] > 2.0:
        print("❌ CRITICAL ERROR: Raw adjacency > 2.0 - missing /15 normalization!")
    elif raw_adj_values[-1] > 0.5:
        print("⚠️ WARNING: Raw adjacency still high - gradients may be stuck")
    elif raw_adj_values[-1] < 0.05:
        print("✅ SUCCESS: Raw adjacency < 0.05 - boundaries should be planar")
    
    reduction = weight_sums[0] / weight_sums[-1] if weight_sums[-1] > 0 else 0
    print(f"\nWeight sum reduction: {weight_sums[0]:.1f} → {weight_sums[-1]:.1f} ({reduction:.1f}x)")
    if reduction < 10:
        print("⚠️ WARNING: Weight sum should drop by 10-100x")
    else:
        print("✅ Good weight sum reduction")
    
    print(f"\nFinal area deviation: {area_deviations[-1]:.4f}")
    if area_deviations[-1] > 0.05:
        print("⚠️ WARNING: Area fractions far from balanced")
    else:
        print("✅ Good area balance")


def main():
    parser = argparse.ArgumentParser(description='Analyze training checkpoints')
    parser.add_argument('checkpoint', type=str, 
                       help='Path to checkpoint file or directory')
    parser.add_argument('--compare', action='store_true',
                       help='Compare all checkpoints in directory')
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    
    if checkpoint_path.is_dir():
        if args.compare:
            compare_checkpoints(checkpoint_path)
        else:
            # Analyze all checkpoints
            for ckpt in sorted(checkpoint_path.glob("*.pt")):
                analyze_checkpoint(ckpt)
    else:
        # Single checkpoint
        analyze_checkpoint(checkpoint_path)


if __name__ == '__main__':
    main()