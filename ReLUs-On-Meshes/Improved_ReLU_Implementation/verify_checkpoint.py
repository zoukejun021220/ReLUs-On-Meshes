"""
Verify checkpoint health based on the recommended criteria.
"""
import torch
import numpy as np
import argparse
from pathlib import Path


def verify_checkpoint(checkpoint_path):
    """Verify a checkpoint meets the health criteria."""
    print(f"\nVerifying checkpoint: {checkpoint_path.name}")
    print("="*60)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract key values
    step = ckpt['step']
    metrics = ckpt['metrics']
    f_values = ckpt['f_values']
    
    beta = metrics['beta']
    lambda_adj = metrics['lambda_adj']
    raw_adj = metrics['raw_adj']
    weight_sum = metrics['weight_sum']
    area_fractions = metrics['area_fractions']
    
    # Compute hardmax distribution
    hard_labels = f_values.argmax(dim=1).numpy()
    label_counts = np.bincount(hard_labels, minlength=6)
    
    print(f"Step: {step}")
    print(f"\nScheduler values:")
    print(f"  β: {beta:.2f}")
    print(f"  λ_adj: {lambda_adj:.2f}")
    
    print(f"\nKey metrics:")
    print(f"  Raw adjacency: {raw_adj:.4f}")
    print(f"  Weight sum: {weight_sum:.1f}")
    
    print(f"\nArea distribution:")
    print(f"  Fractions: {area_fractions}")
    print(f"  Max deviation from 1/6: {np.abs(area_fractions - 1/6).max():.4f}")
    
    print(f"\nHardmax distribution:")
    total_verts = len(hard_labels)
    for i, count in enumerate(label_counts):
        pct = count / total_verts * 100
        print(f"  Region {i}: {count:5d} vertices ({pct:5.1f}%)")
    
    # Health checks
    print(f"\n{'='*40}")
    print("HEALTH CHECKS:")
    print(f"{'='*40}")
    
    checks_passed = []
    
    # Check 1: Beta at 10k steps
    if step == 10000:
        if beta <= 6:
            print("✅ Beta ≈ 6 at 10k steps (good for avoiding saturation)")
            checks_passed.append(True)
        else:
            print(f"⚠️  Beta = {beta:.1f} at 10k (should be ≤ 6)")
            checks_passed.append(False)
    
    # Check 2: Weight sum reduction
    if step == 100000:
        initial_weight = 366660  # Typical initial value
        reduction = initial_weight / weight_sum
        if reduction >= 10:
            print(f"✅ Weight sum reduced {reduction:.1f}x by 100k steps")
            checks_passed.append(True)
        else:
            print(f"⚠️  Weight sum only reduced {reduction:.1f}x (should be ≥10x)")
            checks_passed.append(False)
    
    # Check 3: Lambda_adj when beta >= 12
    if beta >= 12:
        if lambda_adj <= 5:
            print(f"✅ λ_adj = {lambda_adj:.1f} when β ≥ 12 (good)")
            checks_passed.append(True)
        else:
            print(f"⚠️  λ_adj = {lambda_adj:.1f} when β = {beta:.1f} (should be ≤ 5)")
            checks_passed.append(False)
    
    # Check 4: Area balance
    max_dev = np.abs(area_fractions - 1/6).max()
    if max_dev <= 0.04:
        print(f"✅ Area fractions within ±0.04 of 1/6")
        checks_passed.append(True)
    else:
        print(f"⚠️  Area deviation = {max_dev:.3f} (should be ≤ 0.04)")
        checks_passed.append(False)
    
    # Check 5: Raw adjacency
    if raw_adj <= 0.35:
        print(f"✅ Raw adjacency = {raw_adj:.4f} (good convergence)")
        checks_passed.append(True)
    elif raw_adj <= 0.5:
        print(f"⚠️  Raw adjacency = {raw_adj:.4f} (making progress)")
        checks_passed.append(True)
    else:
        print(f"❌ Raw adjacency = {raw_adj:.4f} (too high, check scheduler)")
        checks_passed.append(False)
    
    # Overall assessment
    print(f"\n{'='*40}")
    if all(checks_passed):
        print("✅ HEALTHY CHECKPOINT - Training is converging well")
    elif len([c for c in checks_passed if c]) >= len(checks_passed) * 0.7:
        print("⚠️  MOSTLY HEALTHY - Some minor issues")
    else:
        print("❌ UNHEALTHY - Check scheduler and loss weights")
    
    return checks_passed


def main():
    parser = argparse.ArgumentParser(description='Verify checkpoint health')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    
    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} does not exist")
        return
    
    verify_checkpoint(checkpoint_path)


if __name__ == '__main__':
    main()