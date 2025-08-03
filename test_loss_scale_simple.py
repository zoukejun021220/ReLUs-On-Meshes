#!/usr/bin/env python3
"""
Simple test to understand the loss scale issue.
"""

import torch
import numpy as np

# Simulate the area loss computation
def test_area_loss(f_scale=0.1, beta=2.0):
    """Test area loss with different field scales."""
    C = 6
    N = 1000  # vertices
    
    # Random field initialization
    f_values = torch.randn(N, C) * f_scale
    
    # Compute softmax probabilities
    probs = torch.softmax(beta * f_values, dim=1)
    
    # Average probability per channel
    avg_probs = probs.mean(dim=0)
    
    # Target is uniform distribution
    target = torch.ones(C) / C
    
    # Area loss (L1 norm)
    area_loss = torch.abs(avg_probs - target).sum()
    
    print(f"\nField scale: {f_scale}, Beta: {beta}")
    print(f"Field values range: [{f_values.min():.3f}, {f_values.max():.3f}]")
    print(f"Average probs: {avg_probs.numpy()}")
    print(f"Target: {target[0]:.6f}")
    print(f"Deviations: {torch.abs(avg_probs - target).numpy()}")
    print(f"Area loss: {area_loss.item():.6f}")
    
    return area_loss.item()


# Test different scenarios
print("="*60)
print("Testing area loss with different initializations:")
print("="*60)

# Small initialization (your current case)
test_area_loss(f_scale=0.1, beta=2.0)

# Larger initialization
test_area_loss(f_scale=1.0, beta=2.0)

# Even larger
test_area_loss(f_scale=2.0, beta=2.0)

# With higher beta
test_area_loss(f_scale=0.1, beta=25.0)

# What initialization scale gives reasonable area loss?
print("\n" + "="*60)
print("Finding good initialization scale:")
print("="*60)

for scale in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
    loss = test_area_loss(f_scale=scale, beta=2.0)
    
print("\n" + "="*60)
print("The issue: With small initialization (0.1), softmax gives nearly")
print("uniform distribution, so area loss is tiny (~0.002).")
print("\nSolutions:")
print("1. Use larger initialization (e.g., randn * 1.0)")
print("2. Scale up the area loss by a factor (e.g., * 1000)")
print("3. Use different area loss formulation")
print("="*60)