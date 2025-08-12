#!/usr/bin/env python3
"""Test training with NaN hardening fixes"""
import torch
import sys

# Enable anomaly detection for debugging
torch.autograd.set_detect_anomaly(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False

print("NaN hardening test configuration:")
print("- Anomaly detection: ON")
print("- TF32: OFF")
print("- Benchmark mode: OFF")
print("- Float64 geometry: YES")
print("- NaN guards: ACTIVE")
print("- Reduced LR: 5e-5")
print("- Reduced initial scale: 0.05")
print("- Conservative beta schedule: βc ≤ 8, βa ≤ 4")
print()

# Import and run the main training
from train_improved import main

if __name__ == "__main__":
    # Run with anomaly detection
    try:
        main()
    except RuntimeError as e:
        if "Non-finite" in str(e):
            print(f"\n[CAUGHT] {e}")
            print("The anomaly detector caught the issue!")
            print("Check the traceback above to see which operation caused NaN.")