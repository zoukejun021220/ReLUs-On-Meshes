#!/usr/bin/env python3
"""
Run main.py with kitty mesh for a few iterations by patching the step count.
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch the optimization stages to run only 5 steps per level
from optimization import CoarseToFineSchedule, TrainingConfig

original_init = CoarseToFineSchedule.__init__

def patched_init(self):
    self.stages = [
        TrainingConfig(level=0, num_faces=3000, steps=5,  # Only 5 steps!
                     beta_start=2.0, beta_end=10.0,
                     lambda_adj_start=0.0, lambda_adj_end=5.0),
        TrainingConfig(level=1, num_faces=12000, steps=5,  # Only 5 steps!
                     beta_start=10.0, beta_end=10.0,
                     lambda_adj_start=5.0, lambda_adj_end=5.0),
        TrainingConfig(level=2, num_faces=-1, steps=5,  # Only 5 steps!
                     beta_start=10.0, beta_end=25.0,
                     lambda_adj_start=5.0, lambda_adj_end=8.0),
    ]

CoarseToFineSchedule.__init__ = patched_init

# Also patch direct training config
original_TrainingConfig_init = TrainingConfig.__init__

def patched_TrainingConfig_init(self, level, num_faces, steps, beta_start, beta_end, 
                               lambda_adj_start, lambda_adj_end, lr_max=5e-3, 
                               lambda_area=1.0, lambda_tv=0.1):
    self.level = level
    self.num_faces = num_faces
    self.steps = min(steps, 5)  # Limit to 5 steps max
    self.beta_start = beta_start
    self.beta_end = beta_end
    self.lambda_adj_start = lambda_adj_start
    self.lambda_adj_end = lambda_adj_end
    self.lr_max = lr_max
    self.lambda_area = lambda_area
    self.lambda_tv = lambda_tv

TrainingConfig.__init__ = patched_TrainingConfig_init

# Now import and run main
from main import main

if __name__ == "__main__":
    print("Running main.py with patched configuration (max 5 steps per level)")
    print("=" * 60)
    
    # Set up arguments
    sys.argv = [
        "main.py",
        "--mesh", "../Piecewise Linear Mesh 3D/l1-poly-dat/hex/kitty/orig.tet.vtk",
        "--anchor-method", "axis",
        "--no-grad-norm"
    ]
    
    try:
        main()
        print("\n" + "=" * 60)
        print("✓ Main.py completed successfully with limited iterations!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()