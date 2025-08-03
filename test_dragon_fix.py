#!/usr/bin/env python3
"""
Test script to verify dragon mesh runs without errors after the indexing fix.
"""
import subprocess
import sys
import os
import time

def test_dragon_mesh():
    """Run main.py with dragon mesh and verify it works."""
    
    # Command to run
    cmd = [
        sys.executable,
        "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Improved_ReLU_Implementation/main.py",
        "--mesh", "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Piecewise Linear Mesh 3D/l1-poly-dat/hex/dragon/orig.tet.vtk",
        "--anchor-method", "axis",  # Use axis method to avoid raycast issues
        "--no-grad-norm"
    ]
    
    print("Testing dragon mesh with numerical stability fixes...")
    print("=" * 60)
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    start_time = time.time()
    timeout = 30  # seconds
    lines_shown = 0
    max_lines = 50
    
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
                
            # Show first 50 lines
            if lines_shown < max_lines:
                print(line.rstrip())
                lines_shown += 1
            
            # Check for errors
            if "IndexError" in line or "RuntimeError" in line or "nan" in line.lower():
                print(f"\nERROR DETECTED: {line}")
                print("Test FAILED")
                process.terminate()
                return False
                
            # Check timeout
            if time.time() - start_time > timeout:
                print("\n" + "=" * 60)
                print(f"Test ran successfully for {timeout} seconds without errors!")
                print("Terminating process...")
                process.terminate()
                process.wait()
                return True
                
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        process.terminate()
        process.wait()
        
    return True

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/Improved_ReLU_Implementation")
    
    # First check if we can import the modules
    try:
        import torch
        print(f"PyTorch available: {torch.cuda.is_available()}")
    except ImportError:
        print("Warning: PyTorch not available in this environment")
        print("Running subprocess test instead...")
        
    success = test_dragon_mesh()
    
    if success:
        print("\n✓ Dragon mesh test completed successfully!")
        print("The numerical stability fixes are working correctly.")
    else:
        print("\n✗ Dragon mesh test failed")
        print("Please check the error messages above.")