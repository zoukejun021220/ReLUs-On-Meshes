#!/usr/bin/env python3
"""
Test script to run main.py with kitty mesh and terminate after a few iterations.
"""
import subprocess
import time
import signal
import sys
import os

def run_main_with_timeout(timeout_seconds=10):
    """Run main.py with kitty mesh and terminate after timeout."""
    
    # Path to kitty mesh
    kitty_mesh = "../Piecewise Linear Mesh 3D/l1-poly-dat/hex/kitty/orig.tet.vtk"
    
    # Command to run
    cmd = [
        sys.executable,  # Use current Python interpreter
        "main.py",
        "--mesh", kitty_mesh,
        "--anchor-method", "axis",  # Use axis method instead of raycast
        "--no-grad-norm",  # Disable GradNorm to avoid the bug
        "--device", "cuda"
    ]
    
    print(f"Running main.py with kitty mesh...")
    print(f"Will terminate after {timeout_seconds} seconds")
    print("-" * 60)
    
    # Start the process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    start_time = time.time()
    iteration_count = 0
    
    try:
        # Read output line by line
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line.rstrip())
            
            # Count iterations
            if "Step" in line and "/" in line:
                iteration_count += 1
            
            # Check if timeout reached
            if time.time() - start_time > timeout_seconds:
                print("\n" + "=" * 60)
                print(f"TIMEOUT: Terminating process after {timeout_seconds} seconds")
                print(f"Completed {iteration_count} iterations")
                print("=" * 60)
                process.terminate()
                time.sleep(1)  # Give it time to terminate gracefully
                if process.poll() is None:
                    process.kill()  # Force kill if still running
                break
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == -signal.SIGTERM:
            print("\nProcess terminated successfully by SIGTERM")
        elif return_code == -signal.SIGKILL:
            print("\nProcess killed by SIGKILL")
        else:
            print(f"\nProcess exited with code: {return_code}")
            
    except KeyboardInterrupt:
        print("\n\nKeyboard interrupt - terminating process...")
        process.terminate()
        process.wait()
        print("Process terminated.")
    
    except Exception as e:
        print(f"\nError: {e}")
        if process.poll() is None:
            process.terminate()
            process.wait()
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f} seconds")
    print(f"Total iterations completed: {iteration_count}")

if __name__ == "__main__":
    # Run for 10 seconds to see a few iterations
    run_main_with_timeout(timeout_seconds=10)