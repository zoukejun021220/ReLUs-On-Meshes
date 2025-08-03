#!/usr/bin/env python3
"""
Test script to run main.py with dragon mesh and terminate after a few iterations.
"""
import subprocess
import time
import signal
import sys
import os

def run_main_with_timeout(timeout_seconds=30):
    """Run main.py with dragon mesh and terminate after timeout."""
    
    # Path to dragon mesh
    dragon_mesh = "../Piecewise Linear Mesh 3D/l1-poly-dat/hex/dragon/orig.tet.vtk"
    
    # Command to run
    cmd = [
        sys.executable,  # Use current Python interpreter
        "main.py",
        "--mesh", dragon_mesh,
        "--anchor-method", "axis",  # Use axis method to avoid rtree dependency
        "--device", "cuda"
    ]
    
    print(f"Running main.py with dragon mesh...")
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
    
    try:
        # Read output line by line
        while True:
            line = process.stdout.readline()
            if not line:
                break
            print(line.rstrip())
            
            # Check if timeout reached
            if time.time() - start_time > timeout_seconds:
                print("\n" + "=" * 60)
                print(f"TIMEOUT: Terminating process after {timeout_seconds} seconds")
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

if __name__ == "__main__":
    # Run for 15 seconds to see a few iterations
    run_main_with_timeout(timeout_seconds=15)