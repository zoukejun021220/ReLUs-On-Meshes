#!/usr/bin/env python3
"""
Command-line interface for running improved ReLU mesh optimization.
"""

import argparse
import os
import sys
import numpy as np
import time

from relus_mesh_optimization_improved import optimize_relu_mesh
from mesh_optimization_helpers import auto_select_pins


def load_mesh(filepath):
    """Load mesh from various file formats."""
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        # Try different key combinations
        if 'vertices' in data and 'faces' in data:
            return data['vertices'], data['faces']
        elif 'mesh' in data and 'face' in data:
            return data['mesh'], data['face']
        elif 'points' in data and 'triangles' in data:
            return data['points'], data['triangles']
        else:
            print(f"Available keys in npz: {list(data.keys())}")
            raise ValueError("Could not find vertex/face data in npz file")
    elif filepath.endswith('.npy'):
        # Assume it contains both vertices and faces
        data = np.load(filepath, allow_pickle=True).item()
        return data['vertices'], data['faces']
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimize ReLU fields on 3D meshes for polycube-like segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python run_optimization.py mesh.npz
  
  # Run with more iterations and dynamic reweighting
  python run_optimization.py mesh.npz --iters 100000 --dynamic-reweight
  
  # Use PCA-based anchor selection and custom output
  python run_optimization.py mesh.npz --anchor-method pca --output optimized.npz
  
  # Quick test with visualization
  python run_optimization.py mesh.npz --iters 10000 --visualize
        """
    )
    
    # Required arguments
    parser.add_argument('mesh', type=str, help='Path to input mesh file (.npz)')
    
    # Output options
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file path (default: <input>_optimized.npz)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    
    # Optimization parameters
    parser.add_argument('--iters', type=int, default=50000,
                        help='Number of optimization iterations (default: 50000)')
    parser.add_argument('--lr', type=float, default=2e-3,
                        help='Learning rate for vertex field (default: 2e-3)')
    parser.add_argument('--lr-offset-factor', type=float, default=10.0,
                        help='Learning rate multiplier for plane offsets (default: 10.0)')
    
    # Beta schedule
    parser.add_argument('--beta-start', type=float, default=4.0,
                        help='Starting beta value (default: 4.0)')
    parser.add_argument('--beta-end', type=float, default=20.0,
                        help='Final beta value (default: 20.0)')
    parser.add_argument('--beta-schedule', type=str, default='sigmoid',
                        choices=['linear', 'sigmoid', 'log'],
                        help='Beta scheduling type (default: sigmoid)')
    
    # Lambda weights
    parser.add_argument('--lambda-contour', type=float, nargs=2, default=[1.0, 4.0],
                        help='Contour loss weight range (default: 1.0 4.0)')
    parser.add_argument('--lambda-smooth', type=float, default=0.1,
                        help='Smoothness loss weight (default: 0.1)')
    parser.add_argument('--lambda-area', type=float, nargs=2, default=[0.0, 100.0],
                        help='Area balance loss weight range (default: 0.0 100.0)')
    
    # Advanced options
    parser.add_argument('--dynamic-reweight', action='store_true',
                        help='Enable dynamic loss reweighting')
    parser.add_argument('--reverse-schedule', action='store_true', default=True,
                        help='Start with full contour weight (default: True)')
    parser.add_argument('--no-reverse-schedule', dest='reverse_schedule', action='store_false',
                        help='Use standard lambda scheduling')
    parser.add_argument('--anchor-method', type=str, default='bbox',
                        choices=['bbox', 'pca'],
                        help='Method for selecting anchor vertices (default: bbox)')
    parser.add_argument('--gradient-clip', type=float, default=5.0,
                        help='Gradient clipping value (default: 5.0)')
    
    # Logging
    parser.add_argument('--print-every', type=int, default=1000,
                        help='Print progress every N iterations (default: 1000)')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    args = parser.parse_args()
    
    # Load mesh
    try:
        if not args.quiet:
            print(f"Loading mesh from {args.mesh}...")
        vertices, faces = load_mesh(args.mesh)
        if not args.quiet:
            print(f"Loaded mesh with {len(vertices)} vertices and {len(faces)} faces")
    except Exception as e:
        print(f"Error loading mesh: {e}")
        sys.exit(1)
    
    # Auto-select pinned vertices
    pinned_indices = auto_select_pins(vertices, method=args.anchor_method)
    if not args.quiet:
        print(f"Selected {len(pinned_indices)} anchor vertices using {args.anchor_method} method")
        print(f"Anchor indices: {pinned_indices}")
    
    # Set output path
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.mesh))[0]
        args.output = f"{base}_optimized.npz"
    
    # Print configuration
    if not args.quiet:
        print("\nOptimization Configuration:")
        print(f"  Iterations: {args.iters}")
        print(f"  Learning rate: {args.lr} (vertex), {args.lr * args.lr_offset_factor} (offsets)")
        print(f"  Beta schedule: {args.beta_start} -> {args.beta_end} ({args.beta_schedule})")
        print(f"  Lambda contour: {args.lambda_contour[0]} -> {args.lambda_contour[1]}")
        print(f"  Lambda smooth: {args.lambda_smooth}")
        print(f"  Lambda area: {args.lambda_area[0]} -> {args.lambda_area[1]}")
        print(f"  Dynamic reweighting: {args.dynamic_reweight}")
        print(f"  Reverse schedule: {args.reverse_schedule}")
        print(f"  Output: {args.output}")
        print()
    
    # Run optimization
    start_time = time.time()
    
    try:
        results = optimize_relu_mesh(
            vertices=vertices,
            faces=faces,
            pinned_indices=pinned_indices,
            n_iters=args.iters,
            lr_vertex=args.lr,
            lr_offset=args.lr * args.lr_offset_factor,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            lambda_contour=tuple(args.lambda_contour),
            lambda_smooth=args.lambda_smooth,
            lambda_area=tuple(args.lambda_area),
            reverse_schedule=args.reverse_schedule,
            use_dynamic_reweighting=args.dynamic_reweight,
            gradient_clip=args.gradient_clip,
            print_every=args.print_every if not args.quiet else args.iters + 1,
            save_path=args.output
        )
        
        elapsed = time.time() - start_time
        
        if not args.quiet:
            print(f"\nOptimization completed in {elapsed/60:.1f} minutes")
            print(f"Best loss: {results['best_loss']:.3e} at iteration {results['best_iter']}")
            print(f"Results saved to: {args.output}")
        
        # Generate visualization if requested
        if args.visualize:
            try:
                from test_improved_optimization import visualize_results
                viz_path = os.path.splitext(args.output)[0] + "_visualization.png"
                visualize_results(vertices, faces, results['f_values'], viz_path)
                print(f"Visualization saved to: {viz_path}")
            except ImportError:
                print("Warning: Could not import visualization function")
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())