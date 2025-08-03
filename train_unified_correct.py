#!/usr/bin/env python3
"""
Correct unified training script using the original loss scale.
This matches the mathematical formulation without over-normalization.
"""

import argparse
import os
import sys
import numpy as np
import torch
import time
import json
from datetime import datetime

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Use the original ground-up implementation with correct scaling
from loss_groundup import MeshLossGroundUp
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
    
    elif filepath.endswith('.vtk') or filepath.endswith('.vtu'):
        # Manual VTK parser for tetrahedral meshes
        return load_vtk_manual(filepath)
    
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def load_vtk_manual(filename):
    """Manual VTK ASCII parser for tetrahedral meshes."""
    vertices = []
    cells = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line.startswith('POINTS'):
            parts = line.split()
            num_points = int(parts[1])
            i += 1
            
            point_data = []
            while len(point_data) < num_points * 3:
                values = lines[i].strip().split()
                point_data.extend([float(v) for v in values])
                i += 1
            
            vertices = np.array(point_data).reshape(-1, 3)
            continue
            
        if line.startswith('CELLS'):
            parts = line.split()
            num_cells = int(parts[1])
            i += 1
            
            for _ in range(num_cells):
                cell_line = lines[i].strip().split()
                cell_size = int(cell_line[0])
                if cell_size == 4:  # Tetrahedron
                    cells.append([int(cell_line[j]) for j in range(1, 5)])
                i += 1
            continue
            
        i += 1
    
    # Extract surface from tetrahedra
    face_set = set()
    for tet in cells:
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]]))
        ]
        for face in faces:
            if face in face_set:
                face_set.remove(face)  # Interior face
            else:
                face_set.add(face)  # Boundary face
    
    triangles = np.array(list(face_set))
    return vertices, triangles


def train_relu_mesh(mesh_path, config):
    """Main training function with correct loss scaling."""
    # Load mesh
    print(f"\nLoading mesh from: {mesh_path}")
    vertices, faces = load_mesh(mesh_path)
    print(f"Loaded mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Convert to torch
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    
    verts = torch.from_numpy(vertices).float()
    faces_tensor = torch.from_numpy(faces).long()
    
    # Initialize loss module (using original implementation)
    print("Initializing loss module...")
    mesh_loss = MeshLossGroundUp(verts, faces_tensor, device)
    
    # Auto-select pinned vertices
    pinned_indices = auto_select_pins(vertices, method=config['anchor_method'])
    print(f"Selected {len(pinned_indices)} anchor vertices: {pinned_indices}")
    
    # Initialize field
    N = len(vertices)
    f_values = torch.randn(N, 6, device=device) * 0.1
    
    # Set pinned values
    for i, idx in enumerate(pinned_indices[:6]):
        f_values[idx] = 0.0
        f_values[idx, i] = 1.0
    
    # Make it a parameter
    f_values = torch.nn.Parameter(f_values)
    
    # Initialize optimizer with potentially adjusted learning rate for large meshes
    # Larger meshes have larger loss values, so we might need to adjust LR
    mesh_size_factor = np.sqrt(len(vertices) / 1000.0)  # Normalize to 1000-vertex mesh
    adjusted_lr = config['lr'] / mesh_size_factor
    
    print(f"Adjusted learning rate: {adjusted_lr:.6f} (mesh size factor: {mesh_size_factor:.2f})")
    
    optimizer = torch.optim.Adam([f_values], lr=adjusted_lr)
    
    # Training metrics
    loss_history = []
    best_loss = float('inf')
    best_iter = 0
    best_f_values = None
    
    # Training loop
    print(f"\nStarting training for {config['n_iters']} iterations...")
    print("="*70)
    
    start_time = time.time()
    
    for it in range(config['n_iters']):
        optimizer.zero_grad()
        
        # Get schedules
        schedules = mesh_loss.get_schedules(it, config['n_iters'])
        beta = schedules['beta']
        lambda_adj = schedules['lambda_adj']
        
        # Override with config if specified
        if 'beta_override' in config and config['beta_override'] is not None:
            beta = config['beta_override']
        if 'lambda_adj_override' in config and config['lambda_adj_override'] is not None:
            lambda_adj = config['lambda_adj_override']
        
        # Adaptive parameters for specific mesh types
        lambda_tv = config['lambda_tv']
        if config['mesh_type'] == 'angel' and lambda_tv == 0.05:
            lambda_tv = 0.01  # Angel needs lower TV
        
        # Compute loss (using original scaling)
        loss, components = mesh_loss.compute_loss(
            f_values,
            beta=beta,
            lambda_adj=lambda_adj,
            lambda_tv=lambda_tv,
            lambda_area=config['lambda_area']
        )
        
        # Record metrics
        loss_history.append({
            'iter': it,
            'total': loss.item(),
            'area': components['area'],
            'adj': components['adj'],
            'tv': components['tv'],
            'beta': beta,
            'lambda_adj': lambda_adj
        })
        
        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_iter = it
            best_f_values = f_values.detach().clone()
        
        # Print progress
        if it % config['print_every'] == 0:
            elapsed = time.time() - start_time
            eta = elapsed / (it + 1) * (config['n_iters'] - it - 1)
            
            print(f"Iter {it:6d}/{config['n_iters']} | "
                  f"Loss: {loss.item():8.4f} | "
                  f"Area: {components['area']:6.4f} | "
                  f"Adj: {components['adj']:8.4f} | "
                  f"TV: {components['tv']:6.4f} | "
                  f"β: {beta:4.1f} | "
                  f"λ_adj: {lambda_adj:4.1f} | "
                  f"Time: {elapsed/60:.1f}m | "
                  f"ETA: {eta/60:.1f}m")
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\nERROR: Loss became NaN at iteration {it}")
            break
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if config['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_([f_values], config['gradient_clip'])
        
        # Optimizer step
        optimizer.step()
        
        # Re-pin vertices
        with torch.no_grad():
            for i, idx in enumerate(pinned_indices[:6]):
                f_values[idx] = 0.0
                f_values[idx, i] = 1.0
    
    # Training complete
    elapsed = time.time() - start_time
    print("="*70)
    print(f"\nTraining completed in {elapsed/60:.1f} minutes")
    print(f"Best loss: {best_loss:.6f} at iteration {best_iter}")
    
    # Prepare results
    results = {
        'vertices': vertices,
        'faces': faces,
        'f_values': best_f_values.cpu().numpy(),
        'pinned_indices': pinned_indices,
        'loss_history': loss_history,
        'best_loss': best_loss,
        'best_iter': best_iter,
        'config': config,
        'training_time': elapsed,
        'timestamp': datetime.now().isoformat()
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Correct unified training script for ReLU mesh optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python train_unified_correct.py mesh.vtk
  
  # With custom learning rate for large meshes
  python train_unified_correct.py dragon.vtk --lr 0.002 --iters 100000
  
  # Specific mesh types
  python train_unified_correct.py kitty.vtk --mesh-type kitty
  python train_unified_correct.py angel.vtk --mesh-type angel
        """
    )
    
    # Required arguments
    parser.add_argument('mesh', type=str, help='Path to input mesh file')
    
    # Output options
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output file path (default: <input>_results.npz)')
    
    # Training parameters
    parser.add_argument('--iters', type=int, default=50000,
                        help='Number of training iterations (default: 50000)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Base learning rate (will be adjusted by mesh size) (default: 0.01)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    # Loss weights
    parser.add_argument('--lambda-area', type=float, default=1.0,
                        help='Area balance weight (default: 1.0)')
    parser.add_argument('--lambda-tv', type=float, default=0.05,
                        help='Total variation weight (default: 0.05)')
    
    # Advanced options
    parser.add_argument('--mesh-type', type=str, default='general',
                        choices=['general', 'sphere', 'kitty', 'angel'],
                        help='Mesh type for adaptive parameters (default: general)')
    parser.add_argument('--anchor-method', type=str, default='bbox',
                        choices=['bbox', 'pca'],
                        help='Anchor selection method (default: bbox)')
    parser.add_argument('--gradient-clip', type=float, default=5.0,
                        help='Gradient clipping value (default: 5.0)')
    parser.add_argument('--print-every', type=int, default=1000,
                        help='Print frequency (default: 1000)')
    
    # Manual overrides
    parser.add_argument('--beta-override', type=float, default=None,
                        help='Override beta schedule with fixed value')
    parser.add_argument('--lambda-adj-override', type=float, default=None,
                        help='Override lambda_adj schedule with fixed value')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.mesh))[0]
        args.output = f"{base}_results.npz"
    
    # Build config
    config = {
        'n_iters': args.iters,
        'lr': args.lr,
        'device': args.device,
        'lambda_area': args.lambda_area,
        'lambda_tv': args.lambda_tv,
        'mesh_type': args.mesh_type,
        'anchor_method': args.anchor_method,
        'gradient_clip': args.gradient_clip,
        'print_every': args.print_every,
        'beta_override': args.beta_override,
        'lambda_adj_override': args.lambda_adj_override,
    }
    
    # Print configuration
    print("\nTraining Configuration:")
    print(json.dumps(config, indent=2))
    
    # Run training
    try:
        results = train_relu_mesh(args.mesh, config)
        
        # Save results
        np.savez_compressed(args.output, **results)
        print(f"\nResults saved to: {args.output}")
        
        # Print summary
        print("\nTraining Summary:")
        print(f"  Final loss: {results['loss_history'][-1]['total']:.6f}")
        print(f"  Best loss: {results['best_loss']:.6f} at iteration {results['best_iter']}")
        print(f"  Area loss: {results['loss_history'][-1]['area']:.6f}")
        print(f"  Adjacent loss: {results['loss_history'][-1]['adj']:.6f}")
        print(f"  TV loss: {results['loss_history'][-1]['tv']:.6f}")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()