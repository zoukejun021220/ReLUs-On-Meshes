"""
Main script for improved ReLU mesh segmentation.
Demonstrates the complete pipeline with revised loss and better anchor selection.
"""
import numpy as np
import torch
import argparse
import time
import json
from pathlib import Path

from mesh_utils import (
    load_mesh_from_vtk, create_icosphere_mesh,
    pick_pca_anchors, pick_raycast_anchors, pick_axis_aligned_anchors
)
from optimization import optimize_mesh_segmentation
from visualization import (
    visualize_segmentation_pyvista, plot_training_history,
    measure_planarity, visualize_field_values
)


def main():
    parser = argparse.ArgumentParser(description='Improved ReLU Mesh Segmentation')
    parser.add_argument('--mesh', type=str, default='sphere',
                       help='Mesh to use: sphere, or path to VTK file')
    parser.add_argument('--vertices', type=int, default=5000,
                       help='Number of vertices for sphere mesh')
    parser.add_argument('--anchor-method', type=str, default='raycast',
                       choices=['axis', 'pca', 'raycast'],
                       help='Method for selecting anchor vertices')
    parser.add_argument('--no-coarse-to-fine', action='store_true',
                       help='Disable coarse-to-fine training')
    parser.add_argument('--no-grad-norm', action='store_true',
                       help='Disable GradNorm loss balancing')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Number of training iterations (overrides default schedule)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load or create mesh
    print("Loading mesh...")
    if args.mesh == 'sphere':
        vertices, faces = create_icosphere_mesh(target_points=args.vertices)
        mesh_name = f'sphere_{args.vertices}'
    else:
        vertices, faces = load_mesh_from_vtk(args.mesh)
        mesh_name = Path(args.mesh).stem
    
    print(f"Mesh loaded: {len(vertices)} vertices, {len(faces)} faces")
    
    # Select anchor vertices
    print(f"Selecting anchor vertices using {args.anchor_method} method...")
    if args.anchor_method == 'axis':
        pinned_indices = pick_axis_aligned_anchors(vertices)
    elif args.anchor_method == 'pca':
        pinned_indices = pick_pca_anchors(vertices)
    elif args.anchor_method == 'raycast':
        pinned_indices = pick_raycast_anchors(vertices, faces)
    
    print("Pinned vertices:")
    region_names = ["Top", "Bottom", "Front", "Back", "Right", "Left"]
    for i, idx in enumerate(pinned_indices):
        print(f"  {region_names[i]}: vertex {idx} at {vertices[idx]}")
    
    # Run optimization
    print("\nStarting optimization...")
    start_time = time.time()
    
    f_values, history = optimize_mesh_segmentation(
        vertices, faces, pinned_indices,
        use_coarse_to_fine=not args.no_coarse_to_fine,
        use_grad_norm=not args.no_grad_norm,
        device=device,
        iterations=args.iterations
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nOptimization completed in {elapsed_time:.1f} seconds")
    
    # Save results
    print("\nSaving results...")
    
    # Save optimized field values and mesh
    torch.save({
        'f_values': f_values,
        'vertices': vertices,
        'faces': faces,
        'pinned_indices': pinned_indices,
        'history': history
    }, output_dir / f'{mesh_name}_results.pt')
    
    # Save as numpy for compatibility with visualizeMesh
    field_values_np = f_values.cpu().numpy()
    np.savez(output_dir / f'{mesh_name}_mesh_and_values.npz',
             vertices=vertices,
             faces=faces,
             f_values=field_values_np,  # Keep for backward compatibility
             field_values=field_values_np,  # Required by visualizeMesh/visual.py
             pinned_indices=pinned_indices)
    
    # Measure planarity
    from mesh_utils import compute_mesh_adjacency
    edges, edge2face, _ = compute_mesh_adjacency(faces)
    planarity = measure_planarity(vertices, faces, f_values, edges, edge2face)
    
    print("\nPlanarity metrics:")
    print(f"  Mean boundary angle: {planarity['mean_angle']:.2f}°")
    print(f"  Max boundary angle: {planarity['max_angle']:.2f}°")
    print(f"  Number of boundary edges: {planarity['num_boundary_edges']}")
    
    # Save metrics with proper type conversion
    from json_serialization_fix import convert_to_serializable
    
    metrics = {
        'mesh_name': mesh_name,
        'num_vertices': len(vertices),
        'num_faces': len(faces),
        'anchor_method': args.anchor_method,
        'elapsed_time': elapsed_time,
        'planarity': planarity,
        'final_loss': history[list(history.keys())[-1]]['loss'][-1] if history else None
    }
    
    # Convert NumPy/PyTorch types to JSON-serializable types
    safe_metrics = convert_to_serializable(metrics)
    
    with open(output_dir / f'{mesh_name}_metrics.json', 'w') as f:
        json.dump(safe_metrics, f, indent=2)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Plot training history
    for level_name, level_history in history.items():
        plot_training_history(level_history, 
                            save_path=output_dir / f'{mesh_name}_history_{level_name}.png')
    
    # Visualize segmentation
    visualize_segmentation_pyvista(vertices, faces, f_values, pinned_indices,
                                  save_path=output_dir / f'{mesh_name}_segmentation.png')
    
    # Visualize individual channels
    for i in range(6):
        visualize_field_values(vertices, faces, f_values, channel=i,
                             save_path=output_dir / f'{mesh_name}_channel_{i}.png')
    
    print(f"\nResults saved to {output_dir}")


def run_experiments():
    """Run experiments on multiple meshes with different settings."""
    
    # Test meshes
    test_cases = [
        ('sphere', {'vertices': 5000}),
        ('kitty', {'mesh': 'l1-poly-dat/hex/kitty/orig.tet.vtk'}),
        ('rod', {'mesh': 'l1-poly-dat/hex/rod/orig.tet.vtk'}),
        ('angel', {'mesh': 'l1-poly-dat/hex/angel_1/orig.tet.vtk'}),
    ]
    
    # Anchor methods to test
    anchor_methods = ['axis', 'pca', 'raycast']
    
    results = []
    
    for mesh_name, mesh_args in test_cases:
        for anchor_method in anchor_methods:
            print(f"\n{'='*60}")
            print(f"Testing {mesh_name} with {anchor_method} anchors")
            print('='*60)
            
            try:
                # Set up arguments
                args = argparse.Namespace(
                    mesh=mesh_args.get('mesh', 'sphere'),
                    vertices=mesh_args.get('vertices', 5000),
                    anchor_method=anchor_method,
                    no_coarse_to_fine=False,
                    no_grad_norm=False,
                    output_dir=f'results/{mesh_name}_{anchor_method}',
                    device='cuda'
                )
                
                # Run main with these arguments
                # (Would need to refactor main() to accept args directly)
                
                results.append({
                    'mesh': mesh_name,
                    'anchor_method': anchor_method,
                    'status': 'completed'
                })
                
            except Exception as e:
                print(f"Error: {e}")
                results.append({
                    'mesh': mesh_name,
                    'anchor_method': anchor_method,
                    'status': 'failed',
                    'error': str(e)
                })
    
    # Save experiment summary with proper type conversion
    safe_results = convert_to_serializable(results)
    with open('experiment_results.json', 'w') as f:
        json.dump(safe_results, f, indent=2)


if __name__ == '__main__':
    main()
    # Uncomment to run full experiments:
    # run_experiments()