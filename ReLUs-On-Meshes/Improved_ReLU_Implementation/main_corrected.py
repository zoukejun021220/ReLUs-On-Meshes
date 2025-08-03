"""
CORRECTED main script for ReLU mesh segmentation.
Uses properly implemented loss functions based on paper audit.
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
from optimization_corrected import optimize_mesh_segmentation_corrected
from visualization import (
    visualize_segmentation_pyvista, plot_training_history,
    measure_planarity, visualize_field_values
)


def main():
    parser = argparse.ArgumentParser(description='CORRECTED ReLU Mesh Segmentation')
    parser.add_argument('--mesh', type=str, default='sphere',
                       help='Mesh to use: sphere, or path to VTK file')
    parser.add_argument('--vertices', type=int, default=5000,
                       help='Number of vertices for sphere mesh')
    parser.add_argument('--anchor-method', type=str, default='raycast',
                       choices=['axis', 'pca', 'raycast'],
                       help='Method for selecting anchor vertices')
    parser.add_argument('--no-coarse-to-fine', action='store_true',
                       help='Disable coarse-to-fine training')
    parser.add_argument('--output-dir', type=str, default='results_corrected',
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
    print("\n" + "="*60)
    print("USING CORRECTED LOSS FUNCTIONS")
    print("Expected behavior:")
    print("- Weight sum (w_e) should drop from ~10,000 to ~100")
    print("- Raw adjacency loss should decrease to near 0")
    print("- Boundary angles should converge to <2°")
    print("="*60 + "\n")
    
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
    print("\nStarting CORRECTED optimization...")
    start_time = time.time()
    
    f_values, history = optimize_mesh_segmentation_corrected(
        vertices, faces, pinned_indices,
        use_coarse_to_fine=not args.no_coarse_to_fine,
        device=device,
        iterations=args.iterations
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nOptimization completed in {elapsed_time:.1f} seconds")
    
    # Analyze final weight sum
    if history:
        final_history = history.get('direct', history.get('level_2', {}))
        if 'weight_sum' in final_history:
            initial_weight_sum = final_history['weight_sum'][0] if final_history['weight_sum'] else 0
            final_weight_sum = final_history['weight_sum'][-1] if final_history['weight_sum'] else 0
            print(f"\nWeight sum reduction: {initial_weight_sum:.1f} -> {final_weight_sum:.1f}")
            print(f"Reduction factor: {initial_weight_sum/max(final_weight_sum, 1):.1f}x")
    
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
    field_values_np = f_values.cpu().detach().numpy()
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
    
    if planarity['mean_angle'] < 2.0:
        print("  ✅ SUCCESS: Mean angle < 2° achieved!")
    else:
        print(f"  ⚠️  Mean angle is {planarity['mean_angle']:.1f}° (target < 2°)")
    
    # Save metrics
    from json_serialization_fix import convert_to_serializable
    
    metrics = {
        'mesh_name': mesh_name,
        'num_vertices': len(vertices),
        'num_faces': len(faces),
        'anchor_method': args.anchor_method,
        'elapsed_time': elapsed_time,
        'planarity': planarity,
        'final_loss': history[list(history.keys())[-1]]['loss'][-1] if history else None,
        'corrected_implementation': True  # Flag to identify this used corrected version
    }
    
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
    print("\nDone! Check the weight_sum history and planarity metrics.")


if __name__ == '__main__':
    main()