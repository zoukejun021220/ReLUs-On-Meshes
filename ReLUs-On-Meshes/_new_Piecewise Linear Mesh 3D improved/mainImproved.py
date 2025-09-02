import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse

from Meshsetup import create_icosphere_mesh, load_volume_tet_mesh_and_extract_surface
from MeshParamCalculationImproved import find_axis_vertices_improved
from optimizationImproved import optimization_improved
from SinOptimizationImproved import optimization_sin_improved


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run mesh optimization')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization (requires display)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint file to resume from (without extension)')
    parser.add_argument(
        '--loss',
        type=str,
        default='anchored',
        choices=['anchored', 'soft_pairs', 'free_planes', 'pairwise_planes', 'codex'],
        help='Select contour loss: anchored | soft_pairs | free_planes | pairwise_planes | codex'
    )
    args = parser.parse_args()
    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Configuration
    use_improved_optimizer = True  # Set to False to use sinusoidal optimizer
    use_pca_axes = True            # Set to False to use world axes
    # Loss selection via CLI
    use_soft_pairs = (args.loss == 'soft_pairs')
    use_free_planes = (args.loss == 'free_planes')
    use_pairwise_planes = (args.loss == 'pairwise_planes')
    use_codex_loss = (args.loss == 'codex')  # Codex intrinsic 3D gradient-alignment loss
    use_svd_init_after_warmup = True  # Reinitialize planes with SVD after warmup (where applicable)
    
    # Step 1: Load the mesh
    print("Loading mesh...")
    start_time = time.time()
    
    # Choose one:
    # Option 1: Create sphere mesh
    # target_points = 5000
    # vertices_np, faces_np = create_icosphere_mesh(target_points=target_points, radius=1.0)
    
    # Option 2: Load from VTK file
    input_filename = "/home/kejunzou/Projects/ReLUs on Meshes/ReLUs-On-Meshes/_codex_Piecewise Linear Mesh 3D improved/l1-poly-dat/hex/canewt/orig.tet.vtk"
    vertices_np, faces_np = load_volume_tet_mesh_and_extract_surface(input_filename)
    
    elapsed = time.time() - start_time
    print(f"Loaded mesh in {elapsed:.2f}s with {len(vertices_np)} vertices and {len(faces_np)} faces")
    
    # Step 2: Find improved anchor points using PCA/support points
    print("\nFinding anchor vertices with improved method...")
    start_time = time.time()
    
    pinned_indices, pinned_axes = find_axis_vertices_improved(
        vertices_np, faces_np, use_pca=use_pca_axes
    )
    
    if use_free_planes:
        print("Using free planes - plane orientations will be learned during optimization")
        print("(Anchor points are still used for pinning constraints)")
    elif use_pairwise_planes:
        print("Using channel-pairwise planes - one plane per channel pair will be learned")
        print("(Anchor points are still used for pinning constraints)")
    
    # Convert to torch tensor
    pinned_axes_torch = torch.tensor(pinned_axes, dtype=torch.float32).to(device)
    
    region_names = ["Top", "Bottom", "Front", "Back", "Right", "Left"]
    elapsed = time.time() - start_time
    print(f"Found anchor vertices in {elapsed:.2f}s")
    
    print("Pinning vertices for 6 regions:")
    for i, (name, idx) in enumerate(zip(region_names, pinned_indices)):
        print(f"  {name}: vertex {idx} at position {vertices_np[idx]}")
    print(f"Using {'PCA' if use_pca_axes else 'world'} axes for plane normals")
    
    # Step 3: Optimize with improved methods
    print("\nStarting optimization with improved methods...")
    
    # Ensure exclusivity if Codex loss is selected
    if use_codex_loss:
        use_pairwise_planes = False
        use_free_planes = False
        use_soft_pairs = False

    optimization_params = {
        'vertices_np': vertices_np,
        'faces_np': faces_np,
        'pinned_indices': pinned_indices,
        'pinned_axes': pinned_axes,
        'n_iters': 100000,
        'warmup_iters': 5000,
        'beta_initial': 3.0,
        'beta_warmup': 3.0,
        'beta_final': 100.0,
        'lambda_contour_initial': 0.0,
        'lambda_contour_warmup': 0.01,
        'lambda_contour_final': 30.0,
        'lambda_smooth': 1,
        'lambda_area_initial': 2,
        'lambda_area_final': 20.0,
        'enable_early_stopping': False,
        'patience': 2000,
        'print_every': 1000,
        'use_anchored_loss': not use_soft_pairs and not use_free_planes and not use_pairwise_planes and not use_codex_loss,  # Exclude when using Codex loss
        'use_soft_pairs_loss': use_soft_pairs,    # Use soft pairs loss for stable triple points
        'use_free_planes_loss': use_free_planes,  # Use free planes with learnable normals
        'use_pairwise_planes_loss': use_pairwise_planes,  # Use channel-pairwise planes
        'use_svd_init_after_warmup': use_svd_init_after_warmup,  # Reinit planes after warmup
        'checkpoint_dir': 'checkpoints',
        'checkpoint_interval': 5000,
        'input_filename': input_filename,
        'resume_checkpoint': args.resume,
    }
    
    if use_improved_optimizer:
        # Use shock-therapy style optimizer
        if use_codex_loss:
            loss_type = "codex-grad-alignment"
        elif use_pairwise_planes:
            loss_type = "channel-pairwise planes"
        elif use_free_planes:
            loss_type = "free planes"
        elif use_soft_pairs:
            loss_type = "soft pairs"
        else:
            loss_type = "anchored planes"
        print(f"Using improved shock-therapy optimizer with {loss_type} loss")
        f_optimized, mesh, loss_history, savepath = optimization_improved(
            **optimization_params,
            shock_steps=1000,
            refine_steps=4000,
            shock_lr=1e-3,
            refine_lr=1e-4,
            use_codex_grad_alignment_loss=use_codex_loss
        )
    else:
        # Use sinusoidal optimizer
        if use_codex_loss:
            loss_type = "codex-grad-alignment"
        elif use_pairwise_planes:
            loss_type = "channel-pairwise planes"
        elif use_free_planes:
            loss_type = "free planes"
        elif use_soft_pairs:
            loss_type = "soft pairs"
        else:
            loss_type = "anchored planes"
        print(f"Using sinusoidal optimizer with {loss_type} loss")
        f_optimized, mesh, loss_history, savepath = optimization_sin_improved(
            **optimization_params,
            lr=2e-3,
            num_phases=3,
            lr_min_factor=0.1,
            lr_max_factor=1.0,
            phase_shift=0.0,
            decay_factor=0.5,
        )
    
    # Step 4: Visualize the result
    if args.visualize:
        print("\nVisualizing result...")
        try:
            import pyvista as pv
            import vtk
            from visualization import visualize_segmentation
            visualize_segmentation(
                vertices_np=vertices_np,
                faces_np=faces_np,
                f_values=f_optimized,
                pinned_indices=pinned_indices,
                region_names=region_names,
                subdivisions=4
            )
        except ImportError as e:
            print(f"Warning: Could not import visualization libraries: {e}")
            print("Skipping visualization.")
    else:
        print("\nVisualization skipped (use --visualize flag to enable)")
    
    # Print summary statistics
    print("\nOptimization Summary:")
    print(f"Final loss: {loss_history[-1]['total']:.6f}")
    print(f"Contour loss: {loss_history[-1]['contour']:.6f}")
    print(f"Smoothness loss: {loss_history[-1]['smoothness']:.6f}")
    print(f"Area balance loss: {loss_history[-1]['area_balance']:.6f}")
    
    if 'plane_offsets' in loss_history[-1]:
        print(f"Final plane offsets: {loss_history[-1]['plane_offsets']}")
    
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal execution time: {total_elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
