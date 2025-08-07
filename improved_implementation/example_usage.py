"""
Example usage of the improved ReLU-on-Meshes implementation.
Demonstrates training on both synthetic sphere and complex meshes.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path to import from existing codebase
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from improved_implementation import (
    MeshSegmentationTrainer,
    visualize_segmentation_pyvista,
    plot_training_history,
    save_segmentation_result
)

# Import mesh loading utilities from original codebase
from Piecewise_Linear_Mesh_3D.Meshsetup import (
    create_icosphere_mesh,
    load_volume_tet_mesh_and_extract_surface
)
from Piecewise_Linear_Mesh_3D.MeshParamCalculation import find_axis_vertices


def train_on_sphere():
    """Example 1: Train on synthetic sphere (should work well)."""
    print("=" * 60)
    print("Example 1: Training on Sphere Mesh")
    print("=" * 60)
    
    # Create sphere mesh
    print("Creating sphere mesh...")
    vertices_np, faces_np = create_icosphere_mesh(target_points=5000, radius=1.0)
    
    # Convert to torch tensors
    vertices = torch.tensor(vertices_np, dtype=torch.float32)
    faces = torch.tensor(faces_np, dtype=torch.int64)
    
    # Initialize trainer
    trainer = MeshSegmentationTrainer(
        verts=vertices,
        faces=faces,
        n_channels=6,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Find and set pinned vertices
    print("Setting pinned vertices...")
    pinned_indices = find_axis_vertices(vertices_np)
    trainer.set_pinned_vertices(torch.tensor(pinned_indices))
    
    # Train
    print("Starting training...")
    trainer.train(
        n_steps=50000,
        print_every=1000,
        save_every=10000,
        checkpoint_dir="checkpoints_sphere",
        initial_lr=1e-3,
        stage_transition=0.6,
        beta_update_every=400,
        hard_pin_at=0.9
    )
    
    # Get results
    f_values = trainer.get_field_values().cpu().numpy()
    
    # Save results
    save_segmentation_result(
        vertices_np, faces_np, f_values,
        "results_sphere.npz"
    )
    
    # Visualize
    print("Visualizing results...")
    visualize_segmentation_pyvista(
        vertices_np, faces_np, f_values,
        subdivisions=3,
        pinned_indices=pinned_indices,
        region_names=["Top", "Bottom", "Front", "Back", "Right", "Left"],
        screenshot_path="sphere_segmentation.png"
    )
    
    # Plot training history
    plot_training_history(trainer.history, "sphere_training_history.png")
    
    return trainer


def train_on_complex_mesh(mesh_name="kitty"):
    """Example 2: Train on complex mesh from dataset."""
    print("=" * 60)
    print(f"Example 2: Training on {mesh_name} Mesh")
    print("=" * 60)
    
    # Load mesh
    mesh_path = f"Piecewise Linear Mesh 3D/l1-poly-dat/hex/{mesh_name}/orig.tet.vtk"
    
    if not os.path.exists(mesh_path):
        print(f"Mesh file not found: {mesh_path}")
        print("Please ensure the mesh dataset is in the correct location.")
        return None
    
    print(f"Loading {mesh_name} mesh...")
    vertices_np, faces_np = load_volume_tet_mesh_and_extract_surface(mesh_path)
    
    # Convert to torch tensors
    vertices = torch.tensor(vertices_np, dtype=torch.float32)
    faces = torch.tensor(faces_np, dtype=torch.int64)
    
    print(f"Loaded mesh with {vertices.shape[0]} vertices and {faces.shape[0]} faces")
    
    # Initialize trainer
    trainer = MeshSegmentationTrainer(
        verts=vertices,
        faces=faces,
        n_channels=6,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Find and set pinned vertices
    print("Setting pinned vertices...")
    pinned_indices = find_axis_vertices(vertices_np)
    trainer.set_pinned_vertices(torch.tensor(pinned_indices))
    
    # Train with adjusted parameters for complex mesh
    print("Starting training...")
    trainer.train(
        n_steps=100000,  # More steps for complex mesh
        print_every=2000,
        save_every=20000,
        checkpoint_dir=f"checkpoints_{mesh_name}",
        initial_lr=5e-4,  # Lower learning rate for stability
        stage_transition=0.7,  # Longer Stage A
        beta_update_every=500,
        hard_pin_at=0.95,  # Delay hard pinning
        grad_clip=3.0  # Tighter gradient clipping
    )
    
    # Get results
    f_values = trainer.get_field_values().cpu().numpy()
    
    # Save results
    save_segmentation_result(
        vertices_np, faces_np, f_values,
        f"results_{mesh_name}.npz"
    )
    
    # Visualize
    print("Visualizing results...")
    visualize_segmentation_pyvista(
        vertices_np, faces_np, f_values,
        subdivisions=3,
        pinned_indices=pinned_indices,
        region_names=["Top", "Bottom", "Front", "Back", "Right", "Left"],
        screenshot_path=f"{mesh_name}_segmentation.png"
    )
    
    # Plot training history
    plot_training_history(trainer.history, f"{mesh_name}_training_history.png")
    
    return trainer


def analyze_convergence(trainer):
    """Analyze convergence behavior of the training."""
    print("\n" + "=" * 60)
    print("Convergence Analysis")
    print("=" * 60)
    
    history = trainer.history
    
    # Final losses
    print(f"Final total loss: {history['loss'][-1]:.6f}")
    print(f"Final smooth loss: {history['loss_smooth'][-1]:.6f}")
    print(f"Final contour loss: {history['loss_contour'][-1]:.6f}")
    print(f"Final area loss: {history['loss_area'][-1]:.6f}")
    
    # Temperature evolution
    print(f"\nFinal β_contour: {history['beta_contour'][-1]:.1f}")
    print(f"Final β_area: {history['beta_area'][-1]:.1f}")
    
    # Check if temperatures reached maximum
    temp_ctrl = trainer.temp_controller
    print(f"\nβ_contour reached max: {history['beta_contour'][-1] >= temp_ctrl.beta_contour_max}")
    print(f"β_area reached max: {history['beta_area'][-1] >= temp_ctrl.beta_area_max}")
    
    # Area balance
    f_values = trainer.get_field_values()
    with torch.no_grad():
        # Compute final area fractions
        from improved_implementation.losses import area_fractions_and_kl
        _, frac = area_fractions_and_kl(
            f_values,
            trainer.faces,
            trainer.mesh_data['tri_area'],
            beta_area=temp_ctrl.beta_area
        )
        
        uniform = 1.0 / trainer.n_channels
        max_deviation = (frac - uniform).abs().max().item()
        
        print(f"\nArea fractions: {frac.cpu().numpy()}")
        print(f"Max deviation from uniform: {max_deviation:.4f}")
        print(f"Area balanced: {max_deviation < temp_ctrl.tau_area}")


def main():
    """Run examples."""
    # Example 1: Sphere (should converge well)
    sphere_trainer = train_on_sphere()
    if sphere_trainer:
        analyze_convergence(sphere_trainer)
    
    # Example 2: Complex mesh
    # Try different meshes: "kitty", "rod", "angel_1", "bunny", etc.
    complex_trainer = train_on_complex_mesh("kitty")
    if complex_trainer:
        analyze_convergence(complex_trainer)
    
    print("\n" + "=" * 60)
    print("Summary of Improvements")
    print("=" * 60)
    print("1. Replaced global 3D plane fitting with local 2D triangle alignment")
    print("2. Added cotangent Laplacian for proper smoothness")
    print("3. Progress-gated temperature scheduling")
    print("4. KL divergence for area balance (maintains gradients)")
    print("5. Soft pinning with annealing")
    print("6. Two-stage optimization strategy")
    print("\nThese changes address the convergence issues identified in your report:")
    print("- SVD instability (Section 4.5.11)")
    print("- Non-planar boundaries on complex meshes (Section 4.6)")
    print("- Gradient vanishing with high beta (Section 4.7.4)")
    print("- Interference between loss terms")


if __name__ == "__main__":
    main()