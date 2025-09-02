"""
Visualization utilities for improved mesh segmentation results.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List
import json

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("PyVista not available. Using matplotlib fallback for visualization.")


def subdivide_mesh_with_values(vertices: np.ndarray,
                              faces: np.ndarray,
                              f_values: np.ndarray,
                              num_subdivisions: int = 2) -> tuple:
    """
    Subdivide mesh for smoother visualization of segmentation boundaries.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (T, 3) face indices
        f_values: (N, C) field values
        num_subdivisions: Number of subdivision iterations
        
    Returns:
        new_vertices: Subdivided vertices
        new_faces: Subdivided faces
        new_f_values: Interpolated field values
    """
    current_vertices = vertices.tolist()
    current_faces = faces.tolist()
    current_fvalues = f_values.tolist()
    
    def subdivide_once(verts, facs, fvals):
        new_verts = list(verts)
        new_fvals = list(fvals)
        new_faces = []
        edge_to_mid = {}
        
        for tri in facs:
            i1, i2, i3 = tri
            
            edges = [
                (min(i1, i2), max(i1, i2)),
                (min(i2, i3), max(i2, i3)),
                (min(i3, i1), max(i3, i1))
            ]
            
            for e in edges:
                if e not in edge_to_mid:
                    # Midpoint position
                    vA = np.array(verts[e[0]], dtype=np.float32)
                    vB = np.array(verts[e[1]], dtype=np.float32)
                    mid_pos = 0.5 * (vA + vB)
                    
                    # Midpoint field value
                    fA = np.array(fvals[e[0]], dtype=np.float32)
                    fB = np.array(fvals[e[1]], dtype=np.float32)
                    mid_f = 0.5 * (fA + fB)
                    
                    new_idx = len(new_verts)
                    new_verts.append(mid_pos.tolist())
                    new_fvals.append(mid_f.tolist())
                    
                    edge_to_mid[e] = new_idx
        
        # Create 4 new faces per old face
        for tri in facs:
            i1, i2, i3 = tri
            e1 = (min(i1, i2), max(i1, i2))
            e2 = (min(i2, i3), max(i2, i3))
            e3 = (min(i3, i1), max(i3, i1))
            
            a = edge_to_mid[e1]
            b = edge_to_mid[e2]
            c = edge_to_mid[e3]
            
            new_faces.append([i1, a, c])
            new_faces.append([i2, b, a])
            new_faces.append([i3, c, b])
            new_faces.append([a, b, c])
        
        return new_verts, new_faces, new_fvals
    
    # Repeatedly subdivide
    for _ in range(num_subdivisions):
        current_vertices, current_faces, current_fvalues = subdivide_once(
            current_vertices, current_faces, current_fvalues
        )
    
    new_vertices = np.array(current_vertices, dtype=np.float32)
    new_faces = np.array(current_faces, dtype=np.int32)
    new_f_values = np.array(current_fvalues, dtype=np.float32)
    
    return new_vertices, new_faces, new_f_values


def visualize_segmentation_pyvista(vertices: np.ndarray,
                                  faces: np.ndarray,
                                  f_values: np.ndarray,
                                  pinned_indices: Optional[np.ndarray] = None,
                                  subdivisions: int = 2,
                                  save_path: Optional[Path] = None):
    """
    Visualize segmentation using PyVista.
    
    Args:
        vertices: (N, 3) vertex positions
        faces: (T, 3) face indices
        f_values: (N, C) field values
        pinned_indices: Optional pinned vertex indices
        subdivisions: Number of mesh subdivisions for smoother boundaries
        save_path: Optional path to save screenshot
    """
    if not HAS_PYVISTA:
        print("PyVista not available. Skipping 3D visualization.")
        return
    
    # Subdivide for smoother boundaries
    if subdivisions > 0:
        sub_verts, sub_faces, sub_fvals = subdivide_mesh_with_values(
            vertices, faces, f_values, subdivisions
        )
    else:
        sub_verts, sub_faces, sub_fvals = vertices, faces, f_values
    
    # Compute hard labels via argmax
    hard_labels = np.argmax(sub_fvals, axis=1)
    
    # Create PyVista mesh
    faces_flat = np.column_stack((np.full(len(sub_faces), 3), sub_faces)).flatten()
    mesh = pv.PolyData(sub_verts, faces_flat)
    mesh["Region"] = hard_labels
    
    # Set up plotter
    plotter = pv.Plotter(window_size=(1200, 800))
    plotter.add_text("Improved ReLU Mesh Segmentation", font_size=14)
    
    # Color map for regions
    colors = [
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
        [0.0, 1.0, 0.0],  # Green
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
    ]
    
    # Add mesh
    plotter.add_mesh(
        mesh,
        scalars="Region",
        show_edges=False,
        cmap=colors[:f_values.shape[1]],
        interpolate_before_map=False,
        show_scalar_bar=True,
        scalar_bar_args={'title': 'Region', 'n_labels': f_values.shape[1]}
    )
    
    # Mark pinned vertices
    if pinned_indices is not None:
        for i, idx in enumerate(pinned_indices):
            if idx < len(vertices):
                pos = vertices[idx]
                plotter.add_points(
                    pos.reshape(1, 3),
                    color=colors[i % len(colors)],
                    point_size=20,
                    render_points_as_spheres=True
                )
    
    plotter.view_isometric()
    
    if save_path:
        plotter.screenshot(save_path)
    
    plotter.show()


def plot_training_history(history: Dict[str, List], save_path: Optional[Path] = None):
    """
    Plot training history curves.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Total loss
    ax = axes[0]
    ax.semilogy(history['loss'], label='Total Loss')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Component losses
    ax = axes[1]
    ax.semilogy(history['loss_smooth'], label='Smooth', alpha=0.7)
    ax.semilogy(history['loss_contour'], label='Contour', alpha=0.7)
    ax.semilogy(history['loss_area'], label='Area', alpha=0.7)
    ax.semilogy(history['loss_pin'], label='Pin', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Component Losses')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Area fractions
    ax = axes[2]
    area_fracs = np.array(history['area_fractions'])
    if area_fracs.ndim == 2:
        for i in range(area_fracs.shape[1]):
            ax.plot(area_fracs[:, i], label=f'Channel {i}', alpha=0.7)
    ax.axhline(1.0/6, color='k', linestyle='--', alpha=0.5, label='Uniform')
    ax.set_xlabel('Step')
    ax.set_ylabel('Area Fraction')
    ax.set_title('Area Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Temperature evolution
    ax = axes[3]
    ax.plot(history['beta_contour'], label='β_contour')
    ax.plot(history['beta_area'], label='β_area')
    ax.set_xlabel('Step')
    ax.set_ylabel('Temperature')
    ax.set_title('Temperature Schedule')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Learning rate
    ax = axes[4]
    ax.semilogy(history['lr'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    # Boundary length (if available)
    ax = axes[5]
    if 'boundary_length' in history and history['boundary_length']:
        ax.plot(history['boundary_length'])
        ax.set_xlabel('Step')
        ax.set_ylabel('Boundary Length')
        ax.set_title('Estimated Boundary Length')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No boundary length data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Boundary Length')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def analyze_convergence(history: Dict[str, List], window: int = 1000) -> Dict[str, float]:
    """
    Analyze convergence metrics from training history.
    
    Args:
        history: Training history
        window: Window size for computing statistics
        
    Returns:
        metrics: Dictionary of convergence metrics
    """
    metrics = {}
    
    # Final loss values
    metrics['final_loss'] = history['loss'][-1]
    metrics['final_loss_smooth'] = history['loss_smooth'][-1]
    metrics['final_loss_contour'] = history['loss_contour'][-1]
    metrics['final_loss_area'] = history['loss_area'][-1]
    
    # Loss reduction
    if len(history['loss']) > window:
        early_loss = np.mean(history['loss'][:window])
        late_loss = np.mean(history['loss'][-window:])
        metrics['loss_reduction'] = (early_loss - late_loss) / early_loss
    
    # Area balance quality
    final_areas = history['area_fractions'][-1]
    uniform = 1.0 / len(final_areas)
    metrics['area_deviation'] = np.abs(final_areas - uniform).max()
    metrics['area_variance'] = np.var(final_areas)
    
    # Temperature progression
    metrics['final_beta_contour'] = history['beta_contour'][-1]
    metrics['final_beta_area'] = history['beta_area'][-1]
    
    # Convergence stability (loss variance in final steps)
    if len(history['loss']) > window:
        metrics['final_loss_std'] = np.std(history['loss'][-window:])
    
    return metrics


def load_and_visualize(result_path: Path):
    """
    Load and visualize results from a saved experiment.
    
    Args:
        result_path: Path to results directory
    """
    # Load results
    data = np.load(result_path / 'final_result.npz')
    vertices = data['vertices']
    faces = data['faces']
    field_values = data['field_values']
    pinned_indices = data.get('pinned_indices', None)
    
    # Load history
    with open(result_path / 'training_history.json', 'r') as f:
        history = json.load(f)
    
    # Visualize segmentation
    print("Visualizing segmentation...")
    visualize_segmentation_pyvista(
        vertices, faces, field_values, pinned_indices,
        subdivisions=2,
        save_path=result_path / 'segmentation.png'
    )
    
    # Plot training curves
    print("Plotting training history...")
    plot_training_history(history, save_path=result_path / 'training_curves.png')
    
    # Analyze convergence
    print("\nConvergence Analysis:")
    metrics = analyze_convergence(history)
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    
    # Save metrics
    with open(result_path / 'convergence_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize mesh segmentation results')
    parser.add_argument('result_dir', type=str, help='Path to results directory')
    
    args = parser.parse_args()
    
    result_path = Path(args.result_dir)
    if not result_path.exists():
        print(f"Error: {result_path} does not exist")
        exit(1)
    
    load_and_visualize(result_path)