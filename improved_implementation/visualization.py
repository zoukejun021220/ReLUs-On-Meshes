"""
Visualization utilities for mesh segmentation results.
Includes mesh subdivision for smooth boundary visualization.
"""

import numpy as np
import torch
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

Tensor = torch.Tensor


def subdivide_mesh_with_values(
    vertices: np.ndarray,
    faces: np.ndarray,
    f_values: np.ndarray,
    num_subdivisions: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subdivide mesh and interpolate field values for smoother visualization.
    
    Args:
        vertices: (N, 3) - vertex positions
        faces: (T, 3) - triangle indices
        f_values: (N, C) - multi-channel field values
        num_subdivisions: number of subdivision iterations
        
    Returns:
        new_vertices: (N', 3) - subdivided vertices
        new_faces: (T', 3) - subdivided faces
        new_f_values: (N', C) - interpolated field values
    """
    # Convert to lists for easier manipulation
    current_vertices = vertices.tolist()
    current_faces = faces.tolist()
    current_fvalues = f_values.tolist()
    
    def subdivide_once(verts, facs, fvals):
        """Perform one round of subdivision."""
        new_verts = list(verts)
        new_fvals = list(fvals)
        new_faces = []
        edge_to_mid = {}
        
        # Process each triangle
        for tri in facs:
            i1, i2, i3 = tri
            
            edges = [
                (min(i1, i2), max(i1, i2)),
                (min(i2, i3), max(i2, i3)),
                (min(i3, i1), max(i3, i1))
            ]
            
            # Create midpoints for each edge
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
                    
                    # Add new vertex
                    new_idx = len(new_verts)
                    new_verts.append(mid_pos.tolist())
                    new_fvals.append(mid_f.tolist())
                    
                    edge_to_mid[e] = new_idx
            
            # Create 4 new triangles
            a = edge_to_mid[(min(i1, i2), max(i1, i2))]
            b = edge_to_mid[(min(i2, i3), max(i2, i3))]
            c = edge_to_mid[(min(i3, i1), max(i3, i1))]
            
            new_faces.append([i1, a, c])
            new_faces.append([i2, b, a])
            new_faces.append([i3, c, b])
            new_faces.append([a, b, c])
        
        return new_verts, new_faces, new_fvals
    
    # Apply subdivisions
    for _ in range(num_subdivisions):
        current_vertices, current_faces, current_fvalues = subdivide_once(
            current_vertices, current_faces, current_fvalues
        )
    
    # Convert back to numpy
    new_vertices = np.array(current_vertices, dtype=np.float32)
    new_faces = np.array(current_faces, dtype=np.int32)
    new_f_values = np.array(current_fvalues, dtype=np.float32)
    
    return new_vertices, new_faces, new_f_values


def visualize_segmentation_pyvista(
    vertices: np.ndarray,
    faces: np.ndarray,
    f_values: np.ndarray,
    subdivisions: int = 2,
    pinned_indices: Optional[List[int]] = None,
    region_names: Optional[List[str]] = None,
    screenshot_path: Optional[str] = None
):
    """
    Visualize segmentation using PyVista (if available).
    
    Args:
        vertices: (N, 3) - vertex positions
        faces: (T, 3) - triangle indices
        f_values: (N, C) - field values
        subdivisions: subdivision level for smooth boundaries
        pinned_indices: optional pinned vertex indices
        region_names: optional region names
        screenshot_path: optional path to save screenshot
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not available, falling back to matplotlib")
        return visualize_segmentation_matplotlib(
            vertices, faces, f_values, subdivisions
        )
    
    # Subdivide for smoother visualization
    if subdivisions > 0:
        sub_vertices, sub_faces, sub_fvals = subdivide_mesh_with_values(
            vertices, faces, f_values, num_subdivisions=subdivisions
        )
    else:
        sub_vertices = vertices
        sub_faces = faces
        sub_fvals = f_values
    
    # Get hard labels via argmax
    hard_labels = np.argmax(sub_fvals, axis=1)
    
    # Define colors for regions
    region_colors = np.array([
        [1.0, 0.0, 0.0, 1.0],  # Red
        [0.0, 0.0, 1.0, 1.0],  # Blue
        [0.0, 1.0, 0.0, 1.0],  # Green
        [1.0, 1.0, 0.0, 1.0],  # Yellow
        [1.0, 0.0, 1.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0, 1.0],  # Cyan
        [1.0, 0.5, 0.0, 1.0],  # Orange
        [0.5, 0.0, 1.0, 1.0],  # Purple
    ])
    
    # Create colormap
    n_channels = f_values.shape[1]
    region_cmap = ListedColormap(region_colors[:n_channels])
    
    # Build PyVista mesh
    faces_flat = np.column_stack((
        np.full(len(sub_faces), 3), sub_faces
    )).flatten()
    mesh = pv.PolyData(sub_vertices, faces_flat)
    mesh["Labels"] = hard_labels + 1  # 1-indexed for display
    
    # Create plotter
    pv.set_plot_theme("document")
    plotter = pv.Plotter()
    plotter.add_text(
        f"Mesh Segmentation (subdivisions={subdivisions})",
        font_size=14,
        position='upper_edge'
    )
    
    # Add mesh
    plotter.add_mesh(
        mesh,
        scalars="Labels",
        show_edges=False,
        cmap=region_cmap,
        interpolate_before_map=False,
        show_scalar_bar=True,
        clim=[1, n_channels]
    )
    
    # Add scalar bar
    plotter.add_scalar_bar(
        title="Region",
        n_labels=n_channels,
        fmt="%d",
        font_family="arial"
    )
    
    # Mark pinned vertices if provided
    if pinned_indices is not None:
        for i, vidx in enumerate(pinned_indices):
            if vidx < vertices.shape[0]:
                pin_pos = vertices[vidx]
                color = region_colors[i % len(region_colors)][:3]
                plotter.add_points(
                    pin_pos.reshape(1, 3),
                    color=color,
                    point_size=15
                )
                
                if region_names and i < len(region_names):
                    offset_pos = pin_pos * 1.02
                    plotter.add_point_labels(
                        [offset_pos],
                        [region_names[i]],
                        font_size=10,
                        text_color=color,
                        shape=None
                    )
    
    # Set view
    plotter.view_isometric()
    
    # Save screenshot if requested
    if screenshot_path:
        plotter.screenshot(screenshot_path)
        print(f"Saved screenshot to {screenshot_path}")
    
    plotter.show()


def visualize_segmentation_matplotlib(
    vertices: np.ndarray,
    faces: np.ndarray,
    f_values: np.ndarray,
    subdivisions: int = 0
):
    """
    Fallback visualization using matplotlib 3D scatter plot.
    
    Args:
        vertices: (N, 3) - vertex positions
        faces: (T, 3) - triangle indices
        f_values: (N, C) - field values
        subdivisions: subdivision level (limited for matplotlib)
    """
    # Limit subdivisions for matplotlib
    subdivisions = min(subdivisions, 1)
    
    if subdivisions > 0:
        vertices, faces, f_values = subdivide_mesh_with_values(
            vertices, faces, f_values, num_subdivisions=subdivisions
        )
    
    # Get hard labels
    hard_labels = np.argmax(f_values, axis=1)
    
    # Define colors
    color_list = np.array([
        [1.0, 0.0, 0.0],  # Red
        [0.0, 0.0, 1.0],  # Blue
        [0.0, 1.0, 0.0],  # Green
        [1.0, 1.0, 0.0],  # Yellow
        [1.0, 0.0, 1.0],  # Magenta
        [0.0, 1.0, 1.0],  # Cyan
    ])
    
    # Map labels to colors
    c_idx = np.take(color_list, hard_labels % len(color_list), axis=0)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        c=c_idx,
        s=5
    )
    
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"Mesh Segmentation (matplotlib fallback, subdiv={subdivisions})")
    plt.show()


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """
    Plot training history including losses and temperatures.
    
    Args:
        history: dictionary of training metrics
        save_path: optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    ax = axes[0, 0]
    ax.plot(history['loss'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Total Loss')
    ax.set_title('Training Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Individual losses
    ax = axes[0, 1]
    ax.plot(history['loss_smooth'], label='Smooth', alpha=0.7)
    ax.plot(history['loss_contour'], label='Contour', alpha=0.7)
    ax.plot(history['loss_area'], label='Area', alpha=0.7)
    ax.plot(history['loss_pin'], label='Pin', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Individual Losses')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Beta evolution
    ax = axes[1, 0]
    ax.plot(history['beta_contour'], label='β_contour')
    ax.plot(history['beta_area'], label='β_area')
    ax.set_xlabel('Step')
    ax.set_ylabel('Temperature (β)')
    ax.set_title('Temperature Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Area fractions (if available)
    ax = axes[1, 1]
    if 'area_fractions' in history and history['area_fractions']:
        area_fracs = np.array(history['area_fractions'])
        if area_fracs.ndim == 2:
            for i in range(area_fracs.shape[1]):
                ax.plot(area_fracs[:, i], label=f'Channel {i+1}', alpha=0.7)
            ax.axhline(y=1.0/area_fracs.shape[1], color='k', 
                      linestyle='--', alpha=0.5, label='Uniform')
            ax.set_xlabel('Step')
            ax.set_ylabel('Area Fraction')
            ax.set_title('Area Balance Evolution')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history plot to {save_path}")
    
    plt.show()


def save_segmentation_result(
    vertices: np.ndarray,
    faces: np.ndarray,
    f_values: np.ndarray,
    save_path: str
):
    """
    Save segmentation result to file.
    
    Args:
        vertices: (N, 3) - vertex positions
        faces: (T, 3) - triangle indices
        f_values: (N, C) - field values
        save_path: path to save file
    """
    np.savez(
        save_path,
        vertices=vertices,
        faces=faces,
        field_values=f_values
    )
    print(f"Saved segmentation result to {save_path}")