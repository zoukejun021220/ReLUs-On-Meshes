"""
Visualization utilities for mesh segmentation results.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pyvista as pv
from typing import Optional, Tuple, List


def get_region_colors() -> np.ndarray:
    """Get distinctive colors for 6 regions."""
    colors = np.array([
        [0.8, 0.2, 0.2],  # Red (top)
        [0.2, 0.2, 0.8],  # Blue (bottom)
        [0.2, 0.8, 0.2],  # Green (front)
        [0.8, 0.8, 0.2],  # Yellow (back)
        [0.8, 0.2, 0.8],  # Magenta (right)
        [0.2, 0.8, 0.8],  # Cyan (left)
    ])
    return colors


def compute_region_labels(f_values: torch.Tensor, use_softmax: bool = False, 
                         beta: float = 25.0) -> np.ndarray:
    """
    Compute region labels from field values.
    
    Args:
        f_values: Field values (N, 6)
        use_softmax: If True, use softmax before argmax
        beta: Temperature for softmax
        
    Returns:
        labels: Array of shape (N,) with values 0-5
    """
    if use_softmax:
        probs = torch.softmax(beta * f_values, dim=1)
        labels = probs.argmax(dim=1).cpu().numpy()
    else:
        labels = f_values.argmax(dim=1).cpu().numpy()
    
    return labels


def visualize_segmentation_pyvista(vertices: np.ndarray, 
                                  faces: np.ndarray,
                                  f_values: torch.Tensor,
                                  pinned_indices: Optional[List[int]] = None,
                                  show_edges: bool = True,
                                  save_path: Optional[str] = None) -> None:
    """
    Visualize mesh segmentation using PyVista.
    
    Args:
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (F, 3)
        f_values: Field values (N, 6)
        pinned_indices: Indices of pinned vertices to highlight
        show_edges: Whether to show mesh edges
        save_path: Path to save screenshot
    """
    # Create PyVista mesh
    cells = np.hstack([np.full((len(faces), 1), 3), faces]).flatten()
    mesh = pv.PolyData(vertices, cells)
    
    # Compute region labels
    labels = compute_region_labels(f_values, use_softmax=True)
    mesh["regions"] = labels
    
    # Create plotter
    plotter = pv.Plotter(window_size=(1200, 800))
    
    # Add mesh with regions
    colors = get_region_colors()
    plotter.add_mesh(mesh, scalars="regions", 
                    cmap=ListedColormap(colors),
                    clim=[0, 5],
                    show_edges=show_edges,
                    edge_color='black',
                    line_width=0.5 if show_edges else 0)
    
    # Highlight pinned vertices if provided
    if pinned_indices is not None:
        pin_points = vertices[pinned_indices]
        plotter.add_points(pin_points, color='black', point_size=20, 
                          render_points_as_spheres=True)
        
        # Add labels
        region_names = ["Top", "Bottom", "Front", "Back", "Right", "Left"]
        for i, idx in enumerate(pinned_indices):
            plotter.add_point_labels(vertices[idx:idx+1], [region_names[i]], 
                                   font_size=12, point_color='black')
    
    # Set camera and show
    plotter.camera_position = 'isometric'
    plotter.show_axes()
    
    if save_path:
        plotter.screenshot(save_path)
    
    plotter.show()


def plot_training_history(history: dict, save_path: Optional[str] = None) -> None:
    """
    Plot training history with multiple subplots.
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot total loss
    if 'loss' in history:
        axes[0].plot(history['loss'])
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Steps (x100)')
        axes[0].set_ylabel('Loss')
        axes[0].set_yscale('log')
        axes[0].grid(True)
    
    # Plot individual losses
    loss_names = ['area', 'adjacency', 'tv']
    for i, name in enumerate(loss_names):
        if name in history:
            axes[i+1].plot(history[name])
            axes[i+1].set_title(f'{name.capitalize()} Loss')
            axes[i+1].set_xlabel('Steps (x100)')
            axes[i+1].set_ylabel('Loss')
            axes[i+1].set_yscale('log')
            axes[i+1].grid(True)
    
    # Plot area fractions
    if 'area_fractions' in history:
        area_fracs = np.array(history['area_fractions'])
        for i in range(6):
            axes[4].plot(area_fracs[:, i], label=f'Region {i}')
        axes[4].axhline(y=1/6, color='k', linestyle='--', alpha=0.5)
        axes[4].set_title('Area Fractions')
        axes[4].set_xlabel('Steps (x100)')
        axes[4].set_ylabel('Fraction')
        axes[4].legend()
        axes[4].grid(True)
    
    # Plot learning rate and beta
    if 'lr' in history:
        ax5_twin = axes[5].twinx()
        axes[5].plot(history['lr'], 'b-', label='LR')
        axes[5].set_ylabel('Learning Rate', color='b')
        axes[5].tick_params(axis='y', labelcolor='b')
        
        if 'beta' in history:
            ax5_twin.plot(history['beta'], 'r-', label='Beta')
            ax5_twin.set_ylabel('Beta', color='r')
            ax5_twin.tick_params(axis='y', labelcolor='r')
        
        axes[5].set_xlabel('Steps (x100)')
        axes[5].set_title('Learning Rate & Beta')
        axes[5].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_field_values(vertices: np.ndarray, 
                          faces: np.ndarray,
                          f_values: torch.Tensor,
                          channel: int = 0,
                          save_path: Optional[str] = None) -> None:
    """
    Visualize individual channel values.
    
    Args:
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (F, 3)
        f_values: Field values (N, 6)
        channel: Which channel to visualize (0-5)
        save_path: Path to save screenshot
    """
    # Create PyVista mesh
    cells = np.hstack([np.full((len(faces), 1), 3), faces]).flatten()
    mesh = pv.PolyData(vertices, cells)
    
    # Add channel values
    channel_values = f_values[:, channel].cpu().numpy()
    mesh[f"channel_{channel}"] = channel_values
    
    # Create plotter
    plotter = pv.Plotter(window_size=(800, 600))
    
    # Add mesh
    plotter.add_mesh(mesh, scalars=f"channel_{channel}",
                    cmap='RdBu', clim=[-1, 1],
                    show_edges=True, edge_color='gray', line_width=0.5)
    
    # Add title
    region_names = ["Top", "Bottom", "Front", "Back", "Right", "Left"]
    plotter.add_text(f"Channel {channel}: {region_names[channel]}", 
                    position='upper_edge', font_size=14)
    
    plotter.camera_position = 'isometric'
    plotter.show_axes()
    
    if save_path:
        plotter.screenshot(save_path)
    
    plotter.show()


def compute_boundary_edges(f_values: torch.Tensor, edges: np.ndarray, 
                          threshold: float = 0.5) -> List[Tuple[int, int]]:
    """
    Find edges that lie on region boundaries.
    
    Args:
        f_values: Field values (N, 6)
        edges: Edge list (E, 2)
        threshold: Threshold for detecting boundaries
        
    Returns:
        List of edge indices that lie on boundaries
    """
    labels = compute_region_labels(f_values)
    
    boundary_edges = []
    for i, (v1, v2) in enumerate(edges):
        if labels[v1] != labels[v2]:
            boundary_edges.append(i)
    
    return boundary_edges


def measure_planarity(vertices: np.ndarray, faces: np.ndarray,
                     f_values: torch.Tensor, edges: np.ndarray,
                     edge2face: np.ndarray) -> dict:
    """
    Measure planarity of region boundaries.
    
    Args:
        vertices: Mesh vertices
        faces: Mesh faces
        f_values: Field values
        edges: Edge list
        edge2face: Edge to face mapping
        
    Returns:
        Dictionary with planarity metrics
    """
    from loss_functions import compute_pairwise_differences, compute_face_gradients
    from mesh_utils import compute_barycentric_matrices
    
    # Compute gradients
    d_v, pairs = compute_pairwise_differences(f_values)
    B = compute_barycentric_matrices(vertices, faces)
    B_torch = torch.tensor(B, dtype=torch.float32, device=f_values.device)
    faces_torch = torch.tensor(faces, dtype=torch.int64, device=f_values.device)
    
    grad15 = compute_face_gradients(f_values, faces_torch, B_torch, pairs)
    
    # Find boundary edges
    boundary_edges = compute_boundary_edges(f_values, edges)
    
    # Compute angles between adjacent face normals at boundaries
    angles = []
    for edge_idx in boundary_edges:
        f1, f2 = edge2face[edge_idx]
        if f1 >= 0 and f2 >= 0:
            # Get region pair for this edge
            labels = compute_region_labels(f_values)
            v1, v2 = edges[edge_idx]
            
            # Find which channel pair this edge belongs to
            for pair_idx, (i, j) in enumerate(pairs.cpu().numpy()):
                if (labels[v1] == i and labels[v2] == j) or \
                   (labels[v1] == j and labels[v2] == i):
                    
                    # Get gradients for this channel pair
                    g1 = grad15[f1, :, pair_idx].cpu().numpy()
                    g2 = grad15[f2, :, pair_idx].cpu().numpy()
                    
                    # Compute angle
                    cos_angle = np.dot(g1, g2) / (np.linalg.norm(g1) * np.linalg.norm(g2) + 1e-10)
                    angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                    angles.append(angle)
                    break
    
    if angles:
        return {
            'mean_angle': np.mean(angles),
            'max_angle': np.max(angles),
            'std_angle': np.std(angles),
            'num_boundary_edges': len(boundary_edges)
        }
    else:
        return {
            'mean_angle': 0,
            'max_angle': 0,
            'std_angle': 0,
            'num_boundary_edges': 0
        }