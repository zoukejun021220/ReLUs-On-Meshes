import numpy as np
import torch
import torch.nn as nn


def normalize_mesh(vertices_np):
    """
    Normalize mesh to unit box centered at origin.
    
    Args:
        vertices_np: Array of shape (N, 3) containing vertex positions
        
    Returns:
        normalized_vertices: Array of shape (N, 3) normalized vertices
        center: Original center of mesh
        scale: Scale factor used
    """
    # Compute center and translate to origin
    center = vertices_np.mean(axis=0)
    centered = vertices_np - center
    
    # Scale to unit box
    bbox_min = centered.min(axis=0)
    bbox_max = centered.max(axis=0)
    bbox_diag = np.linalg.norm(bbox_max - bbox_min)
    scale = 2.0 / bbox_diag  # Scale to [-1, 1] box
    
    normalized_vertices = centered * scale
    
    return normalized_vertices, center, scale


def compute_mesh_pca(vertices_np, faces_np):
    """
    Compute area-weighted PCA on triangle centroids to find principal axes.
    
    Args:
        vertices_np: Array of shape (N, 3) containing vertex positions
        faces_np: Array of shape (T, 3) containing triangle indices
        
    Returns:
        axes: Array of shape (3, 3) where each row is a principal axis
        eigenvalues: Array of shape (3,) with corresponding eigenvalues
    """
    # Compute triangle centroids and areas
    v0 = vertices_np[faces_np[:, 0]]
    v1 = vertices_np[faces_np[:, 1]]
    v2 = vertices_np[faces_np[:, 2]]
    
    centroids = (v0 + v1 + v2) / 3.0  # (T, 3)
    
    # Compute areas
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(normals, axis=1)  # (T,)
    
    # Area-weighted mean
    total_area = areas.sum()
    weighted_mean = (centroids * areas[:, np.newaxis]).sum(axis=0) / total_area
    
    # Area-weighted covariance
    centered = centroids - weighted_mean
    weighted_centered = centered * np.sqrt(areas[:, np.newaxis])
    cov = (weighted_centered.T @ weighted_centered) / total_area
    
    # PCA via eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Return axes as rows
    axes = eigenvectors.T
    
    return axes, eigenvalues


def find_support_points(vertices_np, axis):
    """
    Find support points (extremal vertices) along a given axis.
    
    Args:
        vertices_np: Array of shape (N, 3) containing vertex positions
        axis: Array of shape (3,) representing the axis direction
        
    Returns:
        idx_pos: Index of vertex with maximum projection
        idx_neg: Index of vertex with minimum projection
    """
    # Project vertices onto axis
    projections = vertices_np @ axis
    
    # Find extremal vertices
    idx_pos = np.argmax(projections)
    idx_neg = np.argmin(projections)
    
    return idx_pos, idx_neg


def find_axis_vertices_improved(vertices_np, faces_np, use_pca=True):
    """
    Improved anchor point selection using PCA and support points.
    
    Args:
        vertices_np: Array of shape (N, 3) containing vertex positions
        faces_np: Array of shape (T, 3) containing triangle indices
        use_pca: If True, use PCA axes; if False, use world axes
        
    Returns:
        pinned_indices: List of 6 vertex indices [top, bottom, front, back, right, left]
        pinned_axes: Array of shape (6, 3) containing the fixed plane normals
    """
    # Normalize mesh first
    normalized_vertices, _, _ = normalize_mesh(vertices_np)
    
    if use_pca:
        # Compute PCA axes
        axes, _ = compute_mesh_pca(normalized_vertices, faces_np)
        # axes[0] = first principal axis (typically longest)
        # axes[1] = second principal axis
        # axes[2] = third principal axis (typically shortest)
        
        # For consistency with original code, map to [Z, Y, X]
        # (top/bottom, front/back, right/left)
        axis_z = axes[2]  # Third axis for top/bottom
        axis_y = axes[1]  # Second axis for front/back
        axis_x = axes[0]  # First axis for right/left
    else:
        # Use world axes
        axis_x = np.array([1.0, 0.0, 0.0])
        axis_y = np.array([0.0, 1.0, 0.0])
        axis_z = np.array([0.0, 0.0, 1.0])
    
    # Find support points for each axis
    idx_top, idx_bottom = find_support_points(normalized_vertices, axis_z)
    idx_front, idx_back = find_support_points(normalized_vertices, axis_y)
    idx_right, idx_left = find_support_points(normalized_vertices, axis_x)
    
    pinned_indices = [idx_top, idx_bottom, idx_front, idx_back, idx_right, idx_left]
    
    # Build fixed plane normals (don't derive from vertex pairs)
    pinned_axes = np.stack([
        axis_z,   # Top channel
        -axis_z,  # Bottom channel
        axis_y,   # Front channel
        -axis_y,  # Back channel
        axis_x,   # Right channel
        -axis_x   # Left channel
    ], axis=0)  # Shape (6, 3)
    
    return pinned_indices, pinned_axes


def compute_face_areas(vertices_np, faces_np):
    """
    Compute the area of each face in the mesh using vectorized operations.
    
    Args:
        vertices_np: Array of shape (N, 3) containing vertex positions
        faces_np: Array of shape (T, 3) containing triangle indices
        
    Returns:
        areas: Array of shape (T,) containing face areas
    """
    # Get all vertices of all triangles at once
    v0 = vertices_np[faces_np[:, 0]]  # (T, 3)
    v1 = vertices_np[faces_np[:, 1]]  # (T, 3)
    v2 = vertices_np[faces_np[:, 2]]  # (T, 3)
    
    # Compute edge vectors
    e1 = v1 - v0  # (T, 3)
    e2 = v2 - v0  # (T, 3)
    
    # Compute areas using cross product
    cross = np.cross(e1, e2)  # (T, 3)
    areas = 0.5 * np.linalg.norm(cross, axis=1)  # (T,)
    
    return areas


def build_triangle_adjacency(faces_np):
    """
    Find pairs of triangles that share an edge.
    
    Args:
        faces_np: Array of shape (T, 3) containing triangle indices
        
    Returns:
        adjacency: Array of shape (E, 2) containing pairs of adjacent triangle indices
    """
    T = faces_np.shape[0]
    
    # Create an array of all edges (3 edges per triangle)
    all_edges = np.zeros((T * 3, 3), dtype=np.int64)
    
    # For each triangle, create its 3 edges as (min_idx, max_idx, tri_idx)
    for t_idx in range(T):
        tri = faces_np[t_idx]
        i1, i2, i3 = tri
        
        # Sort vertex indices for each edge
        edges = np.array([
            [min(i1, i2), max(i1, i2), t_idx],
            [min(i2, i3), max(i2, i3), t_idx],
            [min(i3, i1), max(i3, i1), t_idx]
        ])
        
        all_edges[t_idx*3:t_idx*3+3] = edges
    
    # Sort by edge (first by min_idx, then by max_idx)
    sorted_edges = all_edges[np.lexsort((all_edges[:, 1], all_edges[:, 0]))]
    
    # Find edges that appear exactly twice (shared by 2 triangles)
    adjacency = []
    i = 0
    while i < len(sorted_edges) - 1:
        if (sorted_edges[i, 0] == sorted_edges[i+1, 0] and 
            sorted_edges[i, 1] == sorted_edges[i+1, 1]):
            # Found a shared edge
            t1 = sorted_edges[i, 2]
            t2 = sorted_edges[i+1, 2]
            adjacency.append((min(t1, t2), max(t1, t2)))
            i += 2
        else:
            i += 1
    
    return np.array(adjacency, dtype=np.int64)


def build_vertex_edges(faces_np):
    """
    Find all unique edges in the mesh.
    
    Args:
        faces_np: Array of shape (T, 3) containing triangle indices
        
    Returns:
        edges: Array of shape (E, 2) containing vertex edge indices
    """
    T = faces_np.shape[0]
    all_edges = np.zeros((T * 3, 2), dtype=np.int64)
    
    # For each triangle, extract sorted edges
    for t_idx in range(T):
        i1, i2, i3 = faces_np[t_idx]
        
        # Sort vertex indices for each edge
        edges = np.array([
            [min(i1, i2), max(i1, i2)],
            [min(i2, i3), max(i2, i3)],
            [min(i3, i1), max(i3, i1)]
        ])
        
        all_edges[t_idx*3:t_idx*3+3] = edges
    
    # Use numpy's unique function on structured arrays to find unique edges
    dtype = [('v1', np.int64), ('v2', np.int64)]
    structured_edges = np.array([(e[0], e[1]) for e in all_edges], dtype=dtype)
    unique_edges = np.unique(structured_edges)
    
    # Convert back to regular array
    edges = np.array([(e[0], e[1]) for e in unique_edges], dtype=np.int64)
    
    return edges


def init_6channels_with_pins(num_vertices, pinned_indices, device):
    """
    Initialize a 6-channel scalar field with pinned vertices.
    
    Args:
        num_vertices: Number of vertices in the mesh
        pinned_indices: List of 6 vertex indices to pin
        device: PyTorch device
        
    Returns:
        f_param: PyTorch parameter of shape (N, 6)
    """
    # Initialize with small random values
    f_init = 0.01 * np.random.randn(num_vertices, 6).astype(np.float32)
    
    # Create pin mask tensor
    pin_mask = np.ones((len(pinned_indices), 6)) * -1.0
    np.fill_diagonal(pin_mask, 1.0)
    
    # Set pinned values: channel c=+1, all others=-1
    for i, v_idx in enumerate(pinned_indices):
        f_init[v_idx] = pin_mask[i]
    
    # Convert to PyTorch parameter
    f_param = nn.Parameter(torch.tensor(f_init, device=device))
    
    return f_param