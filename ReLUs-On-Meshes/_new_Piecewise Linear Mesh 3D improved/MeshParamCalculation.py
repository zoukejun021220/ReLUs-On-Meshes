import numpy as np
import math
import torch
import torch.nn as nn


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
    Find pairs of triangles that share an edge (vectorized NumPy).

    Args:
        faces_np: Array of shape (T, 3) containing triangle indices

    Returns:
        adjacency: Array of shape (E, 2) containing pairs of adjacent triangle indices
    """
    T = faces_np.shape[0]
    # Build all edges and sort endpoints
    edges = np.concatenate(
        [faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]], axis=0
    )  # (3T,2)
    edges_sorted = np.sort(edges, axis=1)
    tri_idx = np.repeat(np.arange(T, dtype=np.int64), 3)

    # Structured array for lexsort/grouping by edge
    dtype = [('v0', np.int64), ('v1', np.int64), ('t', np.int64)]
    es = np.empty(edges_sorted.shape[0], dtype=dtype)
    es['v0'] = edges_sorted[:, 0]
    es['v1'] = edges_sorted[:, 1]
    es['t'] = tri_idx
    order = np.lexsort((es['v1'], es['v0']))
    es = es[order]

    same = (es['v0'][1:] == es['v0'][:-1]) & (es['v1'][1:] == es['v1'][:-1])
    starts = np.concatenate(([0], np.nonzero(~same)[0] + 1, [len(es)]))
    counts = np.diff(starts)
    mask2 = counts == 2
    if not np.any(mask2):
        return np.empty((0, 2), dtype=np.int64)
    s2 = starts[:-1][mask2]
    t0 = es['t'][s2]
    t1 = es['t'][s2 + 1]
    return np.stack([np.minimum(t0, t1), np.maximum(t0, t1)], axis=1)

def build_vertex_edges(faces_np):
    """
    Find all unique undirected edges in the mesh (vectorized NumPy).

    Args:
        faces_np: Array of shape (T, 3) containing triangle indices

    Returns:
        edges: Array of shape (E, 2) containing vertex edge indices
    """
    edges = np.concatenate(
        [faces_np[:, [0, 1]], faces_np[:, [1, 2]], faces_np[:, [2, 0]]], axis=0
    )
    edges_sorted = np.sort(edges, axis=1)
    return np.unique(edges_sorted, axis=0)







def init_6channels_with_pins(num_vertices, pinned_indices, device):
    """
    Initialize a 6-channel scalar field with pinned vertices.
    Vectorized implementation.
    
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
    
    # Set pinned values: channel c=+1, all others=-1 (vectorized)
    pins = np.asarray(pinned_indices, dtype=np.int64)
    valid = (pins >= 0) & (pins < num_vertices)
    if np.any(valid):
        f_init[pins[valid]] = pin_mask[valid]
    
    # Convert to PyTorch parameter
    f_param = nn.Parameter(torch.tensor(f_init, device=device))
    
    return f_param



def find_axis_vertices(vertices_np):
    """
    Force 6 pinned vertices (top,bottom,front,back,right,left)
    to be 90° apart in one rigid transformation:
      1) Z = top->bottom from bounding box center
      2) X = right->left, orthonormal to Z
      3) Y = cross(Z,X)
      4) Build anchor points around bounding-box center
      5) For each anchor, pick nearest real vertex

    Returns
    -------
    pinned_indices_unique : list of int
       [ top, bottom, front, back, right, left ] in original mesh indexing
       with duplicates removed if any.
    """

    # -------------------------------------------------------
    # 1) Find bounding box, center
    # -------------------------------------------------------
    x_min, y_min, z_min = vertices_np.min(axis=0)
    x_max, y_max, z_max = vertices_np.max(axis=0)

    cx = 0.5*(x_min + x_max)
    cy = 0.5*(y_min + y_max)
    cz = 0.5*(z_min + z_max)
    center = np.array([cx, cy, cz], dtype=vertices_np.dtype)

    # "Top–Bottom": fix x=cx, y=cy
    top_pt    = np.array([cx, cy, z_max])
    bottom_pt = np.array([cx, cy, z_min])
    zdir_raw  = top_pt - bottom_pt
    dist_z    = np.linalg.norm(zdir_raw)

    # "Right–Left": fix y=cy, z=cz
    right_pt  = np.array([x_max, cy, cz])
    left_pt   = np.array([x_min, cy, cz])
    xdir_raw  = right_pt - left_pt
    dist_x    = np.linalg.norm(xdir_raw)

    # "Front–Back": fix x=cx, z=cz
    front_pt  = np.array([cx, y_max, cz])
    back_pt   = np.array([cx, y_min, cz])
    ydir_raw  = front_pt - back_pt
    dist_y    = np.linalg.norm(ydir_raw)

    # -------------------------------------------------------
    # 2) Orthonormalize in fixed order: Z, then X, then Y=cross(Z,X)
    # -------------------------------------------------------
    eps = 1e-14
    def safe_norm(v):
        n = np.linalg.norm(v)
        return v / n if n>eps else np.zeros_like(v)

    zdir = safe_norm(zdir_raw)  # exact Z
    xdir_ = safe_norm(xdir_raw) # raw X
    # Remove component of xdir_ along zdir
    dot_xz = np.dot(xdir_, zdir)
    xdir_ = xdir_ - dot_xz*zdir
    xdir = safe_norm(xdir_)
    # Now cross => Y
    ydir = np.cross(zdir, xdir)
    ydir = safe_norm(ydir)

    half_z = 0.5*dist_z
    half_x = 0.5*dist_x
    half_y = 0.5*dist_y

    # -------------------------------------------------------
    # 3) Build anchor points
    # -------------------------------------------------------
    # top/bottom along ±z
    top_anchor    = center + (half_z)*zdir
    bottom_anchor = center - (half_z)*zdir
    # front/back along ±y
    front_anchor  = center + (half_y)*ydir
    back_anchor   = center - (half_y)*ydir
    # right/left along ±x
    right_anchor  = center + (half_x)*xdir
    left_anchor   = center - (half_x)*xdir

    anchors = [
        top_anchor,
        bottom_anchor,
        front_anchor,
        back_anchor,
        right_anchor,
        left_anchor
    ]

    # -------------------------------------------------------
    # 4) Pick nearest real vertex for each anchor
    # -------------------------------------------------------
    pinned_indices = []
    for pt in anchors:
        diffs = vertices_np - pt
        dist_sq = np.einsum('nd,nd->n', diffs, diffs)
        idx_closest = np.argmin(dist_sq)
        pinned_indices.append(idx_closest)

    # -------------------------------------------------------
    # 5) Remove duplicates if degenerate
    # -------------------------------------------------------
    pinned_indices_unique = []
    used = set()
    for idx in pinned_indices:
        if idx not in used:
            pinned_indices_unique.append(idx)
            used.add(idx)

    return pinned_indices_unique


def build_pinned_axes_6(vertices_np, pinned_indices):
    """
    Builds a (6,3) set of plane normals from  pinned vertices:
       pinned_indices = [top, bottom, front, back, right, left]
    We compute:
      nZ = normalize( pos(top) - pos(bottom) )
      nY = normalize( pos(front) - pos(back) )
      nX = normalize( pos(right) - pos(left) )
    Then pinned_axes_6 = [ nZ, -nZ, nY, -nY, nX, -nX ]
    That yields 6 directions, one for each channel.
    """
    # Unpack the pinned indices in order
    top_idx, bottom_idx, front_idx, back_idx, right_idx, left_idx = pinned_indices

    pos_top    = vertices_np[top_idx]
    pos_bottom = vertices_np[bottom_idx]
    pos_front  = vertices_np[front_idx]
    pos_back   = vertices_np[back_idx]
    pos_right  = vertices_np[right_idx]
    pos_left   = vertices_np[left_idx]

    # Compute direction vectors
    vec_z = pos_top - pos_bottom
    vec_y = pos_front - pos_back
    vec_x = pos_right - pos_left

    # Normalize
    def normed(v):
        v = np.array(v, dtype=np.float32)
        mag = np.linalg.norm(v)
        if mag < 1e-14:
            return np.array([0,0,1], dtype=np.float32)  # fallback
        return v / mag

    nZ = normed(vec_z)
    nY = normed(vec_y)
    nX = normed(vec_x)

    # Build a final (6,3) array
    pinned_axes_6 = np.stack([
        nZ,
        -nZ,
        nY,
        -nY,
        nX,
        -nX
    ], axis=0)  # shape (6,3)

    return pinned_axes_6
