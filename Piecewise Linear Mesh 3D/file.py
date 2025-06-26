
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict
import math
import time
import pyvista as pv
###############################################################################
# 1) Generate an icosphere mesh with vectorized operations
###############################################################################

def create_icosphere_mesh(target_points=None, subdivisions=2, radius=1.0):
    """
    Creates an icosahedron-based sphere mesh by subdividing faces.
    Vectorized where possible.
    
    Args:
        target_points: Desired number of vertices (approximate) - if provided, 
                      overrides the subdivisions parameter
        subdivisions: Number of times to subdivide the initial icosahedron
        radius: Radius of the final sphere
        
    Returns:
        vertices_np: Array of shape (N, 3) containing vertex positions
        faces_np: Array of shape (T, 3) containing triangle indices
    """
    # If target_points is provided, calculate appropriate subdivision level
    if target_points is not None:
        # Initial icosahedron has 12 vertices
        # Each subdivision approximately quadruples the number of vertices
        # So we can estimate: num_vertices ≈ 12 * 4^subdivisions
        
        # Solve for subdivisions: log_4(target_points/12)
        if target_points < 12:
            print(f"Warning: Target points {target_points} is less than minimum (12). Using 0 subdivisions.")
            subdivisions = 0
        else:
            # Calculate subdivision level needed to reach target_points
            subdivisions = int(np.log(target_points / 12) / np.log(4))
            
            # Calculate actual number of points with this subdivision
            estimated_points = 12 * (4 ** subdivisions)
            print(f"Using {subdivisions} subdivisions to create approximately {estimated_points} points " +
                  f"(target was {target_points}).")
    # Start with icosahedron vertices
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    base_vertices = np.array([
        [-1,  phi, 0],
        [ 1,  phi, 0],
        [-1, -phi, 0],
        [ 1, -phi, 0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi, 0, -1],
        [ phi, 0,  1],
        [-phi, 0, -1],
        [-phi, 0,  1],
    ], dtype=np.float32)
    
    # Normalize vertices to lie on a unit sphere (vectorized)
    norms = np.linalg.norm(base_vertices, axis=1, keepdims=True)
    base_vertices /= norms
    
    # Initial faces of the icosahedron
    base_faces = np.array([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ], dtype=np.int32)
    
    # Subdivide the mesh
    vertices = base_vertices
    faces = base_faces
    
    for _ in range(subdivisions):
        vertices, faces = subdivide_mesh(vertices, faces)
    
    # Scale by radius (vectorized)
    vertices *= radius
    
    return vertices, faces

def subdivide_mesh(vertices, faces):
    """
    Subdivide each triangle into four smaller triangles.
    Optimized version with more efficient data structures.
    
    Args:
        vertices: Array of shape (N, 3) containing vertex positions
        faces: Array of shape (T, 3) containing triangle indices
        
    Returns:
        new_vertices: Array of shape (N', 3) containing new vertex positions
        new_faces: Array of shape (4T, 3) containing new triangle indices
    """
    vertices = list(vertices)
    new_faces = []
    edge_to_mid = {}
    
    # Extract all edges from faces for batch processing
    all_edges = []
    for face in faces:
        v1, v2, v3 = face
        all_edges.extend([(min(v1, v2), max(v1, v2)), 
                         (min(v2, v3), max(v2, v3)), 
                         (min(v3, v1), max(v3, v1))])
    
    # Find unique edges
    unique_edges = list(set(all_edges))
    
    # Compute midpoints in one batch
    for edge in unique_edges:
        a, b = edge
        mid = 0.5 * (np.array(vertices[a]) + np.array(vertices[b]))
        mid = mid / np.linalg.norm(mid)
        edge_to_mid[edge] = len(vertices)
        vertices.append(mid)
    
    # Create new faces
    for face in faces:
        v1, v2, v3 = face
        e1 = (min(v1, v2), max(v1, v2))
        e2 = (min(v2, v3), max(v2, v3))
        e3 = (min(v3, v1), max(v3, v1))
        
        a = edge_to_mid[e1]
        b = edge_to_mid[e2]
        c = edge_to_mid[e3]
        
        new_faces.extend([
            [v1, a, c],
            [v2, b, a],
            [v3, c, b],
            [a, b, c]
        ])
    
    return np.array(vertices), np.array(new_faces, dtype=np.int32)



def load_volume_tet_mesh_and_extract_surface(file_path):
    """
    Loads a VTK (or VTU) file containing a volumetric tetrahedral mesh,
    extracts its boundary surface, and returns a (vertices, faces) pair
    with all boundary faces triangulated.

    Args:
        file_path (str): Path to the VTK/VTU file.

    Returns:
        vertices_np (np.ndarray): Array of shape (N, 3) containing surface vertex coordinates.
        faces_np (np.ndarray): Array of shape (F, 3) containing triangulated surface faces (vertex indices).
    """
    # 1) Read mesh from file
    mesh = pv.read(file_path)  # PyVista automatically guesses file type (VTK, VTU, etc.)

    # 2) Extract the boundary surface
    surface_mesh = mesh.extract_surface()

    # 3) Triangulate (ensures only triangular cells)
    surface_mesh = surface_mesh.triangulate()

    # surface_mesh.faces is a "face array" of the form [3, i0, i1, i2, 3, i0, i1, i2, ...]
    # which we can reshape into a matrix of shape (num_faces, 4), and drop the first column (the "3")
    faces_array = surface_mesh.faces.reshape(-1, 4)[:, 1:]  # shape: (num_faces, 3)

    # Extract points
    vertices_np = surface_mesh.points  # shape: (N, 3)

    return vertices_np, faces_array
###############################################################################
# 2) Mesh data structures with vectorized operations
###############################################################################

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
    More optimized implementation with vectorized operations where possible.
    
    Args:
        faces_np: Array of shape (T, 3) containing triangle indices
        
    Returns:
        adjacency: Array of shape (E, 2) containing pairs of adjacent triangle indices
    """
    # Use numpy operations to create edges
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
    Find all unique edges in the mesh with vectorized operations.
    
    Args:
        faces_np: Array of shape (T, 3) containing triangle indices
        
    Returns:
        edges: Array of shape (E, 2) containing vertex edge indices
    """
    # Extract all edges from triangles
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

###############################################################################
# 3) Choose pinned vertices for 6 regions
###############################################################################

def find_axis_vertices(vertices_np):
    """
    Find vertices to pin for 6 regions using vectorized operations.
    
    Args:
        vertices_np: Array of shape (N, 3) containing vertex positions
        
    Returns:
        pinned_indices: List of 6 vertex indices
    """
    # Find max/min along each axis in one operation
    max_indices = np.argmax(vertices_np, axis=0)  # [x_max_idx, y_max_idx, z_max_idx]
    min_indices = np.argmin(vertices_np, axis=0)  # [x_min_idx, y_min_idx, z_min_idx]
    
    # Order as [top, bottom, front, back, right, left]
    pinned_indices = [
        max_indices[2],  # top (max z)
        min_indices[2],  # bottom (min z)
        max_indices[1],  # front (max y)
        min_indices[1],  # back (min y)
        max_indices[0],  # right (max x)
        min_indices[0]   # left (min x)
    ]
    
    return pinned_indices

###############################################################################
# 4) Initialize 6-channel scalar field with pinned vertices
###############################################################################

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
    
    # Set pinned values: channel c=+1, all others=-1
    for i, v_idx in enumerate(pinned_indices):
        f_init[v_idx] = pin_mask[i]
    
    # Convert to PyTorch parameter
    f_param = nn.Parameter(torch.tensor(f_init, device=device))
    
    return f_param

###############################################################################
# 5) Highly vectorized loss functions for the 6-channel field
###############################################################################import torch
import torch

def contour_alignment_loss(
    vertices:        torch.Tensor,  # (N,3) float
    faces:           torch.Tensor,  # (T,3) long
    f_values:        torch.Tensor,  # (N,C) float
    beta:            float = 20.0,
    eps:             float = 1e-9,
    soft_inside:     float = 10.0,
    include_triples: bool = True
) -> torch.Tensor:
    """
    A fully vectorized, "vigorous" contour-alignment loss that:
      1) Collects edge intersections for every channel pair (i<j) using a logistic weight
         for sign-changes in (f_i - f_j).
      2) Optionally finds "soft triple intersections" for (c0,c1,c2) if C>=3:
         - We do a closed-form barycentric solve for (f_c0-f_c1)=0, (f_c0-f_c2)=0
         - We multiply by each corner's softmax-prob of channels c0,c1,c2,
           plus a smooth clamp to keep the barycentric coords in [0,1].
      3) Builds a weighted covariance for each pair in one pass. Does an SVD => plane normal.
      4) In a second pass, accumulates the MSE distances of all intersection points
         to that pair's plane. Summation => final scalar loss.

    *No loops* over triangles or pairs in Python. Everything is broadcast/batch in PyTorch.
    *Fully differentiable*—no discrete argmax or masks. If C<2 => returns 0.0 (no pairs).

    Args:
      vertices: (N,3) float
      faces: (T,3) long
      f_values: (N,C) float   # multi-channel field
      beta: float  # logistic sharpness for edge intersection
      eps: float
      soft_inside: float  # how sharply to clamp barycentric coords for triple intersection
      include_triples: bool

    Returns:
      total_loss: scalar float
    """
    device = vertices.device
    N = vertices.shape[0]
    T = faces.shape[0]
    C = f_values.shape[1]

    # 0) If <2 channels => no pairs => return 0.0
    if C < 2:
        return torch.zeros((), device=device, dtype=vertices.dtype)

    # ------------------------------------------------------------------------
    # 1) Build all channel pairs i<j => shape (P,)
    # ------------------------------------------------------------------------
    i2, j2 = torch.triu_indices(C, C, offset=1, device=device)  # each => (P,)
    if i2.numel() == 0:
        # Edge case if somehow no pairs. Return 0.
        return torch.zeros((), device=device, dtype=vertices.dtype)

    P = i2.shape[0]  # number of (i<j) pairs

    # Build a map (cA,cB) => pair index
    pair_idx_mat = torch.full((C,C), -1, device=device, dtype=torch.long)
    p_arange     = torch.arange(P, device=device)
    pair_idx_mat[i2, j2] = p_arange
    pair_idx_mat[j2, i2] = p_arange

    # ------------------------------------------------------------------------
    # 2) Edge Intersections => logistic weighting
    # ------------------------------------------------------------------------
    # faces => (T,3), so gather coords => (T,3,3)
    p_tri = vertices[faces]
    # gather field => (T,3,C)
    f_tri = f_values[faces]

    # d => shape (T,3,P).  d[t,v,p] = f_tri[t,v, i2[p]] - f_tri[t,v, j2[p]]
    d = f_tri[..., i2] - f_tri[..., j2]  # (T,3,P)

    p0 = p_tri[:,0]
    p1 = p_tri[:,1]
    p2 = p_tri[:,2]

    def edge_intersection(dA, dB, vA, vB):
        """
        dA,dB: (T,P)
        vA,vB: (T,3)
        return coords: (T,P,3), w: (T,P)
        """
        prod  = dA*dB
        w_    = torch.sigmoid(-beta*prod)  # logistic weight
        alpha = torch.abs(dA)/(torch.abs(dA)+torch.abs(dB)+eps)
        coords = vA.unsqueeze(1) + alpha.unsqueeze(-1)*(vB - vA).unsqueeze(1)
        return coords, w_

    d0, d1, d2 = d[:,0,:], d[:,1,:], d[:,2,:]
    coords_01, w_01 = edge_intersection(d0, d1, p0, p1)
    coords_12, w_12 = edge_intersection(d1, d2, p1, p2)
    coords_20, w_20 = edge_intersection(d2, d0, p2, p0)

    def flatten_edge(coords_tp3, w_tp):
        # coords_tp3 => (T,P,3), w_tp => (T,P)
        coords_flat = coords_tp3.reshape(-1,3)
        w_flat      = w_tp.reshape(-1)
        pair_idx_arange = torch.arange(P, device=device).view(1,P).expand(coords_tp3.shape[0],-1).reshape(-1)
        return coords_flat, w_flat, pair_idx_arange

    coords_01f, w_01f, pidx_01f = flatten_edge(coords_01, w_01)
    coords_12f, w_12f, pidx_12f = flatten_edge(coords_12, w_12)
    coords_20f, w_20f, pidx_20f = flatten_edge(coords_20, w_20)

    all_coords = torch.cat([coords_01f, coords_12f, coords_20f], dim=0)
    all_w      = torch.cat([w_01f,      w_12f,      w_20f],      dim=0)
    all_pidx   = torch.cat([pidx_01f,   pidx_12f,   pidx_20f],   dim=0)

    # ------------------------------------------------------------------------
    # 3) Soft triple intersection => only if C>=3 and include_triples=True
    # ------------------------------------------------------------------------
    if include_triples and C >= 3:
        f0 = f_tri[:,0,:]  # (T,C)
        f1 = f_tri[:,1,:]
        f2 = f_tri[:,2,:]
        # softmax => each corner has distribution over channels
        pi0 = torch.softmax(f0, dim=1)
        pi1 = torch.softmax(f1, dim=1)
        pi2 = torch.softmax(f2, dim=1)

        # All combos c0< c1< c2
        combs = torch.combinations(torch.arange(C, device=device), r=3)
        # e.g. shape => (#comb,3)
        if combs.numel() > 0:  # ensure there's something
            ncomb = combs.shape[0]
            c0_idx = combs[:,0].view(1,ncomb)
            c1_idx = combs[:,1].view(1,ncomb)
            c2_idx = combs[:,2].view(1,ncomb)

            expand_t = (T, ncomb)
            f0_c0 = torch.gather(f0, 1, c0_idx.expand(expand_t))
            f0_c1 = torch.gather(f0, 1, c1_idx.expand(expand_t))
            f0_c2 = torch.gather(f0, 1, c2_idx.expand(expand_t))
            f1_c0 = torch.gather(f1, 1, c0_idx.expand(expand_t))
            f1_c1 = torch.gather(f1, 1, c1_idx.expand(expand_t))
            f1_c2 = torch.gather(f1, 1, c2_idx.expand(expand_t))
            f2_c0 = torch.gather(f2, 1, c0_idx.expand(expand_t))
            f2_c1 = torch.gather(f2, 1, c1_idx.expand(expand_t))
            f2_c2 = torch.gather(f2, 1, c2_idx.expand(expand_t))

            # differences
            rg0 = f0_c0 - f0_c1
            rg1 = f1_c0 - f1_c1
            rg2 = f2_c0 - f2_c1
            rb0 = f0_c0 - f0_c2
            rb1 = f1_c0 - f1_c2
            rb2 = f2_c0 - f2_c2

            A_xy = rg0 - rg2
            B_xy = rg1 - rg2
            X_xy = rg2
            A_xz = rb0 - rb2
            B_xz = rb1 - rb2
            X_xz = rb2

            det = A_xy*B_xz - B_xy*A_xz
            alpha = (X_xy*B_xz - B_xy*X_xz)/(det+eps)
            beta_  = (A_xy*X_xz - X_xy*A_xz)/(det+eps)
            gamma_ = 1.0 - alpha - beta_

            # "soft inside" factor
            if soft_inside>0.0:
                def smoothstep(x):
                    return torch.sigmoid(soft_inside*x)
                insideFactor = smoothstep(alpha)*smoothstep(beta_)*smoothstep(gamma_)
            else:
                insideFactor = torch.ones_like(alpha)

            # Probability corner0 picks c0, corner1 picks c1, corner2 picks c2
            pi0_c0 = torch.gather(pi0, 1, c0_idx.expand(expand_t))
            pi1_c1 = torch.gather(pi1, 1, c1_idx.expand(expand_t))
            pi2_c2 = torch.gather(pi2, 1, c2_idx.expand(expand_t))
            triple_prob = pi0_c0 * pi1_c1 * pi2_c2  # (T,ncomb)

            triple_w = triple_prob*insideFactor

            # Barycentric => 3D
            p0_3 = p_tri[:,0,:].unsqueeze(1)
            p1_3 = p_tri[:,1,:].unsqueeze(1)
            p2_3 = p_tri[:,2,:].unsqueeze(1)

            alpha_e = alpha.unsqueeze(-1)
            beta_e  = beta_.unsqueeze(-1)
            gamma_e = gamma_.unsqueeze(-1)
            triple_coords = alpha_e*p0_3 + beta_e*p1_3 + gamma_e*p2_3

            # replicate each triple for the 3 pairs => (c0,c1),(c0,c2),(c1,c2)
            pair0 = pair_idx_mat[combs[:,0], combs[:,1]]
            pair1 = pair_idx_mat[combs[:,0], combs[:,2]]
            pair2 = pair_idx_mat[combs[:,1], combs[:,2]]
            triple_pairs = torch.stack([pair0, pair1, pair2], dim=1)

            coords_flat = triple_coords.view(-1,3)
            w_flat      = triple_w.view(-1)

            coords_rep = coords_flat.repeat_interleave(3, dim=0)
            w_rep      = w_flat.repeat_interleave(3, dim=0)
            triple_pairs_flat = triple_pairs.view(-1)
            triple_pairs_full = triple_pairs_flat.unsqueeze(0).expand(T, -1).reshape(-1)

            # for difference=0 => product=0 => logistic(0)=0.5, or pick any factor
            w_rep_final = w_rep*0.5

            all_coords = torch.cat([all_coords, coords_rep], dim=0)
            all_w      = torch.cat([all_w,      w_rep_final], dim=0)
            all_pidx   = torch.cat([all_pidx,   triple_pairs_full], dim=0)

    # ------------------------------------------------------------------------
    # 4) Weighted covariance => plane fits
    # ------------------------------------------------------------------------
    sum_w   = torch.zeros((P,), device=device, dtype=all_coords.dtype)
    sum_x   = torch.zeros((P,3), device=device, dtype=all_coords.dtype)
    sum_xx  = torch.zeros((P,3,3), device=device, dtype=all_coords.dtype)

    weighted_coords = all_coords*all_w.unsqueeze(-1)
    sum_w.index_add_(0, all_pidx, all_w)
    sum_x.index_add_(0, all_pidx, weighted_coords)

    outer_ = weighted_coords.unsqueeze(2)*all_coords.unsqueeze(1)
    sum_xx_flat = sum_xx.view(P,9)
    outer_flat  = outer_.reshape(-1,9)
    sum_xx_flat.index_add_(0, all_pidx, outer_flat)
    sum_xx = sum_xx_flat.view(P,3,3)

    sum_w_clamped = sum_w.clamp_min(eps)
    mean_ = sum_x/sum_w_clamped.unsqueeze(-1)
    mean_outer = mean_.unsqueeze(2)*mean_.unsqueeze(1)
    cov = sum_xx/sum_w_clamped.view(-1,1,1) - mean_outer

    # ------------------------------------------------------------------------
    # 5) Plane from SVD
    # ------------------------------------------------------------------------
    cov_f32 = cov.float()
    U,S,Vt = torch.linalg.svd(cov_f32, full_matrices=False)  # => (P,3,3)
    plane_n = Vt[:, -1, :].to(cov.dtype)
    plane_n = plane_n/(plane_n.norm(dim=1,keepdim=True)+eps)
    plane_d = -(plane_n*mean_).sum(dim=1)

    # ------------------------------------------------------------------------
    # 6) Second pass => MSE
    # ------------------------------------------------------------------------
    sum_sq = torch.zeros((P,), device=device, dtype=all_coords.dtype)
    n_idx = plane_n[all_pidx]
    d_idx = plane_d[all_pidx]
    dist  = (n_idx*all_coords).sum(dim=1)+d_idx
    dist_sq = dist.square()*all_w
    sum_sq.index_add_(0, all_pidx, dist_sq)

    mse_pairs  = sum_sq/(sum_w_clamped+eps)
    total_loss = mse_pairs.sum()
    return total_loss




def contour_alignment_loss_6channels(points, triangles, f_values, adjacency, beta=20.0):
    """
    Compute a differentiable contour alignment loss that avoids the "stuck at constant value" problem.
    This is a wrapper around contour_alignment_loss_6channels_label_subdivide.
    
    Args:
        points: Tensor of shape (N, 3) containing vertex positions
        triangles: Tensor of shape (T, 3) containing triangle vertex indices
        f_values: Tensor of shape (N, 6) containing scalar field values
        adjacency: Tensor of shape (E, 2) containing adjacent triangle pairs
        beta: Temperature parameter for softmax (unused in the subdivide version)
        
    Returns:
        loss: Scalar loss for contour alignment
    """
    # Call the implementation with appropriate parameters
    return contour_alignment_loss(
        points=points, 
        triangles=triangles, 
        f_values=f_values, 
        adjacency=adjacency,
        check_triple=True
    )

def area_balance_loss_optimized(points, triangles, f_values, beta, mesh_area):
    """
    Highly vectorized implementation of area balance loss using softmax instead of sigmoid.
    
    Args:
        points: Tensor of shape (N, 3) containing vertex positions
        triangles: Tensor of shape (T, 3) containing triangle vertex indices
        f_values: Tensor of shape (N, 6) containing scalar field values
        beta: Softmax sharpness parameter
        mesh_area: Total mesh area
        
    Returns:
        loss: Area balance loss value
        fractions: Tensor of shape (6,) with fraction of area for each channel
    """
    device = points.device
    num_channels = f_values.shape[1]
    num_triangles = triangles.shape[0]
    
    # Define barycentric sampling points
    bary_points = torch.tensor([
        [1/3, 1/3, 1/3],  # Center
        [0.5, 0.5, 0.0],  # Edge midpoints
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ], device=device)
    
    num_samples = bary_points.shape[0]
    bary_weights = torch.ones(num_samples, device=device) / num_samples
    
    # Compute triangle areas in one batch
    v0 = triangles[:, 0]  # (T,)
    v1 = triangles[:, 1]  # (T,)
    v2 = triangles[:, 2]  # (T,)
    
    p0 = points[v0]  # (T, 3)
    p1 = points[v1]  # (T, 3)
    p2 = points[v2]  # (T, 3)
    
    e1 = p1 - p0  # (T, 3)
    e2 = p2 - p0  # (T, 3)
    
    normals = torch.cross(e1, e2, dim=1)  # (T, 3)
    areas = 0.5 * torch.norm(normals, dim=1)  # (T,)
    
    # Convert to softmax probabilities
    raw_scores = beta * f_values  # (N, 6)
    
    # Get scalar field values at triangle vertices
    f0 = f_values[v0]  # (T, 6)
    f1 = f_values[v1]  # (T, 6)
    f2 = f_values[v2]  # (T, 6)
    
    # Initialize tensor to accumulate channel areas
    channel_areas = torch.zeros(num_channels, device=device)
    
    # Process all triangles and all barycentric samples at once
    # Reshape for broadcasting:
    # f0: (T, 6) -> (T, 1, 6)
    # f1: (T, 6) -> (T, 1, 6)
    # f2: (T, 6) -> (T, 1, 6)
    # bary_points: (S, 3) -> (1, S, 3)
    f0 = f0.unsqueeze(1)  # (T, 1, 6)
    f1 = f1.unsqueeze(1)  # (T, 1, 6)
    f2 = f2.unsqueeze(1)  # (T, 1, 6)
    
    bary_expanded = bary_points.unsqueeze(0)  # (1, S, 3)
    
    # Extract barycentric coordinates
    b0 = bary_expanded[:, :, 0].unsqueeze(2)  # (1, S, 1)
    b1 = bary_expanded[:, :, 1].unsqueeze(2)  # (1, S, 1)
    b2 = bary_expanded[:, :, 2].unsqueeze(2)  # (1, S, 1)
    
    # Interpolate field at all barycentric points for all triangles
    f_interp = (f0 * b0) + (f1 * b1) + (f2 * b2)  # (T, S, 6)
    
    # Apply softmax to all interpolated values
    f_interp_beta = beta * f_interp  # (T, S, 6)
    exp_vals = torch.exp(f_interp_beta)  # (T, S, 6)
    sum_exp = torch.sum(exp_vals, dim=2, keepdim=True)  # (T, S, 1)
    s = torch.softmax(f_interp_beta, dim=2)
    
    # Apply sample weights and reshape
    s_weighted = s * bary_weights.view(1, num_samples, 1)  # (T, S, 6)
    
    # Sum over samples
    s_mean = torch.sum(s_weighted, dim=1)  # (T, 6)
    
    # Weight by triangle areas and sum
    weighted_areas = s_mean * areas.unsqueeze(1)  # (T, 6)
    channel_areas = torch.sum(weighted_areas, dim=0)  # (6,)
    
    # Compute fractions and loss
    fractions = channel_areas / mesh_area
    target = torch.ones_like(fractions) / num_channels
    loss = torch.sum(torch.abs(fractions - target))
    
    return loss, fractions

def smoothness_loss_optimized(f_values, vertex_edges):
    """
    Vectorized implementation of smoothness loss.
    
    Args:
        f_values: Tensor of shape (N, 6) containing scalar field values
        vertex_edges: Tensor of shape (E, 2) containing vertex edge indices
        
    Returns:
        loss: Smoothness loss value
    """
    # Get field values at all edge endpoints at once
    v1_idx = vertex_edges[:, 0]  # (E,)
    v2_idx = vertex_edges[:, 1]  # (E,)
    
    f1 = f_values[v1_idx]  # (E, 6)
    f2 = f_values[v2_idx]  # (E, 6)
    
    # Compute squared difference for all edges and all channels
    diff = f1 - f2  # (E, 6)
    loss = torch.sum(diff**2)
    
    return loss

def compute_combined_loss_optimized(f_values, points, triangles, adjacency, vertex_edges,
                                  mesh_area, beta=20.0, lambda_contour=1.0, lambda_smooth=1.0,
                                  lambda_area=1.0, use_label_subdivide=True):
    """
    Compute the combined loss with optimized vectorized operations.
    Now using contour_alignment_loss_6channels_label_subdivide for improved boundary detection.
    
    Args:
        f_values: Tensor of shape (N, 6) containing scalar field values
        points: Tensor of shape (N, 3) containing vertex positions
        triangles: Tensor of shape (T, 3) containing triangle vertex indices
        adjacency: Tensor of shape (E, 2) containing adjacent triangle pairs
        vertex_edges: Tensor of shape (E', 2) containing vertex edge indices
        mesh_area: Total mesh area
        beta: Softmax temperature parameter
        lambda_contour: Weight for contour alignment loss
        lambda_smooth: Weight for smoothness loss
        lambda_area: Weight for area balance loss
        use_label_subdivide: If True, use the label subdivision approach, otherwise use the simpler diff approach
        
    Returns:
        total_loss: Combined loss value
        loss_dict: Dictionary containing individual loss components
    """
    # Compute individual losses
    if use_label_subdivide:
        # Use the improved label subdivision approach for contour alignment
        contour_loss = contour_alignment_loss(
            vertices=points,
            faces=triangles,
            f_values=f_values,
            include_triples=True,
            beta=beta,
            eps=1e-9
            )
    else:
        # Use the simpler probability difference approach
        contour_loss = contour_alignment_loss_6channels(points, triangles, f_values, adjacency, beta)
        
    smooth_loss = smoothness_loss_optimized(f_values, vertex_edges)
    area_loss, area_fracs = area_balance_loss_optimized(points, triangles, f_values, beta, mesh_area)
    
    # Combine losses (note: no overlap loss since softmax handles that)
    total_loss = (lambda_contour * contour_loss +
                 lambda_smooth * smooth_loss +
                 lambda_area * area_loss)
    
    # Store individual losses for monitoring
    loss_dict = {
        'contour': contour_loss.item(),
        'smoothness': smooth_loss.item(),
        'area_balance': area_loss.item(),
        'total': total_loss.item(),
        'area_fractions': area_fracs.detach().cpu().numpy()
    }
    
    return total_loss, loss_dict

def visualize_segmentation_hardmax(vertices_np, faces_np, f_values, vis_resolution=30):
    """
    Visualize the segmentation using true hardmax boundaries with barycentric interpolation.
    Each pixel in a triangle is colored based on the argmax of interpolated values,
    allowing for sharp boundaries that cut through triangles.
    
    Args:
        vertices_np: Array of shape (N, 3) containing vertex positions
        faces_np: Array of shape (T, 3) containing triangle indices
        f_values: Array of shape (N, 6) containing scalar field values  
        vis_resolution: Resolution for the visualization sphere
    """
    try:
        import pyvista as pv
        import numpy as np
        import time
        from matplotlib.colors import ListedColormap
        import matplotlib.pyplot as plt
        
        print("Creating hardmax visualization with PyVista...")
        start_time = time.time()
        
        # Set up PyVista theme
        pv.set_plot_theme("document")
        
        # Define region colors
        region_colors = np.array([
            [1.0, 0.0, 0.0, 1.0],  # Red - Region 1
            [0.0, 0.0, 1.0, 1.0],  # Blue - Region 2
            [0.0, 1.0, 0.0, 1.0],  # Green - Region 3
            [1.0, 1.0, 0.0, 1.0],  # Yellow - Region 4
            [1.0, 0.0, 1.0, 1.0],  # Magenta - Region 5
            [0.0, 1.0, 1.0, 1.0]   # Cyan - Region 6
        ])
        
        # Create a proper matplotlib colormap
        region_cmap = ListedColormap(region_colors)
        
        # Create high-resolution sphere for barycentric rendering
        print(f"Creating high-resolution sphere with {vis_resolution*2} resolution...")
        high_res_sphere = pv.Sphere(radius=1.0, 
                                   theta_resolution=vis_resolution*2, 
                                   phi_resolution=vis_resolution*2)
        
        # Get the low-res sphere points (original mesh)
        # Create a triangular mesh using the original vertices and faces
        low_res_mesh = pv.PolyData(vertices_np, 
                                  np.column_stack((np.full(len(faces_np), 3), faces_np)).flatten())
        
        # For interpolation, we need to find which triangle each high-res point belongs to
        # and compute its barycentric coordinates
        print("Finding closest cells for all high-res points...")
        
        # Find closest cell (triangle) for each high-res point
        # This returns the cell index directly as a NumPy array
        triangle_indices = low_res_mesh.find_closest_cell(high_res_sphere.points)
        
        # Initialize arrays to store interpolated values and labels
        interpolated_labels = np.zeros(len(high_res_sphere.points), dtype=np.int32)
        
        print("Interpolating values and applying hardmax...")
        # For each high-res point, interpolate using barycentric coordinates
        for i, point in enumerate(high_res_sphere.points):
            if i % 10000 == 0:
                print(f"Processing point {i} of {len(high_res_sphere.points)}...")
                
            # Get the triangle this point maps to
            tri_idx = triangle_indices[i]
            
            # Get the triangle vertices
            triangle = faces_np[tri_idx]
            v0, v1, v2 = triangle
            
            # Get vertex positions
            p0 = vertices_np[v0]
            p1 = vertices_np[v1]
            p2 = vertices_np[v2]
            
            # Compute barycentric coordinates
            # Method: solve the linear system:
            # point = b0*p0 + b1*p1 + b2*p2
            # with b0 + b1 + b2 = 1
            
            # Create vectors
            v0v1 = p1 - p0
            v0v2 = p2 - p0
            v0p = point - p0
            
            # Create matrix
            d00 = np.dot(v0v1, v0v1)
            d01 = np.dot(v0v1, v0v2)
            d11 = np.dot(v0v2, v0v2)
            d20 = np.dot(v0p, v0v1)
            d21 = np.dot(v0p, v0v2)
            
            # Compute barycentric coordinates
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-10:
                # Degenerate triangle, just use closest vertex
                dist0 = np.sum((point - p0)**2)
                dist1 = np.sum((point - p1)**2)
                dist2 = np.sum((point - p2)**2)
                
                if dist0 <= dist1 and dist0 <= dist2:
                    interpolated_labels[i] = np.argmax(f_values[v0])
                elif dist1 <= dist0 and dist1 <= dist2:
                    interpolated_labels[i] = np.argmax(f_values[v1])
                else:
                    interpolated_labels[i] = np.argmax(f_values[v2])
                continue
                
            b1 = (d11 * d20 - d01 * d21) / denom
            b2 = (d00 * d21 - d01 * d20) / denom
            b0 = 1.0 - b1 - b2
            
            # Clamp barycentric coordinates (if point is slightly outside triangle)
            b0 = max(0.0, min(1.0, b0))
            b1 = max(0.0, min(1.0, b1))
            b2 = max(0.0, min(1.0, b2))
            
            # Normalize to ensure they sum to 1
            total = b0 + b1 + b2
            if total > 0:
                b0 /= total
                b1 /= total
                b2 /= total
            
            # Get the field values at the triangle vertices
            f0 = f_values[v0]
            f1 = f_values[v1]
            f2 = f_values[v2]
            
            # Interpolate the field values using barycentric coordinates
            f_interp = b0 * f0 + b1 * f1 + b2 * f2
            
            # Apply hardmax (argmax) to get the dominant label
            dominant_label = np.argmax(f_interp)
            interpolated_labels[i] = dominant_label
        
        # Map to 1-indexed for plotting
        label_field = interpolated_labels + 1
        high_res_sphere.point_data["Labels"] = label_field
        
        print("Creating visualization...")
        # Create a plotter
        plotter = pv.Plotter(window_size=[1200, 1200])
        
        # Add title
        plotter.add_text("Hardmax Segmentation with Barycentric Interpolation", 
                       font_size=14, position='upper_edge')
        
        # Add the high-res sphere with hardmax labels
        plotter.add_mesh(
            high_res_sphere,
            scalars="Labels",
            show_edges=False,
            cmap=region_cmap,
            interpolate_before_map=False,  # No interpolation for crisp boundaries
            show_scalar_bar=True,
            clim=[1, 6]
        )
        
        # Add scalar bar
        plotter.add_scalar_bar(
            title="Region Labels (1-6)",
            n_labels=6,
            italic=False,
            fmt="%d",  # Integer format
            font_family="arial",
            shadow=True,
            position_x=0.05,
            position_y=0.05,
            width=0.4
        )
        
        # Find and mark pinned vertices
        max_indices = np.argmax(vertices_np, axis=0)
        min_indices = np.argmin(vertices_np, axis=0)
        pinned_indices = [
            max_indices[2],  # top (max z)
            min_indices[2],  # bottom (min z)
            max_indices[1],  # front (max y)
            min_indices[1],  # back (min y)
            max_indices[0],  # right (max x)
            min_indices[0]   # left (min x)
        ]
        
        # Colors for markers
        marker_colors = [
            [1.0, 0.0, 0.0],  # Red - Top (0)
            [0.0, 0.0, 1.0],  # Blue - Bottom (1)
            [0.0, 1.0, 0.0],  # Green - Front (2)
            [1.0, 1.0, 0.0],  # Yellow - Back (3)
            [1.0, 0.0, 1.0],  # Magenta - Right (4)
            [0.0, 1.0, 1.0]   # Cyan - Left (5)
        ]
        
        # Add annotation markers for each pinned vertex
        region_names = ["Top (1)", "Bottom (2)", "Front (3)", "Back (4)", "Right (5)", "Left (6)"]
        for i, (name, idx) in enumerate(zip(region_names, pinned_indices)):
            # Get vertex position
            pos = vertices_np[idx]
            # Add a point marker at the pinned vertex
            plotter.add_points(pos.reshape(1, 3), color=marker_colors[i], point_size=15)
            # Add text label near the point
            offset = pos * 1.1  # Slightly offset from the point
            plotter.add_point_labels([offset], [name], font_size=10, 
                                   shadow=True, shape=None, text_color=marker_colors[i])
        
        # Set view angle
        plotter.view_isometric()
        plotter.camera.zoom(1.5)
        
        # Save a static screenshot
        try:
            plotter.screenshot('hardmax_segmentation.png', transparent_background=False)
            print("Screenshot saved to hardmax_segmentation.png")
        except Exception as e:
            print(f"Warning: Could not save screenshot: {e}")
            
        print("Displaying interactive visualization. Close the window to continue.")
        plotter.show()
        
        # Also create a direct comparison visualization (original vs hardmax)
        print("Creating comparison visualization...")
        
        # Create original mesh with argmax colors for comparison
        mesh = pv.PolyData(vertices_np, 
                          np.column_stack((np.full(len(faces_np), 3), faces_np)).flatten())
        
        # Apply argmax to get the labels
        labels = np.argmax(f_values, axis=1)
        mesh.point_data["Labels"] = labels + 1  # 1-indexed
        
        # Create a two-panel comparison
        plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])
        
        # First panel: Original mesh with vertex labels
        plotter.subplot(0, 0)
        plotter.add_text("Original Mesh (Labels at Vertices)", font_size=14, position="upper_edge")
        
        # Add the original mesh
        plotter.add_mesh(
            mesh,
            scalars="Labels",
            show_edges=True,
            edge_color='black',
            line_width=0.5,
            cmap=region_cmap,
            interpolate_before_map=False,
            show_scalar_bar=False,
            clim=[1, 6]
        )
        
        # Add markers to the first view
        for i, (name, idx) in enumerate(zip(region_names, pinned_indices)):
            pos = vertices_np[idx]
            plotter.add_points(pos.reshape(1, 3), color=marker_colors[i], point_size=15)
            offset = pos * 1.1
            plotter.add_point_labels([offset], [name], font_size=10, 
                                   shadow=True, shape=None, text_color=marker_colors[i])
        
        # Second panel: High-res hardmax visualization
        plotter.subplot(0, 1)
        plotter.add_text("Hardmax with Barycentric Interpolation", font_size=14, position="upper_edge")
        
        # Add the high-res sphere
        plotter.add_mesh(
            high_res_sphere,
            scalars="Labels",
            show_edges=False,
            cmap=region_cmap,
            interpolate_before_map=False,
            show_scalar_bar=True,
            clim=[1, 6]
        )
        
        # Add scalar bar to the second view
        plotter.add_scalar_bar(
            title="Region Labels (1-6)",
            n_labels=6,
            italic=False,
            fmt="%d",
            font_family="arial",
            shadow=True,
            position_x=0.05,
            position_y=0.05,
            width=0.4
        )
        
        # Add markers to the second view too
        for i, (name, idx) in enumerate(zip(region_names, pinned_indices)):
            pos = vertices_np[idx]
            plotter.add_points(pos.reshape(1, 3), color=marker_colors[i], point_size=15)
            offset = pos * 1.1
            plotter.add_point_labels([offset], [name], font_size=10, 
                                   shadow=True, shape=None, text_color=marker_colors[i])
        
        # Link the views
        plotter.link_views()
        plotter.view_isometric()
        
        # Save the comparison
        try:
            plotter.screenshot('hardmax_comparison.png', transparent_background=False)
            print("Comparison saved to hardmax_comparison.png")
        except Exception as e:
            print(f"Warning: Could not save comparison: {e}")
            
        plotter.show()
        
    except ImportError as e:
        print(f"PyVista not available: {e}. Using matplotlib for basic visualization...")
        
        # Fallback to matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        # Get the discrete labels for the original vertices
        labels = np.argmax(f_values, axis=1)
        
        # Create a 3D figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the mesh points with their labels
        colors = ['red', 'blue', 'green', 'yellow', 'magenta', 'cyan']
        ax.scatter(vertices_np[:, 0], vertices_np[:, 1], vertices_np[:, 2],
                  c=[colors[label] for label in labels], s=30)
        
        # Title and axis settings
        ax.set_title('Original Mesh with Vertex Labels')
        ax.set_box_aspect([1, 1, 1])
        
        # Save and show
        plt.savefig('hardmax_basic.png', dpi=300)
        plt.show()
###############################################################################
# 7) Optimized 6-patch segmentation
###############################################################################
from typing import Optional
# --------------------------------------------------------------------------- #
# fast single‑phase optimiser                                                 #
# --------------------------------------------------------------------------- #
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import numpy as np

def optimization(
    vertices_np, faces_np, pinned_indices,
    *,
    n_iters: int = 80_000,           # fewer iters than before
    lr: float = 2e-3,
    beta: float = 1.0,
    target_beta: Optional[float] = 40.0,
    beta_schedule: bool = True,
    lambda_contour_initial: float = 0.0,
    lambda_contour_final: float = 2.0,
    lambda_smooth: float = 0.2,      # slightly higher default
    lambda_area_initial: float = 0.2,
    lambda_area_final: float = 2.0,
    # OneCycleLR parameters:
    pct_start: float = 0.3,
    anneal_strategy: str = 'cos',
    div_factor: float = 25.0,
    # Early stopping:
    enable_early_stopping: bool = True,
    patience: int = 2000,
    print_every: int = 100,
    save_path: str = "final_mesh_and_values.npz"  # Path where we will save the mesh and field values
):
    """
    A faster-converging optimizer using:
      - OneCycleLR (with optional early stopping).
      - Slightly higher smoothness by default.
      - Shorter ramp for contour/area.

    Returns
    -------
    f_final : (N, 6) final field values on CPU
    final_mesh: (N, 3) vertices of the final mesh
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert data to torch
    v  = torch.from_numpy(vertices_np).float().to(device)
    f  = torch.from_numpy(faces_np).long().to(device)

    from collections import defaultdict
    from time import time

    # Build adjacency etc. 
    tri_adj    = torch.from_numpy(build_triangle_adjacency(faces_np)).long().to(device)
    vert_edges = torch.from_numpy(build_vertex_edges(faces_np)).long().to(device)
    mesh_area  = compute_face_areas(vertices_np, faces_np).sum()

    # Initialize the 6-channel field
    f_param = init_6channels_with_pins(len(vertices_np), pinned_indices, device)
    pin_mask = torch.full((6,6), -1.0, device=device)
    torch.diagonal(pin_mask).fill_(1.0)

    # Set up the optimizer (AdamW usually works well)
    opt = optim.AdamW([f_param], lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)

    # OneCycleLR Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        total_steps=n_iters,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        div_factor=div_factor,
        final_div_factor=1e+4,
        three_phase=False,
    )

    if beta_schedule and target_beta is not None:
        beta_vals = np.linspace(beta, target_beta, n_iters+1)
    else:
        beta_vals = np.full(n_iters+1, beta)

    ramp_iters = max(int(0.2 * n_iters), 1)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
    history = []
    t0 = time()

    best_loss = float('inf')
    best_iter = 0

    for it in range(1, n_iters+1):
        frac = min(it / ramp_iters, 1.0)

        beta_now = float(beta_vals[it])
        lambda_c_now = lambda_contour_initial + (lambda_contour_final - lambda_contour_initial)*frac
        lambda_a_now = lambda_area_initial + (lambda_area_final - lambda_area_initial)*frac

        lr_now = scheduler.get_last_lr()[0]

        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')): 
            total, comp = compute_combined_loss_optimized(
                f_param, v, f, tri_adj, vert_edges, mesh_area,
                beta=beta_now,
                lambda_contour = lambda_c_now,
                lambda_smooth  = lambda_smooth,
                lambda_area    = lambda_a_now,
                use_label_subdivide = True,
            )

        scaler.scale(total).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(f_param, 5.0)

        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        scheduler.step()

        with torch.no_grad():
            for k, idx in enumerate(pinned_indices):
                f_param[idx] = pin_mask[k]

        if total.item() < best_loss:
            best_loss = total.item()
            best_iter = it

        if (it % print_every == 0) or (it == 1) or (it == n_iters):
            print(
                f"iter {it:6d}/{n_iters}  total={total.item():.3e} "
                f"contour={comp['contour']:.3e}  smooth={comp['smoothness']:.3e}  "
                f"area={comp['area_balance']:.3e}  β={beta_now:.1f}  λc={lambda_c_now:.2f}  "
                f"λa={lambda_a_now:.2f}  lr={lr_now:.2e}"
            )
            history.append({
                'iter': it,
                'total': total.item(),
                'contour': comp['contour'],
                'smoothness': comp['smoothness'],
                'area_balance': comp['area_balance'],
                'beta': beta_now,
                'lambda_c': lambda_c_now,
                'lambda_a': lambda_a_now,
                'lr': lr_now
            })

        if enable_early_stopping:
            if it - best_iter > patience:
                print(f"Early stopping at iteration {it} (no improvement for {patience} steps).")
                break

    print(f"Finished in {(time()-t0)/60:.1f} min. Best loss={best_loss:.3e} at iter={best_iter}.")

    # After optimization, save the final mesh and its scalar field values
    final_mesh = vertices_np  # The final mesh
    final_field_values = f_param.detach().cpu().numpy()  # The final scalar field values

    # Save the final mesh and scalar field values to a .npz file
    np.savez_compressed(save_path, mesh=final_mesh, field_values=final_field_values)

    print(f"Final mesh and field values saved to {save_path}")

    return final_field_values, final_mesh, history, save_path







###############################################################################
# 8) Main function
###############################################################################

def main():
    total_start_time = time.time()
    
    # Step 1: Create the sphere mesh
    # pecify either target_points s
    target_points = 5000  # Specify desired number of points (approximate)

    
    print(f"Creating icosphere mesh with target of {target_points} points...")
    start_time = time.time()
    vertices_np, faces_np = load_volume_tet_mesh_and_extract_surface("multipatch\l1-poly-dat\hex\canewt\orig.tet.vtk")
    #vertices_np, faces_np = create_icosphere_mesh(target_points=target_points, subdivisions=2)
    elapsed = time.time() - start_time
    print(f"Created sphere mesh in {elapsed:.2f}s with {len(vertices_np)} vertices and {len(faces_np)} faces")
    
    # Step 2: Choose vertices to pin for 6 regions
    print("Finding vertices to pin...")
    start_time = time.time()
    pinned_indices = find_axis_vertices(vertices_np)
    region_names = ["Top", "Bottom", "Front", "Back", "Right", "Left"]
    elapsed = time.time() - start_time
    print(f"Found pin vertices in {elapsed:.2f}s")
    
    print("Pinning vertices for 6 regions:")
    for i, (name, idx) in enumerate(zip(region_names, pinned_indices)):
        print(f"  {name}: vertex {idx} at position {vertices_np[idx]}")
    
    # Step 3: Optimize the 6-channel scalar field with contour alignment loss
    print("\nStarting optimization...")
    n_iters = 1000000  
    f_optimized, final_mesh, history, save_path= optimization(
        vertices_np=vertices_np,
        faces_np=faces_np,
        pinned_indices=pinned_indices,
        n_iters=n_iters,
        lr=1e-3,
        beta=20.0,
        lambda_contour_initial=0.0,
        lambda_contour_final=10.0,
        lambda_smooth=0.1,
        lambda_area_initial=0.2,
        lambda_area_final=30.0,
        print_every=100,
        target_beta=80.0,  # Adjust if needed
        beta_schedule=True,
        patience=1000000

    # Use the new label subdivision approach
    )
    
    # Step 4: Visualize the result with softmax instead of sigmoid
    print("\nVisualizing result...")
    visualize_segmentation_hardmax(
        vertices_np=vertices_np,
        faces_np=faces_np,
        f_values=f_optimized,
       
        vis_resolution=600,  # Higher resolution for better visualization
       
    )
    
    total_elapsed = time.time() - total_start_time
    print(f"Total execution time: {total_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()