"""
SVD-based plane initialization for channel-pairwise planes.
Initializes planes optimally between channel pairs after warm-up.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def svd_init_pairwise_planes(
    vertices: torch.Tensor,
    f_values: torch.Tensor,
    device: torch.device,
    threshold: float = 0.8
) -> Tuple[nn.Parameter, nn.Parameter]:
    """
    Initialize pairwise planes using SVD on vertices that belong to each channel pair.
    
    This function finds the optimal separating plane between each pair of channels
    by analyzing the spatial distribution of vertices where each channel dominates.
    
    Args:
        vertices: (V, 3) vertex positions
        f_values: (V, C) channel values after warm-up
        device: torch device
        threshold: relative threshold for channel dominance (default 0.8)
    
    Returns:
        plane_normals: (num_pairs, 3) initialized plane normals
        plane_offsets: (num_pairs,) initialized plane offsets
    """
    V, C = f_values.shape
    num_pairs = C * (C - 1) // 2
    
    # Find channel assignments based on max value
    channel_assignments = torch.argmax(f_values, dim=1)  # (V,)
    
    # Initialize storage for planes
    plane_normals_list = []
    plane_offsets_list = []
    
    # Create channel pair indices
    pair_idx = 0
    for i in range(C):
        for j in range(i + 1, C):
            # Find vertices where channel i or j dominates
            mask_i = channel_assignments == i
            mask_j = channel_assignments == j
            
            # Also consider vertices where i or j are strong but not necessarily max
            f_max = f_values.max(dim=1)[0]
            strong_i = (f_values[:, i] > threshold * f_max) & mask_i
            strong_j = (f_values[:, j] > threshold * f_max) & mask_j
            
            verts_i = vertices[strong_i]
            verts_j = vertices[strong_j]
            
            if len(verts_i) < 3 or len(verts_j) < 3:
                # Not enough points, use random initialization
                normal = torch.randn(3, device=device)
                normal = F.normalize(normal, p=2, dim=0)
                offset = 0.0
            else:
                # Compute centroids
                center_i = verts_i.mean(dim=0)
                center_j = verts_j.mean(dim=0)
                
                # Initial plane normal points from i to j
                normal = center_j - center_i
                
                if normal.norm() < 1e-6:
                    # Centers coincide, use PCA on combined points
                    all_verts = torch.cat([verts_i, verts_j], dim=0)
                    centered = all_verts - all_verts.mean(dim=0)
                    _, _, v = torch.svd(centered)
                    normal = v[:, 2]  # Smallest principal component
                else:
                    normal = F.normalize(normal, p=2, dim=0)
                
                # Find optimal offset using all boundary vertices
                # Collect vertices near the boundary between channels i and j
                f_diff = (f_values[:, i] - f_values[:, j]).abs()
                f_sum = f_values[:, i] + f_values[:, j]
                boundary_score = f_diff / (f_sum + 1e-6)
                
                # Consider vertices where i and j are competitive
                boundary_mask = (boundary_score < 0.3) & (f_sum > 0.1)
                
                if boundary_mask.sum() > 0:
                    boundary_verts = vertices[boundary_mask]
                    # Project boundary vertices onto normal
                    projections = torch.matmul(boundary_verts, normal)
                    offset = -projections.mean()
                else:
                    # No clear boundary, place plane at midpoint
                    midpoint = (center_i + center_j) / 2
                    offset = -torch.dot(midpoint, normal)
            
            plane_normals_list.append(normal)
            plane_offsets_list.append(offset if isinstance(offset, torch.Tensor) else torch.tensor(offset, device=device, dtype=vertices.dtype))
            pair_idx += 1
    
    # Stack into tensors
    plane_normals = torch.stack(plane_normals_list, dim=0)
    plane_offsets = torch.stack(plane_offsets_list, dim=0)
    
    return nn.Parameter(plane_normals), nn.Parameter(plane_offsets)


def reinit_planes_with_svd(
    vertices: torch.Tensor,
    f_values: torch.Tensor,
    plane_normals: nn.Parameter,
    plane_offsets: nn.Parameter,
    momentum: float = 0.5
) -> None:
    """
    Reinitialize planes using SVD while maintaining some momentum from current values.
    
    Args:
        vertices: (V, 3) vertex positions
        f_values: (V, C) current channel values
        plane_normals: existing plane normals to update in-place
        plane_offsets: existing plane offsets to update in-place
        momentum: how much to blend with existing values (0=full reinit, 1=no change)
    """
    device = plane_normals.device
    
    # Get new initialization
    new_normals, new_offsets = svd_init_pairwise_planes(vertices, f_values, device)
    
    # Blend with existing values
    with torch.no_grad():
        # For normals, normalize after blending
        blended_normals = momentum * plane_normals.data + (1 - momentum) * new_normals.data
        plane_normals.data = F.normalize(blended_normals, p=2, dim=1)
        
        # For offsets, simple blend
        plane_offsets.data = momentum * plane_offsets.data + (1 - momentum) * new_offsets.data


def analyze_channel_topology(
    vertices: torch.Tensor,
    f_values: torch.Tensor,
    connectivity: torch.Tensor
) -> dict:
    """
    Analyze if channels form connected regions.
    
    Args:
        vertices: (V, 3) vertex positions
        f_values: (V, C) channel values
        connectivity: (E, 2) edge connectivity
        
    Returns:
        Dictionary with topology information per channel
    """
    V, C = f_values.shape
    channel_assignments = torch.argmax(f_values, dim=1)
    
    topology_info = {}
    
    for c in range(C):
        # Find vertices belonging to channel c
        channel_mask = channel_assignments == c
        channel_verts = torch.where(channel_mask)[0]
        
        if len(channel_verts) == 0:
            topology_info[c] = {'num_components': 0, 'largest_component': 0}
            continue
        
        # Build subgraph for this channel
        # Find edges where both vertices belong to channel c
        edge_mask = channel_mask[connectivity[:, 0]] & channel_mask[connectivity[:, 1]]
        channel_edges = connectivity[edge_mask]
        
        # Run connected components (simple DFS)
        visited = torch.zeros(V, dtype=torch.bool, device=vertices.device)
        components = []
        
        for start_vert in channel_verts:
            if visited[start_vert]:
                continue
                
            # DFS from this vertex
            component = []
            stack = [start_vert.item()]
            
            while stack:
                v = stack.pop()
                if visited[v]:
                    continue
                    
                visited[v] = True
                component.append(v)
                
                # Find neighbors in channel subgraph
                neighbors = []
                for edge in channel_edges:
                    if edge[0] == v:
                        neighbors.append(edge[1].item())
                    elif edge[1] == v:
                        neighbors.append(edge[0].item())
                
                for n in neighbors:
                    if not visited[n] and channel_mask[n]:
                        stack.append(n)
            
            if component:
                components.append(len(component))
        
        topology_info[c] = {
            'num_components': len(components),
            'largest_component': max(components) if components else 0,
            'total_vertices': len(channel_verts)
        }
    
    return topology_info