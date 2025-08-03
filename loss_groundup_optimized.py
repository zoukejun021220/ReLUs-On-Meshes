#!/usr/bin/env python3
"""
Fully vectorized and optimized implementation of the loss function.
All loops are vectorized for maximum GPU efficiency.
"""

import torch
import math
from typing import Tuple, Dict


def build_edge_topology_vectorized(faces: torch.LongTensor):
    """Fully vectorized edge topology builder."""
    device = faces.device
    F = faces.shape[0]

    # All three directed edges of every face
    v0, v1, v2 = faces[:,0], faces[:,1], faces[:,2]
    directed = torch.stack([torch.stack([v0,v1],1),
                           torch.stack([v1,v2],1),
                           torch.stack([v2,v0],1)],1)        # (F,3,2)

    # Canonical edges
    flat = directed.reshape(-1,2)                             # (3F,2)
    canon = torch.sort(flat, dim=1)[0]                        # Sort to get canonical order
    
    # Unique edges
    canon, inv = torch.unique(canon, return_inverse=True, dim=0)
    E = canon.shape[0]
    
    # face→edge lookup
    face2edge = inv.view(F,3)
    
    # Vectorized edge→face mapping
    edge2face = torch.full((E,2), -1, dtype=torch.long, device=device)
    face_ids = torch.arange(F, device=device).repeat_interleave(3)
    
    # Sort by edge index to handle collisions
    sorted_idx = torch.argsort(inv)
    sorted_edges = inv[sorted_idx]
    sorted_faces = face_ids[sorted_idx]
    
    # Find first occurrence of each edge
    edge_changes = torch.cat([
        torch.tensor([True], device=device),
        sorted_edges[1:] != sorted_edges[:-1]
    ])
    first_idx = torch.where(edge_changes)[0]
    
    # Find second occurrence (if exists)
    second_mask = ~edge_changes
    second_mask[first_idx[1:]-1] = False  # Remove duplicates at boundaries
    second_idx = torch.where(second_mask)[0]
    
    # Assign faces to edges
    edge2face[sorted_edges[first_idx], 0] = sorted_faces[first_idx]
    if len(second_idx) > 0:
        edge2face[sorted_edges[second_idx], 1] = sorted_faces[second_idx]
    
    return canon, edge2face, face2edge


def face_gradient_mats(verts: torch.Tensor, faces: torch.LongTensor):
    """Pre-compute per-face gradient matrices."""
    v = verts[faces]                # (F,3,3)
    v0, v1, v2 = v[:,0], v[:,1], v[:,2]
    e1, e2 = v1 - v0, v2 - v0       # (F,3)

    # Normal and double area
    n   = torch.cross(e1, e2, dim=1)            # (F,3)
    n2  = (n*n).sum(1, keepdim=True)            # (F,1)
    n2  = torch.clamp(n2, min=1e-20)            # avoid zero‑division

    # Gradient computation matrices
    c1 = torch.cross(e2, n, dim=1) / n2         # (F,3)
    c2 = torch.cross(n, e1, dim=1) / n2         # (F,3)

    # Assemble 3×3 matrix
    B = torch.stack([-c1-c2, c1, c2], dim=2)    # (F,3,3)
    return B


def loss_fully_vectorized(Fv, beta, lambda_adj, lambda_tv, lambda_area,
                         verts, faces, edges, edge2face, B, face_areas):
    """
    Fully vectorized loss computation - no Python loops.
    """
    V, C = Fv.shape
    device = Fv.device
    eps = 1e-10

    # (A) Area balance - already vectorized
    Pv = torch.softmax(beta * Fv, dim=1)              # (V,C)
    Pf = Pv[faces].mean(dim=1)                        # (F,C)
    area_fractions = (Pf.T * face_areas).sum(dim=1)   # (C,)
    area_fractions = area_fractions / face_areas.sum()
    area_loss = lambda_area * torch.abs(area_fractions - 1.0/C).sum()

    # Vectorized gradient computation for all channels at once
    Ff = Fv[faces]                                    # (F,3,C)
    # Reshape for batch matrix multiplication
    Ff_reshaped = Ff.transpose(1, 2)                  # (F,C,3)
    B_expanded = B.unsqueeze(1)                       # (F,1,3,3)
    # Batch matrix multiply: (F,C,3,3) @ (F,C,3,1) -> (F,C,3,1)
    grads = torch.matmul(B_expanded, Ff_reshaped.unsqueeze(-1)).squeeze(-1)  # (F,C,3)
    grads = grads.transpose(1, 2)                     # (F,3,C)

    # Vectorized channel pair computation
    num_pairs = C * (C - 1) // 2
    
    # Create channel pair indices
    a_indices = []
    b_indices = []
    for a in range(C-1):
        for b in range(a+1, C):
            a_indices.append(a)
            b_indices.append(b)
    
    a_indices = torch.tensor(a_indices, device=device)
    b_indices = torch.tensor(b_indices, device=device)
    
    # Compute differences for all channel pairs at once
    Fa_all = Fv[:, a_indices]                        # (V, num_pairs)
    Fb_all = Fv[:, b_indices]                        # (V, num_pairs)
    d_all = Fa_all - Fb_all                          # (V, num_pairs)
    
    # Edge values
    v_idx0, v_idx1 = edges[:,0], edges[:,1]
    d_i_all = d_all[v_idx0]                          # (E, num_pairs)
    d_j_all = d_all[v_idx1]                          # (E, num_pairs)
    
    # Boundary weights for all pairs
    w_e_all = torch.sigmoid(-beta * d_i_all * d_j_all)  # (E, num_pairs)
    
    # Adjacent direction term
    f1 = edge2face[:,0]
    f2 = edge2face[:,1]
    valid = (f1>=0) & (f2>=0)
    
    adj_loss = 0.0
    if valid.any():
        # Gradients for all channel pairs
        g1_all = grads[f1[valid], :, a_indices] - grads[f1[valid], :, b_indices]  # (Ev, 3, num_pairs)
        g2_all = grads[f2[valid], :, a_indices] - grads[f2[valid], :, b_indices]  # (Ev, 3, num_pairs)
        
        # Area normalization
        area1 = face_areas[f1[valid]]
        area2 = face_areas[f2[valid]]
        avg_area = (area1 + area2) / 2                           # (Ev,)
        area_scale = avg_area.sqrt().unsqueeze(1).unsqueeze(2)   # (Ev, 1, 1)
        
        g1_all = g1_all * area_scale
        g2_all = g2_all * area_scale
        
        # Compute cosine similarity for all pairs
        dot_all = (g1_all * g2_all).sum(dim=1)                   # (Ev, num_pairs)
        n1_all = g1_all.norm(dim=1) + eps                        # (Ev, num_pairs)
        n2_all = g2_all.norm(dim=1) + eps                        # (Ev, num_pairs)
        cos_all = torch.clamp(dot_all / (n1_all * n2_all), -1.0, 1.0)
        
        # Weighted loss
        w_valid_all = w_e_all[valid]                              # (Ev, num_pairs)
        adj_losses = (w_valid_all * (1 - cos_all)).mean(dim=0)   # (num_pairs,)
        adj_loss = lambda_adj * adj_losses.mean()
    
    # Total variation term
    tv_losses = ((1 - w_e_all) * (d_i_all - d_j_all).pow(2)).mean(dim=0)  # (num_pairs,)
    tv_loss = lambda_tv * tv_losses.mean()
    
    return area_loss + adj_loss + tv_loss, \
           {'area': area_loss.item(),
            'adj' : adj_loss.item() if isinstance(adj_loss, torch.Tensor) else adj_loss,
            'tv'  : tv_loss.item()}


class MeshLossOptimized:
    """Fully optimized loss computation with vectorized operations."""
    
    def __init__(self, verts: torch.Tensor, faces: torch.LongTensor, device='cuda'):
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Ensure everything is on the correct device from the start
        verts = verts.to(self.device)
        faces = faces.to(self.device)
        
        # Build topology directly on device (avoid CPU->GPU transfers)
        edges, edge2face, face2edge = build_edge_topology_vectorized(faces)
        
        self.verts = verts
        self.faces = faces
        self.edges = edges
        self.edge2face = edge2face
        self.face2edge = face2edge
        
        # Pre-compute on device
        self.B = face_gradient_mats(verts, faces)
        
        # Face areas
        v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
        self.face_areas = 0.5 * torch.linalg.norm(
            torch.cross(v1-v0, v2-v0, dim=1), dim=1)
    
    def compute_loss(self, Fv: torch.Tensor, beta: float, 
                     lambda_adj: float = 5.0, lambda_tv: float = 0.05, 
                     lambda_area: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """Compute the fully vectorized loss."""
        return loss_fully_vectorized(
            Fv, beta, lambda_adj, lambda_tv, lambda_area,
            self.verts, self.faces, self.edges, self.edge2face, 
            self.B, self.face_areas
        )
    
    def get_schedules(self, t: int, total_steps: int) -> Dict[str, float]:
        """Get recommended hyperparameter schedules."""
        warmup_steps = int(0.2 * total_steps)
        warm_frac = min(1.0, t / warmup_steps)
        
        beta_start, beta_end = 2.0, 25.0
        adj_start, adj_end = 0.0, 5.0
        
        beta = beta_start + (beta_end - beta_start) * warm_frac
        lambda_adj = adj_start + (adj_end - adj_start) * warm_frac
        
        return {
            'beta': beta,
            'lambda_adj': lambda_adj
        }


def set_pinned_values_vectorized(f_values: torch.Tensor, pinned_indices: list) -> None:
    """Vectorized version of setting pinned values."""
    # Convert to tensor for vectorized operations
    pinned_indices = torch.tensor(pinned_indices[:6], device=f_values.device)
    
    # Create identity assignments
    f_values[pinned_indices] = 0.0
    f_values[pinned_indices, torch.arange(len(pinned_indices), device=f_values.device)] = 1.0