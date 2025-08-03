#!/usr/bin/env python3
"""
Balanced implementation with proper normalization.
Uses sum over edges but normalizes by mesh size to prevent explosion.
"""

import torch
import math
from typing import Tuple, Dict


def build_edge_topology(faces: torch.LongTensor):
    """Build edge connectivity from faces."""
    device = faces.device
    F = faces.shape[0]

    # All three directed edges of every face
    v0, v1, v2 = faces[:,0], faces[:,1], faces[:,2]
    directed = torch.stack([torch.stack([v0,v1],1),
                            torch.stack([v1,v2],1),
                            torch.stack([v2,v0],1)],1)        # (F,3,2)

    # Canonical edges
    flat = directed.reshape(-1,2)                             # (3F,2)
    low  = torch.min(flat[:,0], flat[:,1])
    high = torch.max(flat[:,0], flat[:,1])
    canon = torch.stack([low, high],1)                        # (3F,2)

    # Unique edges
    canon, inv = torch.unique(canon, return_inverse=True, dim=0)
    E = canon.shape[0]

    # face→edge lookup
    face2edge = inv.view(F,3)

    # edge→(face₁,face₂)
    edge2face = torch.full((E,2), -1, dtype=torch.long, device=device)
    face_ids = torch.arange(F, device=device).repeat_interleave(3)
    for k in range(3*F):
        e = inv[k]
        pos = 0 if edge2face[e,0] == -1 else 1
        edge2face[e,pos] = face_ids[k]

    return canon, edge2face, face2edge


def face_gradient_mats(verts: torch.Tensor, faces: torch.LongTensor):
    """Pre-compute per-face gradient matrices."""
    v = verts[faces]                # (F,3,3)
    v0, v1, v2 = v[:,0], v[:,1], v[:,2]
    e1, e2 = v1 - v0, v2 - v0       # (F,3)

    # Normal and double area
    n   = torch.cross(e1, e2, dim=1)            # (F,3)
    n2  = (n*n).sum(1, keepdim=True)            # (F,1)
    n2[n2 < 1e-20] = 1e-20                      # avoid zero‑division

    # Gradient computation matrices
    c1 = torch.cross(e2, n, dim=1) / n2         # (F,3)
    c2 = torch.cross(n, e1, dim=1) / n2         # (F,3)

    # Assemble 3×3 matrix
    B = torch.stack([-c1-c2, c1, c2], dim=2)    # (F,3,3)
    return B


def loss_balanced(Fv, beta, lambda_adj, lambda_tv, lambda_area,
                  verts, faces, edges, edge2face, B, face_areas,
                  normalize_by_edges=True):
    """
    Balanced loss that prevents both explosion and over-normalization.
    
    Args:
        normalize_by_edges: If True, normalize adj and tv losses by number of edges
                           to make loss scale-invariant with mesh resolution
    """
    V, C = Fv.shape
    E = edges.shape[0]
    eps = 1e-10

    # (A) Area balance - this is already properly normalized
    Pv = torch.softmax(beta * Fv, dim=1)              # (V,C)
    Pf = Pv[faces].mean(dim=1)                        # (F,C)

    area_fractions = (Pf.T * face_areas).sum(dim=1)   # (C,)
    area_fractions = area_fractions / face_areas.sum()

    area_loss = lambda_area * torch.abs(area_fractions - 1.0/C).sum()

    # Pre-compute per-face gradients
    Ff = Fv[faces]                                    # (F,3,C)
    grads = []
    for c in range(C):
        gc = torch.einsum('fij,fj->fi', B, Ff[:,:,c]) # (F,3)
        grads.append(gc)
    grads = torch.stack(grads, dim=2)                 # (F,3,C)

    # Iterate over channel pairs
    adj_loss = 0.0
    tv_loss  = 0.0
    num_channel_pairs = C * (C - 1) // 2

    v_idx0, v_idx1 = edges[:,0], edges[:,1]

    for a in range(C-1):
        Fa = Fv[:,a]
        for b in range(a+1, C):
            Fb = Fv[:,b]
            d  = Fa - Fb

            d_i = d[v_idx0]
            d_j = d[v_idx1]

            # Soft boundary weight
            w_e = torch.sigmoid(-beta * d_i * d_j)

            # (B) Adjacent direction term
            f1 = edge2face[:,0]
            f2 = edge2face[:,1]
            valid = (f1>=0) & (f2>=0)
            
            if valid.any():
                g1 = grads[f1[valid], :, a] - grads[f1[valid], :, b]
                g2 = grads[f2[valid], :, a] - grads[f2[valid], :, b]

                dot = (g1 * g2).sum(dim=1)
                n1  = g1.norm(dim=1) + eps
                n2  = g2.norm(dim=1) + eps
                cos = torch.clamp(dot / (n1 * n2), -1.0, 1.0)

                # Sum over valid edges for this channel pair
                adj_loss += (w_e[valid] * (1 - cos)).sum()

            # (C) Gated TV term - sum over all edges
            tv_loss += ((1 - w_e) * (d_i - d_j).pow(2)).sum()

    # Apply lambda weights
    adj_loss = lambda_adj * adj_loss
    tv_loss = lambda_tv * tv_loss
    
    # Normalize by mesh size if requested
    if normalize_by_edges:
        # Normalize by total edge-channel pairs to be scale invariant
        # This prevents explosion on large meshes while maintaining reasonable scale
        normalizer = E * num_channel_pairs
        adj_loss = adj_loss / normalizer
        tv_loss = tv_loss / normalizer

    return area_loss + adj_loss + tv_loss, \
           {'area': area_loss.item(),
            'adj' : adj_loss.item(),
            'tv'  : tv_loss.item()}


class MeshLossBalanced:
    """Balanced loss computation that works well across mesh sizes."""
    
    def __init__(self, verts: torch.Tensor, faces: torch.LongTensor, device='cuda'):
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Build topology
        edges_cpu, edge2face_cpu, face2edge_cpu = build_edge_topology(faces.cpu())
        
        # Move to device
        self.verts = verts.to(self.device)
        self.faces = faces.to(self.device)
        self.edges = edges_cpu.to(self.device)
        self.edge2face = edge2face_cpu.to(self.device)
        self.face2edge = face2edge_cpu.to(self.device)
        
        # Pre-compute
        self.B = face_gradient_mats(self.verts, self.faces).to(self.device)
        
        # Face areas
        v0, v1, v2 = self.verts[self.faces[:,0]], self.verts[self.faces[:,1]], self.verts[self.faces[:,2]]
        self.face_areas = 0.5 * torch.linalg.norm(
            torch.cross(v1-v0, v2-v0, dim=1), dim=1)
        
        # Mesh statistics for adaptive normalization
        self.num_edges = len(self.edges)
        self.num_faces = len(self.faces)
        self.mesh_scale = self.verts.max() - self.verts.min()
        
        print(f"Mesh stats: {self.num_edges} edges, {self.num_faces} faces, scale={self.mesh_scale:.3f}")
    
    def compute_loss(self, Fv: torch.Tensor, beta: float, 
                     lambda_adj: float = 5.0, lambda_tv: float = 0.05, 
                     lambda_area: float = 1.0,
                     normalize_by_edges: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the balanced three-term loss.
        
        Args:
            normalize_by_edges: If True, normalize by mesh size for scale invariance
        """
        return loss_balanced(
            Fv, beta, lambda_adj, lambda_tv, lambda_area,
            self.verts, self.faces, self.edges, self.edge2face, 
            self.B, self.face_areas, normalize_by_edges
        )
    
    def get_schedules(self, t: int, total_steps: int) -> Dict[str, float]:
        """Get recommended hyperparameter schedules with mesh-aware scaling."""
        warmup_steps = int(0.2 * total_steps)
        warm_frac = min(1.0, t / warmup_steps)
        
        beta_start, beta_end = 2.0, 25.0
        
        # Scale lambda_adj based on mesh size
        # Larger meshes need higher lambda_adj to maintain similar effect
        scale_factor = np.sqrt(self.num_edges / 1000.0)  # Normalize to ~1000 edge mesh
        adj_start = 0.0
        adj_end = 5.0 * scale_factor
        
        beta = beta_start + (beta_end - beta_start) * warm_frac
        lambda_adj = adj_start + (adj_end - adj_start) * warm_frac
        
        return {
            'beta': beta,
            'lambda_adj': lambda_adj,
            'scale_factor': scale_factor
        }


if __name__ == "__main__":
    import numpy as np
    
    # Test on meshes of different sizes
    print("Testing balanced loss on different mesh sizes...")
    
    for n_verts in [100, 1000, 10000]:
        print(f"\nTesting with {n_verts} vertices...")
        
        # Create random mesh
        verts = torch.randn(n_verts, 3)
        faces = torch.randint(0, n_verts, (n_verts * 2, 3))
        
        # Initialize
        loss_module = MeshLossBalanced(verts, faces, device='cpu')
        
        # Test field
        Fv = torch.randn(n_verts, 6, requires_grad=True)
        
        # Compute with normalization
        loss_norm, parts_norm = loss_module.compute_loss(Fv, beta=10.0, normalize_by_edges=True)
        print(f"  With normalization: {loss_norm.item():.6f} (area={parts_norm['area']:.4f}, adj={parts_norm['adj']:.4f}, tv={parts_norm['tv']:.4f})")
        
        # Compute without normalization  
        loss_raw, parts_raw = loss_module.compute_loss(Fv, beta=10.0, normalize_by_edges=False)
        print(f"  Without normalization: {loss_raw.item():.6f} (area={parts_raw['area']:.4f}, adj={parts_raw['adj']:.4f}, tv={parts_raw['tv']:.4f})")