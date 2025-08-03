#!/usr/bin/env python3
"""
Fixed ground-up implementation with proper normalization.
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


def loss_revised_normalized(Fv, beta, lambda_adj, lambda_tv, lambda_area,
                           verts, faces, edges, edge2face, B, face_areas):
    """
    Fixed loss with proper normalization.
    """
    V, C = Fv.shape
    eps = 1e-10

    # (A) Area balance
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
    adj_count = 0

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

                # Normalize by face areas for scale invariance
                area1 = face_areas[f1[valid]]
                area2 = face_areas[f2[valid]]
                avg_area = (area1 + area2) / 2
                
                # Scale gradients by sqrt(area) for proper normalization
                g1 = g1 * avg_area.sqrt().unsqueeze(1)
                g2 = g2 * avg_area.sqrt().unsqueeze(1)

                dot = (g1 * g2).sum(dim=1)
                n1  = g1.norm(dim=1) + eps
                n2  = g2.norm(dim=1) + eps
                cos = torch.clamp(dot / (n1 * n2), -1.0, 1.0)

                # Accumulate with normalization
                adj_loss += (w_e[valid] * (1 - cos)).mean()  # Mean instead of sum
                adj_count += 1

            # (C) Gated TV term
            tv_loss += ((1 - w_e) * (d_i - d_j).pow(2)).mean()  # Mean instead of sum

    # Normalize by number of channel pairs
    if adj_count > 0:
        adj_loss = lambda_adj * adj_loss / adj_count
        tv_loss = lambda_tv * tv_loss / adj_count

    return area_loss + adj_loss + tv_loss, \
           {'area': area_loss.item(),
            'adj' : adj_loss.item(),
            'tv'  : tv_loss.item()}


class MeshLossGroundUpFixed:
    """Fixed version with proper normalization."""
    
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
    
    def compute_loss(self, Fv: torch.Tensor, beta: float, 
                     lambda_adj: float = 5.0, lambda_tv: float = 0.05, 
                     lambda_area: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """Compute the normalized three-term loss."""
        return loss_revised_normalized(
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