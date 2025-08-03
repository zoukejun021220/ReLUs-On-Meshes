#!/usr/bin/env python3
"""
Ground-up implementation of the three-term loss function following the guide.
Clean, self-contained implementation with no external dependencies.
"""

import torch
import math
from typing import Tuple, Dict


def build_edge_topology(faces: torch.LongTensor):
    """
    Build edge connectivity from faces.
    
    Returns:
      edges          – (E,2)  int64  unordered vertex pairs
      edge2face      – (E,2)  int64  incident face indices (-1 if none)
      face2edge      – (F,3)  int64  edge indices for the three sides
    """
    device = faces.device
    F = faces.shape[0]

    # 1. All three directed edges of every face
    v0, v1, v2 = faces[:,0], faces[:,1], faces[:,2]
    directed = torch.stack([torch.stack([v0,v1],1),
                            torch.stack([v1,v2],1),
                            torch.stack([v2,v0],1)],1)        # (F,3,2)

    # 2. Canonical (unordered) edges so v_low < v_high
    flat = directed.reshape(-1,2)                             # (3F,2)
    low  = torch.min(flat[:,0], flat[:,1])
    high = torch.max(flat[:,0], flat[:,1])
    canon = torch.stack([low, high],1)                        # (3F,2)

    # 3. Unique edges
    canon, inv = torch.unique(canon, return_inverse=True, dim=0)
    E = canon.shape[0]

    # 4. face→edge lookup
    face2edge = inv.view(F,3)

    # 5. edge→(face₁,face₂)
    edge2face = torch.full((E,2), -1, dtype=torch.long, device=device)
    # Which face produced each directed edge?
    face_ids = torch.arange(F, device=device).repeat_interleave(3)
    # Fill first empty slot per edge
    for k in range(3*F):
        e = inv[k]
        pos = 0 if edge2face[e,0] == -1 else 1
        edge2face[e,pos] = face_ids[k]

    return canon, edge2face, face2edge


def face_gradient_mats(verts: torch.Tensor, faces: torch.LongTensor):
    """
    Pre-compute per-face gradient matrices.
    
    Returns B  – (F,3,3)  float32
    """
    v = verts[faces]                # (F,3,3)
    v0, v1, v2 = v[:,0], v[:,1], v[:,2]
    e1, e2 = v1 - v0, v2 - v0       # (F,3)

    # Normal and double area
    n   = torch.cross(e1, e2, dim=1)            # (F,3)
    n2  = (n*n).sum(1, keepdim=True)            # (F,1)
    n2[n2 < 1e-20] = 1e-20                      # avoid zero‑division

    # Formula: g = ( (f1-f0)*(e2×n) + (f2-f0)*(n×e1) ) / ‖n‖²
    c1 = torch.cross(e2, n, dim=1) / n2         # (F,3)
    c2 = torch.cross(n, e1, dim=1) / n2         # (F,3)

    # Assemble 3×3 matrix multiplying [f0,f1,f2]^T
    B = torch.stack([-c1-c2, c1, c2], dim=2)    # (F,3,3)
    return B                                    # store on CPU


def loss_revised(Fv, beta, lambda_adj, lambda_tv, lambda_area,
                 verts, faces, edges, edge2face, B, face_areas):
    """
    Implements Sections (A) area, (B) adjacent‑direction, (C) gated TV.
    All tensors must share the same device/dtype.
    
    Args:
        Fv: (V, C) vertex field values
        beta: temperature parameter
        lambda_adj: weight for adjacent direction term
        lambda_tv: weight for total variation term
        lambda_area: weight for area balance term
        verts: (V, 3) vertex positions
        faces: (F, 3) face indices
        edges: (E, 2) edge vertex pairs
        edge2face: (E, 2) incident faces for each edge
        B: (F, 3, 3) gradient matrices
        face_areas: (F,) face areas
    
    Returns:
        loss: total loss
        parts: dict with individual loss components
    """
    V, C = Fv.shape
    eps = 1e-10

    # -------------------------------------------------- (A) area balance
    # Soft region probabilities (beta‑scaled softmax)
    Pv = torch.softmax(beta * Fv, dim=1)              # (V,C)

    # Convert vertex probs to face probs by averaging the 3 vertices
    Pf = Pv[faces].mean(dim=1)                        # (F,C)

    area_fractions = (Pf.T * face_areas).sum(dim=1)   # (C,)
    area_fractions = area_fractions / face_areas.sum()

    area_loss = lambda_area * torch.abs(
        area_fractions - 1.0/C).sum()

    # -------------------------------------------------- pre‑compute per‑face grads of each channel
    # Fv_per_face : (F,3,C)
    Ff = Fv[faces]                                    # (F,3,C)
    # Compute grads one channel at a time to limit memory
    grads = []
    for c in range(C):
        gc = torch.einsum('fij,fj->fi', B, Ff[:,:,c]) # (F,3)
        grads.append(gc)
    grads = torch.stack(grads, dim=2)                 # (F,3,C)

    # -------------------------------------------------- iterate over channel pairs
    adj_loss = 0.0
    tv_loss  = 0.0

    v_idx0, v_idx1 = edges[:,0], edges[:,1]

    for a in range(C-1):
        Fa = Fv[:,a]          # (V,)
        for b in range(a+1, C):
            Fb = Fv[:,b]
            d  = Fa - Fb      # (V,)

            d_i = d[v_idx0]   # (E,)
            d_j = d[v_idx1]

            # Soft boundary weight
            w_e = torch.sigmoid(-beta * d_i * d_j)    # (E,)

            # -- (B) adjacent‑direction term
            f1 = edge2face[:,0]
            f2 = edge2face[:,1]
            valid = (f1>=0) & (f2>=0)                 # ignore open edges
            if valid.any():
                g1 = grads[f1[valid], :, a] - grads[f1[valid], :, b]  # (Ev,3)
                g2 = grads[f2[valid], :, a] - grads[f2[valid], :, b]  # (Ev,3)

                dot = (g1 * g2).sum(dim=1)
                n1  = g1.norm(dim=1) + eps
                n2  = g2.norm(dim=1) + eps
                cos = dot / (n1 * n2)

                adj_loss += lambda_adj * \
                    (w_e[valid] * (1 - cos)).sum()

            # -- (C) gated total variation term
            tv_loss += lambda_tv * \
                ((1 - w_e) * (d_i - d_j).pow(2)).sum()

    return area_loss + adj_loss + tv_loss, \
           {'area': area_loss.item(),
            'adj' : adj_loss.item(),
            'tv'  : tv_loss.item()}


class MeshLossGroundUp:
    """
    Convenience class to encapsulate the mesh preprocessing and loss computation.
    """
    
    def __init__(self, verts: torch.Tensor, faces: torch.LongTensor, device='cuda'):
        """
        Initialize with mesh data.
        
        Args:
            verts: (V, 3) vertex positions
            faces: (F, 3) face indices
            device: torch device
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # Build topology on CPU first
        edges_cpu, edge2face_cpu, face2edge_cpu = build_edge_topology(faces.cpu())
        
        # Move everything to device
        self.verts = verts.to(self.device)
        self.faces = faces.to(self.device)
        self.edges = edges_cpu.to(self.device)
        self.edge2face = edge2face_cpu.to(self.device)
        self.face2edge = face2edge_cpu.to(self.device)
        
        # Pre-compute gradient matrices
        self.B = face_gradient_mats(self.verts, self.faces).to(self.device)
        
        # Pre-compute face areas
        v0, v1, v2 = self.verts[self.faces[:,0]], self.verts[self.faces[:,1]], self.verts[self.faces[:,2]]
        self.face_areas = 0.5 * torch.linalg.norm(
            torch.cross(v1-v0, v2-v0, dim=1), dim=1)  # (F,)
    
    def compute_loss(self, Fv: torch.Tensor, beta: float, 
                     lambda_adj: float = 5.0, lambda_tv: float = 0.05, 
                     lambda_area: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the three-term loss.
        
        Args:
            Fv: (V, C) vertex field values
            beta: temperature parameter
            lambda_adj: weight for adjacent direction term
            lambda_tv: weight for total variation term
            lambda_area: weight for area balance term
            
        Returns:
            loss: total loss
            parts: dict with individual loss components
        """
        return loss_revised(
            Fv, beta, lambda_adj, lambda_tv, lambda_area,
            self.verts, self.faces, self.edges, self.edge2face, 
            self.B, self.face_areas
        )
    
    def get_schedules(self, t: int, total_steps: int) -> Dict[str, float]:
        """
        Get recommended hyperparameter schedules.
        
        Args:
            t: current step
            total_steps: total training steps
            
        Returns:
            dict with beta, lambda_adj values
        """
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


if __name__ == "__main__":
    # Unit test on a toy mesh
    print("Running unit test on toy mesh...")
    
    # Tiny square split into two triangles
    verts = torch.tensor([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=torch.float)
    faces = torch.tensor([[0,1,2],[0,2,3]])
    Fv = torch.randn(len(verts), 6, requires_grad=True)
    
    # Build topology on CPU
    edges, e2f, f2e = build_edge_topology(faces)
    B = face_gradient_mats(verts, faces)
    
    # One loss pass
    loss, parts = loss_revised(Fv, beta=25, lambda_adj=5, lambda_tv=0.05,
                               lambda_area=1.0,
                               verts=verts, faces=faces,
                               edges=edges, edge2face=e2f, B=B,
                               face_areas=torch.ones(len(faces)))
    loss.backward()
    print(f"Loss: {float(loss):.6f}")
    print(f"Components: {parts}")
    print("Unit test passed!")
    
    # Test convenience class
    print("\nTesting convenience class...")
    mesh_loss = MeshLossGroundUp(verts, faces, device='cpu')
    Fv_new = torch.randn(len(verts), 6, requires_grad=True)
    loss2, parts2 = mesh_loss.compute_loss(Fv_new, beta=25)
    print(f"Loss: {float(loss2):.6f}")
    print(f"Components: {parts2}")
    print("Convenience class test passed!")