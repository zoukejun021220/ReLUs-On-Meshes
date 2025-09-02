#!/usr/bin/env python3
"""
Analyze contour alignment quality in 3D space.
"""
import numpy as np
import torch
import argparse
from pathlib import Path

def analyze_3d_alignment(checkpoint_path: str):
    """Analyze the quality of boundary alignment in 3D."""
    
    # Load checkpoint
    data = np.load(checkpoint_path)
    
    # Extract data
    F = torch.tensor(data['field_values'], dtype=torch.float32)
    verts = torch.tensor(data['vertices'], dtype=torch.float32)
    faces = torch.tensor(data['faces'], dtype=torch.long)
    edge_idx = torch.tensor(data['edge_idx'], dtype=torch.long)
    
    # Get mesh connectivity
    from utils.mesh_preprocessing import preprocess_mesh
    import meshzoo
    
    # Build edge-triangle connectivity
    T = faces.shape[0]
    E = edge_idx.shape[0]
    
    # Create edge to triangle map
    edge_tris = torch.full((E, 2), -1, dtype=torch.long)
    edge_set = {(min(e[0].item(), e[1].item()), max(e[0].item(), e[1].item())): i 
                for i, e in enumerate(edge_idx)}
    
    for t_idx, tri in enumerate(faces):
        v0, v1, v2 = tri.tolist()
        edges = [(min(v0, v1), max(v0, v1)),
                 (min(v1, v2), max(v1, v2)),
                 (min(v2, v0), max(v2, v0))]
        
        for edge in edges:
            if edge in edge_set:
                e_idx = edge_set[edge]
                if edge_tris[e_idx, 0] == -1:
                    edge_tris[e_idx, 0] = t_idx
                else:
                    edge_tris[e_idx, 1] = t_idx
    
    # Compute active edges
    beta = float(data.get('beta_contour', 8.0))
    C = F.shape[1]
    
    # Get top 2 channels per edge
    f_mid = 0.5 * (F[edge_idx[:, 0]] + F[edge_idx[:, 1]])
    top_vals, top_idx = torch.topk(f_mid, k=2, dim=1)
    chan_i = top_idx[:, 0]
    chan_j = top_idx[:, 1]
    
    # Edge crossing weights
    da = F[edge_idx[:, 0], chan_i] - F[edge_idx[:, 0], chan_j]
    db = F[edge_idx[:, 1], chan_i] - F[edge_idx[:, 1], chan_j]
    w = torch.sigmoid(-beta * da * db)
    
    # Active edges
    active_mask = (w > 0.5) & (edge_tris[:, 0] >= 0) & (edge_tris[:, 1] >= 0)
    active_edges = torch.where(active_mask)[0]
    
    print(f"\nAnalyzing checkpoint: {checkpoint_path}")
    print(f"Step: {data.get('step', 'unknown')}")
    print(f"β_contour: {beta:.2f}")
    print(f"Active edges: {active_mask.sum().item()} / {E} ({100*active_mask.float().mean():.1f}%)")
    
    if len(active_edges) == 0:
        print("No active interior edges found!")
        return
    
    # Compute 3D alignment for active edges
    alignments = []
    
    # Triangle frames
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    
    e0 = v1 - v0
    e1 = v2 - v0
    
    t0 = torch.nn.functional.normalize(e0, dim=1)
    n = torch.nn.functional.normalize(torch.cross(e0, e1, dim=1), dim=1)
    t1 = torch.cross(n, t0, dim=1)
    
    for e_idx in active_edges[:1000]:  # Sample first 1000
        tL = edge_tris[e_idx, 0]
        tR = edge_tris[e_idx, 1]
        
        if tL < 0 or tR < 0:
            continue
        
        # Get channel pair for this edge
        ci = chan_i[e_idx]
        cj = chan_j[e_idx]
        
        # Height values
        h_L = F[faces[tL], ci] - F[faces[tL], cj]
        h_R = F[faces[tR], ci] - F[faces[tR], cj]
        
        # Compute 2D coordinates (simplified - use barycentric)
        # For simplicity, use edge midpoint projection
        
        # Get 3D gradient direction (simplified)
        # This is approximate - for exact, need full 2D triangle coordinates
        g_L = h_L[1] - h_L[0]  # Simplified gradient
        g_R = h_R[1] - h_R[0]
        
        # Get normals
        n_L = n[tL]
        n_R = n[tR]
        
        # Alignment is dot product of normals (simplified metric)
        align = (n_L * n_R).sum().abs().item()
        alignments.append(align)
    
    alignments = np.array(alignments)
    
    print(f"\n3D Alignment Statistics:")
    print(f"  Mean |cos θ|: {alignments.mean():.3f}")
    print(f"  Median |cos θ|: {np.median(alignments):.3f}")
    print(f"  Min |cos θ|: {alignments.min():.3f}")
    print(f"  Max |cos θ|: {alignments.max():.3f}")
    
    # Estimate average misalignment angle
    mean_cos = alignments.mean()
    mean_angle_deg = np.degrees(np.arccos(np.clip(mean_cos, -1, 1)))
    print(f"  Average misalignment: {mean_angle_deg:.1f}°")
    
    # Check margins
    margins = []
    for i in range(F.shape[0]):
        top2 = torch.topk(F[i], k=2)[0]
        margin = (top2[0] - top2[1]).item()
        margins.append(margin)
    
    margins = np.array(margins)
    low_margin_frac = (margins < 0.2).mean()
    
    print(f"\nMargin Statistics:")
    print(f"  Mean margin: {margins.mean():.3f}")
    print(f"  Median margin: {np.median(margins):.3f}")
    print(f"  Vertices with margin < 0.2: {100*low_margin_frac:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint npz file")
    args = parser.parse_args()
    
    analyze_3d_alignment(args.checkpoint)