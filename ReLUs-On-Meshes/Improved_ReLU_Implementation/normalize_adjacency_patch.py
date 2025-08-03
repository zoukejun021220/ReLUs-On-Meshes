#!/usr/bin/env python3
"""
Simple patch to normalize the adjacency loss in your existing code.
This directly addresses why the raw adjacency term is so large.
"""
import torch
from typing import Optional


def adjacency_loss_normalized(grad15: torch.Tensor, edge2face: torch.Tensor, w_e: torch.Tensor,
                              face_mask: Optional[torch.Tensor], lambda_adj: float, 
                              normalize: bool = True) -> torch.Tensor:
    """
    Normalized version of adjacency loss that produces reasonable raw values.
    
    The original loss sums over:
    - E edges (e.g., 24,000)
    - 15 channel pairs
    - penalty up to 2.0
    
    This gives max raw value = 24,000 × 15 × 2 = 720,000!
    
    With normalization, the raw value stays in [0, 2] range.
    """
    eps = 1e-8
    clamp_val = 0.999999
    
    f1, f2 = edge2face.T
    
    # Only consider interior edges (both faces valid)
    interior = (f1 >= 0) & (f2 >= 0)
    
    # Also filter out edges adjacent to degenerate faces
    if face_mask is not None:
        deg1 = (f1 >= 0) & (~face_mask[f1])
        deg2 = (f2 >= 0) & (~face_mask[f2])
        interior = interior & ~(deg1 | deg2)
    
    if interior.sum() == 0:
        return torch.tensor(0.0, device=grad15.device)
    
    # Get gradient pairs for interior edges
    g1 = grad15[f1[interior]]  # (E_interior, 3, 15)
    g2 = grad15[f2[interior]]  # (E_interior, 3, 15)
    
    # Compute norms
    n1 = g1.norm(dim=1)  # (E_interior, 15)
    n2 = g2.norm(dim=1)  # (E_interior, 15)
    
    # Mask out edges whose gradients are both ~0
    valid = (n1 > eps) & (n2 > eps)  # (E_interior, 15)
    
    if valid.sum() == 0:
        return torch.tensor(0.0, device=grad15.device)
    
    # Compute dot product and cosine similarity for all edges
    dot_prod = (g1 * g2).sum(dim=1)  # (E_interior, 15)
    
    # Compute cosine similarity in fp32 for stability
    cos_theta = (dot_prod / (n1 * n2 + 1e-10)).float()
    cos_theta = cos_theta.clamp(-clamp_val, clamp_val)  # Clamp to (-1, 1)
    
    # Get corresponding weights
    w_interior = w_e[interior]  # (E_interior, 15)
    
    # Apply valid mask and compute loss
    valid_contribution = valid.float() * w_interior * (1.0 - cos_theta)
    
    # Sum the raw loss
    L_adj_raw = valid_contribution.sum()
    
    # NORMALIZATION: Divide by what we summed over
    if normalize:
        num_interior_edges = interior.sum().item()
        num_channel_pairs = 15  # 6 choose 2
        normalizer = max(num_interior_edges * num_channel_pairs, 1)
        L_adj_normalized = L_adj_raw / normalizer
        
        # Print diagnostics every so often
        if torch.rand(1).item() < 0.01:  # 1% chance to print
            print(f"  [Adjacency] Raw={L_adj_raw:.1f}, Edges={num_interior_edges}, "
                  f"Normalized={L_adj_normalized:.4f}, λ={lambda_adj:.2f}")
        
        return lambda_adj * L_adj_normalized
    else:
        # Original unnormalized version
        return lambda_adj * L_adj_raw


def patch_compute_total_loss(compute_total_loss_fn):
    """
    Monkey-patch the existing compute_total_loss to use normalized adjacency.
    """
    def compute_total_loss_normalized(f_values, vertices, faces, edges, triangle_adjacency,
                                     face_mask, beta, lambda_adj, lambda_tv, lambda_area,
                                     pinned_axes=None, use_grad_norm=False, step=0,
                                     tv_clip=2e2):
        """
        Patched version with normalized adjacency loss.
        """
        from loss_functions import (
            compute_pairwise_differences,
            compute_edge_weights,
            compute_face_gradients,
            gated_tv_loss,
            area_balance_loss,
            compute_edge2face,
            compute_barycentric_matrices
        )
        
        # Original preprocessing
        d_v, pairs = compute_pairwise_differences(f_values)
        w_e = compute_edge_weights(d_v, edges, beta)
        edge2face = compute_edge2face(faces, edges.cpu().numpy())
        edge2face = torch.from_numpy(edge2face).to(f_values.device)
        B, face_areas, face_mask = compute_barycentric_matrices(vertices, faces)
        grad15 = compute_face_gradients(f_values, faces, B, pairs)
        
        # Use normalized adjacency loss
        L_adj = adjacency_loss_normalized(grad15, edge2face, w_e, face_mask, lambda_adj, normalize=True)
        
        # Original TV and area losses
        L_tv = gated_tv_loss(d_v, edges, w_e, edge2face, face_mask, lambda_tv, tv_clip=tv_clip)
        L_area = area_balance_loss(f_values, faces, face_areas, pairs, beta, lambda_area)
        
        # Handle pinned axes if provided
        if pinned_axes is not None:
            from loss_functions import compute_contour_alignment_loss
            L_contour = compute_contour_alignment_loss(
                f_values, vertices, edges, beta, pinned_axes
            )
            total = L_area + L_adj + L_tv + L_contour
            components = {
                'area_balance': L_area.item() if hasattr(L_area, 'item') else L_area,
                'adjacency': L_adj.item() if hasattr(L_adj, 'item') else L_adj,
                'tv': L_tv.item() if hasattr(L_tv, 'item') else L_tv,
                'contour': L_contour.item() if hasattr(L_contour, 'item') else L_contour
            }
        else:
            total = L_area + L_adj + L_tv
            components = {
                'area_balance': L_area.item() if hasattr(L_area, 'item') else L_area,
                'adjacency': L_adj.item() if hasattr(L_adj, 'item') else L_adj,
                'tv': L_tv.item() if hasattr(L_tv, 'item') else L_tv
            }
        
        return total, components
    
    return compute_total_loss_normalized


# Simple test
if __name__ == "__main__":
    print("="*60)
    print("ADJACENCY LOSS NORMALIZATION PATCH")
    print("="*60)
    print("\nProblem: Raw adjacency = E × 15 × 2 ≈ 720,000")
    print("Solution: Normalize by (num_edges × num_pairs)")
    print("Result: Raw adjacency ∈ [0, 2] instead of [0, 720,000]")
    print("\nTo use this patch in your code:\n")
    print("from normalize_adjacency_patch import patch_compute_total_loss")
    print("from loss_functions import compute_total_loss")
    print("compute_total_loss = patch_compute_total_loss(compute_total_loss)")
    print("\nOr directly replace adjacency_loss with adjacency_loss_normalized")