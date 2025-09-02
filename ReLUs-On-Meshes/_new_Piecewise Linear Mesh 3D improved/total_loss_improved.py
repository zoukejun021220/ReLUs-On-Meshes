from anchoredPlaneCL import contour_alignment_loss_anchored
from smoothnessArea import smoothness_loss_optimized, area_balance_loss_optimized


def compute_combined_loss_improved(
    f_values, points, triangles, adjacency, vertex_edges,
    mesh_area, pinned_axes, plane_offsets,
    beta=20.0, lambda_contour=1.0, lambda_smooth=1.0,
    lambda_area=1.0, use_anchored_loss=True, include_triples=False
):
    """
    Compute the combined loss with improved stability using anchored planes.
    
    Args:
        f_values: Tensor of shape (N, 6) containing scalar field values
        points: Tensor of shape (N, 3) containing vertex positions
        triangles: Tensor of shape (T, 3) containing triangle vertex indices
        adjacency: Tensor of shape (E, 2) containing adjacent triangle pairs
        vertex_edges: Tensor of shape (E', 2) containing vertex edge indices
        mesh_area: Total mesh area
        pinned_axes: Tensor of shape (6, 3) containing fixed plane normals
        plane_offsets: Tensor of shape (6,) containing learnable plane offsets
        beta: Softmax temperature parameter
        lambda_contour: Weight for contour alignment loss
        lambda_smooth: Weight for smoothness loss
        lambda_area: Weight for area balance loss
        use_anchored_loss: If True, use anchored planes; if False, use original SVD
        include_triples: Whether to include triple intersections
        
    Returns:
        total_loss: Combined loss value
        loss_dict: Dictionary containing individual loss components
    """
    # Compute contour loss
    if use_anchored_loss:
        # Use the stable anchored planes loss
        contour_loss = contour_alignment_loss_anchored(
            points, triangles, f_values, pinned_axes, plane_offsets,
            beta_edge=beta, include_triples=include_triples
        )
    else:
        # Fall back to original loss if requested
        from pairPLaneCL import contour_alignment_loss
        contour_loss = contour_alignment_loss(
            points, triangles, f_values, pinned_axes,
            beta=beta, include_triples=include_triples,
            adajancy=adjacency, plane_offsets=plane_offsets
        )
    
    # Compute other losses
    smooth_loss = smoothness_loss_optimized(f_values, vertex_edges)
    area_loss, area_fracs = area_balance_loss_optimized(
        points, triangles, f_values, beta, mesh_area
    )
    
    # Combine losses
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