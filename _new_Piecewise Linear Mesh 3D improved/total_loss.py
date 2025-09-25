#from channelwiseplaneCL import contour_alignment_loss
#from channelwiseSegmentPLaneCL import contour_alignment_loss
#from pairwise_segmentCL import contour_alignment_loss
#from PairwisePlaneCL import contour_alignment_loss
from pairPLaneCL import contour_alignment_loss
from smoothnessArea import smoothness_loss_optimized, area_balance_loss_optimized




def compute_combined_loss_optimized(f_values, points, triangles, adjacency, vertex_edges,
                                  mesh_area, beta=20.0, lambda_contour=1.0, lambda_smooth=1.0,
                                  lambda_area=1.0,pinned_axes=None,plane_offsets=None):
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
    
        # Use the improved label subdivision approach for contour alignment
    contour_loss = contour_alignment_loss(
        points, triangles, f_values, beta=beta, include_triples=True, adajancy=adjacency,
        beta_edge=beta, beta_triple=beta,  eps=1e-9, lambda_plane=1.0,
        lambda_contour=lambda_contour, pinned_axes=pinned_axes, plane_offsets=plane_offsets,
            )
    
        
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