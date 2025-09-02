#!/usr/bin/env python3
"""
Debug script to understand why losses are growing.
"""
import torch
import numpy as np
from mesh_utils import load_mesh_from_file, compute_mesh_data
from loss_functions import compute_total_loss

def debug_loss_computation():
    """Debug the loss computation step by step."""
    
    print("="*70)
    print("DEBUGGING LOSS COMPUTATION")
    print("="*70)
    
    # Load a small test mesh
    from mesh_utils import create_icosphere_mesh
    vertices, faces = create_icosphere_mesh(target_points=1000)
    
    print(f"\nMesh stats:")
    print(f"  Vertices: {vertices.shape[0]}")
    print(f"  Faces: {faces.shape[0]}")
    
    # Compute mesh data
    mesh_data = compute_mesh_data(vertices, faces)
    
    # Convert to torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vertices = torch.from_numpy(vertices).float().to(device)
    faces = torch.from_numpy(faces).long().to(device)
    edges = torch.from_numpy(mesh_data['edges']).long().to(device)
    edge2face = torch.from_numpy(mesh_data['edge2face']).long().to(device)
    face_areas = torch.from_numpy(mesh_data['face_areas']).float().to(device)
    B = torch.from_numpy(mesh_data['B']).float().to(device)
    
    print(f"  Edges: {edges.shape[0]}")
    
    # Initialize field values with different patterns
    num_vertices = vertices.shape[0]
    
    print("\n" + "="*70)
    print("TESTING DIFFERENT FIELD INITIALIZATIONS")
    print("="*70)
    
    test_cases = [
        ("Random small", torch.randn(num_vertices, 6, device=device) * 0.1),
        ("Random normal", torch.randn(num_vertices, 6, device=device)),
        ("Random large", torch.randn(num_vertices, 6, device=device) * 5.0),
        ("Zeros", torch.zeros(num_vertices, 6, device=device)),
        ("Axis-aligned", torch.eye(6, device=device).unsqueeze(0).expand(num_vertices, -1, -1)[:, :, 0])
    ]
    
    for name, f_values in test_cases:
        print(f"\n{name} initialization:")
        print(f"  f_values range: [{f_values.min():.3f}, {f_values.max():.3f}]")
        print(f"  f_values std: {f_values.std():.3f}")
        
        # Test with different parameters
        for beta in [1.0, 10.0]:
            for lambda_adj in [0.0, 0.1, 1.0]:
                loss_dict = compute_total_loss(
                    f_values,
                    vertices,
                    faces,
                    edges,
                    edge2face,
                    face_areas,
                    B,
                    face_mask=mesh_data.get('face_mask'),
                    beta=beta,
                    lambda_area=1.0,
                    lambda_adj=lambda_adj,
                    lambda_tv=0.1,
                    return_components=True
                )
                
                print(f"  β={beta:4.1f}, λ_adj={lambda_adj:3.1f}: "
                      f"Total={loss_dict['total'].item():8.4f}, "
                      f"Area={loss_dict['area'].item():8.4f}, "
                      f"Adj={loss_dict['adjacency'].item():8.4f}, "
                      f"TV={loss_dict['tv'].item():8.4f}")
                
                # Calculate raw adjacency
                if lambda_adj > 0:
                    raw_adj = loss_dict['adjacency'].item() / lambda_adj
                    print(f"                    Raw Adj={raw_adj:8.4f}")
    
    print("\n" + "="*70)
    print("CHECKING GRADIENT BEHAVIOR")
    print("="*70)
    
    # Test gradient computation
    f_values = torch.randn(num_vertices, 6, device=device, requires_grad=True)
    
    loss_dict = compute_total_loss(
        f_values,
        vertices,
        faces,
        edges,
        edge2face,
        face_areas,
        B,
        beta=10.0,
        lambda_area=1.0,
        lambda_adj=0.5,
        lambda_tv=0.1,
        return_components=True
    )
    
    total_loss = loss_dict['total']
    total_loss.backward()
    
    grad_norm = f_values.grad.norm().item()
    grad_max = f_values.grad.abs().max().item()
    
    print(f"\nGradient stats:")
    print(f"  Grad norm: {grad_norm:.4f}")
    print(f"  Grad max: {grad_max:.4f}")
    print(f"  Grad mean: {f_values.grad.abs().mean().item():.4f}")
    
    # Check if gradients are reasonable
    if grad_norm > 100:
        print("  ⚠️ WARNING: Gradient norm is very large!")
    if grad_max > 10:
        print("  ⚠️ WARNING: Maximum gradient is very large!")
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Calculate theoretical bounds
    num_edges = edges.shape[0]
    num_interior = (edge2face[:, 0] >= 0) & (edge2face[:, 1] >= 0)
    num_interior_edges = num_interior.sum().item()
    
    print(f"\nTheoretical adjacency loss bounds:")
    print(f"  Number of edges: {num_edges}")
    print(f"  Number of interior edges: {num_interior_edges}")
    print(f"  Number of channel pairs: 15")
    print(f"  Max contribution per edge-pair: 2.0 (when cos(θ) = -1)")
    print(f"  Unnormalized max: {num_interior_edges * 15 * 2.0:.0f}")
    print(f"  Normalized max: {2.0:.1f}")
    
    # Check actual normalization
    from loss_functions import adjacency_loss, compute_pairwise_differences, compute_edge_weights, compute_face_gradients
    
    f_values_test = torch.randn(num_vertices, 6, device=device)
    d_v, pairs = compute_pairwise_differences(f_values_test)
    w_e = compute_edge_weights(d_v, edges, beta=10.0)
    grad15 = compute_face_gradients(f_values_test, faces, B, pairs)
    
    # Test with lambda=1 to get raw value
    adj_loss = adjacency_loss(grad15, edge2face, w_e, None, lambda_adj=1.0)
    print(f"\nActual adjacency loss (λ=1.0): {adj_loss.item():.4f}")
    
    if adj_loss.item() > 10:
        print("  ⚠️ ERROR: Adjacency loss is not properly normalized!")
        print("  Expected range: [0, 2]")
        print(f"  Actual value: {adj_loss.item():.4f}")

if __name__ == "__main__":
    debug_loss_computation()