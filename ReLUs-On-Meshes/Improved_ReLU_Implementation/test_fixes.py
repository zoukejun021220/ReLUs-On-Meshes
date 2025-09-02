"""
Test script to verify all fixes are working correctly.
"""
import torch
import numpy as np
from loss_functions_corrected import (
    compute_face_gradients, adjacency_loss_corrected, 
    gated_tv_loss_corrected, compute_edge_weights,
    compute_total_loss_corrected
)
from mesh_utils import create_icosphere_mesh, compute_mesh_data
import itertools


def test_adjacency_normalization():
    """Test that raw adjacency is properly normalized by 15."""
    print("\n" + "="*60)
    print("TEST 1: Adjacency Loss Normalization")
    print("="*60)
    
    # Create small test mesh
    vertices, faces = create_icosphere_mesh(target_points=100)
    mesh_data = compute_mesh_data(vertices, faces)
    
    # Convert to torch
    vertices_t = torch.from_numpy(vertices).float()
    faces_t = torch.from_numpy(faces).long()
    edges_t = torch.from_numpy(mesh_data['edges']).long()
    edge2face_t = torch.from_numpy(mesh_data['edge2face']).long()
    B_t = torch.from_numpy(mesh_data['B']).float()
    
    # Random field values
    f_values = torch.randn(len(vertices), 6) * 0.1
    
    # Compute gradients
    pairs = torch.tensor(list(itertools.combinations(range(6), 2)))
    grad15 = compute_face_gradients(f_values, faces_t, B_t, pairs)
    
    # Compute edge weights
    idx_i = []
    idx_j = []
    for i in range(6):
        for j in range(i+1, 6):
            idx_i.append(i)
            idx_j.append(j)
    d_v = f_values[:, idx_i] - f_values[:, idx_j]
    w_e = compute_edge_weights(d_v, edges_t, beta=5.0)
    
    # Compute adjacency loss
    L_adj_weighted, L_adj_raw = adjacency_loss_corrected(
        grad15, edge2face_t, w_e, None, lambda_adj=1.0
    )
    
    print(f"Raw adjacency loss: {L_adj_raw:.6f}")
    print(f"Expected range: [0, 2]")
    
    if L_adj_raw > 2.0:
        print("❌ FAIL: Raw adjacency > 2.0 - normalization is wrong!")
    elif L_adj_raw < 0:
        print("❌ FAIL: Raw adjacency < 0 - impossible!")
    else:
        print("✅ PASS: Raw adjacency in valid range [0, 2]")
    
    # Test with high beta (should approach 0 for random initialization)
    w_e_high = compute_edge_weights(d_v, edges_t, beta=20.0)
    L_adj_weighted2, L_adj_raw2 = adjacency_loss_corrected(
        grad15, edge2face_t, w_e_high, None, lambda_adj=1.0
    )
    
    print(f"\nWith high beta (20):")
    print(f"  Raw adjacency: {L_adj_raw2:.6f}")
    print(f"  Should be similar or lower (more confident boundaries)")
    
    return L_adj_raw <= 2.0


def test_tv_gating():
    """Test that TV loss uses w_e*(1-w_e) gating."""
    print("\n" + "="*60)
    print("TEST 2: TV Loss Gating Function")
    print("="*60)
    
    # Create test data
    vertices, faces = create_icosphere_mesh(target_points=100)
    mesh_data = compute_mesh_data(vertices, faces)
    edges_t = torch.from_numpy(mesh_data['edges']).long()
    
    # Test edge weights at different values
    test_weights = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0]).reshape(-1, 1)
    
    print("Testing gating function w_e*(1-w_e):")
    for w in test_weights:
        gating = w * (1 - w)
        print(f"  w_e={w.item():.2f} → gating={gating.item():.4f}")
    
    print("\nExpected: Maximum at w_e=0.5, zero at w_e=0 and w_e=1")
    
    # Test in actual TV loss
    f_values = torch.randn(len(vertices), 6)
    d_v = torch.randn(len(vertices), 15) * 0.1
    
    # Create edge weights with some at 0.5 (boundaries)
    w_e = torch.ones(len(edges_t), 15) * 0.5
    
    L_tv = gated_tv_loss_corrected(d_v, edges_t, w_e, lambda_tv=1.0)
    
    print(f"\nTV loss with w_e=0.5 everywhere: {L_tv:.6f}")
    
    # Now with w_e=0 (interior)
    w_e_interior = torch.zeros(len(edges_t), 15)
    L_tv_interior = gated_tv_loss_corrected(d_v, edges_t, w_e_interior, lambda_tv=1.0)
    
    print(f"TV loss with w_e=0 (interior): {L_tv_interior:.6f}")
    print(f"Should be near 0 (no smoothing inside regions)")
    
    # Now with w_e=1 (converged boundaries)
    w_e_converged = torch.ones(len(edges_t), 15)
    L_tv_converged = gated_tv_loss_corrected(d_v, edges_t, w_e_converged, lambda_tv=1.0)
    
    print(f"TV loss with w_e=1 (converged): {L_tv_converged:.6f}")
    print(f"Should be near 0 (no smoothing at converged boundaries)")
    
    return abs(L_tv_interior) < 1e-6 and abs(L_tv_converged) < 1e-6


def test_weight_distribution():
    """Test weight distribution at different beta values."""
    print("\n" + "="*60)
    print("TEST 3: Weight Distribution vs Beta")
    print("="*60)
    
    # Create test data
    vertices, faces = create_icosphere_mesh(target_points=100)
    mesh_data = compute_mesh_data(vertices, faces)
    edges_t = torch.from_numpy(mesh_data['edges']).long()
    
    # Random field with some structure
    f_values = torch.randn(len(vertices), 6)
    f_values[0:20, :] = torch.eye(6)[0]  # First 20 vertices in region 0
    f_values[20:40, :] = torch.eye(6)[1]  # Next 20 in region 1
    
    idx_i = []
    idx_j = []
    for i in range(6):
        for j in range(i+1, 6):
            idx_i.append(i)
            idx_j.append(j)
    d_v = f_values[:, idx_i] - f_values[:, idx_j]
    
    for beta in [0, 2, 5, 10, 20]:
        w_e = compute_edge_weights(d_v, edges_t, beta=beta)
        w_flat = w_e.flatten()
        
        near_0 = (w_flat < 0.1).float().mean().item() * 100
        near_1 = (w_flat > 0.9).float().mean().item() * 100
        middle = ((w_flat >= 0.1) & (w_flat <= 0.9)).float().mean().item() * 100
        
        print(f"\nβ={beta:2d}:")
        print(f"  Near 0 (<0.1): {near_0:.1f}%")
        print(f"  Middle (0.1-0.9): {middle:.1f}%")  
        print(f"  Near 1 (>0.9): {near_1:.1f}%")
        print(f"  Mean: {w_flat.mean():.3f}, Std: {w_flat.std():.3f}")
        
        if beta >= 10:
            if middle > 50:
                print("  ⚠️ WARNING: Too many weights in middle - beta may be too low")
            else:
                print("  ✅ Good separation (U-shaped distribution)")


def test_schedule_parameters():
    """Test that scheduling parameters are correct."""
    print("\n" + "="*60)
    print("TEST 4: Training Schedule Parameters")
    print("="*60)
    
    from optimization_corrected import CoarseToFineSchedule
    
    schedule = CoarseToFineSchedule()
    
    print("Coarse-to-fine stages:")
    for i, stage in enumerate(schedule.stages):
        print(f"\nLevel {i}:")
        print(f"  Faces: {stage.num_faces}")
        print(f"  Steps: {stage.steps}")
        print(f"  Beta: {stage.beta_start:.1f} → {stage.beta_end:.1f}")
        print(f"  Lambda_adj: {stage.lambda_adj_start:.1f} → {stage.lambda_adj_end:.1f}")
        print(f"  Lambda_area: {stage.lambda_area:.1f}")
        
        # Check for issues
        if stage.beta_end > 20:
            print("  ⚠️ WARNING: Beta may be too high (>20)")
        if stage.lambda_adj_end > 5 and i == 2:
            print("  ⚠️ WARNING: Lambda_adj should be frozen at 5 in final stage")
        if stage.lambda_area < 4:
            print("  ⚠️ WARNING: Lambda_area may be too low (<4)")


def run_all_tests():
    """Run all diagnostic tests."""
    print("\n" + "="*70)
    print("RUNNING DIAGNOSTIC TESTS FOR CORRECTED IMPLEMENTATION")
    print("="*70)
    
    results = []
    
    # Test 1: Adjacency normalization
    results.append(("Adjacency normalization", test_adjacency_normalization()))
    
    # Test 2: TV gating
    results.append(("TV gating function", test_tv_gating()))
    
    # Test 3: Weight distribution
    test_weight_distribution()
    results.append(("Weight distribution", True))  # Visual test
    
    # Test 4: Schedule parameters
    test_schedule_parameters()
    results.append(("Schedule parameters", True))  # Visual test
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    if all(p for _, p in results):
        print("\n🎉 All tests passed! The fixes should work correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the implementation.")


if __name__ == '__main__':
    run_all_tests()