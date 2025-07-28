#!/usr/bin/env python3
"""
Test script for improved ReLU mesh optimization.
Demonstrates usage on various test meshes.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Import the optimization functions
from relus_mesh_optimization_improved import optimize_relu_mesh, optimize_multiscale
from mesh_optimization_helpers import auto_select_pins


def create_test_sphere(n_subdivisions=2):
    """Create a unit sphere mesh for testing."""
    # Simple icosphere generation
    t = (1.0 + np.sqrt(5.0)) / 2.0
    
    # Base icosahedron vertices
    vertices = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1]
    ])
    
    # Normalize to unit sphere
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    
    # Base icosahedron faces
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ])
    
    # Simple subdivision (optional)
    for _ in range(n_subdivisions):
        # Subdivide each triangle into 4
        new_vertices = list(vertices)
        new_faces = []
        
        edge_midpoints = {}
        
        for face in faces:
            # Get vertices
            v0, v1, v2 = face
            
            # Create midpoints
            edges = [(v0, v1), (v1, v2), (v2, v0)]
            mids = []
            
            for e in edges:
                key = tuple(sorted(e))
                if key not in edge_midpoints:
                    # Create new vertex at midpoint
                    mid = 0.5 * (vertices[e[0]] + vertices[e[1]])
                    mid = mid / np.linalg.norm(mid)  # Project to sphere
                    edge_midpoints[key] = len(new_vertices)
                    new_vertices.append(mid)
                mids.append(edge_midpoints[key])
            
            # Create 4 new faces
            m01, m12, m20 = mids
            new_faces.extend([
                [v0, m01, m20],
                [v1, m12, m01],
                [v2, m20, m12],
                [m01, m12, m20]
            ])
        
        vertices = np.array(new_vertices)
        faces = np.array(new_faces)
    
    return vertices, faces


def create_test_cube():
    """Create a unit cube mesh for testing."""
    # Cube vertices
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Bottom
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # Top
    ]) * 0.5
    
    # Cube faces (triangulated)
    faces = np.array([
        # Bottom
        [0, 1, 2], [0, 2, 3],
        # Top
        [4, 6, 5], [4, 7, 6],
        # Front
        [0, 4, 5], [0, 5, 1],
        # Back
        [2, 6, 7], [2, 7, 3],
        # Left
        [0, 3, 7], [0, 7, 4],
        # Right
        [1, 5, 6], [1, 6, 2]
    ])
    
    return vertices, faces


def load_mesh_from_file(filepath):
    """Load mesh from .npz or .obj file."""
    if filepath.endswith('.npz'):
        data = np.load(filepath)
        if 'vertices' in data and 'faces' in data:
            return data['vertices'], data['faces']
        elif 'mesh' in data and 'face' in data:
            return data['mesh'], data['face']
        else:
            raise ValueError(f"Unknown npz format: {list(data.keys())}")
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def visualize_results(vertices, faces, f_values, save_path=None):
    """Visualize the segmentation results."""
    fig = plt.figure(figsize=(15, 5))
    
    # Get channel assignments
    channel_assignments = f_values.argmax(axis=1)
    
    # Plot 1: 3D mesh with channel colors
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Segmented Mesh')
    
    # Create face colors based on vertex assignments
    face_colors = []
    for face in faces:
        # Average channel of face vertices
        avg_channel = np.mean(channel_assignments[face])
        face_colors.append(avg_channel)
    
    # Plot triangles
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    triangles = vertices[faces]
    collection = Poly3DCollection(triangles, alpha=0.8)
    collection.set_array(np.array(face_colors))
    collection.set_cmap('tab10')
    ax1.add_collection3d(collection)
    
    # Set limits
    ax1.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax1.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax1.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot 2: Channel distribution
    ax2 = fig.add_subplot(132)
    ax2.set_title('Channel Distribution')
    unique, counts = np.unique(channel_assignments, return_counts=True)
    ax2.bar(unique, counts)
    ax2.set_xlabel('Channel')
    ax2.set_ylabel('Number of Vertices')
    ax2.set_xticks(range(6))
    
    # Plot 3: Field values heatmap
    ax3 = fig.add_subplot(133)
    ax3.set_title('Field Values Heatmap')
    im = ax3.imshow(f_values.T, aspect='auto', cmap='RdBu_r')
    ax3.set_xlabel('Vertex Index')
    ax3.set_ylabel('Channel')
    ax3.set_yticks(range(6))
    plt.colorbar(im, ax=ax3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def test_on_sphere():
    """Test optimization on a sphere mesh."""
    print("\n" + "="*60)
    print("Testing on SPHERE mesh")
    print("="*60)
    
    # Create sphere
    vertices, faces = create_test_sphere(n_subdivisions=2)
    print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Auto-select pins
    pinned_indices = auto_select_pins(vertices, method='bbox')
    print(f"Pinned vertices: {pinned_indices}")
    
    # Run optimization
    results = optimize_relu_mesh(
        vertices, faces, pinned_indices,
        n_iters=20000,
        lr_vertex=2e-3,
        lr_offset=2e-2,
        beta_start=4.0,
        beta_end=15.0,
        beta_schedule="sigmoid",
        lambda_contour=(1.0, 4.0),
        lambda_smooth=0.1,
        lambda_area=(0.0, 100.0),
        reverse_schedule=True,
        use_dynamic_reweighting=True,
        print_every=1000,
        save_path="sphere_optimized.npz"
    )
    
    print(f"\nOptimization complete!")
    print(f"Final loss: {results['best_loss']:.3e} at iteration {results['best_iter']}")
    
    # Visualize
    visualize_results(vertices, faces, results['f_values'], "sphere_results.png")
    
    return results


def test_on_cube():
    """Test optimization on a cube mesh."""
    print("\n" + "="*60)
    print("Testing on CUBE mesh")
    print("="*60)
    
    # Create cube
    vertices, faces = create_test_cube()
    print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Auto-select pins (should naturally pick corners)
    pinned_indices = auto_select_pins(vertices, method='bbox')
    print(f"Pinned vertices: {pinned_indices}")
    
    # Run optimization
    results = optimize_relu_mesh(
        vertices, faces, pinned_indices,
        n_iters=10000,  # Cube should converge faster
        lr_vertex=2e-3,
        lr_offset=2e-2,
        beta_start=4.0,
        beta_end=15.0,
        beta_schedule="sigmoid",
        lambda_contour=(1.0, 4.0),
        lambda_smooth=0.1,
        lambda_area=(0.0, 100.0),
        reverse_schedule=True,
        use_dynamic_reweighting=True,
        print_every=500,
        save_path="cube_optimized.npz"
    )
    
    print(f"\nOptimization complete!")
    print(f"Final loss: {results['best_loss']:.3e} at iteration {results['best_iter']}")
    
    # Visualize
    visualize_results(vertices, faces, results['f_values'], "cube_results.png")
    
    return results


def test_on_complex_mesh(mesh_path):
    """Test optimization on a complex mesh from file."""
    print("\n" + "="*60)
    print(f"Testing on COMPLEX mesh: {mesh_path}")
    print("="*60)
    
    # Load mesh
    vertices, faces = load_mesh_from_file(mesh_path)
    print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
    
    # Auto-select pins using PCA method for complex shapes
    pinned_indices = auto_select_pins(vertices, method='pca')
    print(f"Pinned vertices: {pinned_indices}")
    
    # Run optimization with more iterations for complex shapes
    results = optimize_relu_mesh(
        vertices, faces, pinned_indices,
        n_iters=50000,
        lr_vertex=1e-3,  # Lower learning rate for stability
        lr_offset=1e-2,
        beta_start=4.0,
        beta_end=20.0,
        beta_schedule="sigmoid",
        lambda_contour=(1.0, 4.0),
        lambda_smooth=0.1,
        lambda_area=(0.0, 100.0),
        reverse_schedule=True,
        use_dynamic_reweighting=True,
        print_every=2000,
        save_path=f"{os.path.splitext(os.path.basename(mesh_path))[0]}_optimized.npz"
    )
    
    print(f"\nOptimization complete!")
    print(f"Final loss: {results['best_loss']:.3e} at iteration {results['best_iter']}")
    
    # Visualize
    output_name = os.path.splitext(os.path.basename(mesh_path))[0]
    visualize_results(vertices, faces, results['f_values'], f"{output_name}_results.png")
    
    return results


def compare_variants():
    """Compare different contour alignment variants."""
    print("\n" + "="*60)
    print("Comparing contour alignment variants")
    print("="*60)
    
    # Create test mesh
    vertices, faces = create_test_sphere(n_subdivisions=1)
    pinned_indices = auto_select_pins(vertices, method='bbox')
    
    variants = ['v1', 'v2', 'v3']
    results = {}
    
    for variant in variants:
        print(f"\nTesting variant: {variant}")
        
        # Note: This would require modifying the main optimization function
        # to accept a variant parameter. For now, we just test V1.
        if variant == 'v1':
            result = optimize_relu_mesh(
                vertices, faces, pinned_indices,
                n_iters=5000,
                print_every=1000,
                save_path=f"sphere_{variant}_optimized.npz"
            )
            results[variant] = result
    
    # Compare results
    print("\n" + "-"*40)
    print("Comparison of variants:")
    for variant, result in results.items():
        print(f"{variant}: Best loss = {result['best_loss']:.3e} at iter {result['best_iter']}")


if __name__ == "__main__":
    # Test on simple meshes
    test_on_sphere()
    test_on_cube()
    
    # Test on complex meshes if available
    complex_mesh_paths = [
        "/home/kejunzou/Projects/ReLUs on Meshes/visualizeMesh/Rodmesh.npz",
        "/home/kejunzou/Projects/ReLUs on Meshes/visualizeMesh/KittyMesh.npz",
        "/home/kejunzou/Projects/ReLUs on Meshes/visualizeMesh/Angelmesh.npz"
    ]
    
    for mesh_path in complex_mesh_paths:
        if os.path.exists(mesh_path):
            try:
                test_on_complex_mesh(mesh_path)
            except Exception as e:
                print(f"Error processing {mesh_path}: {e}")
        else:
            print(f"Mesh file not found: {mesh_path}")
    
    # Compare variants
    # compare_variants()
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)