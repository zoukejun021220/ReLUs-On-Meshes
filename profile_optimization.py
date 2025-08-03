#!/usr/bin/env python3
"""
Profile the optimization to find bottlenecks.
"""

import numpy as np
import torch
import time
from mesh_optimization_helpers import auto_select_pins, build_triangle_adjacency, build_vertex_edges
from improved_loss_v2_fast import improved_loss_function_fast

# Create a moderate test mesh
n = 20
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
z = np.linspace(-1, 1, n)

vertices = []
for i in range(n):
    for j in range(n):
        for k in range(n):
            vertices.append([x[i], y[j], z[k]])
vertices = np.array(vertices, dtype=np.float32)

# Create a simple cube mesh faces (this is simplified)
faces = []
for i in range(n-1):
    for j in range(n-1):
        for k in range(n-1):
            # Create a cube from 8 vertices
            base = i * n * n + j * n + k
            # Just add a few triangles for testing
            faces.extend([
                [base, base + 1, base + n],
                [base + 1, base + n + 1, base + n]
            ])
faces = np.array(faces[:1000], dtype=np.int32)  # Limit to 1000 faces

print(f"Test mesh: {len(vertices)} vertices, {len(faces)} faces")

# Move to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
v = torch.from_numpy(vertices).float().to(device)
f = torch.from_numpy(faces).long().to(device)

# Build edges
edges = torch.from_numpy(build_vertex_edges(faces)).long().to(device)
print(f"Edges: {len(edges)}")

# Initialize f_values
f_values = torch.randn(len(vertices), 6, device=device, requires_grad=True)

# Profile the loss function
print("\nProfiling loss function components...")

# Warm up
for _ in range(5):
    loss, _ = improved_loss_function_fast(v, f, f_values, edges, beta=5.0)

# Time each component
num_runs = 20

torch.cuda.synchronize()
start = time.time()
for _ in range(num_runs):
    loss, components = improved_loss_function_fast(v, f, f_values, edges, beta=5.0)
    loss.backward()
torch.cuda.synchronize()
total_time = time.time() - start

print(f"\nTotal time for {num_runs} iterations: {total_time:.3f}s")
print(f"Average time per iteration: {total_time/num_runs*1000:.1f}ms")
print(f"Estimated iterations per second: {num_runs/total_time:.1f}")

# Profile individual components
from improved_loss_v2_fast import (
    compute_soft_area_fractions_fast,
    compute_boundary_weights_fast,
    build_triangle_edge_adjacency_fast
)

# Area computation
torch.cuda.synchronize()
start = time.time()
for _ in range(num_runs):
    p, areas = compute_soft_area_fractions_fast(v, f, f_values, beta=5.0)
torch.cuda.synchronize()
print(f"\nArea computation: {(time.time()-start)/num_runs*1000:.1f}ms")

# Boundary weights
torch.cuda.synchronize()
start = time.time()
for _ in range(num_runs):
    w_e = compute_boundary_weights_fast(f_values, edges, beta=5.0)
torch.cuda.synchronize()
print(f"Boundary weights: {(time.time()-start)/num_runs*1000:.1f}ms")

# Adjacency building
torch.cuda.synchronize()
start = time.time()
for _ in range(num_runs):
    adj_pairs = build_triangle_edge_adjacency_fast(f, len(vertices))
torch.cuda.synchronize()
print(f"Adjacency building: {(time.time()-start)/num_runs*1000:.1f}ms")

print(f"\nNumber of adjacent triangle pairs: {len(adj_pairs)}")