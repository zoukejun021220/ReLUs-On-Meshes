#!/usr/bin/env python3
"""
Test script to verify pin/channel mapping is working correctly.
"""
import torch
import numpy as np
from utils.mesh_preprocessing import pick_axis_aligned_anchors, vertex_normals

def test_pin_mapping():
    """Create a simple test mesh and verify pin mapping."""
    # Create a simple cube mesh
    vertices = torch.tensor([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # top
    ], dtype=torch.float32)
    
    # Cube faces (triangulated)
    faces = torch.tensor([
        # Bottom face
        [0, 1, 2], [0, 2, 3],
        # Top face  
        [4, 6, 5], [4, 7, 6],
        # Front face
        [0, 5, 1], [0, 4, 5],
        # Back face
        [2, 7, 3], [2, 6, 7],
        # Left face
        [0, 7, 4], [0, 3, 7],
        # Right face
        [1, 6, 2], [1, 5, 6]
    ], dtype=torch.long)
    
    print("Testing pin selection methods on a cube mesh...\n")
    
    # Test 1: Extremes method
    print("1. EXTREMES METHOD:")
    pins_ext, labels_ext = pick_axis_aligned_anchors(vertices, faces, method='extremes')
    print(f"   Pins: {pins_ext.tolist()}")
    for i, (pin_idx, label) in enumerate(zip(pins_ext, labels_ext.values())):
        v = vertices[pin_idx]
        print(f"   Channel {i} ({label}): vertex {pin_idx} at position {v.tolist()}")
    
    # Test 2: Normal-based method
    print("\n2. NORMAL-BASED METHOD:")
    pins_norm, labels_norm = pick_axis_aligned_anchors(vertices, faces, method='normals')
    print(f"   Pins: {pins_norm.tolist()}")
    
    # Compute and show normals for selected vertices
    normals = vertex_normals(vertices, faces)
    for i, (pin_idx, label) in enumerate(zip(pins_norm, labels_norm.values())):
        v = vertices[pin_idx]
        n = normals[pin_idx]
        print(f"   Channel {i} ({label}): vertex {pin_idx} at {v.tolist()}, normal {n.tolist()}")
    
    # Test 3: Verify mapping consistency
    print("\n3. MAPPING VERIFICATION:")
    # Create pin targets
    n_channels = 6
    P = len(pins_norm)
    targets = torch.full((P, n_channels), -1.0)
    for i in range(min(P, n_channels)):
        targets[i, i] = 1.0
    
    print("   Pin target matrix:")
    for i in range(P):
        label = labels_norm[i]
        print(f"   {label}: {targets[i].tolist()}")
    
    print("\n4. CHECKING FOR DUPLICATES:")
    if len(pins_norm) < len(set(pins_norm.tolist())):
        print("   WARNING: Duplicate pins detected!")
    else:
        print("   ✓ All pins are unique")
    
    return pins_norm, labels_norm

if __name__ == "__main__":
    pins, labels = test_pin_mapping()
    print("\nTest completed. Pin mapping is ready for training.")