#!/usr/bin/env python3
"""Test different edge detection strategies"""
import torch


def current_method(F, edges):
    """Current method: top-2 channels at edge midpoint"""
    f_mid = 0.5 * (F[edges[:, 0]] + F[edges[:, 1]])
    top_vals, top_idx = torch.topk(f_mid, k=2, dim=1)
    chan_i = top_idx[:, 0]
    chan_j = top_idx[:, 1]
    
    da = F[edges[:, 0], chan_i] - F[edges[:, 0], chan_j]
    db = F[edges[:, 1], chan_i] - F[edges[:, 1], chan_j]
    w = torch.sigmoid(-10.0 * da * db)
    
    return w, chan_i, chan_j


def vertex_argmax_method(F, edges):
    """Detect when vertices have different argmax channels"""
    # Get argmax channel for each vertex
    argmax_a = F[edges[:, 0]].argmax(dim=1)
    argmax_b = F[edges[:, 1]].argmax(dim=1)
    
    # Active when argmax differs
    active = (argmax_a != argmax_b).float()
    
    # For each edge, use the two different argmax channels
    chan_i = argmax_a
    chan_j = argmax_b
    
    return active, chan_i, chan_j


def test_methods():
    # Test case: 4 vertices, 3 channels
    F = torch.tensor([
        [0.9, 0.05, 0.05],  # vertex 0: channel 0 wins
        [0.05, 0.9, 0.05],  # vertex 1: channel 1 wins  
        [0.05, 0.05, 0.9],  # vertex 2: channel 2 wins
        [0.8, 0.1, 0.1],    # vertex 3: channel 0 wins
    ])
    
    edges = torch.tensor([
        [0, 1],  # should be active (0 vs 1)
        [1, 2],  # should be active (1 vs 2)
        [0, 3],  # should NOT be active (both 0)
        [2, 3],  # should be active (2 vs 0)
    ])
    
    print("Field values:")
    print(F)
    print("\nEdges:", edges)
    
    # Current method
    w1, ci1, cj1 = current_method(F, edges)
    print("\nCurrent method (midpoint top-2):")
    print(f"  Weights: {w1}")
    print(f"  Chan pairs: {list(zip(ci1.tolist(), cj1.tolist()))}")
    
    # Vertex argmax method  
    w2, ci2, cj2 = vertex_argmax_method(F, edges)
    print("\nVertex argmax method:")
    print(f"  Active: {w2}")
    print(f"  Chan pairs: {list(zip(ci2.tolist(), cj2.tolist()))}")
    
    print("\nExpected active edges: [1, 1, 0, 1]")


if __name__ == "__main__":
    test_methods()