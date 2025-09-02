import torch
import numpy as np
from svdPlaneInit import svd_init_pairwise_planes

# Create dummy data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
V = 100
C = 6

# Random vertices
vertices = torch.randn(V, 3, device=device)

# Create channel values with clear regions
f_values = torch.zeros(V, C, device=device)
for i in range(V):
    channel = i % C
    f_values[i, channel] = 1.0
    # Add some noise
    f_values[i] += 0.1 * torch.randn(C, device=device)

# Test SVD initialization
print("Testing SVD initialization for pairwise planes...")
plane_normals, plane_offsets = svd_init_pairwise_planes(vertices, f_values, device)

print(f"Initialized {plane_normals.shape[0]} planes (should be 12 for 6 channels)")
print(f"Plane normals shape: {plane_normals.shape}")
print(f"Plane offsets shape: {plane_offsets.shape}")

# Check that normals are normalized
norms = torch.norm(plane_normals, dim=1)
print(f"Norm range: [{norms.min():.4f}, {norms.max():.4f}] (should be ~1.0)")

# Test channel pairs configuration
from channelPairsConfig import get_valid_channel_pairs, describe_channel_pair
i_indices, j_indices = get_valid_channel_pairs(6)
print(f"\nValid channel pairs ({len(i_indices)} total):")
for idx, (i, j) in enumerate(zip(i_indices.tolist(), j_indices.tolist())):
    print(f"  Plane {idx}: {describe_channel_pair(i, j)} (channels {i}-{j})")

print("\nTest completed successfully!")