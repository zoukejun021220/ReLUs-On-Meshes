"""
Configuration for valid channel pairs in 6-channel segmentation.
Excludes opposite pairs that should never meet.
"""
import torch
from typing import Tuple, List


def get_valid_channel_pairs(n_channels: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get valid channel pairs excluding opposite pairs.
    
    For 6-channel segmentation:
    - Channels 0-1: Top/Bottom (opposites)
    - Channels 2-3: Front/Back (opposites)
    - Channels 4-5: Right/Left (opposites)
    
    Returns:
        i_indices: tensor of first channel indices in valid pairs
        j_indices: tensor of second channel indices in valid pairs
    """
    if n_channels != 6:
        # For non-6 channel case, return all pairs
        return torch.triu_indices(n_channels, n_channels, offset=1)
    
    # Define opposite pairs
    opposite_pairs = [(0, 1), (2, 3), (4, 5)]
    opposite_set = set(opposite_pairs + [(b, a) for a, b in opposite_pairs])
    
    # Generate all valid pairs
    valid_pairs = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            if (i, j) not in opposite_set:
                valid_pairs.append((i, j))
    
    # Convert to tensors
    i_indices = torch.tensor([p[0] for p in valid_pairs], dtype=torch.long)
    j_indices = torch.tensor([p[1] for p in valid_pairs], dtype=torch.long)
    
    return i_indices, j_indices


def get_num_valid_pairs(n_channels: int = 6) -> int:
    """
    Get number of valid channel pairs.
    
    For 6 channels: 15 total pairs - 3 opposite pairs = 12 valid pairs
    """
    if n_channels != 6:
        return n_channels * (n_channels - 1) // 2
    
    return 12  # 15 - 3 opposite pairs


def describe_channel_pair(i: int, j: int) -> str:
    """
    Get a human-readable description of a channel pair.
    """
    names = ["Top", "Bottom", "Front", "Back", "Right", "Left"]
    if i < len(names) and j < len(names):
        return f"{names[i]}-{names[j]}"
    return f"Channel{i}-Channel{j}"


def is_valid_pair(i: int, j: int, n_channels: int = 6) -> bool:
    """
    Check if a channel pair is valid (not opposites).
    """
    if n_channels != 6:
        return True
    
    opposite_pairs = [(0, 1), (2, 3), (4, 5)]
    return (i, j) not in opposite_pairs and (j, i) not in opposite_pairs