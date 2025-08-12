#!/usr/bin/env python3
"""
Convert old .pt checkpoint to new .npz format and extract field values.
"""
import torch
import numpy as np
import argparse
from pathlib import Path

def convert_pt_checkpoint(pt_path: str, mesh_npz_path: str = None):
    """Convert .pt checkpoint to .npz format."""
    
    # Load the .pt checkpoint (weights_only=False for custom classes)
    checkpoint = torch.load(pt_path, map_location='cpu', weights_only=False)
    
    # Extract step from filename
    step = int(Path(pt_path).stem.split('_')[-1])
    
    # Load mesh data if provided (for vertices/faces)
    mesh_data = {}
    if mesh_npz_path and Path(mesh_npz_path).exists():
        mesh = np.load(mesh_npz_path)
        if 'vertices' in mesh:
            mesh_data['vertices'] = mesh['vertices']
        if 'faces' in mesh:
            mesh_data['faces'] = mesh['faces']
    
    # Convert field values
    if 'F' in checkpoint:
        field_values = checkpoint['F'].numpy()
    else:
        print("Error: No field values found in checkpoint")
        return None
    
    # Extract temperature values if available
    beta_contour = 8.0  # default
    beta_area = 4.0     # default
    
    if 'temp_ctrl' in checkpoint:
        temp_ctrl = checkpoint['temp_ctrl']
        if hasattr(temp_ctrl, 'beta_contour'):
            beta_contour = temp_ctrl.beta_contour
        if hasattr(temp_ctrl, 'beta_area'):
            beta_area = temp_ctrl.beta_area
    
    # Create output data
    output_data = {
        'field_values': field_values,
        'step': step,
        'beta_contour': beta_contour,
        'beta_area': beta_area,
        **mesh_data
    }
    
    # Save field values only
    output_path = f"field_values_{step:06d}.npz"
    np.savez_compressed(output_path, **output_data)
    
    print(f"Converted checkpoint from step {step}")
    print(f"Field values shape: {field_values.shape}")
    print(f"Beta values: contour={beta_contour}, area={beta_area}")
    print(f"Saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .pt checkpoint to .npz")
    parser.add_argument("checkpoint", type=str, help="Path to .pt checkpoint file")
    parser.add_argument("--mesh", type=str, default=None, help="Path to mesh npz file for vertices/faces")
    
    args = parser.parse_args()
    convert_pt_checkpoint(args.checkpoint, args.mesh)