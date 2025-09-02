#!/usr/bin/env python3
"""
Extract field values from checkpoint for upload.
"""
import numpy as np
import argparse
from pathlib import Path

def extract_field_values(checkpoint_path: str, output_path: str = None):
    """Extract field values from checkpoint npz file."""
    
    # Load checkpoint
    data = np.load(checkpoint_path)
    
    # Get step number from filename if not in data
    if 'step' in data:
        step = int(data['step'])
    else:
        # Extract from filename like checkpoint_245000.npz
        step = int(Path(checkpoint_path).stem.split('_')[-1])
    
    if output_path is None:
        output_path = f"field_values_{step:06d}.npz"
    
    # Extract key data
    save_data = {
        'field_values': data['field_values'],
        'step': step,
    }
    
    # Add beta values if present
    if 'beta_contour' in data:
        save_data['beta_contour'] = float(data['beta_contour'])
    if 'beta_area' in data:
        save_data['beta_area'] = float(data['beta_area'])
    
    # Add other useful info if available
    if 'vertices' in data:
        save_data['vertices'] = data['vertices']
    if 'faces' in data:
        save_data['faces'] = data['faces']
    if 'pinned_indices' in data:
        save_data['pinned_indices'] = data['pinned_indices']
    
    # Save
    np.savez_compressed(output_path, **save_data)
    
    print(f"Extracted field values from step {step}")
    print(f"Shape: {data['field_values'].shape}")
    print(f"Beta values: contour={save_data.get('beta_contour', 'N/A')}, area={save_data.get('beta_area', 'N/A')}")
    print(f"Saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract field values from checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint npz file")
    parser.add_argument("--output", type=str, default=None, help="Output filename")
    
    args = parser.parse_args()
    extract_field_values(args.checkpoint, args.output)