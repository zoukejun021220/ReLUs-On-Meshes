#!/usr/bin/env python3
"""
Test that the JSON serialization fix works properly.
"""
import json
import numpy as np
import torch
from json_serialization_fix import convert_to_serializable, safe_json_dump


def test_json_serialization():
    """Test that all common NumPy/PyTorch types can be serialized."""
    
    print("="*60)
    print("Testing JSON Serialization Fix")
    print("="*60)
    
    # Create test data with problematic types
    test_metrics = {
        'mesh_name': 'test_mesh',
        'num_vertices': np.int32(1000),
        'num_faces': np.int64(2000),
        'elapsed_time': np.float32(123.456),
        'planarity': {
            'mean_angle': np.float64(45.5),
            'max_angle': np.float32(89.9),
            'num_boundary_edges': np.int32(150),
            'angles_array': np.array([30.0, 45.0, 60.0], dtype=np.float32)
        },
        'final_loss': torch.tensor(0.0123),
        'history': {
            'losses': [np.float32(1.0), np.float32(0.5), np.float32(0.1)],
            'steps': np.arange(3, dtype=np.int32).tolist()
        },
        'bool_value': np.bool_(True),
        'none_value': None
    }
    
    print("\nOriginal data types:")
    print_types(test_metrics, indent=2)
    
    # Convert to serializable
    print("\nConverting to JSON-serializable types...")
    safe_metrics = convert_to_serializable(test_metrics)
    
    print("\nConverted data types:")
    print_types(safe_metrics, indent=2)
    
    # Test JSON serialization
    print("\nTesting JSON serialization...")
    try:
        json_str = json.dumps(safe_metrics, indent=2)
        print("✓ Successfully serialized to JSON string")
        
        # Test deserialization
        reloaded = json.loads(json_str)
        print("✓ Successfully deserialized from JSON string")
        
        # Save to file
        test_file = 'test_metrics.json'
        safe_json_dump(test_metrics, test_file)
        print(f"✓ Successfully saved to {test_file}")
        
        # Load and verify
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        print(f"✓ Successfully loaded from {test_file}")
        
        # Verify values are preserved
        assert loaded_data['num_vertices'] == 1000
        assert abs(loaded_data['elapsed_time'] - 123.456) < 0.001
        assert loaded_data['planarity']['num_boundary_edges'] == 150
        assert loaded_data['bool_value'] == True
        assert loaded_data['none_value'] is None
        print("✓ All values preserved correctly")
        
        print("\n" + "="*60)
        print("SUCCESS: JSON serialization fix working properly!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\n" + "="*60)
        print("FAILED: JSON serialization still has issues")
        print("="*60)
        return False


def print_types(obj, indent=0):
    """Recursively print types of nested objects."""
    prefix = " " * indent
    
    if isinstance(obj, dict):
        for key, val in obj.items():
            if isinstance(val, dict):
                print(f"{prefix}{key}: dict")
                print_types(val, indent + 2)
            elif isinstance(val, (list, tuple)):
                print(f"{prefix}{key}: {type(val).__name__} of {type(val[0]).__name__ if val else 'empty'}")
            else:
                print(f"{prefix}{key}: {type(val).__name__}")
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} of {type(obj[0]).__name__ if obj else 'empty'}")
    else:
        print(f"{prefix}{type(obj).__name__}")


if __name__ == "__main__":
    success = test_json_serialization()
    
    if success:
        print("\nThe fix has been applied successfully!")
        print("Your main.py should now save metrics without errors.")
    else:
        print("\nThere might still be issues. Please check the error messages.")