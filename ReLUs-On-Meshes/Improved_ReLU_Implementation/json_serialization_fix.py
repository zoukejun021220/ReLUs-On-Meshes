"""
Fix for JSON serialization of NumPy/PyTorch types.
"""
import json
import numpy as np
import torch


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy and PyTorch types."""
    
    def default(self, obj):
        # Handle NumPy types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        
        # Handle PyTorch types
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif hasattr(obj, 'item'):  # Handles 0-d arrays/tensors
            return obj.item()
        
        # Handle other types that might cause issues
        elif isinstance(obj, (bytes, bytearray)):
            return obj.decode('utf-8')
        
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


def convert_to_serializable(obj):
    """
    Recursively convert NumPy/PyTorch types to JSON-serializable Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    else:
        return obj


def safe_json_dump(data, file_path, indent=2):
    """
    Safely dump data to JSON, handling NumPy/PyTorch types.
    
    Args:
        data: Data to serialize
        file_path: Path to save JSON file
        indent: JSON indentation level
    """
    # Convert data to serializable format
    serializable_data = convert_to_serializable(data)
    
    # Write to file
    with open(file_path, 'w') as f:
        json.dump(serializable_data, f, indent=indent, cls=NumpyEncoder)


def patch_main_json_serialization():
    """
    Returns a patched version of the metrics saving code.
    """
    def save_metrics_safely(metrics, output_path):
        """
        Save metrics with proper type conversion.
        """
        # Convert all NumPy/PyTorch types to Python types
        safe_metrics = convert_to_serializable(metrics)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(safe_metrics, f, indent=2)
        
        return safe_metrics
    
    return save_metrics_safely


# Example usage
if __name__ == "__main__":
    print("JSON Serialization Fix Module")
    print("="*50)
    
    # Test with problematic types
    test_data = {
        'float32': np.float32(1.5),
        'float64': np.float64(2.5),
        'int32': np.int32(10),
        'int64': np.int64(20),
        'bool': np.bool_(True),
        'array': np.array([1, 2, 3]),
        'nested': {
            'tensor': torch.tensor([1.0, 2.0, 3.0]),
            'scalar': np.float32(3.14)
        }
    }
    
    print("\nOriginal data types:")
    for key, val in test_data.items():
        if isinstance(val, dict):
            for k, v in val.items():
                print(f"  {key}.{k}: {type(v)}")
        else:
            print(f"  {key}: {type(val)}")
    
    # Convert to serializable
    safe_data = convert_to_serializable(test_data)
    
    print("\nConverted data types:")
    for key, val in safe_data.items():
        if isinstance(val, dict):
            for k, v in val.items():
                print(f"  {key}.{k}: {type(v)}")
        else:
            print(f"  {key}: {type(val)}")
    
    # Test JSON serialization
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(safe_data, f, indent=2)
        print(f"\n✓ Successfully serialized to {f.name}")
    
    print("\nTo use in your code:")
    print("  from json_serialization_fix import convert_to_serializable")
    print("  safe_metrics = convert_to_serializable(metrics)")
    print("  json.dump(safe_metrics, f, indent=2)")