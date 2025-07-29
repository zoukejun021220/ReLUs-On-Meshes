# ReLU Mesh Optimization

This project implements an advanced optimization framework for learning piecewise linear functions on 3D meshes using ReLU activations. The implementation is based on the paper "ReLUs On Meshes" and includes significant improvements for better gradient flow and optimization stability.

## Overview

The framework optimizes plane parameters to create piecewise linear approximations of target functions on 3D meshes. It uses a novel loss function that addresses gradient flow issues and provides stable optimization even for complex geometries.

## Features

- **Improved Loss Function**: Enhanced loss formulation that prevents gradient vanishing
- **VTK Mesh Support**: Native support for loading and processing VTK mesh files
- **3D Visualization**: Real-time visualization of optimization progress
- **Flexible Architecture**: Support for both channel-wise and pairwise plane configurations
- **Comprehensive Testing**: Test scripts for validating loss functions and mesh loading

## Installation

1. Clone the repository:
```bash
git clone https://github.com/zoukejun021220/ReLUs-On-Meshes.git
cd ReLUs-On-Meshes
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- numpy
- torch
- vtk
- matplotlib
- meshio (optional, for additional mesh format support)

## Project Structure

```
ReLUs-On-Meshes/
├── relus_mesh_optimization_improved.py  # Main optimization class with improved loss
├── improved_loss_v2.py                  # Enhanced loss function implementation
├── run_optimization.py                  # Script to run optimization
├── test_new_loss.py                     # Test script for loss validation
├── test_vtk_loading.py                  # Test script for VTK mesh loading
├── requirements.txt                     # Python dependencies
└── ReLUs-On-Meshes/                    # Original reference implementation
    ├── Piecewise Linear Mesh 2D/        # 2D mesh optimization examples
    ├── Piecewise Linear Mesh 3D/        # 3D mesh optimization examples
    └── visualizeMesh/                   # Visualization utilities
```

## Usage

### Basic Optimization

To run the optimization on a mesh:

```python
from relus_mesh_optimization_improved import ReLUMeshOptimizer
import torch

# Initialize optimizer
optimizer = ReLUMeshOptimizer(
    mesh_path="path/to/your/mesh.vtk",
    num_planes=10,
    learning_rate=0.01,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Define target function
def target_function(points):
    # Example: sphere distance function
    return torch.norm(points, dim=1) - 1.0

# Run optimization
optimizer.optimize(
    target_function=target_function,
    num_iterations=1000,
    visualize=True
)

# Save results
optimizer.save_results("optimization_results.npz")
```

### Using the Run Script

For quick testing, use the provided run script:

```bash
python run_optimization.py --mesh path/to/mesh.vtk --planes 10 --iterations 1000
```

### Testing Loss Functions

To validate the improved loss function:

```bash
python test_new_loss.py
```

This will:
- Compare gradients between old and new loss functions
- Visualize gradient flow for different scenarios
- Generate plots showing optimization behavior

### Working with VTK Meshes

Test VTK mesh loading:

```bash
python test_vtk_loading.py
```

## Key Improvements

### 1. Enhanced Loss Function

The improved loss function addresses gradient vanishing issues:

```python
def improved_loss(pred, target, epsilon=1e-6):
    diff = pred - target
    loss = torch.where(
        torch.abs(diff) < epsilon,
        0.5 * diff**2 / epsilon,
        torch.abs(diff) - 0.5 * epsilon
    )
    return loss.mean()
```

### 2. Better Gradient Flow

- Smooth transition between L1 and L2 losses
- Prevents zero gradients near optimal solutions
- Adaptive epsilon for different scales

### 3. Stable Optimization

- Gradient clipping to prevent instabilities
- Learning rate scheduling
- Regularization terms for smooth solutions

## Advanced Features

### Channel-wise Planes

For independent optimization per coordinate:

```python
optimizer = ReLUMeshOptimizer(
    mesh_path="mesh.vtk",
    num_planes=10,
    plane_type='channel-wise'
)
```

### Pairwise Planes

For coupled coordinate optimization:

```python
optimizer = ReLUMeshOptimizer(
    mesh_path="mesh.vtk",
    num_planes=10,
    plane_type='pairwise'
)
```

### Custom Initialization

```python
# Initialize planes with specific orientations
initial_planes = torch.randn(10, 4)  # [a, b, c, d] for ax + by + cz + d = 0
optimizer.set_initial_planes(initial_planes)
```

## Visualization

The framework includes real-time visualization during optimization:

- **3D Mesh Display**: Shows the mesh with current approximation
- **Loss Curves**: Plots training loss over iterations
- **Gradient Norms**: Monitors gradient magnitudes
- **Plane Visualizations**: Shows learned plane configurations

## Examples

### 2D Snake Mesh

```bash
cd "ReLUs-On-Meshes/Piecewise Linear Mesh 2D"
python snake_optimization.py
```

### 3D Kitty Mesh

```bash
cd "ReLUs-On-Meshes/Piecewise Linear Mesh 3D"
python main.py --mesh l1-poly-dat/hex/kitty/orig.tet.vtk
```

## Performance Tips

1. **GPU Acceleration**: Use CUDA for faster optimization
2. **Batch Processing**: Process multiple points simultaneously
3. **Adaptive Learning Rate**: Use schedulers for better convergence
4. **Early Stopping**: Monitor validation loss to prevent overfitting

## Troubleshooting

### Common Issues

1. **Gradient Vanishing**: Increase epsilon in loss function
2. **Slow Convergence**: Adjust learning rate or use different optimizer
3. **Memory Issues**: Reduce batch size or number of planes
4. **Visualization Errors**: Ensure VTK is properly installed

### Debug Mode

Enable detailed logging:

```python
optimizer = ReLUMeshOptimizer(mesh_path="mesh.vtk", debug=True)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use this code in your research, please cite:

```bibtex
@article{relus-on-meshes,
  title={ReLUs On Meshes: Learning Piecewise Linear Functions on 3D Meshes},
  author={Original Authors},
  journal={Conference/Journal Name},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original ReLUs On Meshes paper authors
- PyTorch team for the deep learning framework
- VTK community for mesh processing tools

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.