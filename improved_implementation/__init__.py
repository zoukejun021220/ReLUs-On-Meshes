"""
Improved ReLU-on-Meshes implementation with stable convergence.
"""

from .trainer import MeshSegmentationTrainer
from .mesh_utils import precompute_mesh_data
from .visualization import (
    visualize_segmentation_pyvista,
    visualize_segmentation_matplotlib,
    plot_training_history,
    save_segmentation_result
)

__all__ = [
    'MeshSegmentationTrainer',
    'precompute_mesh_data',
    'visualize_segmentation_pyvista',
    'visualize_segmentation_matplotlib',
    'plot_training_history',
    'save_segmentation_result'
]