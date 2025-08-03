"""
Optimized training pipeline for ReLU mesh segmentation.
Implements coarse-to-fine schedule with improved convergence.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
from loss_functions import compute_total_loss, GradNorm


@dataclass
class TrainingConfig:
    """Configuration for training stages."""
    level: int
    num_faces: int
    steps: int
    beta_start: float
    beta_end: float
    lambda_adj_start: float
    lambda_adj_end: float
    lr_max: float = 5e-3
    lambda_area: float = 1.0
    lambda_tv: float = 0.1
    

class CoarseToFineSchedule:
    """Manages coarse-to-fine training schedule."""
    
    def __init__(self):
        self.stages = [
            TrainingConfig(level=0, num_faces=3000, steps=30000,
                         beta_start=2.0, beta_end=10.0,
                         lambda_adj_start=0.0, lambda_adj_end=5.0),
            TrainingConfig(level=1, num_faces=12000, steps=60000,
                         beta_start=10.0, beta_end=10.0,
                         lambda_adj_start=5.0, lambda_adj_end=5.0),
            TrainingConfig(level=2, num_faces=-1, steps=120000,  # -1 means full resolution
                         beta_start=10.0, beta_end=25.0,
                         lambda_adj_start=5.0, lambda_adj_end=8.0),
        ]
        
    def get_stage(self, level: int) -> TrainingConfig:
        return self.stages[level]
    
    def interpolate_param(self, start: float, end: float, progress: float) -> float:
        """Linear interpolation of parameters."""
        return start + (end - start) * progress


class OneCycleLR:
    """Custom One-Cycle learning rate scheduler."""
    
    def __init__(self, optimizer, max_lr: float, total_steps: int, 
                 pct_start: float = 0.3, div_factor: float = 25.0, final_div: float = 1e4):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div
        self.step_num = 0
        
    def step(self):
        """Update learning rate."""
        self.step_num += 1
        
        if self.step_num <= self.pct_start * self.total_steps:
            # Increasing phase
            progress = self.step_num / (self.pct_start * self.total_steps)
            lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Decreasing phase
            progress = (self.step_num - self.pct_start * self.total_steps) / \
                      ((1 - self.pct_start) * self.total_steps)
            lr = self.max_lr - (self.max_lr - self.final_lr) * progress
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
    def reset(self, total_steps: Optional[int] = None):
        """Reset scheduler for new stage."""
        self.step_num = 0
        if total_steps is not None:
            self.total_steps = total_steps


def downsample_mesh(vertices: np.ndarray, faces: np.ndarray, 
                   target_faces: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample mesh to target number of faces using quadric decimation.
    
    Returns:
        new_vertices, new_faces, vertex_mapping (from new to original)
    """
    import trimesh
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Simplify mesh
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
    
    # Find mapping from simplified to original vertices
    from scipy.spatial import cKDTree
    tree = cKDTree(vertices)
    _, vertex_mapping = tree.query(simplified.vertices)
    
    return simplified.vertices, simplified.faces, vertex_mapping


def soft_pinning(f_values: torch.Tensor, pinned_indices: List[int], 
                 pinned_values: torch.Tensor, decay_rate: float = 0.99) -> None:
    """
    Apply soft pinning with exponential decay.
    
    Args:
        f_values: Field values to modify
        pinned_indices: Indices of pinned vertices
        pinned_values: Target values for pinned vertices (6, 6) matrix
        decay_rate: Exponential decay rate
    """
    with torch.no_grad():
        for i, idx in enumerate(pinned_indices):
            current = f_values[idx]
            target = pinned_values[i]
            f_values[idx] = decay_rate * current + (1 - decay_rate) * target


def train_stage(config: TrainingConfig,
                f_values: nn.Parameter,
                mesh_data: Dict,
                pinned_indices: List[int],
                schedule: CoarseToFineSchedule,
                use_grad_norm: bool = True,
                device: torch.device = torch.device('cuda')) -> Dict[str, List]:
    """
    Train one stage of the coarse-to-fine pipeline.
    
    Args:
        config: Training configuration for this stage
        f_values: Field values parameter
        mesh_data: Dictionary containing mesh information
        pinned_indices: Indices of pinned vertices
        schedule: Coarse-to-fine schedule manager
        use_grad_norm: Whether to use GradNorm for loss balancing
        device: Torch device
        
    Returns:
        Dictionary of training history
    """
    # Setup optimizer
    optimizer = optim.AdamW([f_values], lr=config.lr_max, weight_decay=1e-4)
    lr_scheduler = OneCycleLR(optimizer, config.lr_max, config.steps)
    
    # Setup GradNorm if requested
    grad_norm = GradNorm(num_tasks=3) if use_grad_norm else None
    
    # Setup mixed precision training
    scaler = GradScaler()
    
    # Create pinned values matrix
    pinned_values = torch.zeros(len(pinned_indices), 6, device=device)
    for i in range(len(pinned_indices)):
        pinned_values[i, i] = 1.0
        pinned_values[i, (i+1)%6:] = -1.0
        pinned_values[i, :i] = -1.0
    
    # Training history
    history = {
        'loss': [], 'area': [], 'adjacency': [], 'tv': [],
        'area_fractions': [], 'lr': [], 'beta': [], 'lambda_adj': []
    }
    
    # Early stopping
    best_loss = float('inf')
    patience_counter = 0
    patience = 5000
    min_improvement = 0.01  # 1% improvement threshold
    
    # Training loop
    for step in range(config.steps):
        # Interpolate parameters
        progress = step / config.steps
        beta = schedule.interpolate_param(config.beta_start, config.beta_end, progress)
        lambda_adj = schedule.interpolate_param(config.lambda_adj_start, config.lambda_adj_end, progress)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            loss_dict = compute_total_loss(
                f_values,
                mesh_data['vertices'],
                mesh_data['faces'],
                mesh_data['edges'],
                mesh_data['edge2face'],
                mesh_data['face_areas'],
                mesh_data['B'],
                beta=beta,
                lambda_area=config.lambda_area,
                lambda_adj=lambda_adj,
                lambda_tv=config.lambda_tv,
                return_components=True
            )
        
        # Apply GradNorm if enabled
        if grad_norm is not None and step > 100:
            grad_norm.update_weights(loss_dict, f_values)
            loss = grad_norm.get_weighted_loss(loss_dict)
        else:
            loss = loss_dict['total']
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([f_values], max_norm=5.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Learning rate update
        lr_scheduler.step()
        
        # Soft pinning
        soft_pinning(f_values, pinned_indices, pinned_values, decay_rate=0.995)
        
        # Record history
        if step % 100 == 0:
            with torch.no_grad():
                history['loss'].append(loss.item())
                history['area'].append(loss_dict['area'].item())
                history['adjacency'].append(loss_dict['adjacency'].item())
                history['tv'].append(loss_dict['tv'].item())
                history['area_fractions'].append(loss_dict['area_fractions'].cpu().numpy())
                history['lr'].append(optimizer.param_groups[0]['lr'])
                history['beta'].append(beta)
                history['lambda_adj'].append(lambda_adj)
            
            # Early stopping check
            if loss.item() < best_loss * (1 - min_improvement):
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 100
                
            if patience_counter >= patience:
                print(f"Early stopping at step {step}")
                break
                
            # Print progress
            if step % 1000 == 0:
                print(f"Step {step}/{config.steps}: Loss={loss.item():.4f}, "
                      f"Area={loss_dict['area'].item():.4f}, "
                      f"Adj={loss_dict['adjacency'].item():.4f}, "
                      f"TV={loss_dict['tv'].item():.4f}")
                print(f"Area fractions: {loss_dict['area_fractions'].cpu().numpy()}")
    
    return history


def optimize_mesh_segmentation(vertices: np.ndarray,
                             faces: np.ndarray,
                             pinned_indices: List[int],
                             num_channels: int = 6,
                             use_coarse_to_fine: bool = True,
                             use_grad_norm: bool = True,
                             device: Optional[torch.device] = None) -> Tuple[torch.Tensor, Dict]:
    """
    Main optimization function for mesh segmentation.
    
    Args:
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (F, 3)
        pinned_indices: Indices of anchor vertices
        num_channels: Number of segmentation channels
        use_coarse_to_fine: Whether to use coarse-to-fine schedule
        use_grad_norm: Whether to use GradNorm for loss balancing
        device: Torch device
        
    Returns:
        Optimized field values and training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to torch tensors
    from mesh_utils import compute_mesh_adjacency, compute_face_areas, compute_barycentric_matrices
    
    vertices_torch = torch.tensor(vertices, dtype=torch.float32, device=device)
    faces_torch = torch.tensor(faces, dtype=torch.int64, device=device)
    
    # Compute mesh data
    edges, edge2face, _ = compute_mesh_adjacency(faces)
    edges_torch = torch.tensor(edges, dtype=torch.int64, device=device)
    edge2face_torch = torch.tensor(edge2face, dtype=torch.int64, device=device)
    
    face_areas = compute_face_areas(vertices, faces)
    face_areas_torch = torch.tensor(face_areas, dtype=torch.float32, device=device)
    
    B = compute_barycentric_matrices(vertices, faces)
    B_torch = torch.tensor(B, dtype=torch.float32, device=device)
    
    # Initialize field values
    f_values = nn.Parameter(torch.randn(len(vertices), num_channels, device=device) * 0.01)
    
    # Set up coarse-to-fine schedule
    schedule = CoarseToFineSchedule()
    
    # Training history
    full_history = {}
    
    if use_coarse_to_fine:
        # Coarse-to-fine training
        for level in range(3):
            print(f"\n=== Training Level {level} ===")
            config = schedule.get_stage(level)
            
            # Prepare mesh for this level
            if config.num_faces > 0 and config.num_faces < len(faces):
                # Downsample mesh
                coarse_vertices, coarse_faces, vertex_mapping = downsample_mesh(
                    vertices, faces, config.num_faces
                )
                
                # Map field values to coarse mesh
                coarse_f_values = nn.Parameter(f_values[vertex_mapping].clone())
                
                # Map pinned indices
                coarse_pinned = []
                for idx in pinned_indices:
                    # Find closest vertex in coarse mesh
                    distances = np.linalg.norm(coarse_vertices - vertices[idx], axis=1)
                    coarse_pinned.append(np.argmin(distances))
                
                # Prepare coarse mesh data
                coarse_edges, coarse_edge2face, _ = compute_mesh_adjacency(coarse_faces)
                mesh_data = {
                    'vertices': torch.tensor(coarse_vertices, dtype=torch.float32, device=device),
                    'faces': torch.tensor(coarse_faces, dtype=torch.int64, device=device),
                    'edges': torch.tensor(coarse_edges, dtype=torch.int64, device=device),
                    'edge2face': torch.tensor(coarse_edge2face, dtype=torch.int64, device=device),
                    'face_areas': torch.tensor(compute_face_areas(coarse_vertices, coarse_faces), 
                                             dtype=torch.float32, device=device),
                    'B': torch.tensor(compute_barycentric_matrices(coarse_vertices, coarse_faces),
                                    dtype=torch.float32, device=device)
                }
                
                # Train on coarse mesh
                history = train_stage(config, coarse_f_values, mesh_data, 
                                    coarse_pinned, schedule, use_grad_norm, device)
                
                # Interpolate back to fine mesh
                with torch.no_grad():
                    f_values.data = coarse_f_values[vertex_mapping].data
                    
            else:
                # Train on full resolution
                mesh_data = {
                    'vertices': vertices_torch,
                    'faces': faces_torch,
                    'edges': edges_torch,
                    'edge2face': edge2face_torch,
                    'face_areas': face_areas_torch,
                    'B': B_torch
                }
                
                history = train_stage(config, f_values, mesh_data,
                                    pinned_indices, schedule, use_grad_norm, device)
            
            full_history[f'level_{level}'] = history
            
    else:
        # Direct training on full resolution
        config = TrainingConfig(level=0, num_faces=-1, steps=200000,
                               beta_start=2.0, beta_end=25.0,
                               lambda_adj_start=0.0, lambda_adj_end=8.0)
        
        mesh_data = {
            'vertices': vertices_torch,
            'faces': faces_torch,
            'edges': edges_torch,
            'edge2face': edge2face_torch,
            'face_areas': face_areas_torch,
            'B': B_torch
        }
        
        history = train_stage(config, f_values, mesh_data,
                            pinned_indices, schedule, use_grad_norm, device)
        
        full_history['direct'] = history
    
    return f_values.detach(), full_history