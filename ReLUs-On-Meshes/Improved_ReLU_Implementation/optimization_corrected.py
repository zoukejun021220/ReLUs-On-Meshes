"""
CORRECTED training pipeline for ReLU mesh segmentation.
Implements proper loss functions and scheduling based on paper audit.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from dataclasses import dataclass
from loss_functions_corrected import compute_total_loss_corrected


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
    """Manages coarse-to-fine training schedule (CORRECTED parameters)."""
    
    def __init__(self):
        # Parameters from the paper
        self.stages = [
            TrainingConfig(level=0, num_faces=3000, steps=30000,
                         beta_start=0.0, beta_end=10.0,  # 0->10
                         lambda_adj_start=0.0, lambda_adj_end=5.0,  # 0->5
                         lr_max=5e-3),
            TrainingConfig(level=1, num_faces=12000, steps=60000,
                         beta_start=10.0, beta_end=15.0,  # 10->15
                         lambda_adj_start=5.0, lambda_adj_end=5.0,  # Stay at 5
                         lr_max=5e-3),  
            TrainingConfig(level=2, num_faces=-1, steps=120000,  # Full resolution
                         beta_start=15.0, beta_end=25.0,  # 15->25
                         lambda_adj_start=5.0, lambda_adj_end=8.0,  # 5->8
                         lr_max=5e-3),  
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


def soft_pinning(f_values: torch.Tensor, pinned_indices: List[int], 
                 pinned_values: torch.Tensor, decay_rate: float = 0.99) -> None:
    """
    Apply soft pinning with exponential decay.
    Using 0.99 to allow more flexibility (1% change per step).
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
                device: torch.device) -> Dict:
    """
    Train a single stage with CORRECTED loss functions.
    """
    # Create pinned values
    pinned_values = torch.eye(6, device=device)
    
    # Setup optimizer
    optimizer = optim.AdamW([f_values], lr=config.lr_max, weight_decay=1e-4)
    lr_scheduler = OneCycleLR(optimizer, config.lr_max, config.steps)
    scaler = GradScaler()
    
    # History tracking
    history = {
        'loss': [], 'area': [], 'adjacency': [], 'tv': [],
        'area_fractions': [], 'lr': [], 'beta': [], 'lambda_adj': [],
        'weight_sum': []  # Track this - should drop by 2 orders of magnitude
    }
    
    # Training loop
    for step in range(config.steps):
        # Parameter scheduling
        progress = step / config.steps
        beta = schedule.interpolate_param(config.beta_start, config.beta_end, progress)
        
        # CRITICAL: Keep lambda_adj = 0 until beta >= 2 to avoid gradient explosion
        if beta < 2.0:
            lambda_adj = 0.0
        else:
            lambda_adj = schedule.interpolate_param(config.lambda_adj_start, config.lambda_adj_end, progress)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            loss_dict = compute_total_loss_corrected(
                f_values,
                mesh_data['vertices'],
                mesh_data['faces'],
                mesh_data['edges'],
                mesh_data['edge2face'],
                mesh_data['face_areas'],
                mesh_data['B'],
                face_mask=mesh_data.get('face_mask', None),
                beta=beta,
                lambda_area=config.lambda_area,
                lambda_adj=lambda_adj,
                lambda_tv=config.lambda_tv,
                return_components=True
            )
        
        loss = loss_dict['total']
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping (increased to 10 for more freedom)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([f_values], max_norm=10.0)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Learning rate update
        lr_scheduler.step()
        
        # Soft pinning (reduced strength)
        soft_pinning(f_values, pinned_indices, pinned_values, decay_rate=0.99)
        
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
                history['weight_sum'].append(loss_dict.get('weight_sum', 0).item())
            
            # Print progress
            if step % 1000 == 0:
                print(f"Step {step}/{config.steps}: Loss={loss.item():.4f}, "
                      f"Area={loss_dict['area'].item():.4f}, "
                      f"Adj={loss_dict['adjacency'].item():.4f}, "
                      f"TV={loss_dict['tv'].item():.4f}")
                
                # Show important metrics
                raw_adj = loss_dict.get('raw_adj_normalized', torch.tensor(0.0)).item()
                weight_sum = loss_dict.get('weight_sum', 0).item()
                print(f"  Raw adj: {raw_adj:.4f}, Weight sum: {weight_sum:.1f}")
                print(f"  β={beta:.1f}, λ_adj={lambda_adj:.2f}, λ_tv={config.lambda_tv}")
                print(f"  Area fractions: {loss_dict['area_fractions'].detach().cpu().numpy()}")
    
    return history


def optimize_mesh_segmentation_corrected(vertices: np.ndarray,
                                        faces: np.ndarray,
                                        pinned_indices: List[int],
                                        num_channels: int = 6,
                                        use_coarse_to_fine: bool = True,
                                        device: Optional[torch.device] = None,
                                        iterations: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
    """
    CORRECTED optimization function using proper loss formulations.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to torch tensors
    vertices_torch = torch.from_numpy(vertices).float().to(device)
    faces_torch = torch.from_numpy(faces).long().to(device)
    
    # Compute mesh data
    from mesh_utils import compute_mesh_data
    mesh_data_np = compute_mesh_data(vertices, faces)
    
    edges_torch = torch.from_numpy(mesh_data_np['edges']).long().to(device)
    edge2face_torch = torch.from_numpy(mesh_data_np['edge2face']).long().to(device)
    face_areas_torch = torch.from_numpy(mesh_data_np['face_areas']).float().to(device)
    B_torch = torch.from_numpy(mesh_data_np['B']).float().to(device)
    
    # Handle degenerate faces
    face_mask_torch = None
    if 'face_mask' in mesh_data_np:
        face_mask_torch = torch.from_numpy(mesh_data_np['face_mask']).bool().to(device)
    
    # Initialize field values
    f_values = nn.Parameter(torch.randn(len(vertices), num_channels, device=device) * 0.1)
    
    schedule = CoarseToFineSchedule()
    full_history = {}
    
    if use_coarse_to_fine:
        # Coarse-to-fine training
        for level in range(3):
            config = schedule.get_stage(level)
            print(f"\n{'='*60}")
            print(f"Training Level {level}: {config.num_faces} faces, {config.steps} steps")
            print(f"Beta: {config.beta_start:.1f} -> {config.beta_end:.1f}")
            print(f"Lambda_adj: {config.lambda_adj_start:.1f} -> {config.lambda_adj_end:.1f}")
            print('='*60)
            
            # For now, train on full resolution
            # (Mesh decimation would go here)
            mesh_data = {
                'vertices': vertices_torch,
                'faces': faces_torch,
                'edges': edges_torch,
                'edge2face': edge2face_torch,
                'face_areas': face_areas_torch,
                'B': B_torch,
                'face_mask': face_mask_torch
            }
            
            history = train_stage(config, f_values, mesh_data, pinned_indices, schedule, device)
            full_history[f'level_{level}'] = history
            
    else:
        # Direct training on full resolution
        steps = iterations if iterations is not None else 200000
        config = TrainingConfig(level=0, num_faces=-1, steps=steps,
                               beta_start=0.0, beta_end=25.0,  # Full range
                               lambda_adj_start=0.0, lambda_adj_end=8.0,  # Full range
                               lr_max=5e-3)
        
        print(f"\n{'='*60}")
        print(f"Direct Training: Full resolution, {config.steps} steps")
        print(f"Beta: {config.beta_start:.1f} -> {config.beta_end:.1f}")
        print(f"Lambda_adj: {config.lambda_adj_start:.1f} -> {config.lambda_adj_end:.1f}")
        print('='*60)
        
        mesh_data = {
            'vertices': vertices_torch,
            'faces': faces_torch,
            'edges': edges_torch,
            'edge2face': edge2face_torch,
            'face_areas': face_areas_torch,
            'B': B_torch,
            'face_mask': face_mask_torch
        }
        
        history = train_stage(config, f_values, mesh_data, pinned_indices, schedule, device)
        full_history['direct'] = history
    
    return f_values, full_history