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
from pathlib import Path
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
        # CORRECTED: Slower beta ramp, frozen lambda_adj in later stages
        self.stages = [
            TrainingConfig(level=0, num_faces=3000, steps=30000,
                         beta_start=0.0, beta_end=6.0,  # Slower: 0->6
                         lambda_adj_start=0.0, lambda_adj_end=5.0,  # 0->5
                         lr_max=5e-3, lambda_area=4.0),  # Increased area weight
            TrainingConfig(level=1, num_faces=12000, steps=60000,
                         beta_start=6.0, beta_end=12.0,  # Slower: 6->12
                         lambda_adj_start=5.0, lambda_adj_end=5.0,  # Stay at 5
                         lr_max=5e-3, lambda_area=4.0),  
            TrainingConfig(level=2, num_faces=-1, steps=180000,  # More steps
                         beta_start=12.0, beta_end=20.0,  # Slower: 12->20
                         lambda_adj_start=5.0, lambda_adj_end=5.0,  # FROZEN at 5
                         lr_max=5e-3, lambda_area=4.0),  
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
                device: torch.device,
                checkpoint_dir: Optional[Path] = None,
                stage_name: str = "stage") -> Dict:
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
        'weight_sum': [], 'raw_adj': []  # Track raw adjacency for freezing
    }
    
    # Track when to freeze lambda_adj
    lambda_adj_frozen = False
    lambda_adj_frozen_value = 5.0
    
    # Checkpoint tracking
    checkpoint_steps = [100, 500, 1000, 5000, 10000, 20000, 50000, 100000, 200000]
    diagnostic_steps = [0, 100, 1000, 10000]  # Steps for detailed diagnostics
    
    # Training loop
    for step in range(config.steps):
        # Parameter scheduling
        progress = step / config.steps
        beta = schedule.interpolate_param(config.beta_start, config.beta_end, progress)
        
        # CRITICAL: Keep lambda_adj = 0 until beta >= 2 to avoid gradient explosion
        # Then freeze it once raw_adj drops below 0.15
        if beta < 2.0:
            lambda_adj = 0.0
        elif lambda_adj_frozen:
            lambda_adj = lambda_adj_frozen_value
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
                raw_adj = loss_dict.get('raw_adj_normalized', 0).item()
                
                # Freeze lambda_adj if raw_adj drops below threshold
                if not lambda_adj_frozen and raw_adj < 0.15 and lambda_adj > 0:
                    lambda_adj_frozen = True
                    lambda_adj_frozen_value = lambda_adj
                    print(f"\n>>> Freezing lambda_adj at {lambda_adj:.2f} (raw_adj={raw_adj:.4f} < 0.15)")
                
                history['loss'].append(loss.item())
                history['area'].append(loss_dict['area'].item())
                history['adjacency'].append(loss_dict['adjacency'].item())
                history['tv'].append(loss_dict['tv'].item())
                history['area_fractions'].append(loss_dict['area_fractions'].cpu().numpy())
                history['lr'].append(optimizer.param_groups[0]['lr'])
                history['beta'].append(beta)
                history['lambda_adj'].append(lambda_adj)
                history['weight_sum'].append(loss_dict.get('weight_sum', 0).item())
                history['raw_adj'].append(raw_adj)
            
            # Save checkpoints at key steps
            if checkpoint_dir and step in checkpoint_steps and step <= config.steps:
                checkpoint = {
                    'step': step,
                    'stage': stage_name,
                    'f_values': f_values.detach().cpu(),
                    'optimizer_state': optimizer.state_dict(),
                    'metrics': {
                        'loss': loss.item(),
                        'area': loss_dict['area'].item(),
                        'adjacency': loss_dict['adjacency'].item(),
                        'tv': loss_dict['tv'].item(),
                        'raw_adj': raw_adj,
                        'weight_sum': loss_dict.get('weight_sum', 0).item(),
                        'beta': beta,
                        'lambda_adj': lambda_adj,
                        'area_fractions': loss_dict['area_fractions'].cpu().numpy()
                    },
                    'config': config.__dict__
                }
                torch.save(checkpoint, checkpoint_dir / f'{stage_name}_step_{step}.pt')
                print(f"  ✓ Checkpoint saved: {stage_name}_step_{step}.pt")
            
            # Detailed diagnostics at key steps
            if step in diagnostic_steps:
                print(f"\n{'='*60}")
                print(f"DIAGNOSTIC at step {step}:")
                print('='*60)
                
                # Weight histogram analysis
                w_e = loss_dict.get('weight_sum', 0)
                if isinstance(w_e, torch.Tensor) and w_e.numel() > 1:
                    w_e_flat = w_e.detach().cpu().numpy().flatten()
                    print(f"Weight distribution:")
                    print(f"  Min: {w_e_flat.min():.4f}")
                    print(f"  Mean: {w_e_flat.mean():.4f}")
                    print(f"  Max: {w_e_flat.max():.4f}")
                    print(f"  Near 0 (< 0.1): {(w_e_flat < 0.1).sum() / len(w_e_flat) * 100:.1f}%")
                    print(f"  Near 1 (> 0.9): {(w_e_flat > 0.9).sum() / len(w_e_flat) * 100:.1f}%")
                
                # Gradient analysis
                if f_values.grad is not None:
                    grad_norm = f_values.grad.norm().item()
                    grad_max = f_values.grad.abs().max().item()
                    print(f"\nGradient stats:")
                    print(f"  Norm: {grad_norm:.6f}")
                    print(f"  Max: {grad_max:.6f}")
                
                # Field value statistics
                f_vals = f_values.detach()
                print(f"\nField value stats:")
                print(f"  Min: {f_vals.min().item():.4f}")
                print(f"  Max: {f_vals.max().item():.4f}")
                print(f"  Std: {f_vals.std().item():.4f}")
                
                # Softmax probabilities
                probs = torch.softmax(beta * f_vals, dim=1)
                max_probs = probs.max(dim=1)[0]
                print(f"\nSoftmax confidence:")
                print(f"  Mean max prob: {max_probs.mean().item():.4f}")
                print(f"  Min max prob: {max_probs.min().item():.4f}")
                print(f"  % confident (>0.9): {(max_probs > 0.9).float().mean().item() * 100:.1f}%")
                
                print('='*60 + '\n')
            
            # Print progress
            if step % 1000 == 0:
                print(f"Step {step}/{config.steps}: Loss={loss.item():.4f}, "
                      f"Area={loss_dict['area'].item():.4f}, "
                      f"Adj={loss_dict['adjacency'].item():.4f}, "
                      f"TV={loss_dict['tv'].item():.4f}")
                
                # Show important metrics - ensure we get the RIGHT raw value
                if 'raw_adj_normalized' in loss_dict:
                    raw_adj = loss_dict['raw_adj_normalized'].item()
                else:
                    # Fallback: compute from weighted loss
                    raw_adj = loss_dict['adjacency'].item() / lambda_adj if lambda_adj > 0 else 0
                    
                weight_sum = loss_dict.get('weight_sum', 0).item()
                
                # Clear output showing both values with better diagnostics
                print(f"  Raw adj (normalized): {raw_adj:.4f} (target < 0.05)")
                print(f"  Weighted adj (λ*norm): {loss_dict['adjacency'].item():.4f}")
                print(f"  Weight sum: {weight_sum:.1f}, β={beta:.1f}, λ_adj={lambda_adj:.2f}")
                
                # Clear warning about expected behavior
                if step == 0:
                    print(f"  Expected: raw_adj should start ~0.45 and drop to <0.05")
                elif raw_adj > 2.0:
                    print(f"  ERROR: Raw adj > 2 indicates missing /15 normalization!")
                elif raw_adj > 0.5 and step > 10000:
                    print(f"  WARNING: Raw adj still high - check gradients are flowing")
                elif raw_adj < 0.05:
                    print(f"  ✓ GOOD: Raw adj < 0.05 - boundaries are planar!")
                    
                print(f"  Area fractions: {loss_dict['area_fractions'].detach().cpu().numpy()}")
    
    return history


def optimize_mesh_segmentation_corrected(vertices: np.ndarray,
                                        faces: np.ndarray,
                                        pinned_indices: List[int],
                                        num_channels: int = 6,
                                        use_coarse_to_fine: bool = True,
                                        device: Optional[torch.device] = None,
                                        iterations: Optional[int] = None,
                                        checkpoint_dir: Optional[Path] = None) -> Tuple[torch.Tensor, Dict]:
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
            
            history = train_stage(config, f_values, mesh_data, pinned_indices, schedule, device,
                                checkpoint_dir=checkpoint_dir, stage_name=f'level_{level}')
            full_history[f'level_{level}'] = history
            
    else:
        # Direct training on full resolution (CORRECTED schedule)
        steps = iterations if iterations is not None else 300000
        config = TrainingConfig(level=0, num_faces=-1, steps=steps,
                               beta_start=0.0, beta_end=20.0,  # Slower ramp
                               lambda_adj_start=0.0, lambda_adj_end=5.0,  # Cap at 5
                               lr_max=5e-3, lambda_area=4.0)  # Increased area weight
        
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
        
        history = train_stage(config, f_values, mesh_data, pinned_indices, schedule, device,
                            checkpoint_dir=checkpoint_dir, stage_name='direct')
        full_history['direct'] = history
    
    return f_values, full_history