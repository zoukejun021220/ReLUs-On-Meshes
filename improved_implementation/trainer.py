"""
Main training loop with two-stage optimization strategy.
Stage A: Coarse segmentation with low beta, focus on smoothness
Stage B: Boundary refinement with progress-gated beta increases
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple
import time
from pathlib import Path

from .mesh_utils import precompute_mesh_data
from .losses import compute_total_loss
from .temperature_control import (
    TempController, 
    maybe_raise_betas, 
    approx_boundary_length,
    get_adaptive_weights,
    get_learning_rate
)

Tensor = torch.Tensor


class MeshSegmentationTrainer:
    """Trainer for mesh segmentation with ReLU fields."""
    
    def __init__(
        self,
        verts: Tensor,
        faces: Tensor,
        n_channels: int = 6,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize trainer.
        
        Args:
            verts: (N, 3) - vertex positions
            faces: (T, 3) - triangle indices
            n_channels: number of segmentation channels
            device: computation device
        """
        self.device = torch.device(device)
        self.n_channels = n_channels
        
        # Move mesh data to device
        self.verts = verts.to(self.device)
        self.faces = faces.to(self.device)
        
        # Precompute mesh data
        print("Precomputing mesh data...")
        self.mesh_data = precompute_mesh_data(self.verts, self.faces)
        
        # Initialize field
        self.F = None
        self.initialize_field()
        
        # Training state
        self.optimizer = None
        self.temp_controller = TempController()
        self.history = {
            'loss': [],
            'loss_smooth': [],
            'loss_contour': [],
            'loss_area': [],
            'loss_pin': [],
            'beta_contour': [],
            'beta_area': [],
            'area_fractions': [],
            'boundary_length': []
        }
        
        print(f"Initialized trainer with {verts.shape[0]} vertices, "
              f"{faces.shape[0]} triangles, {n_channels} channels")
    
    def initialize_field(self, init_std: float = 0.01):
        """Initialize the field with small random values."""
        n_verts = self.verts.shape[0]
        self.F = nn.Parameter(
            torch.randn(n_verts, self.n_channels, device=self.device) * init_std
        )
    
    def set_pinned_vertices(
        self, 
        pin_idx: Tensor, 
        pin_values: Optional[Tensor] = None
    ):
        """
        Set pinned vertices for the optimization.
        
        Args:
            pin_idx: (P,) - indices of vertices to pin
            pin_values: (P, C) - target values, or None for one-hot encoding
        """
        self.pin_idx = pin_idx.to(self.device)
        
        if pin_values is None:
            # Create one-hot encoding
            P = len(pin_idx)
            C = self.n_channels
            if P != C:
                raise ValueError(f"Number of pins ({P}) must match channels ({C}) for one-hot")
            
            # One-hot: +1 for assigned channel, -1 for others
            pin_values = torch.full((P, C), -1.0, device=self.device)
            for i in range(P):
                pin_values[i, i] = 1.0
        
        self.pin_target = pin_values.to(self.device)
        
        # Initialize field at pinned vertices
        with torch.no_grad():
            self.F.data[self.pin_idx] = self.pin_target
    
    def train(
        self,
        n_steps: int = 100000,
        print_every: int = 1000,
        save_every: int = 10000,
        checkpoint_dir: Optional[str] = None,
        initial_lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 5.0,
        stage_transition: float = 0.6,
        beta_update_every: int = 400,
        hard_pin_at: float = 0.9
    ):
        """
        Train the segmentation field.
        
        Args:
            n_steps: total training steps
            print_every: print progress interval
            save_every: checkpoint save interval
            checkpoint_dir: directory for checkpoints
            initial_lr: starting learning rate
            weight_decay: AdamW weight decay
            grad_clip: gradient clipping value
            stage_transition: fraction of steps for Stage A
            beta_update_every: steps between beta updates
            hard_pin_at: fraction of steps to switch to hard pinning
        """
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            [self.F], 
            lr=initial_lr, 
            weight_decay=weight_decay
        )
        
        # Checkpoint directory
        if checkpoint_dir:
            ckpt_path = Path(checkpoint_dir)
            ckpt_path.mkdir(exist_ok=True)
        
        start_time = time.time()
        
        for step in range(n_steps):
            # Get adaptive weights and learning rate
            weights = get_adaptive_weights(step, n_steps, stage_transition)
            lr = get_learning_rate(step, n_steps, initial_lr, stage_transition=stage_transition)
            
            # Update learning rate
            for g in self.optimizer.param_groups:
                g['lr'] = lr
            
            # Forward pass
            self.optimizer.zero_grad(set_to_none=True)
            
            losses, total_loss = compute_total_loss(
                self.F,
                self.mesh_data,
                self.faces,
                self.pin_idx if hasattr(self, 'pin_idx') else None,
                self.pin_target if hasattr(self, 'pin_target') else None,
                weights,
                self.temp_controller.beta_contour,
                self.temp_controller.beta_area
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([self.F], max_norm=grad_clip)
            
            # Optimizer step
            self.optimizer.step()
            
            # Progress-based temperature updates
            if step > 0 and (step % beta_update_every) == 0:
                boundary_len = approx_boundary_length(
                    self.F, 
                    self.mesh_data['edge_idx'],
                    self.temp_controller.beta_contour,
                    self.verts,
                    self.mesh_data['edge_len']
                )
                
                beta_increased = maybe_raise_betas(
                    self.temp_controller,
                    losses['_frac'],
                    boundary_len,
                    self.mesh_data['bbox_diag']
                )
                
                if beta_increased and print_every > 0:
                    print(f"  Beta increased: contour={self.temp_controller.beta_contour:.1f}, "
                          f"area={self.temp_controller.beta_area:.1f}")
            
            # Hard pinning in final stage
            if step == int(hard_pin_at * n_steps) and hasattr(self, 'pin_idx'):
                with torch.no_grad():
                    self.F.data[self.pin_idx] = self.pin_target
                print(f"  Switched to hard pinning at step {step}")
            
            # Record history
            self.history['loss'].append(total_loss.item())
            self.history['loss_smooth'].append(losses['smooth'].item())
            self.history['loss_contour'].append(losses['contour'].item())
            self.history['loss_area'].append(losses['area'].item())
            self.history['loss_pin'].append(losses['pin'].item())
            self.history['beta_contour'].append(self.temp_controller.beta_contour)
            self.history['beta_area'].append(self.temp_controller.beta_area)
            
            # Print progress
            if print_every > 0 and (step % print_every) == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (step + 1) * (n_steps - step - 1)
                
                print(f"Step {step:6d}/{n_steps} | "
                      f"Loss: {total_loss.item():.5f} | "
                      f"Smooth: {losses['smooth'].item():.4f} | "
                      f"Contour: {losses['contour'].item():.4f} | "
                      f"Area: {losses['area'].item():.4f} | "
                      f"βc: {self.temp_controller.beta_contour:.1f} | "
                      f"βa: {self.temp_controller.beta_area:.1f} | "
                      f"LR: {lr:.2e} | "
                      f"ETA: {eta/60:.1f}m")
                
                # Area fractions
                if '_frac' in losses:
                    frac_str = ", ".join([f"{f:.3f}" for f in losses['_frac'].detach().cpu().numpy()])
                    print(f"  Area fractions: [{frac_str}]")
            
            # Save checkpoint
            if checkpoint_dir and save_every > 0 and (step % save_every) == 0:
                self.save_checkpoint(ckpt_path / f"checkpoint_{step:06d}.pt", step)
        
        # Final checkpoint
        if checkpoint_dir:
            self.save_checkpoint(ckpt_path / "final.pt", n_steps)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.1f} minutes")
    
    def save_checkpoint(self, path: Path, step: int):
        """Save training checkpoint."""
        torch.save({
            'step': step,
            'F': self.F.data,
            'optimizer': self.optimizer.state_dict(),
            'temp_controller': self.temp_controller,
            'history': self.history,
            'mesh_info': {
                'n_verts': self.verts.shape[0],
                'n_faces': self.faces.shape[0],
                'n_channels': self.n_channels
            }
        }, path)
        print(f"  Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path):
        """Load training checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.F.data = ckpt['F']
        if self.optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        self.temp_controller = ckpt['temp_controller']
        self.history = ckpt['history']
        print(f"Loaded checkpoint from {path} (step {ckpt['step']})")
        return ckpt['step']
    
    def get_field_values(self) -> Tensor:
        """Get current field values."""
        return self.F.data.detach()
    
    def get_segmentation(self, beta: Optional[float] = None) -> Tensor:
        """
        Get hard segmentation labels.
        
        Args:
            beta: temperature for softmax (None = use argmax)
            
        Returns:
            (N,) tensor of segment labels
        """
        if beta is None:
            return self.F.data.argmax(dim=1)
        else:
            probs = torch.softmax(beta * self.F.data, dim=1)
            return probs.argmax(dim=1)