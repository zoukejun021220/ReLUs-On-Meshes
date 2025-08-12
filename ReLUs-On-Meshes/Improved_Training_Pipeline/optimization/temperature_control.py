"""
Progress-gated temperature scheduling and optimization utilities.
Addresses the issue of premature hardening (report sections 4.4.3, 4.4.5).
"""
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import numpy as np


@dataclass
class TempController:
    """
    Progress-gated temperature controller.
    Instead of time-based ramping, increase beta based on optimization progress.
    """
    # Current temperatures (SVD-based schedule)
    beta_contour: float = 0.8  # Start low for contour
    beta_area: float = 2.0     # Start higher for area to get meaningful KL divergence
    
    # Maximum temperatures for SVD approach
    beta_contour_max: float = 16.0   # Higher max for final sharp boundaries
    beta_area_max: float = 4.0       # Keep area temp moderate
    
    # Step sizes for increases (gradual for SVD)
    step_up_contour: float = 0.2   # Smaller steps for stability
    step_up_area: float = 0.1      # Keep area steps small
    
    # Progress thresholds
    tau_area: float = 0.05  # Stricter - only increase β when area deviation < 0.05
    min_boundary_fraction: float = 0.02  # Lower threshold
    contour_improve_threshold: float = 0.005  # Reduced to allow more β increases
    
    # Cooldown tracking
    last_beta_update_step: int = -10000
    min_steps_between_updates: int = 1000  # Reduced cooldown for faster β ramping
    best_contour_loss_since_update: float = float('inf')
    
    # History tracking
    history: Dict[str, List[float]] = field(default_factory=lambda: {
        'beta_contour': [],
        'beta_area': [],
        'area_deviation': [],
        'boundary_length': []
    })
    
    def check_and_update(self, 
                        area_fractions: torch.Tensor,
                        boundary_length: float,
                        bbox_diagonal: float,
                        step: int,
                        contour_loss: float) -> bool:
        """
        Check progress and potentially increase temperatures.
        Now requires contour improvement and enforces cooldown.
        
        Args:
            area_fractions: (C,) current area distribution
            boundary_length: Estimated boundary length
            bbox_diagonal: Mesh bounding box diagonal
            step: Current optimization step
            contour_loss: Current contour alignment loss
            
        Returns:
            updated: Whether temperatures were increased
        """
        # Compute area deviation from uniform
        uniform = 1.0 / area_fractions.shape[0]
        deviation = (area_fractions - uniform).abs().max().item()
        
        # Normalized boundary length
        boundary_fraction = boundary_length / bbox_diagonal
        
        # Store history
        self.history['area_deviation'].append(deviation)
        self.history['boundary_length'].append(boundary_fraction)
        self.history['beta_contour'].append(self.beta_contour)
        self.history['beta_area'].append(self.beta_area)
        
        # Check cooldown
        if step - self.last_beta_update_step < self.min_steps_between_updates:
            # Update best contour loss in this window
            self.best_contour_loss_since_update = min(
                self.best_contour_loss_since_update, contour_loss
            )
            return False
        
        # Handle first call or invalid best loss
        if not torch.isfinite(torch.tensor(self.best_contour_loss_since_update)):
            self.best_contour_loss_since_update = float(contour_loss)
        
        # Check improvement
        improvement = (self.best_contour_loss_since_update - contour_loss) / max(self.best_contour_loss_since_update, 1e-9)
        has_improved = improvement >= self.contour_improve_threshold
        
        # Adaptive thresholds based on current beta
        # Early on (low beta), be more lenient with area deviation
        adaptive_tau = self.tau_area * (1.0 + max(0, 5.0 - self.beta_contour) * 0.1)
        
        # Check all conditions
        updated = False
        if (has_improved and
            deviation < adaptive_tau and 
            boundary_fraction > self.min_boundary_fraction):
            
            # Adaptive step sizes - smaller steps at higher temperatures
            contour_step = self.step_up_contour * (1.0 - self.beta_contour / self.beta_contour_max * 0.5)
            area_step = self.step_up_area * (1.0 - self.beta_area / self.beta_area_max * 0.5)
            
            # Increase contour beta
            if self.beta_contour < self.beta_contour_max:
                self.beta_contour = min(
                    self.beta_contour + contour_step,
                    self.beta_contour_max
                )
                updated = True
            
            # Increase area beta (slower)
            if self.beta_area < self.beta_area_max:
                self.beta_area = min(
                    self.beta_area + area_step,
                    self.beta_area_max
                )
                updated = True
            
            if updated:
                self.last_beta_update_step = step
                self.best_contour_loss_since_update = float('inf')  # Reset
        else:
            # Track best loss in window
            self.best_contour_loss_since_update = min(
                self.best_contour_loss_since_update, contour_loss
            )
        
        return updated


@dataclass 
class OptimizationStage:
    """Configuration for a single optimization stage."""
    name: str
    start_fraction: float  # When to start (fraction of total steps)
    end_fraction: float    # When to end
    
    # Loss weights
    lambda_smooth: float = 1.0
    lambda_contour: float = 0.1
    lambda_area: float = 0.1
    lambda_pin: float = 0.01
    lambda_tv: Optional[float] = None  # Optional total variation
    
    # Learning rate
    lr: float = 1e-3
    cosine_decay: bool = False  # Use cosine decay for this stage
    
    # Pin constraint
    use_hard_pins: bool = False
    
    # Temperature control
    allow_temp_increase: bool = True


class TwoStageScheduler:
    """
    Two-stage optimization scheduler.
    Stage A: Coarse segmentation with low beta, strong smoothness
    Stage B: Refinement with higher beta, stronger alignment
    """
    
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        
        # Define SVD-based training stages
        # Stage 0: Warm start (2-5k iters) - coarse regions
        # Stage 1: Frozen SVD planes (15-30k iters) - let planes teach the field
        # Stage 2: Plane trust ramp (30-70k iters) - balanced learning
        # Stage 3: Lock normals, learn offsets (70-120k iters) - fine-tune positions
        # Stage 4: Crisp snap (final 20-40k) - final sharpening
        self.stages = [
            OptimizationStage(
                name="Stage 0: Warm Start",
                start_fraction=0.0,
                end_fraction=0.0167,  # First 5k steps (5k/300k)
                lambda_smooth=1.0,    # High cotan smoothness
                lambda_contour=0.0,   # NO SVD loss yet
                lambda_area=1.0,      # Area balance (reverse-KL + box barrier)
                lambda_pin=0.5,       # Soft pins
                lambda_tv=None,       # No TV
                lr=1e-4,              # AdamW with gradient clipping
                use_hard_pins=False,
                allow_temp_increase=True  # beta_c: 0.8 → 1.8 linear
            ),
            OptimizationStage(
                name="Stage 1: Frozen SVD Planes",
                start_fraction=0.0167,
                end_fraction=0.1,     # 5k-30k steps
                lambda_smooth=0.8,    # Still strong smoothness
                lambda_contour=0.1,   # Low SVD weight
                lambda_area=0.8,      # Maintain area balance
                lambda_pin=0.3,       # Reduce pin weight
                lambda_tv=None,
                lr=8e-5,              # Slightly reduced
                use_hard_pins=False,
                allow_temp_increase=True  # Update planes every K=20-50 steps
            ),
            OptimizationStage(
                name="Stage 2: Plane Trust Ramp",
                start_fraction=0.1,
                end_fraction=0.233,   # 30k-70k steps  
                lambda_smooth=0.5,    # Reduce smoothness
                lambda_contour=0.3,   # Increase SVD weight (ramp 0.1→0.5)
                lambda_area=0.5,      # Balanced
                lambda_pin=0.1,       # Low pin
                lambda_tv=None,
                lr=5e-5,              # Lower LR
                use_hard_pins=False,
                allow_temp_increase=True  # beta_c: 1.8 → 4.0, update planes K=10-20
            ),
            OptimizationStage(
                name="Stage 3: Lock Normals",
                start_fraction=0.233,
                end_fraction=0.4,     # 70k-120k steps
                lambda_smooth=0.3,    # Lower smoothness
                lambda_contour=0.5,   # Strong SVD (can go to 0.8)
                lambda_area=0.3,      # Lower area
                lambda_pin=0.05,      # Minimal pin
                lambda_tv=None,
                lr=3e-5,              # Even lower
                cosine_decay=True,    # Start decay
                use_hard_pins=False,
                allow_temp_increase=True  # beta_c → 8.0, only update offsets
            ),
            OptimizationStage(
                name="Stage 4: Crisp Snap",
                start_fraction=0.4,
                end_fraction=1.0,     # 120k-300k steps
                lambda_smooth=0.1,    # Minimal smoothness
                lambda_contour=0.8,   # Very high SVD (can go to 1.0+)
                lambda_area=0.2,      # Some area balance
                lambda_pin=0.01,      # Almost no pin
                lambda_tv=None,
                lr=1e-5,              # Very low LR
                cosine_decay=True,    # Continue decay
                use_hard_pins=False,  # Could use hard pins in final 10k
                allow_temp_increase=True  # beta_c → 12-16 for sharp boundaries
            )
        ]
        
        self.current_stage_idx = 0
        
    def get_stage(self, step: int) -> OptimizationStage:
        """Get current optimization stage based on step."""
        fraction = step / self.total_steps
        
        # Find appropriate stage
        for i, stage in enumerate(self.stages):
            if stage.start_fraction <= fraction < stage.end_fraction:
                if i != self.current_stage_idx:
                    print(f"\n=== Switching to {stage.name} ===")
                    self.current_stage_idx = i
                return stage
        
        # Default to last stage
        return self.stages[-1]
    
    def get_lr(self, step: int, base_lr: float) -> float:
        """Get learning rate for current step with minimum clamp."""
        stage = self.get_stage(step)
        lr = stage.lr
        
        # Apply decay if specified (cosine schedule)
        if stage.cosine_decay:
            stage_start = int(stage.start_fraction * self.total_steps)
            stage_end = int(stage.end_fraction * self.total_steps)
            steps_in_stage = step - stage_start
            total_stage_steps = stage_end - stage_start
            
            # Cosine annealing within stage
            progress = min(steps_in_stage / max(total_stage_steps, 1), 1.0)
            min_lr = 1e-5  # Minimum learning rate
            lr = min_lr + (lr - min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        # Ensure minimum learning rate
        return max(lr, 1e-5)


class GradientMonitor:
    """
    Monitor gradient statistics to detect vanishing/exploding gradients.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history = {
            'total': [],
            'smooth': [],
            'contour': [],
            'area': [],
            'pin': []
        }
        
    def log_gradients(self, F: torch.Tensor, loss_grads: Dict[str, torch.Tensor]):
        """Log gradient norms for each loss component."""
        # Total gradient norm
        total_grad_norm = F.grad.norm().item() if F.grad is not None else 0.0
        self.history['total'].append(total_grad_norm)
        
        # Per-loss gradient norms (if available)
        for name, grad in loss_grads.items():
            if grad is not None:
                self.history[name].append(grad.norm().item())
        
        # Keep only recent history
        for key in self.history:
            if len(self.history[key]) > self.window_size:
                self.history[key] = self.history[key][-self.window_size:]
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get gradient statistics."""
        stats = {}
        
        for name, values in self.history.items():
            if values:
                values_array = np.array(values)
                stats[name] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'median': float(np.median(values_array))
                }
            else:
                stats[name] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'median': 0.0}
        
        return stats
    
    def check_health(self) -> Dict[str, bool]:
        """Check for gradient issues."""
        health = {}
        
        if self.history['total']:
            recent = np.array(self.history['total'][-20:])  # Last 20 steps
            
            # Check for vanishing gradients
            health['vanishing'] = np.mean(recent) < 1e-7
            
            # Check for exploding gradients  
            health['exploding'] = np.max(recent) > 1e3
            
            # Check for high variance
            if len(recent) > 1:
                health['unstable'] = np.std(recent) / (np.mean(recent) + 1e-8) > 10.0
            else:
                health['unstable'] = False
        else:
            health = {'vanishing': False, 'exploding': False, 'unstable': False}
        
        return health


class EarlyStopping:
    """
    Early stopping based on loss plateau detection.
    """
    
    def __init__(self, patience: int = 5000, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None
        
    def check(self, loss: float, state_dict: Optional[Dict] = None) -> bool:
        """
        Check if training should stop.
        
        Args:
            loss: Current loss value
            state_dict: Optional model state to save
            
        Returns:
            should_stop: Whether to stop training
        """
        if loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = loss
            self.counter = 0
            if state_dict is not None:
                self.best_state = {k: v.clone() for k, v in state_dict.items()}
            return False
        else:
            # No improvement
            self.counter += 1
            return self.counter >= self.patience
    
    def get_best_state(self) -> Optional[Dict]:
        """Get the best model state."""
        return self.best_state