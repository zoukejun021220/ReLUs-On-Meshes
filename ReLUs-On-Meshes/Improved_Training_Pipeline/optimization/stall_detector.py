"""
Stall detection and dynamic lambda adjustment for improved convergence.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque


class StallDetector:
    """
    Detects optimization stalls and suggests parameter adjustments.
    """
    
    def __init__(self, 
                 window_size: int = 500,  # Reduced from 1000
                 patience: int = 1500,    # Reduced from 3000
                 improvement_threshold: float = 0.005,  # Reduced from 0.01
                 contour_loss_threshold: float = 0.002):
        """
        Args:
            window_size: Size of moving average window
            patience: Steps to wait before declaring stall
            improvement_threshold: Minimum relative improvement to not be stalled
            contour_loss_threshold: Minimum contour loss improvement
        """
        self.window_size = window_size
        self.patience = patience
        self.improvement_threshold = improvement_threshold
        self.contour_loss_threshold = contour_loss_threshold
        
        # Loss history
        self.total_loss_history = deque(maxlen=window_size * 2)
        self.contour_loss_history = deque(maxlen=window_size * 2)
        self.area_dev_history = deque(maxlen=window_size * 2)
        
        # Stall tracking
        self.steps_since_improvement = 0
        self.best_loss = float('inf')
        self.best_contour_loss = float('inf')
        self.last_adjustment_step = -patience
        
        # Lambda tracking
        self.lambda_contour_adjustments = []
        
    def update(self, 
               total_loss: float, 
               contour_loss: float,
               area_deviation: float,
               step: int) -> Tuple[bool, Optional[float]]:
        """
        Update stall detector with current metrics.
        
        Returns:
            (is_stalled, suggested_lambda_contour_multiplier)
        """
        self.total_loss_history.append(total_loss)
        self.contour_loss_history.append(contour_loss)
        self.area_dev_history.append(area_deviation)
        
        # Not enough history yet
        if len(self.total_loss_history) < self.window_size:
            return False, None
        
        # Compute moving averages
        recent_start = len(self.total_loss_history) - self.window_size
        old_start = max(0, len(self.total_loss_history) - 2 * self.window_size)
        
        # Ensure we have data for old window
        if old_start >= recent_start:
            return False, None
            
        recent_total = np.mean(list(self.total_loss_history)[recent_start:])
        old_total = np.mean(list(self.total_loss_history)[old_start:recent_start])
        
        recent_contour = np.mean(list(self.contour_loss_history)[recent_start:])
        old_contour = np.mean(list(self.contour_loss_history)[old_start:recent_start])
        
        recent_area_dev = np.mean(list(self.area_dev_history)[recent_start:])
        
        # Check for improvement - focus on CONTOUR loss only
        contour_improvement = (old_contour - recent_contour) / max(old_contour, 1e-9)
        
        # Only consider contour loss for stall detection
        has_improved = contour_improvement > self.contour_loss_threshold
        
        if has_improved:
            self.steps_since_improvement = 0
            self.best_loss = min(self.best_loss, recent_total)
            self.best_contour_loss = min(self.best_contour_loss, recent_contour)
        else:
            self.steps_since_improvement += 1
        
        # Check if we're stalled
        is_stalled = (
            self.steps_since_improvement > self.patience and
            step - self.last_adjustment_step > self.patience // 2  # Can adjust more frequently
        )
        
        suggested_multiplier = None
        
        # Only ramp up after patches have formed (low area deviation) and enough steps
        patches_formed = recent_area_dev < 0.05  # Stricter threshold
        min_steps_before_ramp = 20000  # Don't ramp in first 20k steps
        
        if is_stalled and patches_formed and step > min_steps_before_ramp:
            # Only increase if contour loss is not already very low
            if recent_contour > 0.01:  # Only if there's room for improvement
                # Very small increase
                suggested_multiplier = 1.02  # 2% increase only
            else:
                # Contour loss already low, no need to increase
                suggested_multiplier = 1.0
        else:
            # No increase if patches not formed or too early
            suggested_multiplier = None
        
        # Only record adjustment if we're suggesting an increase
        if suggested_multiplier is not None and suggested_multiplier > 1.0:
            self.last_adjustment_step = step
            self.lambda_contour_adjustments.append({
                'step': step,
                'multiplier': suggested_multiplier,
                'contour_loss': recent_contour,
                'area_dev': recent_area_dev
            })
        
        return is_stalled, suggested_multiplier
    
    def get_adaptive_lambda_contour(self, 
                                   base_lambda: float,
                                   current_step: int,
                                   max_lambda: float = 5.0,
                                   smooth_transitions: bool = True,
                                   total_steps: int = 300000) -> float:
        """
        Get adaptively adjusted lambda_contour based on stall history.
        
        Args:
            base_lambda: Base lambda from schedule
            current_step: Current optimization step
            max_lambda: Maximum allowed lambda_contour
            smooth_transitions: Whether to smooth stage transitions
            
        Returns:
            Adjusted lambda_contour
        """
        # Apply all historical adjustments
        adjusted_lambda = base_lambda
        
        for adjustment in self.lambda_contour_adjustments:
            if adjustment['step'] <= current_step:
                adjusted_lambda *= adjustment['multiplier']
        
        # NO automatic growth - only increase when stalled
        # This makes the optimization more conservative
        
        # Cap at maximum
        adjusted_lambda = min(adjusted_lambda, max_lambda)
        
        # Smooth transitions: limit how much lambda can grow relative to base
        if smooth_transitions:
            # Maximum growth factor relative to base lambda
            max_growth = 25.0  # λ can be at most 25x the base value (increased)
            adjusted_lambda = min(adjusted_lambda, base_lambda * max_growth)
        
        return adjusted_lambda
    
    def get_stats(self) -> Dict:
        """Get current stall detector statistics."""
        if len(self.total_loss_history) < self.window_size:
            return {
                'steps_since_improvement': self.steps_since_improvement,
                'num_adjustments': len(self.lambda_contour_adjustments),
                'is_warming_up': True
            }
        
        recent_start = len(self.total_loss_history) - self.window_size
        recent_total = np.mean(list(self.total_loss_history)[recent_start:])
        recent_contour = np.mean(list(self.contour_loss_history)[recent_start:])
        recent_area_dev = np.mean(list(self.area_dev_history)[recent_start:])
        
        return {
            'steps_since_improvement': self.steps_since_improvement,
            'num_adjustments': len(self.lambda_contour_adjustments),
            'recent_total_loss': recent_total,
            'recent_contour_loss': recent_contour,
            'recent_area_dev': recent_area_dev,
            'is_warming_up': False
        }