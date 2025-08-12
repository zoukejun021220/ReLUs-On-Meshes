"""
Smooth lambda scheduling with gradual transitions.
"""
import torch
import numpy as np
from typing import Dict, Optional


class SmoothLambdaScheduler:
    """
    Provides smooth transitions for lambda values between stages.
    """
    
    def __init__(self, transition_steps: int = 2000):
        """
        Args:
            transition_steps: Number of steps to smoothly transition between stages
        """
        self.transition_steps = transition_steps
        self.last_stage_name = None
        self.transition_start_step = None
        self.old_lambdas = {}
        self.new_lambdas = {}
        
    def get_smooth_lambda(self,
                         lambda_name: str,
                         base_value: float,
                         current_step: int,
                         stage_name: str) -> float:
        """
        Get smoothly transitioned lambda value.
        
        Args:
            lambda_name: Name of lambda (e.g., 'contour', 'smooth', 'area')
            base_value: Base value from current stage (or adaptive value)
            current_step: Current optimization step
            stage_name: Current stage name
            
        Returns:
            Smoothly transitioned lambda value
        """
        # For lambda_contour, the base_value is already adaptive - just return it
        # The smooth scheduler was incorrectly caching and overriding adaptive values
        if lambda_name == 'contour':
            return base_value
            
        # For other lambdas, do smooth transitions between stages
        # Check if we're in a new stage
        if stage_name != self.last_stage_name:
            if self.last_stage_name is not None:
                # Starting a transition
                self.transition_start_step = current_step
                self.old_lambdas = self.new_lambdas.copy()
                self.new_lambdas[lambda_name] = base_value
            else:
                # First stage - no transition
                self.new_lambdas[lambda_name] = base_value
                self.old_lambdas[lambda_name] = base_value
            
            self.last_stage_name = stage_name
        
        # Update target value
        if lambda_name not in self.new_lambdas:
            self.new_lambdas[lambda_name] = base_value
            self.old_lambdas[lambda_name] = base_value
        else:
            # Update new_lambdas to track the current base value
            self.new_lambdas[lambda_name] = base_value
        
        # Check if we're in a transition
        if (self.transition_start_step is not None and 
            current_step < self.transition_start_step + self.transition_steps and
            lambda_name in self.old_lambdas):
            
            # Compute transition progress
            progress = (current_step - self.transition_start_step) / self.transition_steps
            progress = np.clip(progress, 0.0, 1.0)
            
            # Smooth transition using cosine interpolation
            smooth_progress = 0.5 * (1 - np.cos(np.pi * progress))
            
            # Interpolate between old and new values
            old_val = self.old_lambdas.get(lambda_name, base_value)
            new_val = self.new_lambdas.get(lambda_name, base_value)
            
            return old_val + (new_val - old_val) * smooth_progress
        
        # No transition - return current value
        return self.new_lambdas.get(lambda_name, base_value)
    
    def reset(self):
        """Reset the scheduler state."""
        self.last_stage_name = None
        self.transition_start_step = None
        self.old_lambdas = {}
        self.new_lambdas = {}