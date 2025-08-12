"""
Improved training pipeline for ReLU mesh segmentation.
Addresses all convergence issues identified in the report.
"""
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

# Import our modules
from utils.mesh_preprocessing import (
    load_volume_tet_mesh_and_extract_surface,
    preprocess_mesh,
    pick_axis_aligned_anchors
)
from losses.improved_losses import (
    contour_alignment_intrinsic,
    contour_alignment_intrinsic_v2,
    smoothness_cotan,
    area_fractions_and_kl,
    area_balance_loss,
    compute_hard_area_fractions,
    pin_loss,
    compute_boundary_stats,
    compute_boundary_stats_v2,
    compute_boundary_length_estimate,
    total_variation_loss,
    non_boundary_margin_loss,
    potts_smoothness_loss,
    boundary_length_regularizer,
    normal_axis_losses,
    area_kl_to_prior,
    triple_point_barrier,
    margin_separation_loss
)
from losses.soft_pairs_contour import contour_alignment_soft_pairs
from losses.svd_contour_loss import contour_alignment_svd
from optimization.temperature_control import (
    TempController,
    TwoStageScheduler,
    GradientMonitor
)
from optimization.stall_detector import StallDetector
from optimization.smooth_scheduler import SmoothLambdaScheduler


def initialize_field(n_vertices: int, n_channels: int, 
                    pinned_indices: torch.Tensor,
                    device: torch.device,
                    verts_xyz: torch.Tensor,
                    channel_labels: Optional[Dict[int, str]] = None) -> torch.Tensor:
    """
    Initialize multi-channel field with coordinate bias to reduce early ties.
    
    Args:
        n_vertices: Number of vertices
        n_channels: Number of channels (6 for axis-aligned segmentation)
        pinned_indices: Indices of pinned vertices
        device: PyTorch device
        verts_xyz: (N, 3) vertex coordinates
        
    Returns:
        F: (N, C) initialized field values
    """
    F = torch.zeros(n_vertices, n_channels, device=device)
    
    # Normalize coordinates
    xyz = (verts_xyz - verts_xyz.mean(0, keepdim=True)) / (verts_xyz.std(0, keepdim=True) + 1e-6)
    
    # Channels: [+X, -X, +Y, -Y, +Z, -Z] - aligned with channel_labels
    if n_channels >= 6:
        # Scale down initial values even more to prevent early instability
        scale = 0.05  # reduced from 0.1
        
        # Initialize based on channel labels if provided
        if channel_labels:
            for c, label in channel_labels.items():
                if c < n_channels:
                    if label == '+X':
                        F[:, c] = xyz[:, 0] * scale
                    elif label == '-X':
                        F[:, c] = -xyz[:, 0] * scale
                    elif label == '+Y':
                        F[:, c] = xyz[:, 1] * scale
                    elif label == '-Y':
                        F[:, c] = -xyz[:, 1] * scale
                    elif label == '+Z':
                        F[:, c] = xyz[:, 2] * scale
                    elif label == '-Z':
                        F[:, c] = -xyz[:, 2] * scale
        else:
            # Default initialization if no labels
            F[:, 0] =  xyz[:, 0] * scale  # +X
            F[:, 1] = -xyz[:, 0] * scale  # -X
            F[:, 2] =  xyz[:, 1] * scale  # +Y
            F[:, 3] = -xyz[:, 1] * scale  # -Y
            F[:, 4] =  xyz[:, 2] * scale  # +Z
            F[:, 5] = -xyz[:, 2] * scale  # -Z
    
    # Add very small noise
    F += 0.0005 * torch.randn_like(F)  # reduced from 0.001
    
    # Keep anchor contrast - now properly mapped to channels
    # Each pin gets +1 on its channel, -1 on all others
    for c in range(min(n_channels, len(pinned_indices))):
        F[pinned_indices[c], :] = -1.0
        F[pinned_indices[c], c] = 1.0
    
    return F


def compute_pin_targets(pinned_indices: torch.Tensor, 
                       n_channels: int,
                       device: torch.device,
                       channel_labels: Optional[Dict[int, str]] = None) -> torch.Tensor:
    """
    Compute target values for pinned vertices.
    
    Args:
        pinned_indices: (P,) indices of pinned vertices
        n_channels: Number of channels
        device: PyTorch device
        
    Returns:
        targets: (P, C) target values
    """
    P = len(pinned_indices)
    targets = torch.full((P, n_channels), -1.0, device=device)
    
    # Each pin index i corresponds to channel i (order preserved)
    for i in range(min(P, n_channels)):
        targets[i, i] = 1.0
    
    return targets


def train_mesh_segmentation(
    mesh_data: Dict[str, torch.Tensor],
    n_channels: int = 6,
    n_steps: int = 100000,
    device: str = 'cuda',
    output_dir: Optional[Path] = None,
    checkpoint_freq: int = 5000,
    log_freq: int = 500,
    verbose: bool = True,
    resume_from: Optional[str] = None,
    use_soft_pairs: bool = False,
    use_5_patch_prior: bool = False,
    use_v2_contour: bool = True,
    use_improved_area: bool = True,
    use_svd_contour: bool = False
) -> Tuple[torch.Tensor, Dict]:
    """
    Main training function with improved optimization.
    
    Args:
        mesh_data: Preprocessed mesh data from preprocess_mesh()
        n_channels: Number of segmentation channels
        n_steps: Number of optimization steps
        device: PyTorch device
        output_dir: Directory for saving checkpoints
        checkpoint_freq: Frequency of checkpointing
        log_freq: Frequency of logging
        verbose: Whether to print progress
        
    Returns:
        F: Optimized field values
        history: Training history
    """
    # Extract mesh data
    verts = mesh_data['vertices']
    faces = mesh_data['faces']
    tri_area = mesh_data['tri_area']
    tri_xy = mesh_data['tri_xy']
    edge_idx = mesh_data['edge_idx']
    edge_tris = mesh_data['edge_tris']
    I = mesh_data['cotan_I']
    J = mesh_data['cotan_J']
    W = mesh_data['cotan_W']
    pinned_indices = mesh_data['pinned_indices']
    channel_labels = mesh_data.get('channel_labels', {i: f'Ch{i}' for i in range(6)})
    stats = mesh_data['stats']
    
    # Initialize or load from checkpoint
    start_step = 0
    opt_path = None
    if resume_from:
        print(f"\nResuming from checkpoint: {resume_from}")
        ckpt = np.load(resume_from)
        F = torch.tensor(ckpt['field_values'], device=device, dtype=torch.float32)
        F = nn.Parameter(F)
        start_step = int(ckpt['step'])
        
        # Initialize controllers with saved state
        temp_ctrl = TempController()
        temp_ctrl.beta_contour = float(ckpt.get('beta_contour', 4.0))
        temp_ctrl.beta_area = float(ckpt.get('beta_area', 2.5))
        temp_ctrl.last_beta_update_step = start_step - 1000  # Allow immediate updates
        stall_detector = StallDetector()
        smooth_scheduler = SmoothLambdaScheduler(transition_steps=2000)
        
        # Plane memory for SVD approach
        plane_memory = {} if use_svd_contour else None
        K_update = 20  # Update planes every K iterations
        
        print(f"  Resuming from step {start_step}")
        print(f"  Current β: contour={temp_ctrl.beta_contour:.2f}, area={temp_ctrl.beta_area:.2f}")
        
        # Check if there's an optimizer state
        opt_path = Path(resume_from).parent / f"{Path(resume_from).stem}_optimizer.pt"
        if opt_path.exists():
            opt_state = torch.load(opt_path, map_location=device)
            if 'temp_ctrl_state' in opt_state:
                # Update temp controller history
                for k, v in opt_state['temp_ctrl_state'].items():
                    if hasattr(temp_ctrl, k):
                        setattr(temp_ctrl, k, v)
    else:
        # Initialize field with coordinate bias
        F = initialize_field(verts.shape[0], n_channels, pinned_indices, device, verts, channel_labels)
        F = nn.Parameter(F)
        temp_ctrl = TempController()
        stall_detector = StallDetector()
        smooth_scheduler = SmoothLambdaScheduler(transition_steps=2000)
        
        # Plane memory for SVD approach
        plane_memory = {} if use_svd_contour else None
        K_update = 20  # Update planes every K iterations
    
    # Pin targets
    pin_targets = compute_pin_targets(pinned_indices, n_channels, device, channel_labels)
    
    # Optimizer - reduced learning rate for better stability
    optimizer = torch.optim.AdamW([F], lr=5e-5, weight_decay=0.0)  # reduced from 1e-4
    
    # Load optimizer state if resuming
    if resume_from and opt_path and opt_path.exists():
        opt_state = torch.load(opt_path, map_location=device)
        if 'optimizer_state' in opt_state:
            optimizer.load_state_dict(opt_state['optimizer_state'])
            print("  Loaded optimizer state")
    
    # Controllers
    scheduler = TwoStageScheduler(n_steps)
    grad_monitor = GradientMonitor()
    
    # History tracking
    history = {
        'loss': [],
        'loss_smooth': [],
        'loss_contour': [],
        'loss_area': [],
        'loss_pin': [],
        'loss_potts': [],
        'loss_boundary_length': [],
        'loss_normal_align': [],
        'loss_normal_disp': [],
        'loss_triple': [],
        'area_fractions': [],
        'beta_contour': [],
        'beta_area': [],
        'boundary_length': [],
        'active_edge_fraction': [],
        'lr': []
    }
    
    # Training loop
    start_time = time.time()
    
    for step in range(start_step, n_steps):
        # Get current stage
        prev_stage_name = scheduler.stages[scheduler.current_stage_idx].name if step > 0 else None
        stage = scheduler.get_stage(step)
        if prev_stage_name and stage.name != prev_stage_name:
            print(f"\n{'='*80}")
            print(f"Stage transition: {prev_stage_name} -> {stage.name}")
            print(f"New weights: λ_smooth={stage.lambda_smooth}, λ_contour={stage.lambda_contour}, λ_area={stage.lambda_area}")
            print(f"{'='*80}\n")
        
        # Update learning rate
        current_lr = scheduler.get_lr(step, stage.lr)
        
        # Temperature-aware LR scaling to prevent instability at high β
        temp_scale = min(1.0, 2.0 / max(1.0, float(temp_ctrl.beta_contour)))
        area_scale = min(1.0, 1.5 / max(1.0, float(temp_ctrl.beta_area)))
        scaled_lr = max(1e-5, current_lr * temp_scale * area_scale)
        
        for g in optimizer.param_groups:
            g['lr'] = scaled_lr
        
        # Zero gradients
        optimizer.zero_grad(set_to_none=True)
        
        # Compute losses
        losses = {}
        
        # Smoothness loss (cotangent Laplacian)
        losses['smooth'] = smoothness_cotan(F, I, J, W)
        
        # Contour alignment (3D) - get edge weights too for margin loss
        # Add debug logging every 1000 steps
        debug_contour = (step % 1000 == 0) and verbose
        
        if use_svd_contour:
            # Use SVD-based contour alignment with stable plane fitting
            # Get edge coordinates for SVD
            edge_verts = verts[edge_idx]  # (E, 2, 3)
            
            # Update planes only every K steps or in certain stages
            update_planes = (step % K_update == 0) or (stage.name in ["Stage 1: Frozen SVD Planes", "Stage 2: Plane Trust Ramp"])
            
            losses['contour'], svd_info = contour_alignment_svd(
                F, edge_verts, edge_idx,
                beta=temp_ctrl.beta_contour,
                min_weight=0.01,
                plane_memory=plane_memory,
                ema=0.2,
                use_triple_points=False,  # Can enable later
                K_update=K_update if update_planes else 1000000  # Large K means no update
            )
            edge_weights = None  # SVD doesn't return edge weights in same format
            
        elif use_v2_contour:
            # Use improved v2 contour with all-pairs mixing and soft-OR coverage
            contour_result = contour_alignment_intrinsic_v2(
                F, faces, edge_idx, edge_tris, verts,
                beta_contour=temp_ctrl.beta_contour,
                return_weights=True,
                debug_log=debug_contour
            )
            losses['contour'], edge_weights = contour_result
        elif use_soft_pairs:
            # Use soft all-pairs mixing for stable triple points
            contour_result = contour_alignment_soft_pairs(
                F, faces, edge_idx, edge_tris,
                beta_contour=temp_ctrl.beta_contour,
                return_weights=True,
                verts=verts
            )
            losses['contour'], edge_weights = contour_result
        else:
            # Use standard top-2 channel selection
            contour_result = contour_alignment_intrinsic(
                F, faces, edge_idx, edge_tris,
                beta_contour=temp_ctrl.beta_contour,
                return_weights=True,
                verts=verts,
                debug_log=debug_contour
            )
            losses['contour'], edge_weights = contour_result
        
        # Area balance
        if use_improved_area:
            # Use improved area loss with reverse KL and straight-through
            # Enable ST earlier to align with hard assignments
            use_straight_through = temp_ctrl.beta_area > 2.0 and step > 2000  # Enable ST much earlier
            barrier_weight = 0.1 if step < 10000 else 0.05 if step < 20000 else 0.02  # Stronger barrier early
            
            # For 5-patch prior, adjust the min_frac to allow one channel to be small
            if use_5_patch_prior:
                min_frac_val = 0.01 / n_channels  # Allow one channel to be very small
                max_frac_val = 2.5 / n_channels   # Allow larger channels
            else:
                min_frac_val = None  # Use defaults
                max_frac_val = None
            
            # Force straight-through from the beginning for better gradients
            use_straight_through = step > 100  # Enable ST almost immediately
            
            # Use L2 method for more direct gradient signal
            # Add entropy regularization to encourage decisive assignments
            entropy_w = 0.5 if step < 5000 else 0.2 if step < 10000 else 0.1
            losses['area'], area_frac = area_balance_loss(
                F, faces, tri_area, beta_area=temp_ctrl.beta_area,
                use_straight_through=use_straight_through,
                method="l2",  # L2 gives clearer gradients than reverse KL
                barrier_w=barrier_weight,
                min_frac=min_frac_val,
                max_frac=max_frac_val,
                entropy_weight=entropy_w
            )
            
            # Also compute hard fractions for monitoring
            area_frac_hard = compute_hard_area_fractions(F, faces, tri_area)
        else:
            # Original area loss
            use_entropy = stage.name == "Stage 0: Smoothness + Area + Normal Warmup"
            losses['area'], area_frac = area_fractions_and_kl(
                F, faces, tri_area, beta_area=temp_ctrl.beta_area,
                use_entropy_regularization=use_entropy
            )
            
            # Optionally use 5-patch prior instead of uniform
            if use_5_patch_prior and not use_entropy:
                # Build a 5+1 prior (last channel optional)
                prior = torch.full((n_channels,), 1.0/5.0, device=device, dtype=F.dtype)
                prior[-1] = 1e-3  # Let the last channel be nearly empty
                prior = prior / prior.sum()  # Renormalize
                
                # Replace area loss with KL to non-uniform prior
                losses['area'] = area_kl_to_prior(area_frac, prior)
            
            area_frac_hard = area_frac  # For compatibility
        
        # Pin constraint (soft or hard)
        if stage.use_hard_pins:
            # Hard projection - enforce exact values
            with torch.no_grad():
                F.data[pinned_indices] = pin_targets
            losses['pin'] = torch.tensor(0.0, device=device)
        else:
            # Soft penalty with stronger weight during early stages
            losses['pin'] = pin_loss(F, pinned_indices, pin_targets)
        
        # Optional total variation
        if stage.lambda_tv is not None:
            losses['tv'] = total_variation_loss(F, edge_idx)
        
        # Margin sharpening loss for Stage A
        if stage.name.startswith("Stage A") and edge_weights is not None:
            losses['sharp'] = non_boundary_margin_loss(F, edge_idx, edge_weights, tau=0.2)
        else:
            losses['sharp'] = torch.tensor(0.0, device=device)
        
        # Potts smoothness on probabilities (to reduce speckles)
        if edge_weights is not None:
            losses['potts'] = potts_smoothness_loss(F, edge_idx, edge_weights, 
                                                    temp_ctrl.beta_area, gamma=2.0)
            # Boundary length regularizer (to reduce ragged seams)
            losses['boundary_length'] = boundary_length_regularizer(edge_idx, edge_weights, verts)
        else:
            losses['potts'] = torch.tensor(0.0, device=device)
            losses['boundary_length'] = torch.tensor(0.0, device=device)
        
        # Normal axis alignment losses (for axis-oriented patches)
        # Define axis vectors for each channel: [+X, -X, +Y, -Y, +Z, -Z]
        axis_vectors = torch.tensor([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ], device=device, dtype=F.dtype)[:n_channels]
        
        losses['normal_align'], losses['normal_disp'] = normal_axis_losses(
            verts, faces, tri_area, F, temp_ctrl.beta_area, axis_vectors
        )
        
        # Triple point barrier to reduce Y-junctions and speckles
        losses['triple'] = triple_point_barrier(F, faces, tri_area, beta_area=temp_ctrl.beta_area, margin=0.10)
        
        # Margin separation loss to break symmetry (only in early stages)
        if step < 20000:
            losses['margin_sep'] = margin_separation_loss(F, tau=0.3)
        else:
            losses['margin_sep'] = torch.tensor(0.0, device=device)
        
        # Debug: print area fractions and normal losses
        if step % 1000 == 0 and verbose:
            if use_improved_area:
                st_status = "ON" if use_straight_through else "OFF"
                print(f"[DEBUG] Area fractions (soft): {area_frac.detach().cpu().numpy()}")
                print(f"[DEBUG] Area fractions (hard): {area_frac_hard.detach().cpu().numpy()}")
                print(f"[DEBUG] β_area={temp_ctrl.beta_area}, area_loss={losses['area'].item():.6f}, "
                      f"ST={st_status}, barrier_w={barrier_weight}, "
                      f"normal_align={losses['normal_align'].item():.4f}, normal_disp={losses['normal_disp'].item():.4f}")
            else:
                print(f"[DEBUG] Area fractions: {area_frac.detach().cpu().numpy()}, β_area={temp_ctrl.beta_area}, area_loss={losses['area'].item():.6f}, "
                      f"normal_align={losses['normal_align'].item():.4f}, normal_disp={losses['normal_disp'].item():.4f}")
            
            # Also print pin values to verify channel mapping
            if step == 0:
                print(f"\n[DEBUG] Pin mapping:")
                for i, idx in enumerate(pinned_indices[:6]):
                    label = channel_labels.get(i, f'Ch{i}')
                    values = F.data[idx].detach().cpu().numpy()
                    print(f"  Pin {i} ({label}) at vertex {idx}: {values}")
        
        # Check for stalls and get adaptive lambda_contour
        area_dev_soft = (area_frac - 1.0/n_channels).abs().max().item()
        area_dev_hard = (area_frac_hard - 1.0/n_channels).abs().max().item() if use_improved_area else area_dev_soft
        # Use hard deviation for stall detection when using improved area loss
        area_dev = area_dev_hard if use_improved_area else area_dev_soft
        is_stalled, suggested_multiplier = stall_detector.update(
            total_loss=losses['smooth'].item() + losses['contour'].item() + losses['area'].item(),
            contour_loss=losses['contour'].item(),
            area_deviation=area_dev,
            step=step
        )
        
        # Get adaptive lambda_contour (can grow up to 5.0)
        adaptive_lambda_contour = stall_detector.get_adaptive_lambda_contour(
            base_lambda=stage.lambda_contour,
            current_step=step,
            max_lambda=5.0,
            total_steps=n_steps  # Pass total steps for proper scaling
        )
        
        # Debug: log stage info every 5000 steps
        if step % 5000 == 0 and verbose:
            fraction = step / n_steps
            print(f"[DEBUG] Step {step}, fraction={fraction:.3f}, stage={stage.name}, "
                  f"base_lambda={stage.lambda_contour}, adaptive_lambda={adaptive_lambda_contour:.3f}")
        
        # Apply smooth transitions for all lambdas
        smooth_lambda_smooth = smooth_scheduler.get_smooth_lambda(
            'smooth', stage.lambda_smooth, step, stage.name
        )
        smooth_lambda_contour = smooth_scheduler.get_smooth_lambda(
            'contour', adaptive_lambda_contour, step, stage.name
        )
        smooth_lambda_area = smooth_scheduler.get_smooth_lambda(
            'area', stage.lambda_area, step, stage.name
        )
        smooth_lambda_pin = smooth_scheduler.get_smooth_lambda(
            'pin', stage.lambda_pin, step, stage.name
        )
        
        # Check for non-finite losses before combining
        for k, v in list(losses.items()):
            if not torch.isfinite(v):
                print(f"[DEBUG] Non-finite in {k}: {v.item()}")
                # Optional: dump some quick stats from contour module
                raise RuntimeError(f"Non-finite loss detected in {k}")
        
        # Total loss with smooth lambdas
        total_loss = (
            smooth_lambda_smooth * losses['smooth'] +
            smooth_lambda_contour * losses['contour'] +
            smooth_lambda_area * losses['area'] +
            smooth_lambda_pin * losses['pin']
        )
        
        if 'tv' in losses and stage.lambda_tv is not None:
            total_loss = total_loss + stage.lambda_tv * losses['tv']
        
        # Add margin loss with appropriate weight
        if stage.name.startswith("Stage A"):
            lambda_sharp = 0.08 if "A1" in stage.name or "A2" in stage.name else 0.04
            total_loss = total_loss + lambda_sharp * losses['sharp']
        
        # Add Potts smoothness (stronger early, weaker late)
        lambda_potts = 0.15 if "Stage A" in stage.name else 0.07
        total_loss = total_loss + lambda_potts * losses['potts']
        
        # Add boundary length regularizer (increased weight)
        lambda_boundary = 1e-3  # Stronger to remove zigzags
        total_loss = total_loss + lambda_boundary * losses['boundary_length']
        
        # Add normal losses
        if stage.name == "Stage 0: Smoothness + Area + Normal Warmup":
            # High normal losses during warmup to establish axis alignment early
            lambda_normal_align = 0.5
            lambda_normal_disp = 0.2
            total_loss = total_loss + lambda_normal_align * losses['normal_align']
            total_loss = total_loss + lambda_normal_disp * losses['normal_disp']
        elif step > 30000:  # Continue after initial stages
            lambda_normal_align = 0.2 if step > 60000 else 0.05
            lambda_normal_disp = 0.1 if step > 60000 else 0.02
            total_loss = total_loss + lambda_normal_align * losses['normal_align']
            total_loss = total_loss + lambda_normal_disp * losses['normal_disp']
        
        # Add triple point barrier with schedule
        if step < 20000:
            lambda_triple = 0.05  # Light during early stages
        elif step < 60000:
            lambda_triple = 0.05  # Maintain during middle
        else:
            lambda_triple = 0.03  # Reduce late
        total_loss = total_loss + lambda_triple * losses['triple']
        
        # Add margin separation loss (strong early, then decay)
        if step < 20000:
            lambda_margin = 0.5 if step < 5000 else 0.2
            total_loss = total_loss + lambda_margin * losses['margin_sep']
        
        # Check for NaN before backward
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"\nWARNING: NaN/Inf detected at step {step}")
            print(f"  Total loss: {total_loss.item()}")
            print(f"  Individual losses: {[(k, v.item()) for k, v in losses.items()]}")
            print(f"  Field stats: min={F.min().item():.6f}, max={F.max().item():.6f}")
            
            # Skip this step
            optimizer.zero_grad()
            
            # Try to recover by reducing learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"  Reduced learning rate to {optimizer.param_groups[0]['lr']:.2e}")
            
            # Reset field to last checkpoint if too many NaN
            nan_count = getattr(train_mesh_segmentation, 'nan_count', 0) + 1
            train_mesh_segmentation.nan_count = nan_count
            
            if nan_count > 10:
                print("  Too many NaN occurrences, stopping training")
                break
                
            continue
        
        # Backward pass
        total_loss.backward()
        
        # Defensive gradient handling to keep training moving
        if (F.grad is None) or torch.isnan(F.grad).any() or torch.isinf(F.grad).any():
            with torch.no_grad():
                if F.grad is None:
                    F.grad = torch.zeros_like(F)
                    print(f"\nWARNING: No gradient computed at step {step}")
                else:
                    nan_count = torch.isnan(F.grad).sum().item()
                    inf_count = torch.isinf(F.grad).sum().item()
                    if nan_count > 0 or inf_count > 0:
                        print(f"\nWARNING: NaN/Inf in gradients at step {step} (NaN: {nan_count}, Inf: {inf_count})")
                        print(f"  Grad norm before fix: {F.grad.norm().item()}")
                        
                torch.nan_to_num_(F.grad, nan=0.0, posinf=0.0, neginf=0.0)
                F.grad.clamp_(-1.0, 1.0)  # Clip-by-value as safety
                
                # Halve LR for a few steps after a NaN event
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"  Reduced LR to {optimizer.param_groups[0]['lr']:.2e} after gradient issue")
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_([F], max_norm=5.0)
        
        # Additional gradient check
        if grad_norm > 100:
            print(f"\nWARNING: Large gradient norm {grad_norm:.2f} at step {step}")
        
        # Optimizer step
        optimizer.step()
        
        # Post-step NaN check with recovery attempt
        if not torch.isfinite(F).all():
            print(f"[DEBUG] F blew up at step {step}. Clamping and continuing once...")
            with torch.no_grad():
                torch.nan_to_num_(F, nan=0.0, posinf=10.0, neginf=-10.0)
                F.clamp_(-20.0, 20.0)  # keeps softmax & sigmoids sane
            # Only break if it happens repeatedly
            nan_count = getattr(train_mesh_segmentation, 'post_step_nan_count', 0) + 1
            train_mesh_segmentation.post_step_nan_count = nan_count
            if nan_count > 3:
                print(f"\nERROR: Repeated NaN in field after optimizer step")
                break
        
        # Update history
        history['loss'].append(total_loss.item())
        history['loss_smooth'].append(losses['smooth'].item())
        history['loss_contour'].append(losses['contour'].item()) 
        history['loss_area'].append(losses['area'].item())
        history['loss_pin'].append(losses['pin'].item())
        history['loss_potts'].append(losses['potts'].item())
        history['loss_boundary_length'].append(losses['boundary_length'].item())
        history['loss_normal_align'].append(losses['normal_align'].item())
        history['loss_normal_disp'].append(losses['normal_disp'].item())
        history['loss_triple'].append(losses['triple'].item())
        history['area_fractions'].append(area_frac.detach().cpu().numpy())
        history['beta_contour'].append(temp_ctrl.beta_contour)
        history['beta_area'].append(temp_ctrl.beta_area)
        history['lr'].append(current_lr)
        
        # Progress-based temperature update (every 400 steps)
        if step % 400 == 0 and stage.allow_temp_increase:
            # Store old best before update
            prev_best = temp_ctrl.best_contour_loss_since_update
            
            if use_v2_contour:
                boundary_length, active_frac, median_phi = compute_boundary_stats_v2(
                    F, edge_idx, verts, temp_ctrl.beta_contour
                )
            else:
                boundary_length, active_frac = compute_boundary_stats(
                    F, edge_idx, verts, temp_ctrl.beta_contour
                )
            history['boundary_length'].append(boundary_length)
            history['active_edge_fraction'].append(active_frac)
            
            updated = temp_ctrl.check_and_update(
                area_frac, boundary_length, stats['bbox_diagonal'], step,
                contour_loss=losses['contour'].item()
            )
            
            if updated and verbose:
                print(f"  -> Temperature increased: βc={temp_ctrl.beta_contour:.1f}, βa={temp_ctrl.beta_area:.1f}")
                if torch.isfinite(torch.tensor(prev_best)):
                    imp = (prev_best - losses['contour'].item()) / max(prev_best, 1e-9)
                    print(f"     (improvement: {imp:.1%})")
        
        # Logging
        if step % log_freq == 0 and verbose:
            # Get current boundary stats for logging
            if step > 0:
                if use_v2_contour:
                    _, active_frac, _ = compute_boundary_stats_v2(
                        F, edge_idx, verts, temp_ctrl.beta_contour
                    )
                else:
                    _, active_frac = compute_boundary_stats(
                        F, edge_idx, verts, temp_ctrl.beta_contour
                    )
            else:
                active_frac = 0.0
                
            # Calculate weighted losses to show actual contribution
            weighted_smooth = smooth_lambda_smooth * losses['smooth'].item()
            weighted_contour = smooth_lambda_contour * losses['contour'].item()
            weighted_area = smooth_lambda_area * losses['area'].item()
            
            # Add stall detection info
            if is_stalled and suggested_multiplier and suggested_multiplier > 1.0:
                print(f"  -> CONTOUR STALL DETECTED! λ_contour: {stage.lambda_contour:.3f} → {adaptive_lambda_contour:.3f} (×{suggested_multiplier:.2f})")
                
            # Calculate weighted normal losses
            weighted_normal_align = 0.0
            weighted_normal_disp = 0.0
            if stage.name == "Stage 0: Smoothness + Area + Normal Warmup":
                weighted_normal_align = 0.5 * losses['normal_align'].item()
                weighted_normal_disp = 0.2 * losses['normal_disp'].item()
            elif step > 30000:
                lambda_normal_align = 0.2 if step > 60000 else 0.05
                lambda_normal_disp = 0.1 if step > 60000 else 0.02
                weighted_normal_align = lambda_normal_align * losses['normal_align'].item()
                weighted_normal_disp = lambda_normal_disp * losses['normal_disp'].item()
            
            # Prepare area dev string
            if use_improved_area:
                area_dev_str = f"AreaDev: soft={area_dev_soft:.3f}, hard={area_dev_hard:.3f}"
            else:
                area_dev_str = f"AreaDev: {area_dev:.3f}"
                
            # Add SVD info if available
            svd_str = ""
            if use_svd_contour and 'svd_info' in locals():
                svd_str = f" | ActivePairs: {svd_info['active_pairs']}"
            
            print(f"Step {step:6d}/{n_steps} | Loss: {total_loss.item():.6f} | "
                  f"Smooth: {losses['smooth'].item():.4f} ({weighted_smooth:.4f}) | "
                  f"Contour: {losses['contour'].item():.4f} ({weighted_contour:.4f}) [λ={smooth_lambda_contour:.2f}] | "
                  f"Area: {losses['area'].item():.4f} ({weighted_area:.4f}) | "
                  f"NormAlign: {losses['normal_align'].item():.4f} ({weighted_normal_align:.4f}) | "
                  f"NormDisp: {losses['normal_disp'].item():.4f} ({weighted_normal_disp:.4f}) | "
                  f"{area_dev_str} | "
                  f"ActiveEdge: {active_frac:.1%} | "
                  f"βc: {temp_ctrl.beta_contour:.1f} | "
                  f"LR: {scaled_lr:.2e}{svd_str}")
            
            # Check gradient health
            grad_monitor.log_gradients(F, {})
            health = grad_monitor.check_health()
            if any(health.values()):
                print(f"  -> Gradient issues: {health}")
        
        # Checkpointing
        if output_dir and step % checkpoint_freq == 0 and step > 0:
            # Save comprehensive checkpoint as npz
            checkpoint_data = {
                # Core data
                'step': step,
                'field_values': F.data.cpu().numpy(),
                'vertices': verts.cpu().numpy(),
                'faces': faces.cpu().numpy(),
                'pinned_indices': pinned_indices.cpu().numpy(),
                'channel_labels': channel_labels,
                
                # Current state
                'beta_contour': temp_ctrl.beta_contour,
                'beta_area': temp_ctrl.beta_area,
                'stage_name': stage.name,
                'learning_rate': current_lr,
                
                # Loss values
                'total_loss': total_loss.item(),
                'loss_smooth': losses['smooth'].item(),
                'loss_contour': losses['contour'].item(),
                'loss_area': losses['area'].item(),
                'loss_pin': losses['pin'].item(),
                'loss_potts': losses['potts'].item(),
                'loss_boundary_length': losses['boundary_length'].item(),
                'loss_triple': losses['triple'].item(),
                
                # Metrics
                'area_fractions': area_frac.detach().cpu().numpy(),
                'area_deviation': area_dev,
                
                # Mesh data
                'edge_idx': edge_idx.cpu().numpy(),
                'tri_area': tri_area.cpu().numpy(),
                
                # Stage info
                'lambda_smooth': stage.lambda_smooth,
                'lambda_contour': stage.lambda_contour,
                'lambda_area': stage.lambda_area,
                'lambda_pin': stage.lambda_pin,
            }
            
            # Add boundary stats if available
            if step % 400 == 0:
                if use_v2_contour:
                    boundary_length, active_frac, median_phi = compute_boundary_stats_v2(
                        F, edge_idx, verts, temp_ctrl.beta_contour
                    )
                    checkpoint_data['median_phi'] = median_phi
                else:
                    boundary_length, active_frac = compute_boundary_stats(
                        F, edge_idx, verts, temp_ctrl.beta_contour
                    )
                checkpoint_data['boundary_length'] = boundary_length
                checkpoint_data['active_edge_fraction'] = active_frac
            
            # Save as compressed npz
            np.savez_compressed(
                output_dir / f'checkpoint_{step:06d}.npz',
                **checkpoint_data
            )
            
            # Also save optimizer state separately (for resuming)
            torch.save({
                'optimizer_state': optimizer.state_dict(),
                'temp_ctrl_state': temp_ctrl.__dict__,
                'scheduler_state': scheduler.current_stage_idx
            }, output_dir / f'checkpoint_{step:06d}_optimizer.pt')
        
    
    # Final hard pinning to ensure exact values
    with torch.no_grad():
        F.data[pinned_indices] = pin_targets
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\nTraining completed in {elapsed:.1f} seconds")
    
    return F.data, history


def main():
    parser = argparse.ArgumentParser(description='Improved ReLU Mesh Segmentation Training')
    parser.add_argument('--mesh', type=str, required=True,
                       help='Path to mesh file (VTK/VTU format)')
    parser.add_argument('--output-dir', type=str, default='results_improved',
                       help='Output directory for results')
    parser.add_argument('--n-steps', type=int, default=200000,
                       help='Number of training steps')
    parser.add_argument('--n-channels', type=int, default=6,
                       help='Number of segmentation channels')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='PyTorch device (cuda/cpu)')
    parser.add_argument('--checkpoint-freq', type=int, default=5000,
                       help='Checkpoint frequency')
    parser.add_argument('--log-freq', type=int, default=500,
                       help='Logging frequency')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume from checkpoint npz file')
    parser.add_argument('--use-soft-pairs', action='store_true',
                       help='Use soft all-pairs contour loss for better stability')
    parser.add_argument('--use-5-patch-prior', action='store_true',
                       help='Use non-uniform area prior for 5 patches instead of 6')
    parser.add_argument('--no-v2-contour', action='store_true',
                       help='Disable improved v2 contour loss (use original instead)')
    parser.add_argument('--use-svd-contour', action='store_true',
                       help='Use SVD-based contour alignment for stable plane fitting')
    parser.add_argument('--no-improved-area', action='store_true',
                       help='Disable improved area loss with reverse KL (use original instead)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check CUDA availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    print(f"Using device: {args.device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load mesh
    print(f"Loading mesh from {args.mesh}...")
    vertices_np, faces_np = load_volume_tet_mesh_and_extract_surface(args.mesh)
    print(f"Loaded mesh: {vertices_np.shape[0]} vertices, {faces_np.shape[0]} faces")
    
    # Preprocess mesh
    print("Preprocessing mesh...")
    mesh_data = preprocess_mesh(vertices_np, faces_np, device=args.device)
    
    # Print mesh statistics
    stats = mesh_data['stats']
    print(f"Mesh statistics:")
    print(f"  - Bounding box diagonal: {stats['bbox_diagonal']:.3f}")
    print(f"  - Total area: {stats['total_area']:.3f}")
    print(f"  - Mean edge length: {stats['mean_edge_length']:.3f}")
    
    # Train
    if args.resume_from:
        print(f"\nResuming training from checkpoint...")
    else:
        print(f"\nStarting training for {args.n_steps} steps...")
    
    F_optimized, history = train_mesh_segmentation(
        mesh_data,
        n_channels=args.n_channels,
        n_steps=args.n_steps,
        device=args.device,
        output_dir=output_dir,
        checkpoint_freq=args.checkpoint_freq,
        log_freq=args.log_freq,
        verbose=True,
        resume_from=args.resume_from,
        use_soft_pairs=args.use_soft_pairs,
        use_5_patch_prior=args.use_5_patch_prior,
        use_v2_contour=not args.no_v2_contour,
        use_improved_area=not args.no_improved_area,
        use_svd_contour=args.use_svd_contour
    )
    
    # Save results
    print("\nSaving results...")
    
    # Get final boundary stats
    if not args.no_v2_contour:
        final_boundary_length, final_active_frac, final_median_phi = compute_boundary_stats_v2(
            F_optimized, mesh_data['edge_idx'], mesh_data['vertices'], 
            history['beta_contour'][-1] if history['beta_contour'] else 8.0
        )
    else:
        final_boundary_length, final_active_frac = compute_boundary_stats(
            F_optimized, mesh_data['edge_idx'], mesh_data['vertices'], 
            history['beta_contour'][-1] if history['beta_contour'] else 8.0
        )
    
    # Comprehensive final result
    final_data = {
        # Mesh data
        'vertices': vertices_np,
        'faces': faces_np,
        'edge_idx': mesh_data['edge_idx'].cpu().numpy(),
        'tri_area': mesh_data['tri_area'].cpu().numpy(),
        
        # Optimization result
        'field_values': F_optimized.cpu().numpy(),
        'pinned_indices': mesh_data['pinned_indices'].cpu().numpy(),
        
        # Final metrics
        'final_loss': history['loss'][-1] if history['loss'] else 0.0,
        'final_beta_contour': history['beta_contour'][-1] if history['beta_contour'] else 8.0,
        'final_beta_area': history['beta_area'][-1] if history['beta_area'] else 4.0,
        'final_boundary_length': final_boundary_length,
        'final_active_edge_fraction': final_active_frac,
        'final_area_fractions': history['area_fractions'][-1] if history['area_fractions'] else None,
        
        # Training info
        'total_steps': args.n_steps,
        'mesh_file': args.mesh,
        'n_channels': args.n_channels,
        
        # Mesh statistics
        'bbox_diagonal': mesh_data['stats']['bbox_diagonal'],
        'total_area': mesh_data['stats']['total_area'],
        'num_vertices': mesh_data['stats']['num_vertices'],
        'num_faces': mesh_data['stats']['num_faces'],
    }
    
    # Save comprehensive npz
    np.savez_compressed(
        output_dir / 'final_result.npz',
        **final_data
    )
    
    # Save history as JSON
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json_history = convert_to_json_serializable(history)
        json.dump(json_history, f, indent=2)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"Results saved to {output_dir}")


if __name__ == '__main__':
    main()