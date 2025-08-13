from typing import Optional
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from datetime import datetime
from MeshParamCalculationImproved import (
    compute_face_areas, build_triangle_adjacency, build_vertex_edges,
    init_6channels_with_pins, normalize_mesh
)


def optimization_sin_improved(
    vertices_np, faces_np, pinned_indices, pinned_axes,
    *,
    n_iters: int = 50_000,
    warmup_iters: int = 5000,
    lr: float = 2e-3,
    beta_initial: float = 1.0,
    beta_warmup: float = 3.0,
    beta_final: float = 15.0,
    lambda_contour_initial: float = 0.0,
    lambda_contour_warmup: float = 0.1,
    lambda_contour_final: float = 2.0,
    lambda_smooth: float = 0.2,
    lambda_area_initial: float = 0.2,
    lambda_area_final: float = 2.0,
    # Sinusoidal LR parameters
    num_phases: int = 3,
    lr_min_factor: float = 0.1,
    lr_max_factor: float = 1.0,
    phase_shift: float = 0.0,
    decay_factor: float = 0.5,
    # Early stopping
    enable_early_stopping: bool = True,
    patience: int = 2000,
    print_every: int = 100,
    save_path: str = "optimized_mesh_and_values.npz",
    use_anchored_loss: bool = True,
    use_soft_pairs_loss: bool = False,
    use_free_planes_loss: bool = False,
    checkpoint_dir: str = "checkpoints",
    checkpoint_interval: int = 500,
    input_filename: Optional[str] = None,
):
    """
    Improved sinusoidal optimizer with proper plane_offsets optimization.
    
    Key improvements:
    - Fixes plane_offsets optimization bug
    - Uses normalized mesh
    - Staged training with warmup
    - Anchored planes loss for stability
    
    Returns:
        f_final: (N, 6) final field values on CPU
        final_mesh: (N, 3) vertices of the final mesh
        history: list of dict with per-iteration logs
        save_path: path where the final mesh and field values were saved
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"run_sin_{timestamp}")
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_path}")
    
    # Normalize mesh
    vertices_norm, center, scale = normalize_mesh(vertices_np)
    print(f"Mesh normalized: center={center}, scale={scale:.4f}")
    
    # Convert data to torch
    v = torch.from_numpy(vertices_norm.copy()).float().to(device)
    f = torch.from_numpy(faces_np.copy()).long().to(device)
    
    # Build adjacency
    tri_adj = torch.from_numpy(build_triangle_adjacency(faces_np)).long().to(device)
    vert_edges = torch.from_numpy(build_vertex_edges(faces_np)).long().to(device)
    mesh_area = compute_face_areas(vertices_norm, faces_np).sum()
    
    # Initialize the 6-channel field and plane offsets
    f_param = init_6channels_with_pins(len(vertices_norm), pinned_indices, device)
    plane_offsets = nn.Parameter(torch.zeros(6, device=device))
    
    # Handle plane initialization based on loss type
    if use_free_planes_loss:
        # Initialize learnable plane normals from pinned axes
        from freePlanesContourCL import init_free_plane_normals
        plane_normals = init_free_plane_normals(6, device, init_scale=0.1, pinned_axes=pinned_axes)
        pinned_axes_torch = torch.from_numpy(pinned_axes).float().to(device)  # Still needed for initialization
        
        # Include all parameters in optimizer
        opt_params = [f_param, plane_offsets, plane_normals]
    else:
        # Use provided fixed axes
        pinned_axes_torch = torch.from_numpy(pinned_axes).float().to(device)
        plane_normals = None
        
        # Include field and offset parameters
        opt_params = [f_param, plane_offsets]
    
    pin_mask = torch.full((6, 6), -1.0, device=device)
    torch.diagonal(pin_mask).fill_(1.0)
    
    # Create optimizer with appropriate parameters
    opt = optim.AdamW(opt_params, lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
    
    # Custom sinusoidal multiphase learning rate scheduler
    class SinusoidalMultiphaseLR(torch.optim.lr_scheduler._LRScheduler):
        def __init__(self, optimizer, total_iters, num_phases, lr_min_factor, lr_max_factor,
                     phase_shift=0.0, decay_factor=0.5, last_epoch=-1):
            self.total_iters = total_iters
            self.num_phases = num_phases
            self.lr_min_factor = lr_min_factor
            self.lr_max_factor = lr_max_factor
            self.phase_shift = phase_shift
            self.decay_factor = decay_factor
            self.base_lrs = None
            super(SinusoidalMultiphaseLR, self).__init__(optimizer, last_epoch)
        
        def get_lr(self):
            if self.last_epoch <= 0:
                return self.base_lrs
            
            # Skip sinusoidal during warmup
            if self.last_epoch < warmup_iters:
                return self.base_lrs
            
            # Adjust for post-warmup iterations
            adjusted_epoch = self.last_epoch - warmup_iters
            adjusted_total = self.total_iters - warmup_iters
            
            # Calculate which phase we're in and the progress within that phase
            iters_per_phase = adjusted_total / self.num_phases
            current_phase = min(int(adjusted_epoch / iters_per_phase), self.num_phases - 1)
            phase_progress = (adjusted_epoch - current_phase * iters_per_phase) / iters_per_phase
            
            # Calculate the amplitude decay for the current phase
            amplitude_decay = self.decay_factor ** current_phase
            
            # Calculate min and max LR for current phase with decay
            lr_min = [base_lr * self.lr_min_factor * amplitude_decay for base_lr in self.base_lrs]
            lr_max = [base_lr * self.lr_max_factor * amplitude_decay for base_lr in self.base_lrs]
            
            # Calculate sine wave value (0 to 1 range)
            sine_val = 0.5 + 0.5 * math.sin(2 * math.pi * phase_progress + self.phase_shift)
            
            # Calculate learning rate
            return [lr_min[i] + sine_val * (lr_max[i] - lr_min[i]) for i in range(len(self.base_lrs))]
    
    # Apply the sinusoidal multiphase LR scheduler
    scheduler = SinusoidalMultiphaseLR(
        opt,
        total_iters=n_iters,
        num_phases=num_phases,
        lr_min_factor=lr_min_factor,
        lr_max_factor=lr_max_factor,
        phase_shift=phase_shift,
        decay_factor=decay_factor
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    history = []
    t0 = time.time()
    
    best_loss = float('inf')
    best_iter = 0
    
    # Import loss functions
    if use_free_planes_loss:
        from freePlanesContourCL import contour_alignment_free_planes
        contour_fn = contour_alignment_free_planes
        print("Using free planes loss (learnable normals)")
    elif use_soft_pairs_loss:
        from softPairsContourCL import contour_alignment_soft_pairs
        contour_fn = contour_alignment_soft_pairs
        print("Using soft pairs contour loss (stable triple points)")
    elif use_anchored_loss:
        from anchoredPlaneCL import contour_alignment_loss_anchored
        contour_fn = contour_alignment_loss_anchored
        print("Using anchored planes loss (stable)")
    else:
        from pairPLaneCL import contour_alignment_loss
        contour_fn = contour_alignment_loss
        print("Using original SVD-based loss")
    
    from smoothnessArea import smoothness_loss_optimized, area_balance_loss_optimized
    
    for it in range(1, n_iters + 1):
        # Determine stage parameters
        if it <= warmup_iters:
            stage = "warmup"
            progress = it / warmup_iters
            beta_now = beta_initial + (beta_warmup - beta_initial) * progress
            lambda_c_now = 0.0  # No contour loss during warmup
            lambda_a_now = lambda_area_initial
        else:
            stage = "main"
            progress = (it - warmup_iters) / (n_iters - warmup_iters)
            beta_now = beta_warmup + (beta_final - beta_warmup) * progress
            lambda_c_now = lambda_contour_warmup + (lambda_contour_final - lambda_contour_warmup) * progress
            lambda_a_now = lambda_area_initial + (lambda_area_final - lambda_area_initial) * progress
        
        # Get current learning rate
        lr_now = scheduler.get_last_lr()[0]
        
        # Compute losses
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            # Contour loss
            if lambda_c_now > 0:
                if use_free_planes_loss:
                    contour_loss = contour_fn(
                        v, f, f_param, plane_normals, plane_offsets, pinned_indices,
                        beta_edge=beta_now, include_triples=(it > n_iters - 5000)
                    )
                elif use_soft_pairs_loss:
                    contour_loss = contour_fn(
                        v, f, f_param, pinned_axes_torch, plane_offsets,
                        beta_edge=beta_now, include_triples=(it > n_iters - 5000)
                    )
                elif use_anchored_loss:
                    contour_loss = contour_fn(
                        v, f, f_param, pinned_axes_torch, plane_offsets,
                        beta_edge=beta_now, include_triples=(it > n_iters - 5000)
                    )
                else:
                    contour_loss = contour_fn(
                        v, f, f_param, pinned_axes_torch,
                        beta=beta_now, include_triples=(it > n_iters - 5000),
                        adajancy=tri_adj, plane_offsets=plane_offsets
                    )
            else:
                contour_loss = torch.tensor(0.0, device=device)
            
            # Other losses
            smooth_loss = smoothness_loss_optimized(f_param, vert_edges)
            area_loss, area_fracs = area_balance_loss_optimized(v, f, f_param, beta_now, mesh_area)
            
            # Total loss
            total = (lambda_c_now * contour_loss +
                    lambda_smooth * smooth_loss +
                    lambda_a_now * area_loss)
        
        # Backward pass
        scaler.scale(total).backward()
        scaler.unscale_(opt)
        if use_free_planes_loss:
            grad_norm = nn.utils.clip_grad_norm_([f_param, plane_offsets, plane_normals], 5.0)
        else:
            grad_norm = nn.utils.clip_grad_norm_([f_param, plane_offsets], 5.0)
        
        mean_grad = f_param.grad.mean() if f_param.grad is not None else 0.0
        
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        
        # Advance scheduler
        scheduler.step()
        
        # Re-pin anchors
        with torch.no_grad():
            for k, idx in enumerate(pinned_indices):
                f_param[idx] = pin_mask[k]
        
        if total.item() < best_loss:
            best_loss = total.item()
            best_iter = it
        
        # Logging
        if (it % print_every == 0) or (it == 1) or (it == n_iters):
            # Calculate current phase for logging
            if it > warmup_iters:
                adjusted_it = it - warmup_iters
                iters_per_phase = (n_iters - warmup_iters) / num_phases
                current_phase = min(int(adjusted_it / iters_per_phase), num_phases - 1) + 1
            else:
                current_phase = 0  # Warmup
            
            print(
                f"[{stage}] iter {it:6d}/{n_iters}  total={total.item():.3e} "
                f"contour={contour_loss.item():.3e}  smooth={smooth_loss.item():.3e}  "
                f"area={area_loss.item():.3e}  β={beta_now:.1f}  λc={lambda_c_now:.2f}  "
                f"λa={lambda_a_now:.2f}  lr={lr_now:.2e}  phase={current_phase}/{num_phases}  "
                f"grad_norm={grad_norm:.3e}  offsets_norm={plane_offsets.norm().item():.3f}"
            )
            
            history.append({
                'iter': it,
                'stage': stage,
                'total': total.item(),
                'contour': contour_loss.item(),
                'smoothness': smooth_loss.item(),
                'area_balance': area_loss.item(),
                'beta': beta_now,
                'lambda_c': lambda_c_now,
                'lambda_a': lambda_a_now,
                'lr': lr_now,
                'phase': current_phase,
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                'grad_mean': mean_grad.item() if isinstance(mean_grad, torch.Tensor) else mean_grad,
                'plane_offsets': plane_offsets.detach().cpu().numpy().copy()
            })
        
        # Save checkpoint
        if it % checkpoint_interval == 0:
            checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_step_{it:06d}")
            
            # Save NPZ with field values and mesh data
            save_dict = {
                'vertices': vertices_np,
                'faces': faces_np,
                'f_values': f_param.detach().cpu().numpy(),
                'plane_offsets': plane_offsets.detach().cpu().numpy(),
                'pinned_indices': pinned_indices,
                'normalization': {'center': center, 'scale': scale},
                'step': it,
                'stage': stage,
                'beta': beta_now,
                'lambda_c': lambda_c_now,
                'lambda_a': lambda_a_now,
                'loss_components': {
                    'total': total.item(),
                    'contour': contour_loss.item(),
                    'smoothness': smooth_loss.item(),
                    'area_balance': area_loss.item()
                },
                'use_free_planes': use_free_planes_loss
            }
            
            if use_free_planes_loss:
                save_dict['plane_normals'] = plane_normals.detach().cpu().numpy()
            else:
                save_dict['pinned_axes'] = pinned_axes
            
            np.savez_compressed(f"{checkpoint_file}.npz", **save_dict)
            
            # Save PyTorch model state
            pt_dict = {
                'step': it,
                'f_param': f_param.state_dict() if hasattr(f_param, 'state_dict') else f_param,
                'plane_offsets': plane_offsets.state_dict() if hasattr(plane_offsets, 'state_dict') else plane_offsets,
                'optimizer_state_dict': opt.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss,
                'best_iter': best_iter,
                'history': history,
                'use_free_planes': use_free_planes_loss
            }
            
            if use_free_planes_loss:
                pt_dict['plane_normals'] = plane_normals.state_dict() if hasattr(plane_normals, 'state_dict') else plane_normals
            
            torch.save(pt_dict, f"{checkpoint_file}.pt")
            
            print(f"  -> Checkpoint saved at step {it}")
        
        # Early stopping
        if enable_early_stopping:
            if it - best_iter > patience:
                print(f"Early stopping at iteration {it} (no improvement for {patience} steps).")
                break
    
    print(f"Finished in {(time.time() - t0) / 60:.1f} min. Best loss={best_loss:.3e} at iter={best_iter}.")
    print(f"Final plane offsets: {plane_offsets.detach().cpu().numpy()}")
    
    # Save results
    final_mesh = vertices_np  # Keep original mesh
    final_field_values = f_param.detach().cpu().numpy()
    
    # Save with optimized_ prefix
    if input_filename:
        base_name = os.path.basename(input_filename)
        name_without_ext = os.path.splitext(base_name)[0]
        save_path = f"optimized_{name_without_ext}.npz"
    
    # Save with additional metadata
    final_save_dict = {
        'mesh': final_mesh,
        'face': faces_np,
        'f_values': final_field_values,
        'plane_offsets': plane_offsets.detach().cpu().numpy(),
        'normalization': {'center': center, 'scale': scale},
        'use_free_planes': use_free_planes_loss
    }
    
    if use_free_planes_loss:
        final_save_dict['plane_normals'] = plane_normals.detach().cpu().numpy()
    else:
        final_save_dict['pinned_axes'] = pinned_axes
    
    np.savez_compressed(save_path, **final_save_dict)
    
    print(f"Final mesh and field values saved to {save_path}")
    
    # Also save final checkpoint
    final_checkpoint = os.path.join(checkpoint_path, f"checkpoint_final_step_{it:06d}")
    
    final_ckpt_dict = {
        'vertices': vertices_np,
        'faces': faces_np,
        'f_values': final_field_values,
        'plane_offsets': plane_offsets.detach().cpu().numpy(),
        'pinned_indices': pinned_indices,
        'normalization': {'center': center, 'scale': scale},
        'step': it,
        'stage': "final",
        'history': history,
        'use_free_planes': use_free_planes_loss
    }
    
    if use_free_planes_loss:
        final_ckpt_dict['plane_normals'] = plane_normals.detach().cpu().numpy()
    else:
        final_ckpt_dict['pinned_axes'] = pinned_axes
    
    np.savez_compressed(f"{final_checkpoint}.npz", **final_ckpt_dict)
    
    pt_final_dict = {
        'step': it,
        'f_param': f_param.state_dict() if hasattr(f_param, 'state_dict') else f_param,
        'plane_offsets': plane_offsets.state_dict() if hasattr(plane_offsets, 'state_dict') else plane_offsets,
        'optimizer_state_dict': opt.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_loss': best_loss,
        'best_iter': best_iter,
        'history': history,
        'use_free_planes': use_free_planes_loss
    }
    
    if use_free_planes_loss:
        pt_final_dict['plane_normals'] = plane_normals.state_dict() if hasattr(plane_normals, 'state_dict') else plane_normals
    
    torch.save(pt_final_dict, f"{final_checkpoint}.pt")
    print(f"Final checkpoint saved to {checkpoint_path}")
    
    return final_field_values, final_mesh, history, save_path