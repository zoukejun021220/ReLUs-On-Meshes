import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from total_loss import compute_combined_loss_optimized
import os
from datetime import datetime


def optimization_improved(
    vertices_np,
    faces_np,
    pinned_indices,
    pinned_axes,
    *,
    n_iters: int = 50_000,
    warmup_iters: int = 5000,
    refine_lr: float = 2e-3,
    shock_lr: float = 2e-2,
    beta_initial: float = 1.0,
    beta_warmup: float = 3.0,
    beta_final: float = 15.0,
    lambda_contour_initial: float = 0.0,
    lambda_contour_warmup: float = 0.1,
    lambda_contour_final: float = 2.0,
    lambda_smooth: float = 0.2,
    lambda_area_initial: float = 0.2,
    lambda_area_final: float = 2.0,
    enable_early_stopping: bool = True,
    patience: int = 2000,
    print_every: int = 100,
    save_path: str = "optimized_mesh_and_values.npz",
    shock_steps: int = 1000,
    refine_steps: int = 4000,
    use_anchored_loss: bool = True,
    use_soft_pairs_loss: bool = False,
    use_free_planes_loss: bool = False,
    use_pairwise_planes_loss: bool = False,
    use_codex_grad_alignment_loss: bool = False,
    use_svd_init_after_warmup: bool = True,
    checkpoint_dir: str = "checkpoints",
    checkpoint_interval: int = 500,
    input_filename: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
):
    """
    Improved optimizer with proper plane_offsets optimization and staged training.
    
    Stages:
    1. Warmup: Only smoothness + area, small beta
    2. Main: Add anchored plane loss, gradually increase beta
    3. Refine: Higher beta, full weights
    
    Args:
        vertices_np, faces_np: mesh data
        pinned_indices: Indices of pinned vertices
        pinned_axes: Fixed plane normals (C, 3)
        n_iters: Total iterations
        warmup_iters: Iterations for warmup phase
        use_anchored_loss: If True, use anchored planes; if False, use original
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create checkpoint directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = os.path.join(checkpoint_dir, f"run_{timestamp}")
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_path}")
    
    # Import mesh utilities
    from MeshParamCalculationImproved import (
        compute_face_areas, build_triangle_adjacency,
        build_vertex_edges, init_6channels_with_pins,
        normalize_mesh
    )
    
    # Normalize mesh
    vertices_norm, center, scale = normalize_mesh(vertices_np)
    print(f"Mesh normalized: center={center}, scale={scale:.4f}")
    
    # Build adjacency
    tri_adj = torch.from_numpy(build_triangle_adjacency(faces_np)).long().to(device)
    vert_edges = torch.from_numpy(build_vertex_edges(faces_np)).long().to(device)
    mesh_area = compute_face_areas(vertices_norm, faces_np).sum()
    
    # Convert to torch (copy arrays to ensure they're writable)
    v = torch.from_numpy(vertices_norm.copy()).float().to(device)
    f = torch.from_numpy(faces_np.copy()).long().to(device)
    
    # Initialize parameters
    f_param = init_6channels_with_pins(len(vertices_norm), pinned_indices, device)
    plane_offsets = nn.Parameter(torch.zeros(6, device=device))
    
    # Handle plane initialization based on loss type
    if use_pairwise_planes_loss:
        # Initialize learnable plane normals and offsets for channel pairs
        from freePlanesContourCLPairwise import init_free_plane_normals_pairwise, init_free_plane_offsets_pairwise
        from channelPairsConfig import get_num_valid_pairs
        num_pairs = get_num_valid_pairs(6)  # 12 valid pairs (excluding opposites)
        plane_normals = init_free_plane_normals_pairwise(6, device, init_scale=0.1)
        plane_offsets = init_free_plane_offsets_pairwise(6, device, init_scale=0.1)
        pinned_axes_torch = torch.from_numpy(pinned_axes).float().to(device)  # Still needed for initialization
        
        # Include all parameters in optimizer
        opt_params = [f_param, plane_offsets, plane_normals]
    elif use_free_planes_loss:
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
    
    # Pin mask
    pin_mask = torch.full((6, 6), -1.0, device=device)
    torch.diagonal(pin_mask).fill_(1.0)
    
    # Resume from checkpoint if provided
    start_step = 1
    if resume_checkpoint:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        
        # Load NPZ checkpoint
        ckpt_data = np.load(f"{resume_checkpoint}.npz")
        
        # Load field values
        f_values_loaded = ckpt_data['f_values']
        f_param.data = torch.from_numpy(f_values_loaded).float().to(device)
        
        # Load iteration number
        if 'iteration' in ckpt_data:
            start_step = int(ckpt_data['iteration']) + 1
            print(f"Resuming from iteration {start_step}")
        
        # Load PyTorch checkpoint
        pt_ckpt = torch.load(f"{resume_checkpoint}.pt", map_location=device)
        
        # Load plane offsets
        if 'plane_offsets' in pt_ckpt:
            plane_offsets.data = pt_ckpt['plane_offsets']
        
        # Load plane normals for free planes
        if use_free_planes_loss and 'plane_normals' in pt_ckpt:
            plane_normals.data = pt_ckpt['plane_normals']
        elif use_pairwise_planes_loss and 'plane_normals' in pt_ckpt:
            plane_normals.data = pt_ckpt['plane_normals']
        
        print("Checkpoint loaded successfully")
    
    # Create optimizer with appropriate parameters
    opt = optim.AdamW(opt_params, lr=refine_lr, betas=(0.9, 0.99), weight_decay=1e-4)
    
    # Load optimizer state if resuming
    if resume_checkpoint and 'optimizer' in pt_ckpt:
        opt.load_state_dict(pt_ckpt['optimizer'])
        print("Optimizer state restored")
    
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))
    
    # Logging
    best_loss = float('inf')
    best_step = 0
    history = []
    start_time = time.time()
    
    # Import the appropriate loss function
    if use_pairwise_planes_loss:
        from freePlanesContourCLPairwise import contour_alignment_free_planes_pairwise
        contour_fn = contour_alignment_free_planes_pairwise
        print("Using channel-pairwise planes loss (one plane per channel pair)")
    elif use_free_planes_loss:
        from freePlanesContourCL import contour_alignment_free_planes
        contour_fn = contour_alignment_free_planes
        print("Using free planes loss (learnable normals)")
    elif use_codex_grad_alignment_loss:
        # Import Codex loss from sibling _codex_ directory
        import os, sys
        this_dir = os.path.dirname(__file__)
        codex_dir = os.path.join(os.path.dirname(this_dir), "_codex_Piecewise Linear Mesh 3D improved")
        if codex_dir not in sys.path:
            sys.path.insert(0, codex_dir)
        from codex_grad_alignment import contour_alignment_codex
        contour_fn = contour_alignment_codex
        print("Using Codex intrinsic 3D gradient-alignment loss")
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
    
    def compute_loss(beta, lambda_c, lambda_a, include_triples=False):
        """Helper to compute loss with current parameters."""
        if use_pairwise_planes_loss:
            # Use pairwise planes loss with learnable normals and offsets
            # Get separate contour and pinning losses
            contour_loss, pinning_loss = contour_fn(
                v, f, f_param, plane_normals, plane_offsets, pinned_indices,
                beta_edge=beta, include_triples=include_triples
            )
            # Add pinning loss directly without lambda_c scaling
            pin_weight = 10.0  # Fixed weight for pinning
        elif use_free_planes_loss:
            # Use free planes loss with learnable normals
            # Get separate contour and pinning losses
            contour_loss, pinning_loss = contour_fn(
                v, f, f_param, plane_normals, plane_offsets, pinned_indices,
                beta_edge=beta, include_triples=include_triples
            )
            # Add pinning loss directly without lambda_c scaling
            pin_weight = 10.0  # Fixed weight for pinning
        elif use_codex_grad_alignment_loss:
            # Codex intrinsic 3D gradient-alignment loss (no planes/pins)
            contour_loss = contour_fn(
                v, f, f_param, beta_edge=beta, include_triples=include_triples
            )
        elif use_soft_pairs_loss:
            # Use soft pairs loss
            contour_loss = contour_fn(
                v, f, f_param, pinned_axes_torch, plane_offsets,
                beta_edge=beta, include_triples=include_triples
            )
        elif use_anchored_loss:
            # Use anchored loss
            contour_loss = contour_fn(
                v, f, f_param, pinned_axes_torch, plane_offsets,
                beta_edge=beta, include_triples=include_triples
            )
        else:
            # Use original loss
            contour_loss = contour_fn(
                v, f, f_param, pinned_axes_torch,
                beta=beta, include_triples=include_triples,
                adajancy=tri_adj, plane_offsets=plane_offsets
            )
        
        # Compute other losses
        from smoothnessArea import smoothness_loss_optimized, area_balance_loss_optimized
        smooth_loss = smoothness_loss_optimized(f_param, vert_edges)
        area_loss, area_fracs = area_balance_loss_optimized(v, f, f_param, beta, mesh_area)
        
        if use_pairwise_planes_loss or use_free_planes_loss:
            # For pairwise/free planes, add pinning loss separately (not scaled by lambda_c)
            total_loss = (lambda_c * contour_loss +
                         lambda_smooth * smooth_loss +
                         lambda_a * area_loss +
                         pin_weight * pinning_loss)
        else:
            total_loss = (lambda_c * contour_loss +
                         lambda_smooth * smooth_loss +
                         lambda_a * area_loss)
        
        components = {
            'contour': contour_loss.item() if lambda_c > 0 else 0.0,
            'smoothness': smooth_loss.item(),
            'area_balance': area_loss.item(),
            'total': total_loss.item()
        }
        
        if use_pairwise_planes_loss or use_free_planes_loss:
            components['pinning'] = pinning_loss.item()
        
        return total_loss, components
    
    # Training loop with stages
    # Track if we need to reinit planes after warmup
    warmup_complete = False
    
    for step in range(start_step, n_iters + 1):
        # Determine current stage and parameters
        if step <= warmup_iters:
            # Warmup: no contour loss, small beta
            stage = "warmup"
            beta = beta_initial + (beta_warmup - beta_initial) * (step / warmup_iters)
            lambda_c = 0.0
            lambda_a = lambda_area_initial
            lr = shock_lr
        elif step <= n_iters - 10000:  # Leave last 10k for refinement
            # Check if we just finished warmup and need to reinit planes
            if not warmup_complete and use_pairwise_planes_loss and use_svd_init_after_warmup:
                warmup_complete = True
                print(f"\n=== Reinitializing planes with SVD after warmup ===")
                # Reinitialize planes based on current channel values
                from svdPlaneInit import reinit_planes_with_svd
                with torch.no_grad():
                    reinit_planes_with_svd(v, f_param, plane_normals, plane_offsets, momentum=0.3)
                print(f"Planes reinitialized with SVD-based positions\n")
            
            # Main training
            stage = "main"
            progress = (step - warmup_iters) / (n_iters - warmup_iters - 10000)
            beta = beta_warmup + (beta_final - beta_warmup) * progress
            lambda_c = lambda_contour_warmup + (lambda_contour_final - lambda_contour_warmup) * progress
            lambda_a = lambda_area_initial + (lambda_area_final - lambda_area_initial) * progress
            lr = refine_lr
        else:
            # Final refinement
            stage = "refine"
            beta = beta_final
            lambda_c = lambda_contour_final
            lambda_a = lambda_area_final
            lr = refine_lr * 0.1  # Lower LR for refinement
        
        # Update learning rate
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            total_loss, comp = compute_loss(beta, lambda_c, lambda_a, 
                                          include_triples=(stage == "refine"))
        
        # Backward pass
        scaler.scale(total_loss).backward()
        scaler.unscale_(opt)
        if use_free_planes_loss:
            nn.utils.clip_grad_norm_([f_param, plane_offsets, plane_normals], 5.0)
        else:
            nn.utils.clip_grad_norm_([f_param, plane_offsets], 5.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        
        # Re-pin anchors
        with torch.no_grad():
            for ch, idx in enumerate(pinned_indices):
                f_param[idx] = pin_mask[ch]
        
        # Track best
        loss_val = total_loss.item()
        if loss_val < best_loss:
            best_loss = loss_val
            best_step = step
        
        # Logging
        if (step == 1) or (step % print_every == 0) or (step == n_iters):
            log_msg = (
                f"[{stage}] step {step}/{n_iters} "
                f"loss={loss_val:.3e} contour={comp['contour']:.3e} "
                f"smooth={comp['smoothness']:.3e} area={comp['area_balance']:.3e} "
            )
            if 'pinning' in comp:
                log_msg += f"pinning={comp['pinning']:.3e} "
            log_msg += (
                f"β={beta:.1f} λc={lambda_c:.2f} λa={lambda_a:.2f} lr={lr:.2e} "
                f"offsets_norm={plane_offsets.norm().item():.3f}"
            )
            print(log_msg)
            history.append({
                'step': step,
                'stage': stage,
                'total': loss_val,
                'contour': comp['contour'],
                'smoothness': comp['smoothness'],
                'area_balance': comp['area_balance'],
                'beta': beta,
                'lambda_c': lambda_c,
                'lambda_a': lambda_a,
                'lr': lr,
                'plane_offsets': plane_offsets.detach().cpu().numpy().copy()
            })
        
        # Save checkpoint
        if step % checkpoint_interval == 0:
            checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_step_{step:06d}")
            
            # Save NPZ with field values and mesh data
            save_dict = {
                'vertices': vertices_np,
                'faces': faces_np,
                'f_values': f_param.detach().cpu().numpy(),
                'plane_offsets': plane_offsets.detach().cpu().numpy(),
                'pinned_indices': pinned_indices,
                'normalization': {'center': center, 'scale': scale},
                'step': step,
                'stage': stage,
                'beta': beta,
                'lambda_c': lambda_c,
                'lambda_a': lambda_a,
                'loss_components': comp,
                'use_free_planes': use_free_planes_loss
            }
            
            if use_free_planes_loss:
                save_dict['plane_normals'] = plane_normals.detach().cpu().numpy()
            else:
                save_dict['pinned_axes'] = pinned_axes
            
            np.savez_compressed(f"{checkpoint_file}.npz", **save_dict)
            
            # Save PyTorch model state
            pt_dict = {
                'step': step,
                'f_param': f_param.state_dict() if hasattr(f_param, 'state_dict') else f_param,
                'plane_offsets': plane_offsets.state_dict() if hasattr(plane_offsets, 'state_dict') else plane_offsets,
                'optimizer_state_dict': opt.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_loss': best_loss,
                'best_step': best_step,
                'history': history,
                'use_free_planes': use_free_planes_loss
            }
            
            if use_free_planes_loss:
                pt_dict['plane_normals'] = plane_normals.state_dict() if hasattr(plane_normals, 'state_dict') else plane_normals
            
            torch.save(pt_dict, f"{checkpoint_file}.pt")
            
            print(f"  -> Checkpoint saved at step {step}")
        
        # Early stopping
        if enable_early_stopping and (step - best_step > patience):
            print(f"Early stopping at step {step}, no improvement for {patience} steps.")
            break
    
    # Finish
    elapsed = (time.time() - start_time) / 60.0
    print(f"Finished {step}/{n_iters} steps in {elapsed:.1f} min. "
          f"Best loss={best_loss:.3e} at step={best_step}.")
    print(f"Final plane offsets: {plane_offsets.detach().cpu().numpy()}")
    
    # Convert back to original space (denormalize)
    final_field_values = f_param.detach().cpu().numpy()
    final_mesh = vertices_np  # Keep original mesh
    
    # Save final results with optimized_ prefix
    if input_filename:
        base_name = os.path.basename(input_filename)
        name_without_ext = os.path.splitext(base_name)[0]
        save_path = f"optimized_{name_without_ext}.npz"
    
    final_save_dict = {
        'vertices': vertices_np,
        'faces': faces_np,
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
    final_checkpoint = os.path.join(checkpoint_path, f"checkpoint_final_step_{step:06d}")
    
    final_ckpt_dict = {
        'vertices': vertices_np,
        'faces': faces_np,
        'f_values': final_field_values,
        'plane_offsets': plane_offsets.detach().cpu().numpy(),
        'pinned_indices': pinned_indices,
        'normalization': {'center': center, 'scale': scale},
        'step': step,
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
        'step': step,
        'f_param': f_param.state_dict() if hasattr(f_param, 'state_dict') else f_param,
        'plane_offsets': plane_offsets.state_dict() if hasattr(plane_offsets, 'state_dict') else plane_offsets,
        'optimizer_state_dict': opt.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_loss': best_loss,
        'best_step': best_step,
        'history': history,
        'use_free_planes': use_free_planes_loss
    }
    
    if use_free_planes_loss:
        pt_final_dict['plane_normals'] = plane_normals.state_dict() if hasattr(plane_normals, 'state_dict') else plane_normals
    
    torch.save(pt_final_dict, f"{final_checkpoint}.pt")
    print(f"Final checkpoint saved to {checkpoint_path}")
    
    return final_field_values, final_mesh, history, save_path
