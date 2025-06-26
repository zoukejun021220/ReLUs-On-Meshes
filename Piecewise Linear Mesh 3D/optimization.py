import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
from total_loss import compute_combined_loss_optimized



def optimization(
    vertices_np,
    plane_offsets,
    faces_np,
    pinned_indices,
    pinned_axes=None,
    *,
    n_iters: int = 1_000_000,
    refine_lr: float = 2e-3,
    shock_lr: float = 2e-2,          # Not used in shock approach
    beta: float = 1.0,
    target_beta: Optional[float] = 40.0,
    beta_schedule: bool = True,                # Not used in shock approach
    lambda_contour_initial: float = 0.0,
    lambda_contour_final: float = 10.0,
    lambda_smooth: float = 0.2,
    lambda_area_initial: float = 0.2,
    lambda_area_final: float = 100.0,
    pct_start: float = 0.3,                   # Not used in shock approach
    anneal_strategy: str = 'cos',             # Not used in shock approach
    div_factor: float = 25.0,                 # Not used
    enable_early_stopping: bool = False,
    patience: int = 2000,
    print_every: int = 100,
    save_path: str = "final_mesh_and_values.npz",
    shock_steps=1000,
    refine_steps=9000):
    """
    Shock-therapy optimizer that does repeated short "shock" phases
    and longer "refine" phases

    Args:
        vertices_np, faces_np: mesh data
        plane_offsets: Extra parameter for the loss
        pinned_indices: Indices of pinned vertices
        pinned_axes: Extra geometry info (optional)
        n_iters: Total number of steps
        lr: We interpret this as the refine LR
        beta: We interpret this as shock beta
        target_beta: We interpret this as refine beta
        lambda_contour_initial: contour weight in shock
        lambda_contour_final:   contour weight in refine
        lambda_smooth: constant smoothness
        lambda_area_initial: area weight in shock
        lambda_area_final:   area weight in refine
        ...
        enable_early_stopping, patience, print_every, etc.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1) Build adjacency etc.
    # ------------------------------------------------------------------
    from MeshParamCalculation import (compute_face_areas, build_triangle_adjacency,
                                      build_vertex_edges, init_6channels_with_pins)

    tri_adj    = torch.from_numpy(build_triangle_adjacency(faces_np)).long().to(device)
    vert_edges = torch.from_numpy(build_vertex_edges(faces_np)).long().to(device)
    mesh_area  = compute_face_areas(vertices_np, faces_np).sum()

    # ------------------------------------------------------------------
    # 2) Convert mesh to torch
    # ------------------------------------------------------------------
    v = torch.from_numpy(vertices_np).float().to(device)
    f = torch.from_numpy(faces_np).long().to(device)

    # ------------------------------------------------------------------
    # 3) Initialize 6-channel field + pin mask
    # ------------------------------------------------------------------
    f_param = init_6channels_with_pins(len(vertices_np), pinned_indices, device)
    pin_mask = torch.full((6, 6), -1.0, device=device)
    torch.diagonal(pin_mask).fill_(1.0)

    # ------------------------------------------------------------------
    # 4) Define "shock" vs "refine" parameters
    # ------------------------------------------------------------------
   
    shock_beta    = beta        # e.g. 1.0
    refine_beta   = target_beta if target_beta is not None else 20.0
    shock_lc      = lambda_contour_initial  # contour in shock
    refine_lc     = lambda_contour_final    # contour in refine
    shock_la      = lambda_area_initial     # area in shock
    refine_la     = lambda_area_final       # area in refine

    # ------------------------------------------------------------------
    # 5) Build optimizer (AdamW)
    # ------------------------------------------------------------------
    opt = optim.AdamW([f_param], lr=shock_lr, betas=(0.9, 0.99), weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))

    # Logging / early stopping
    best_loss = float('inf')
    best_step = 0
    history = []
    start_time = time.time()

    # ------------------------------------------------------------------
    # Helper function to run a sub-phase (shock or refine)
    # ------------------------------------------------------------------
    def run_phase(phase_name, steps, phase_lr, phase_beta, c_wt, a_wt):
        nonlocal global_step, best_loss, best_step
        # Manually set LR
        for param_group in opt.param_groups:
            param_group['lr'] = phase_lr

        for local in range(steps):
            global_step += 1
            if global_step > n_iters:
                return  # do not exceed total steps

            with torch.cuda.amp.autocast(enabled=(device.type=='cuda')):
                total_loss, comp = compute_combined_loss_optimized(
                    f_param, v, f, tri_adj, vert_edges, mesh_area,
                    beta=phase_beta,
                    lambda_contour=c_wt,
                    lambda_smooth=lambda_smooth,
                    lambda_area=a_wt,
                    plane_offsets=plane_offsets,
                    pinned_axes=pinned_axes
                )

            # backward
            scaler.scale(total_loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(f_param, 5.0)
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
                best_step = global_step

            # Logging
            if (global_step == 1) or (global_step % print_every == 0) or (global_step == n_iters):
                lr_now = opt.param_groups[0]['lr']
                print(
                    f"[{phase_name}] step {global_step}/{n_iters} "
                    f"loss={loss_val:.3e} contour={comp['contour']:.3e} "
                    f"smooth={comp['smoothness']:.3e} area={comp['area_balance']:.3e} "
                    f"beta={phase_beta:.1f} λc={c_wt:.2f} λa={a_wt:.2f} lr={lr_now:.2e}"
                )
                history.append({
                    'step': global_step,
                    'phase': phase_name,
                    'total': loss_val,
                    'contour': comp['contour'],
                    'smoothness': comp['smoothness'],
                    'area_balance': comp['area_balance'],
                    'beta': phase_beta,
                    'lambda_c': c_wt,
                    'lambda_a': a_wt,
                    'lr': lr_now,
                })

            # Early stopping
            if enable_early_stopping and (global_step - best_step > patience):
                print(f"Early stopping at step {global_step}, no improvement for {patience} steps.")
                return

    # ------------------------------------------------------------------
    # 6) Main training loop in "shock therapy" style
    # ------------------------------------------------------------------
    # In each "round" we do shock_steps + refine_steps
    round_size = shock_steps + refine_steps
    num_rounds = n_iters // round_size
    leftover   = n_iters % round_size
    global_step = 0

    for rd in range(num_rounds):
        # Shock phase
        run_phase("shock", shock_steps, shock_lr, shock_beta, shock_lc, shock_la)
        if global_step >= n_iters:
            break
        # Refine phase
        run_phase("refine", refine_steps, refine_lr, refine_beta, refine_lc, refine_la)
        if global_step >= n_iters:
            break

    # Handle leftover steps
    leftover_steps = n_iters - global_step
    if leftover_steps > 0:
        # We can do a partial shock
        shock_left = min(shock_steps, leftover_steps)
        run_phase("shock", shock_left, shock_lr, shock_beta, shock_lc, shock_la)

        leftover2 = n_iters - global_step
        if leftover2 > 0:
            refine_left = min(refine_steps, leftover2)
            run_phase("refine", refine_left, refine_lr, refine_beta, refine_lc, refine_la)

    # ------------------------------------------------------------------
    # 7) Finish and save results
    # ------------------------------------------------------------------
    elapsed = (time.time() - start_time) / 60.0
    print(f"Finished {global_step}/{n_iters} steps in {elapsed:.1f} min. "
          f"Best loss={best_loss:.3e} at step={best_step}.")

    final_field_values = f_param.detach().cpu().numpy()
    final_mesh = vertices_np  # same as input, if not deformed

    # Save
    np.savez_compressed(save_path, vertices=vertices_np,faces=faces_np, f_values=final_field_values)
    print(f"Final mesh and field values saved to {save_path}")

    return final_field_values, final_mesh, history, save_path
