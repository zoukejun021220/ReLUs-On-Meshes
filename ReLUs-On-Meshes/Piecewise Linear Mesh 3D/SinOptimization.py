from typing import Optional
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional
import numpy as np
from total_loss import compute_combined_loss_optimized
from MeshParamCalculation import compute_face_areas, build_triangle_adjacency, build_vertex_edges
from MeshParamCalculation import init_6channels_with_pins

def optimization(
    vertices_np, faces_np, pinned_indices,
    *,
    n_iters: int = 80_000,           # fewer iters than before
    lr: float = 2e-3,
    beta: float = 1.0,
    target_beta: Optional[float] = 40.0,
    beta_schedule: bool = True,
    lambda_contour_initial: float = 0.0,
    lambda_contour_final: float = 2.0,
    lambda_smooth: float = 0.2,      # slightly higher default
    lambda_area_initial: float = 0.2,
    lambda_area_final: float = 2.0,
    # Sinusoidal LR parameters:
    num_phases: int = 3,             # Number of sine wave cycles
    lr_min_factor: float = 0.1,      # Minimum LR as a factor of initial LR
    lr_max_factor: float = 1.0,      # Maximum LR as a factor of initial LR
    phase_shift: float = 0.0,        # Shift the sine wave (in radians)
    decay_factor: float = 0.5,       # How much to decrease amplitude per phase
    # Early stopping:
    enable_early_stopping: bool = True,
    patience: int = 2000,
    print_every: int = 100,
    save_path: str = "final_mesh_and_values.npz",
    pinned_axes=None  # Path where we will save the mesh and field values
):
    """
    A faster-converging optimizer using:
      - Sinusoidal, multiphased learning rate scheduling
      - Slightly higher smoothness by default
      - Shorter ramp for contour/area

    Returns
    -------
    f_final : (N, 6) final field values on CPU
    final_mesh: (N, 3) vertices of the final mesh
    history: list of dict with per-iteration logs
    save_path: path where the final mesh and field values were saved
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert data to torch
    v  = torch.from_numpy(vertices_np).float().to(device)
    f  = torch.from_numpy(faces_np).long().to(device)

    from collections import defaultdict
    from time import time

    # Build adjacency 
    tri_adj    = torch.from_numpy(build_triangle_adjacency(faces_np)).long().to(device)
    vert_edges = torch.from_numpy(build_vertex_edges(faces_np)).long().to(device)
    mesh_area  = compute_face_areas(vertices_np, faces_np).sum()

    # Initialize the 6-channel field
    f_param = init_6channels_with_pins(len(vertices_np), pinned_indices, device)
    pin_mask = torch.full((6,6), -1.0, device=device)
    torch.diagonal(pin_mask).fill_(1.0)

    # Set up the optimizer (AdamW usually works well)
    opt = optim.AdamW([f_param], lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)

    # Custom sinusoidal, multiphased learning rate scheduler
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
            
            # Calculate which phase we're in and the progress within that phase
            iters_per_phase = self.total_iters / self.num_phases
            current_phase = min(int(self.last_epoch / iters_per_phase), self.num_phases - 1)
            phase_progress = (self.last_epoch - current_phase * iters_per_phase) / iters_per_phase
            
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

    if beta_schedule and target_beta is not None:
        beta_vals = np.linspace(beta, target_beta, n_iters+1)
    else:
        beta_vals = np.full(n_iters+1, beta)

    ramp_iters = max(int(0.2 * n_iters), 1)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=='cuda'))
    history = []
    t0 = time()

    best_loss = float('inf')
    best_iter = 0

    for it in range(1, n_iters+1):
        frac = min(it / ramp_iters, 1.0)

        beta_now = float(beta_vals[it])
        lambda_c_now = lambda_contour_initial + (lambda_contour_final - lambda_contour_initial)*frac
        lambda_a_now = lambda_area_initial + (lambda_area_final - lambda_area_initial)*frac

        # Get current learning rate
        lr_now = scheduler.get_last_lr()[0]

        with torch.cuda.amp.autocast(enabled=(device.type=='cuda')): 
            total, comp = compute_combined_loss_optimized(
                f_param, v, f, tri_adj, vert_edges, mesh_area,
                beta=beta_now,
                lambda_contour = lambda_c_now,
                lambda_smooth  = lambda_smooth,
                lambda_area    = lambda_a_now,
                pinned_axes = pinned_axes
            )

        scaler.scale(total).backward()
        scaler.unscale_(opt)
        grad_norm = nn.utils.clip_grad_norm_(f_param, 5.0)

        mean_grad = f_param.grad.mean()

        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        # Advance scheduler
        scheduler.step()

        with torch.no_grad():
            for k, idx in enumerate(pinned_indices):
                f_param[idx] = pin_mask[k]

        if total.item() < best_loss:
            best_loss = total.item()
            best_iter = it

        if (it % print_every == 0) or (it == 1) or (it == n_iters):
            # Calculate current phase for logging
            iters_per_phase = n_iters / num_phases
            current_phase = min(int(it / iters_per_phase), num_phases - 1) + 1
            
            print(
                f"iter {it:6d}/{n_iters}  total={total.item():.3e} "
                f"contour={comp['contour']:.3e}  smooth={comp['smoothness']:.3e}  "
                f"area={comp['area_balance']:.3e}  β={beta_now:.1f}  λc={lambda_c_now:.2f}  "
                f"λa={lambda_a_now:.2f}  lr={lr_now:.2e}  phase={current_phase}/{num_phases}  "
                f"grad_mean={mean_grad:.3e}  grad_norm={grad_norm:.3e}"
            )
            history.append({
                'iter': it,
                'total': total.item(),
                'contour': comp['contour'],
                'smoothness': comp['smoothness'],
                'area_balance': comp['area_balance'],
                'beta': beta_now,
                'lambda_c': lambda_c_now,
                'lambda_a': lambda_a_now,
                'lr': lr_now,
                'phase': current_phase,
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                'grad_mean': mean_grad.item() if isinstance(mean_grad, torch.Tensor) else mean_grad
            })

        if enable_early_stopping:
            if it - best_iter > patience:
                print(f"Early stopping at iteration {it} (no improvement for {patience} steps).")
                break

    print(f"Finished in {(time()-t0)/60:.1f} min. Best loss={best_loss:.3e} at iter={best_iter}.")

    # After optimization, save the final mesh and its scalar field values
    final_mesh = vertices_np  # The final mesh
    final_field_values = f_param.detach().cpu().numpy()  # The final scalar field values

    # Save the final mesh and scalar field values to a .npz file
    np.savez_compressed(save_path, mesh=final_mesh, face=faces_np, f_values=final_field_values)

    print(f"Final mesh and field values saved to {save_path}")

    return final_field_values, final_mesh, history, save_path