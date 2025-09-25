"""Training loop for intrinsic Voronoi fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from .gating import EdgeGateConfig, compute_edge_gates
from .geometry import (
    bisector_mask_from_phi,
    edge_mean_pair_norms,
    face_gradients,
    pairwise_gradients,
)
from .initialization import initialize_fields
from .inference import VoronoiInference
from .losses import (
    LossTerms,
    LossWeights,
    eikonal_loss,
    gradient_alignment_loss,
    hamilton_jacobi_loss,
    lipschitz_hinge_loss,
    tv_along_isolines,
)
from .mesh import MeshData, PrecomputedGeometry, precompute_geometry
from .masks import eikonal_mask, seed_face_mask
from .utils import pair_indices


@dataclass
class TrainingConfig:
    warmup_steps: int = 5000
    main_steps: int = 35000
    refine_steps: int = 10000
    stage_a_steps: Optional[int] = None
    stage_b_steps: Optional[int] = None
    max_steps: int = 0
    lr: float = 2e-3
    lr_warmup: Optional[float] = 5e-3
    lr_main: Optional[float] = 1e-4
    lr_refine: Optional[float] = 1e-4
    weight_decay: float = 0
    grad_clip: float = 1.0
    beta_start: float = 1.0
    beta_mid: float = 6.0
    beta_end: float = 10.0
    beta_refine: float = 70.0
    charbonnier_delta: float = 1e-6
    epsilon_g: float = 1e-12
    seed_margin_factor: float = 1.0
    dominance: float = 0.5
    lip_weight: float = 2.0
    seed_weight: float = 1.0
    eik_weight: float = 4.0
    hj_weight: float = 1.0
    interface_loss: str = "hj"
    grad_align_beta: float = 6.0
    grad_align_beta_start: Optional[float] = None
    grad_align_beta_end: Optional[float] = None
    grad_align_include_triples: bool = False
    eik_upper_weight: float = 1.0
    eik_lower_weight: float = 0.25
    stage_a_eik_weight: Optional[float] = 0.0
    stage_a_lip_weight: Optional[float] = 0.0
    stage_b_eik_weight: Optional[float] = 4.0
    stage_b_lip_weight: Optional[float] = 1.0
    stage_c_eik_weight: Optional[float] = 6.0
    stage_c_lip_weight: Optional[float] = 0.75
    stage_c_interface_weight: Optional[float] = None
    stage_a_tv_weight: float = 0.0
    stage_b_tv_weight: float = 0.0
    stage_c_tv_weight: float = 0.05
    tau_bis: float = 0.3
    log_interval: int = 100
    checkpoint_interval: Optional[int] = 2500
    checkpoint_dir: Optional[Path] = None
    init_method: str = "heat"
    gate_config: EdgeGateConfig = field(default_factory=EdgeGateConfig)

    def __post_init__(self) -> None:
        if self.stage_a_steps is None:
            self.stage_a_steps = max(0, self.warmup_steps)
        else:
            self.warmup_steps = max(0, self.stage_a_steps)

        if self.stage_b_steps is None:
            self.stage_b_steps = self.stage_a_steps + max(0, self.main_steps)
        else:
            self.stage_b_steps = max(self.stage_b_steps, self.stage_a_steps)
            self.main_steps = max(0, self.stage_b_steps - self.stage_a_steps)

        if self.max_steps <= 0:
            self.max_steps = self.stage_b_steps + max(0, self.refine_steps)
        else:
            self.max_steps = max(self.max_steps, self.stage_b_steps)
        self.refine_steps = max(0, self.max_steps - self.stage_b_steps)

        if self.lr_warmup is None:
            self.lr_warmup = max(self.lr * 5.0, self.lr)
        if self.lr_main is None:
            self.lr_main = self.lr
        if self.lr_refine is None:
            self.lr_refine = max(self.lr * 0.1, self.lr / 10.0)

        self.beta_mid = max(self.beta_mid, self.beta_start)
        self.beta_end = max(self.beta_end, self.beta_mid)
        self.beta_refine = max(self.beta_refine, self.beta_end)

        if self.grad_align_beta_start is None:
            self.grad_align_beta_start = self.beta_start
        if self.grad_align_beta_end is None:
            self.grad_align_beta_end = self.beta_refine


class VoronoiTrainer:
    def __init__(
        self,
        mesh: MeshData,
        seeds: Sequence[int],
        precomp: Optional[PrecomputedGeometry] = None,
        *,
        device: torch.device,
        config: Optional[TrainingConfig] = None,
    ) -> None:
        self.mesh = mesh
        self.seeds = list(seeds)
        self.precomp = precomp
        self.device = device
        self.config = config or TrainingConfig()
        self.edge_pairs = pair_indices(len(self.seeds), device)
        self.field_param: Optional[torch.nn.Parameter] = None
        self.optimizer: Optional[AdamW] = None
        self.history: List[LossTerms] = []
        self.seed_rings_torch: List[torch.Tensor] = []
        self.seed_mask: Optional[torch.Tensor] = None
        self.current_beta: float = self.config.beta_start
        self.current_lr: float = self.config.lr_warmup
        self.checkpoint_dir: Optional[Path] = None
        self._seed_rows: Optional[torch.Tensor] = None
        self._seed_cols: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if self.precomp is None:
            self.precomp = precompute_geometry(self.mesh, self.seeds, device=self.device)

        self.seed_rings_torch = [
            torch.from_numpy(ring).to(self.device, dtype=torch.long)
            for ring in self.precomp.seed_rings
        ]
        self.seed_mask = seed_face_mask(
            self.precomp.faces,
            self.seed_rings_torch,
            num_vertices=self.precomp.vertices.shape[0],
        ).to(self.device)

        initial_field = initialize_fields(
            self.mesh,
            self.seeds,
            self.precomp,
            device=self.device,
            method=self.config.init_method,
        )
        self.field_param = nn.Parameter(initial_field)
        self.optimizer = AdamW(
            [self.field_param],
            lr=self.config.lr_warmup,
            weight_decay=self.config.weight_decay,
        )
        self.current_lr = self.config.lr_warmup

        if self.seeds:
            self._seed_rows = torch.tensor(self.seeds, dtype=torch.long, device=self.device)
            self._seed_cols = torch.arange(len(self.seeds), dtype=torch.long, device=self.device)
        else:
            self._seed_rows = None
            self._seed_cols = None

        if self.config.checkpoint_dir is not None:
            self.checkpoint_dir = Path(self.config.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

    def _beta(self, step: int) -> float:
        return float(self.config.beta_refine)

    def _stage_loss_weights(self, step: int) -> LossWeights:
        return LossWeights(
            seed=0.0,
            eikonal=self.config.eik_weight,
            lipschitz=self.config.lip_weight,
            interface=self.config.hj_weight,
        )

    def _stage_tv_weight(self, step: int) -> float:
        return self.config.stage_c_tv_weight

    def _grad_align_beta(self, step: int) -> float:
        return float(self.config.grad_align_beta)

    def train(self) -> List[LossTerms]:
        if self.field_param is None or self.optimizer is None:
            self.setup()

        assert self.precomp is not None
        assert self.seed_mask is not None

        history: List[LossTerms] = []

        seed_rows = self._seed_rows
        seed_cols = self._seed_cols
        seed_fixed = None
        if seed_rows is not None and seed_cols is not None:
            seed_fixed = self.field_param.data[seed_rows, seed_cols].clone()

        face_areas = self.precomp.face_areas.to(self.device, dtype=self.field_param.dtype)
        face_normals = self.precomp.face_normals.to(self.device, dtype=self.field_param.dtype)
        edge_lengths = self.precomp.edge_lengths.to(self.device, dtype=self.field_param.dtype)
        interface_mode = self.config.interface_loss.lower()
        grad_aliases = {"gradient_alignment", "grad_alignment", "grad_align"}
        if interface_mode not in {"hj"} | grad_aliases:
            raise ValueError(f"Unsupported interface_loss '{self.config.interface_loss}'")
        use_grad_align = interface_mode in grad_aliases

        for step in range(1, self.config.max_steps + 1):
            stage_tag = self._stage_tag(step)
            self._update_learning_rate(stage_tag)
            beta = self._grad_align_beta(step) if use_grad_align else self._beta(step)
            self.current_beta = beta

            grads = face_gradients(self.field_param, self.precomp)
            pair_grads = pairwise_gradients(grads, self.edge_pairs)
            g_left, g_right = edge_mean_pair_norms(pair_grads, self.precomp.edge_faces)

            gates = compute_edge_gates(
                self.field_param,
                self.precomp.edge_indices,
                edge_lengths,
                g_left,
                g_right,
                config=self.config.gate_config,
            )

            bis_mask = bisector_mask_from_phi(gates.phi, self.precomp.face_edges, self.config.tau_bis)
            eik_mask = eikonal_mask(self.seed_mask.float(), bis_mask.to(self.device, dtype=self.field_param.dtype))

            if seed_rows is not None:
                seed_loss = torch.zeros((), device=self.device, dtype=self.field_param.dtype)
            else:
                seed_loss = torch.zeros((), device=self.device, dtype=self.field_param.dtype)
            lip_loss = lipschitz_hinge_loss(
                self.field_param,
                self.precomp.edge_indices,
                edge_lengths,
            )
            eik_loss = eikonal_loss(
                grads,
                eik_mask,
                face_areas,
                self.config.charbonnier_delta,
                upper_weight=self.config.eik_upper_weight,
                lower_weight=self.config.eik_lower_weight,
            )
            if not use_grad_align:
                interface_loss_val = hamilton_jacobi_loss(
                    grads,
                    pair_grads,
                    face_normals,
                    self.precomp.edge_faces,
                    gates,
                    self.edge_pairs,
                    self.precomp.edge_unit_vectors,
                    self.precomp.edge_dihedral_cos,
                    self.precomp.edge_dihedral_sin,
                    delta=self.config.charbonnier_delta,
                    epsilon_g=self.config.epsilon_g,
                )
            else:
                interface_loss_val = gradient_alignment_loss(
                    self.field_param,
                    self.precomp.vertices,
                    self.precomp.faces,
                    self.precomp.edge_indices,
                    self.precomp.edge_faces,
                    beta_edge=beta,
                    include_triples=self.config.grad_align_include_triples,
                )
            tv_weight = self._stage_tv_weight(step)
            weighted_tv = tv_along_isolines(
                grads,
                face_areas,
                weight=tv_weight,
            )

            weights = self._stage_loss_weights(step)
            weighted_seed = weights.seed * seed_loss
            weighted_lip = weights.lipschitz * lip_loss
            weighted_eik = weights.eikonal * eik_loss
            weighted_interface = weights.interface * interface_loss_val
            total_loss = (
                weighted_seed
                + weighted_lip
                + weighted_eik
                + weighted_interface
                + weighted_tv
            )

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if (
                self.field_param.grad is not None
                and seed_rows is not None
                and seed_cols is not None
            ):
                self.field_param.grad[seed_rows, seed_cols] = 0.0
            torch.nn.utils.clip_grad_norm_([self.field_param], self.config.grad_clip)
            self.optimizer.step()
            if seed_fixed is not None:
                with torch.no_grad():
                    self.field_param.data[seed_rows, seed_cols] = seed_fixed

            terms = LossTerms(
                seed=weighted_seed.detach(),
                eikonal=weighted_eik.detach(),
                lipschitz=weighted_lip.detach(),
                interface=weighted_interface.detach(),
                tv=weighted_tv.detach(),
            )
            history.append(terms)

            total_val = float(total_loss.detach().item())
            if self._should_checkpoint(step):
                self._save_checkpoint(step, total_val, terms)

            if self._should_log_step(step):
                stage = stage_tag
                seed_val = float(terms.seed.item())
                eik_val = float(terms.eikonal.item())
                lip_val = float(terms.lipschitz.item())
                interface_val = float(terms.interface.item())
                tv_val = float(terms.tv.item())
                interface_tag = "hj" if not use_grad_align else "grad_align"
                print(
                    f"[step {step:5d}/{self.config.max_steps:5d}] stage={stage} "
                    f"total={total_val:.6f} seed={seed_val:.6f} "
                    f"eik={eik_val:.6f} lip={lip_val:.6f} {interface_tag}={interface_val:.6f} "
                    f"tv={tv_val:.6f} beta={self.current_beta:.2f} lr={self.current_lr:.2e}"
                )

        self.history = history
        return history

    def _should_log_step(self, step: int) -> bool:
        interval = max(1, self.config.log_interval)
        return step == 1 or step == self.config.max_steps or step % interval == 0

    def _stage_tag(self, step: int) -> str:
        return "main"

    def _update_learning_rate(self, stage: str) -> None:
        if self.optimizer is None:
            return
        target = self.config.lr_main if self.config.lr_main is not None else self.config.lr
        if abs(self.current_lr - target) < 1e-12:
            return
        for group in self.optimizer.param_groups:
            group["lr"] = target
        self.current_lr = target

    def _should_checkpoint(self, step: int) -> bool:
        if self.checkpoint_dir is None:
            return False
        interval = self.config.checkpoint_interval
        return interval is not None and interval > 0 and step % interval == 0

    def _checkpoint_path(self, step: int) -> Optional[Path]:
        if self.checkpoint_dir is None:
            return None
        filename = f"checkpoint_step{step:05d}.pt"
        return self.checkpoint_dir / filename

    def _serialize_loss_terms(self, terms: LossTerms, total_val: float) -> Dict[str, float]:
        return {
            "seed": float(terms.seed.item()),
            "eikonal": float(terms.eikonal.item()),
            "lipschitz": float(terms.lipschitz.item()),
            "interface": float(terms.interface.item()),
            "tv": float(terms.tv.item()),
            "total": float(total_val),
        }

    def _save_checkpoint(self, step: int, total_val: float, terms: LossTerms) -> None:
        if self.field_param is None or self.optimizer is None:
            return
        if self.precomp is None:
            return
        path = self._checkpoint_path(step)
        if path is None:
            return
        config_dict = asdict(self.config)
        if config_dict.get("checkpoint_dir") is not None:
            config_dict["checkpoint_dir"] = str(config_dict["checkpoint_dir"])

        field_current = self.field_param.detach()
        optimizer_state = self.optimizer.state_dict()
        loss_dict = self._serialize_loss_terms(terms, total_val)

        inference = VoronoiInference(self.precomp, tau_bis=self.config.tau_bis)
        inference_result = inference.run(field_current)

        field_cpu = field_current.cpu()
        vertices_np = np.asarray(self.mesh.vertices, dtype=np.float64)
        faces_np = np.asarray(self.mesh.faces, dtype=np.int64)
        labels_np = inference_result.vertex_labels.cpu().numpy()
        distances_np = inference_result.vertex_distances.cpu().numpy()
        boundary_np = inference_result.edge_boundaries.cpu().numpy()
        stage = self._stage_tag(step)

        state = {
            "step": step,
            "beta": self.current_beta,
            "seeds": self.seeds,
            "stage": stage,
            "loss": loss_dict,
            "config": config_dict,
            "model_state": field_cpu,
            "optimizer_state": optimizer_state,
        }
        torch.save(state, path)

        npz_path = path.with_suffix(".npz")
        np.savez_compressed(
            npz_path,
            step=np.array(step, dtype=np.int32),
            beta=np.array(self.current_beta, dtype=np.float32),
            field=field_cpu.numpy(),
            seeds=np.asarray(self.seeds, dtype=np.int64),
            loss_seed=np.array(loss_dict["seed"], dtype=np.float32),
            loss_eikonal=np.array(loss_dict["eikonal"], dtype=np.float32),
            loss_lipschitz=np.array(loss_dict["lipschitz"], dtype=np.float32),
            loss_interface=np.array(loss_dict["interface"], dtype=np.float32),
            loss_tv=np.array(loss_dict["tv"], dtype=np.float32),
            loss_total=np.array(loss_dict["total"], dtype=np.float32),
            stage=np.array(stage, dtype=np.str_),
            vertices=vertices_np,
            faces=faces_np,
            labels=labels_np,
            distances=distances_np,
            boundary_edges=boundary_np,
            scale_factor=np.array(self.precomp.scale_factor, dtype=np.float64),
            original_mean_edge=np.array(self.precomp.original_mean_edge, dtype=np.float64),
        )
