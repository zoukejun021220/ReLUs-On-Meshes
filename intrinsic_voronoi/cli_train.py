"""Command-line entry for training the intrinsic Voronoi pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import torch

from .inference import VoronoiInference
from .mesh import MeshData, load_mesh, load_surface_mesh, precompute_geometry
from .trainer import TrainingConfig, VoronoiTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train intrinsic Voronoi distance fields")
    parser.add_argument("mesh", type=Path, help="Path to npz file with vertices/faces arrays")
    parser.add_argument("seeds", type=Path, help="Path to text file listing seed vertex indices")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--config", type=Path, help="Optional JSON config override")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to the final npz results file (default derived from mesh/interface)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory to store periodic training checkpoints (default derived from --output)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        help="Number of steps between checkpoints (default 2500; set to 0 to disable)",
    )
    parser.add_argument(
        "--init-method",
        choices=["dijkstra", "heat"],
        help="Initialization strategy for the Voronoi field (default heat)",
    )
    return parser.parse_args()


def load_npz_mesh(path: Path) -> MeshData:
    data = np.load(path)
    vertices = data["vertices"]
    faces = data["faces"]
    return load_mesh(vertices, faces)


def load_seed_indices(path: Path) -> Sequence[int]:
    with path.open("r", encoding="utf8") as handle:
        return [int(line.strip()) for line in handle if line.strip()]


def load_input_mesh(path: Path) -> MeshData:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return load_npz_mesh(path)
    if suffix in {".vtk", ".vtp", ".vtu", ".ply", ".obj", ".stl"}:
        return load_surface_mesh(path)
    raise ValueError(f"Unsupported mesh extension '{suffix}'. Provide .npz or a VTK-like file.")


def apply_config_overrides(config: TrainingConfig, data: Dict[str, Any]) -> TrainingConfig:
    gate_overrides = data.pop("gate_config", None)
    allowed: Dict[str, Any] = {}
    for key, value in data.items():
        if not hasattr(config, key):
            continue
        if key == "checkpoint_dir" and value is not None:
            allowed[key] = Path(value)
        else:
            allowed[key] = value
    updated = replace(config, **allowed)
    if gate_overrides:
        gate_cfg = replace(updated.gate_config, **{k: v for k, v in gate_overrides.items() if hasattr(updated.gate_config, k)})
        updated = replace(updated, gate_config=gate_cfg)
    return updated


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    mesh = load_input_mesh(args.mesh)
    seeds = load_seed_indices(args.seeds)
    precomp = precompute_geometry(mesh, seeds, device=device)

    config = TrainingConfig()
    if args.config:
        with args.config.open("r", encoding="utf8") as handle:
            overrides = json.load(handle)
        config = apply_config_overrides(config, overrides)

    if args.checkpoint_interval is not None:
        interval_override = max(0, args.checkpoint_interval)
        config = replace(config, checkpoint_interval=interval_override)

    if args.init_method is not None:
        config = replace(config, init_method=args.init_method)

    interface_mode = config.interface_loss.lower()
    grad_aliases = {"gradient_alignment", "grad_alignment", "grad_align"}
    interface_tag = "grad_align" if interface_mode in grad_aliases else "hj"

    mesh_stem = args.mesh.stem
    default_save_dir = Path(f"{mesh_stem}_{interface_tag}")
    if args.output is None:
        output_path = default_save_dir / f"{mesh_stem}_{interface_tag}.npz"
    else:
        output_path = args.output

    checkpoint_dir = args.checkpoint_dir or config.checkpoint_dir
    interval = config.checkpoint_interval
    if interval is not None and interval <= 0:
        checkpoint_dir = None
        config = replace(config, checkpoint_interval=None)
    elif checkpoint_dir is None and interval is not None and interval > 0:
        if args.output is None:
            checkpoint_dir = default_save_dir / "checkpoints"
        else:
            checkpoint_dir = output_path.parent / f"{output_path.stem}_checkpoints"

    if checkpoint_dir is not None:
        config = replace(config, checkpoint_dir=Path(checkpoint_dir))
    else:
        config = replace(config, checkpoint_dir=None)

    trainer = VoronoiTrainer(mesh, seeds, precomp=precomp, device=device, config=config)
    history = trainer.train()

    if not history:
        raise RuntimeError("Training did not produce any history records")

    final_terms = history[-1]
    print(
        f"Final loss total={final_terms.total.item():.6f} "
        f"seed={final_terms.seed.item():.6f} "
        f"eikonal={final_terms.eikonal.item():.6f} "
        f"lipschitz={final_terms.lipschitz.item():.6f} "
        f"{interface_tag}={final_terms.interface.item():.6f} "
        f"tv={final_terms.tv.item():.6f}"
    )

    field = trainer.field_param.detach()
    inference = VoronoiInference(precomp, tau_bis=config.tau_bis)
    result = inference.run(field)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        vertices=mesh.vertices,
        faces=mesh.faces,
        field=field.detach().cpu().numpy(),
        labels=result.vertex_labels.cpu().numpy(),
        distances=result.vertex_distances.cpu().numpy(),
        boundary_edges=result.edge_boundaries.cpu().numpy(),
        scale_factor=precomp.scale_factor,
        original_mean_edge=precomp.original_mean_edge,
    )
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
