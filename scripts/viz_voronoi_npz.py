#!/usr/bin/env python3
"""Convert Voronoi training results (.npz) to VTK PolyData and launch viz_tool."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
import warnings
from typing import Iterable, List

import numpy as np

try:
    import pyvista as pv
except ImportError as exc:  # pragma: no cover - helpful message for CLI usage
    raise SystemExit("pyvista is required; install it via `pip install pyvista`." ) from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _convert_faces(triangles: np.ndarray) -> np.ndarray:
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces array must have shape (F, 3)")
    prefix = np.full((triangles.shape[0], 1), 3, dtype=triangles.dtype)
    return np.hstack((prefix, triangles))


def _available_arrays(npz_data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key in ("field", "distances", "labels", "boundary_edges"):
        if key in npz_data.files:
            arrays[key] = npz_data[key]
    return arrays


def _write_polydata(npz_path: Path, destination: Path) -> List[str]:
    data = np.load(npz_path)
    vertices = data["vertices"].astype(np.float32)
    faces = data["faces"].astype(np.int64)

    mesh = pv.PolyData(vertices, _convert_faces(faces))

    avail = _available_arrays(data)
    n_points = vertices.shape[0]
    n_cells = faces.shape[0]

    if "field" in avail and avail["field"].shape[0] == n_points:
        mesh.point_data["field"] = avail["field"].astype(np.float32)
    if "distances" in avail and avail["distances"].shape[0] == n_points:
        mesh.point_data["distances"] = avail["distances"].astype(np.float32)
    if "labels" in avail and avail["labels"].shape[0] == n_points:
        mesh.point_data["labels"] = avail["labels"].astype(np.int32)
    if "boundary_edges" in avail:
        edges = avail["boundary_edges"]
        if edges.shape[0] == n_cells:
            mesh.cell_data["boundary_edges"] = edges.astype(np.int8)
        else:
            warnings.warn(
                "Skipping 'boundary_edges' because its length does not match the cell count",
                RuntimeWarning,
                stacklevel=2,
            )

    mesh.save(destination)
    return list(mesh.point_data.keys())


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load voronoi_results.npz, emit a temporary .vtp, and forward to viz_tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("npz", type=Path, help="Path to voronoi_results.npz (or similar)")
    parser.add_argument(
        "--arrays",
        nargs="+",
        default=["field"],
        help="Point arrays to expose as f0..fN in viz_tool (must exist in the NPZ)",
    )
    parser.add_argument(
        "--expr",
        default="argmin(f0)",
        help="Expression passed to viz_tool (see viz_tool --help)",
    )
    parser.add_argument(
        "--mode",
        choices=["scalar", "label", "rgb"],
        default="label",
        help="Shading mode for viz_tool",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Do not delete the intermediate .vtp file (printed on exit)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save the converted .vtp to this path instead of a temporary file",
    )
    parser.add_argument(
        "--viz-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments forwarded directly to viz_tool after '--'",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = make_arg_parser()
    args = parser.parse_args(argv)

    if not args.npz.exists():
        parser.error(f"NPZ file not found: {args.npz}")

    tmp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.output is not None:
        tmp_path = args.output
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tmp_dir = tempfile.TemporaryDirectory(prefix="voronoi_viz_")
        tmp_path = Path(tmp_dir.name) / (args.npz.stem + ".vtp")

    available = _write_polydata(args.npz, tmp_path)

    missing = [name for name in args.arrays if name not in available]
    if missing:
        parser.error(f"Array(s) {missing} not present in converted VTP. Available: {available}")

    viz_cli: List[str] = [str(tmp_path), "--arrays", *args.arrays, "--expr", args.expr, "--mode", args.mode]
    if args.viz_args:
        viz_cli.extend(args.viz_args)

    from viz_tool.main import main as viz_main

    try:
        viz_main(viz_cli)
    finally:
        if args.keep_temp:
            print(f"Temporary VTP preserved at {tmp_path}")
        elif tmp_dir is not None:
            tmp_dir.cleanup()


if __name__ == "__main__":
    main(sys.argv[1:])
