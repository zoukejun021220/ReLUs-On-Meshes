#!/usr/bin/env python3
"""Compute heat-method geodesics and visualize the cut locus on a surface mesh.

This script mirrors the reference workflow from the "Geodesics in Heat" paper:
  * load/triangulate a surface (.vtp/.vtk/.mesh)
  * solve for heat-method distances from user-provided seeds
  * extract a cut-locus proxy by thresholding |Δphi| (cotan Laplacian)
  * overlay iso-distance fronts for visual validation
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import vtk
from vtk.util import numpy_support as nps

# Allow running the script without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geodesic_in_heat.cut_locus import cut_locus_by_laplacian, cut_locus_by_gradient_jump
from geodesic_in_heat.fronts import determine_front_levels, fronts_polydata
from geodesic_in_heat.heat import HeatGeodesic
from geodesic_in_heat.io import load_polydata, mean_edge_length
from geodesic_in_heat.meshio_support import load_surface_from_mesh_file
from geodesic_in_heat.sources import gfps_geodesic_seeds
from geodesic_in_heat.volume import extract_surface, polydata_to_VF, read_unstructured_grid


def _load_surface(mesh_path: str):
    """Load and triangulate a surface from .vtp/.vtk/.mesh/.vtu/.vtk(volume)."""
    try:
        return load_polydata(mesh_path)
    except Exception:
        ext = Path(mesh_path).suffix.lower()
        if ext == ".mesh":
            return load_surface_from_mesh_file(mesh_path)
        ug = read_unstructured_grid(mesh_path)
        pd = extract_surface(ug, keep_quads=False, keep_ids=True)
        V, F = polydata_to_VF(pd)
        return pd, V, F


def _nearest_vertex(V: np.ndarray, target: Sequence[float]) -> int:
    p = np.asarray(target, dtype=np.float64)
    d = np.linalg.norm(V - p[None, :], axis=1)
    return int(np.argmin(d))


def _visualize(
    pd: vtk.vtkPolyData,
    V: np.ndarray,
    F: np.ndarray,
    phi: np.ndarray,
    cut_lines: vtk.vtkPolyData,
    fronts: vtk.vtkPolyData | None,
    seeds: Sequence[int],
    screenshot: str | None,
) -> None:
    try:
        import pyvista as pv
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            "PyVista is required for visualization. Install with `pip install pyvista`."
        ) from exc

    mesh = pv.wrap(pd)
    mesh.point_data["phi_geodesic"] = phi

    plotter = pv.Plotter(window_size=(1300, 900))
    plotter.set_background("white")
    plotter.add_mesh(
        mesh,
        scalars="phi_geodesic",
        smooth_shading=True,
        show_scalar_bar=True,
        lighting=False,
    )

    if fronts is not None and fronts.GetNumberOfCells() > 0:
        plotter.add_mesh(pv.wrap(fronts), color="white", line_width=2)

    if cut_lines is not None and cut_lines.GetNumberOfCells() > 0:
        plotter.add_mesh(pv.wrap(cut_lines), color="red", line_width=3)

    if len(seeds) > 0:
        radius = 2.0 * mean_edge_length(V, F) if F.size > 0 else 0.01
        for sid in seeds:
            center = V[int(sid)]
            plotter.add_mesh(pv.Sphere(radius=radius, center=center), color="magenta", specular=0.0)

    plotter.add_text("white: fronts  |  red: cut locus", font_size=10, color="black")
    plotter.show(screenshot=screenshot)


def _unwrap_cut_locus(res):
    """Return (polydata, threshold, num_components) regardless of helper return type."""
    if hasattr(res, "lines"):
        poly = res.lines
        thresh = getattr(res, "threshold", float("nan"))
        ncomp = getattr(res, "num_components", -1)
    else:
        poly = res
        thresh = float("nan")
        ncomp = -1
    return poly, thresh, ncomp


def _build_candidate_values(base: float, *, defaults: Sequence[float]) -> list[float]:
    """Mix user input with sensible fallback values, remove duplicates, keep order."""
    extras = [base, base * 0.5, base * 1.5, base * 2.0]
    candidates = []
    seen: set[float] = set()

    def add(val: float) -> None:
        try:
            v = float(val)
        except (TypeError, ValueError):
            return
        if v < 0:
            return
        if v in seen:
            return
        seen.add(v)
        candidates.append(v)

    for val in extras + list(defaults):
        add(val)
    return candidates


def _extract_cut_lines_autotune(
    pd: vtk.vtkPolyData,
    V: np.ndarray,
    F: np.ndarray,
    phi: np.ndarray,
    seeds: Sequence[int],
    *,
    method: str,
    pct_grid: Sequence[float] = (0.6, 0.8, 1.0, 1.5, 2.5, 4.0, 6.0),
    rmult_grid: Sequence[float] = (6.0, 8.0, 10.0, 12.0),
    len_grid: Sequence[float] = (10.0, 20.0, 30.0),
) -> tuple[vtk.vtkPolyData, float, int]:
    """Try a small parameter grid until some cut-locus lines survive."""

    def run_once(pct: float, rmult: float, lmult: float):
        if method == "gradjump":
            return cut_locus_by_gradient_jump(
                pd,
                V,
                F,
                phi,
                top_percent=float(pct),
                seeds=seeds,
                exclude_radius_multiplier=float(rmult),
                min_component_length_multiplier=float(lmult),
            )
        return cut_locus_by_laplacian(
            pd,
            V,
            F,
            phi,
            top_percent=float(pct),
            seeds=seeds,
            exclude_radius_multiplier=float(rmult),
            min_component_length_multiplier=float(lmult),
        )

    last_poly = vtk.vtkPolyData()
    last_thresh = float("nan")
    last_comp = 0
    for rmult in rmult_grid:
        for pct in pct_grid:
            if pct <= 0.0:
                continue
            for lmult in sorted(len_grid, reverse=True):
                res = run_once(pct, rmult, lmult)
                poly, thresh, ncomp = _unwrap_cut_locus(res)
                last_poly, last_thresh, last_comp = poly, thresh, ncomp
                n_lines = poly.GetNumberOfCells()
                if n_lines > 0:
                    print(
                        f"[auto] method={method} top%={pct} exclude_mult={rmult} "
                        f"min_len_mult={lmult} -> {n_lines} lines"
                    )
                    return poly, thresh, ncomp
    print("[auto] no cut-locus lines survived filters; returning last attempt")
    return last_poly, last_thresh, last_comp


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Heat geodesic + cut-locus visualizer for surface meshes")
    ap.add_argument("mesh", nargs="?", help="Input surface (.vtp/.vtk/.mesh) or volume (.vtu/.vtk) file")
    ap.add_argument("--mesh", dest="mesh_opt", help="Input surface (.vtp/.vtk/.mesh) or volume (.vtu/.vtk) file")

    src = ap.add_mutually_exclusive_group()
    src.add_argument("--seed", type=int, help="Single source vertex id")
    src.add_argument("--seed-xyz", type=float, nargs=3, metavar=("X", "Y", "Z"), help="Seed as 3D coordinates (nearest vertex is used)")
    src.add_argument("--seeds", type=int, nargs="+", help="Multiple seed vertex ids")
    src.add_argument("--gfps", type=int, metavar="K", help="Geodesic farthest-point sampling with K seeds")

    ap.add_argument("--time-mult", type=float, default=1.0, help="Set diffusion time to time_mult * h^2 (default 1)")
    ap.add_argument("--front-levels", type=int, default=10, help="Number of iso-distance fronts to overlay")
    ap.add_argument(
        "--method",
        choices=("lap", "gradjump"),
        default="lap",
        help="Cut-locus extractor: 'lap' for |Δphi| band, 'gradjump' for edge gradient jumps",
    )
    ap.add_argument("--top-percent", type=float, default=1.0, help="Top percent of signal retained for the cut locus")
    ap.add_argument("--exclude-radius-mult", type=float, default=10.0, help="Exclude multiplier ×h ball around each seed before thresholding")
    ap.add_argument("--min-length-mult", type=float, default=10.0, help="Discard cut-locus components shorter than multiplier ×h")
    ap.add_argument("--screenshot", type=Path, default=None, help="Optional path to save a screenshot")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    mesh_arg = args.mesh_opt or args.mesh
    if mesh_arg is None:
        raise SystemExit("Please provide a mesh path (positional or --mesh)")
    mesh_path = Path(mesh_arg)
    if not mesh_path.exists():
        raise SystemExit(f"Input file not found: {mesh_arg}")

    pd, V, F = _load_surface(str(mesh_path))
    h = mean_edge_length(V, F)
    t = float(args.time_mult) * (h * h)

    if args.seeds is not None:
        seeds = np.asarray(args.seeds, dtype=np.int32)
    elif args.seed is not None:
        seeds = np.array([int(args.seed)], dtype=np.int32)
    elif args.seed_xyz is not None:
        sid = _nearest_vertex(V, args.seed_xyz)
        print(f"[info] seed_xyz snapped to vertex id {sid}")
        seeds = np.array([sid], dtype=np.int32)
    elif args.gfps is not None:
        seeds = gfps_geodesic_seeds(V, F, int(args.gfps))
    else:
        centroid_seed = _nearest_vertex(V, np.mean(V, axis=0))
        print(f"[info] defaulting to centroid nearest vertex id {centroid_seed}")
        seeds = np.array([centroid_seed], dtype=np.int32)

    geo = HeatGeodesic(V, F, t=t)
    phi = geo.phi_to_subset(seeds)

    try:
        seed_mask = np.zeros(V.shape[0], dtype=np.uint8)
        seed_mask[np.asarray(seeds, dtype=np.intp)] = 1
        arr_seed = nps.numpy_to_vtk(seed_mask, deep=True)
        arr_seed.SetName("seed_mask")
        pd.GetPointData().AddArray(arr_seed)
    except Exception:
        pass

    if args.method == "gradjump":
        poly, thresh, ncomp = _extract_cut_lines_autotune(
            pd,
            V,
            F,
            phi,
            seeds,
            method="gradjump",
            pct_grid=_build_candidate_values(
                args.top_percent,
                defaults=(0.6, 0.8, 1.0, 2.0, 4.0, 8.0),
            ),
            rmult_grid=_build_candidate_values(
                args.exclude_radius_mult,
                defaults=(0.0, 4.0, 6.0, 8.0, 10.0, 12.0),
            ),
            len_grid=_build_candidate_values(
                args.min_length_mult,
                defaults=(0.0, 3.0, 5.0, 10.0, 15.0),
            ),
        )
    else:
        poly, thresh, ncomp = _extract_cut_lines_autotune(
            pd,
            V,
            F,
            phi,
            seeds,
            method="lap",
            pct_grid=_build_candidate_values(
                args.top_percent,
                defaults=(0.6, 0.8, 1.0, 2.0, 4.0, 8.0),
            ),
            rmult_grid=_build_candidate_values(
                args.exclude_radius_mult,
                defaults=(0.0, 4.0, 6.0, 8.0, 10.0, 12.0),
            ),
            len_grid=_build_candidate_values(
                args.min_length_mult,
                defaults=(0.0, 5.0, 10.0, 20.0, 30.0),
            ),
        )

    levels = determine_front_levels(
        phi,
        num_levels=max(int(args.front_levels), 0),
        include_zero=False,
        spacing=None,
    )
    fronts = fronts_polydata(pd, phi, levels)

    # Attach phi array for downstream viewers
    arr_phi = nps.numpy_to_vtk(phi.astype(np.float64), deep=True)
    arr_phi.SetName("phi_geodesic")
    pd.GetPointData().AddArray(arr_phi)
    pd.GetPointData().SetActiveScalars("phi_geodesic")

    print(
        f"[cut-locus] method={args.method} threshold={thresh:.6g} components={ncomp} "
        f"seeds={seeds.tolist()}"
    )

    cut_lines = poly

    screenshot = str(args.screenshot) if args.screenshot is not None else None
    _visualize(pd, V, F, phi, cut_lines, fronts, seeds, screenshot=screenshot)


if __name__ == "__main__":  # pragma: no cover
    main()
