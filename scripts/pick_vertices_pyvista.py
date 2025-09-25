#!/usr/bin/env python3
"""Interactive vertex picker using PyVista's point-picking callback."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

try:
    import pyvista as pv
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "PyVista is required for this utility. Install it with 'pip install pyvista'."
    ) from exc


def _load_mesh(path: str) -> pv.PolyData:
    """Read the mesh with PyVista and ensure point IDs are available."""
    dataset = pv.read(path)
    if dataset.n_points == 0:
        raise ValueError(f"Loaded mesh contains no points: {path}")
    if isinstance(dataset, pv.PolyData):
        mesh = dataset.copy(deep=True)
        if not mesh.is_all_triangles:
            mesh = mesh.triangulate()
        return mesh
    if not hasattr(dataset, "extract_surface"):
        raise ValueError("Dataset does not provide an extractable surface. Convert to a surface mesh first.")
    surface = dataset.extract_surface()
    if surface.n_points == 0:
        raise ValueError("Surface extraction produced an empty mesh; is the dataset volumetric only?")
    if not surface.is_all_triangles:
        surface = surface.triangulate()
    return surface


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pick vertex indices using a PyVista viewer")
    ap.add_argument("mesh", help="Path to a mesh that PyVista can read (.vtp/.vtk/.stl/.obj/...)")
    ap.add_argument("--screenshot", help="Optional PNG path to save once you close the window")
    ap.add_argument("--point-size", type=float, default=8.0, help="Size of picked point markers")
    ap.add_argument("--show-edges", action="store_true", help="Overlay mesh edges during viewing")
    return ap.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.exists(args.mesh):
        print(f"Mesh not found: {args.mesh}", file=sys.stderr)
        return 1
    try:
        mesh = _load_mesh(args.mesh)
    except Exception as exc:
        print(f"Failed to load mesh: {exc}", file=sys.stderr)
        return 1

    print("\nControls: left click to rotate, scroll to zoom, and press P to enable point picking.\n"
          "Once picking is active, click a vertex to print its index. Press Esc to exit.\n")

    plotter = pv.Plotter()
    plotter.set_background("white")

    picked_ids: list[int] = []

    def _callback(point: object, picker: object | None = None) -> None:
        # PyVista passes the picked coordinate as a numpy array; use picker for the index when available
        idx = None
        if picker is not None and hasattr(picker, "GetPointId"):
            idx = int(picker.GetPointId())
        if idx is None or idx < 0:
            # fall back to an on-mesh lookup if picker lacks an id
            try:
                import numpy as _np
                coords = _np.asarray(point, dtype=float)
                idx = int(_np.linalg.norm(mesh.points - coords, axis=1).argmin())
            except Exception:  # pragma: no cover - best-effort fallback
                print("Picked point but could not determine vertex index.")
                return
        picked_ids.append(idx)
        xyz = mesh.points[idx]
        print(f"Picked vertex {idx} at {xyz}")

    plotter.add_mesh(
        mesh,
        show_edges=args.show_edges,
        color="lightgray",
        edge_color="black",
        line_width=1.0,
        smooth_shading=True,
    )
    # Configure a fresh point picker explicitly to capture vertex IDs when available
    import vtk
    picker = vtk.vtkPointPicker()

    plotter.enable_point_picking(
        callback=_callback,
        show_message=True,
        use_picker=True,
        picker=picker,
        left_clicking=False,
        show_point=True,
        point_size=args.point_size,
        color="red",
    )

    plotter.show(screenshot=args.screenshot)

    if picked_ids:
        print("\nPicked vertex indices:", " ".join(str(i) for i in picked_ids))
    else:
        print("\nNo vertices were picked.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
