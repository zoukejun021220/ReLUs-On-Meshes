from __future__ import annotations

import argparse
import os

import numpy as np

from .io import load_polydata, write_results, mean_edge_length
from .volume import read_unstructured_grid, extract_surface, polydata_to_VF
from .meshio_support import load_surface_from_mesh_file
from .sources import boundary_vertex_ids, gfps_geodesic_seeds
from .heat import HeatGeodesic
from .fronts import determine_front_levels, fronts_polydata


def _load_surface(mesh_path: str):
    try:
        return load_polydata(mesh_path)
    except Exception:
        ext = mesh_path.lower()
        if ext.endswith(".mesh"):
            return load_surface_from_mesh_file(mesh_path)
        ug = read_unstructured_grid(mesh_path)
        pd = extract_surface(ug, keep_quads=False, keep_ids=True)
        V, F = polydata_to_VF(pd)
        return pd, V, F


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Compute geodesic distance and visualize front iso-curves")
    ap.add_argument("mesh", help="Input surface (.vtp/.vtk) or volume (.vtu/.vtk/.mesh)")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--single", type=int, help="Single source vertex id")
    src.add_argument("--seeds", type=int, nargs="+", help="Multiple source vertex ids")
    src.add_argument("--boundary", action="store_true", help="Use all boundary vertices as sources")
    src.add_argument("--gfps", type=int, metavar="K", help="Geodesic farthest-point sampling with K seeds")

    ap.add_argument("--time-step", type=float, default=None, help="Diffusion time t; default t≈h²")
    ap.add_argument("--levels", type=float, nargs="+", help="Explicit front distances")
    ap.add_argument("--spacing", type=float, default=None, help="Spacing between fronts (overridden by --levels)")
    ap.add_argument("--num-levels", type=int, default=10, help="Number of fronts when spacing not specified")
    ap.add_argument("--max-distance", type=float, default=None, help="Maximum distance to include in fronts")
    ap.add_argument("--include-zero", action="store_true", help="Include the zero-distance front (through seeds)")
    ap.add_argument("--start-distance", type=float, default=None, help="Override first non-zero front distance")

    ap.add_argument("--out-mesh", default="mesh_fronts.vtp", help="Output mesh with phi scalar")
    ap.add_argument("--out-fronts", default="phi_fronts.vtp", help="Output polylines for the fronts")
    ap.add_argument("--viewer", action="store_true", help="Launch viewer after writing files")
    ap.add_argument("--viewer-offscreen", action="store_true", help="Use off-screen rendering when viewer is enabled")
    ap.add_argument("--viewer-show-edges", action="store_true", help="Render mesh edges in viewer")
    ap.add_argument("--viewer-edge-color", default="black", help="Viewer edge color")

    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.mesh):
        raise SystemExit(f"Input file not found: {args.mesh}")

    pd, V, F = _load_surface(args.mesh)
    h = mean_edge_length(V, F)
    t = float(args.time_step) if args.time_step is not None else (h * h)

    if args.single is not None:
        seeds = np.array([int(args.single)], dtype=np.int32)
    elif args.seeds is not None:
        seeds = np.asarray(args.seeds, dtype=np.int32)
    elif args.boundary:
        seeds = boundary_vertex_ids(pd)
    else:
        seeds = gfps_geodesic_seeds(V, F, int(args.gfps))

    geo = HeatGeodesic(V, F, t=t)
    phi = geo.phi_to_subset(seeds)

    # Mark source vertices for visualization
    try:
        import numpy as _np
        from vtk.util import numpy_support as _nps

        seed_mask = _np.zeros(V.shape[0], dtype=_np.int32)
        seed_mask[_np.asarray(seeds, dtype=_np.intp)] = 1
        arr_seed = _nps.numpy_to_vtk(seed_mask, deep=True)
        arr_seed.SetName("seed_mask")
        pd.GetPointData().AddArray(arr_seed)
    except Exception:
        pass

    levels = determine_front_levels(
        phi,
        levels=args.levels,
        spacing=args.spacing if args.levels is None else None,
        num_levels=args.num_levels,
        max_distance=args.max_distance,
        include_zero=args.include_zero,
        start_distance=args.start_distance,
    )

    fronts = fronts_polydata(pd, phi, levels)

    write_results(pd, phi, labels=None, contours_pd=fronts, out_mesh=args.out_mesh, out_contours=args.out_fronts)

    if args.viewer:
        mesh_paths = [args.out_mesh]
        contour_paths = [args.out_fronts] if fronts is not None and fronts.GetNumberOfPoints() > 0 else None
        try:
            from .view import _try_pyvista_show, _vtk_show
        except Exception as exc:  # pragma: no cover - import failure fallback
            print(f"Viewer import failed: {exc}")
            return

        shown = _try_pyvista_show(
            mesh_paths=mesh_paths,
            contours_paths=contour_paths,
            scalars="phi_geodesic",
            screenshot=None,
            offscreen=bool(args.viewer_offscreen),
            show_edges=bool(args.viewer_show_edges),
            edge_color=args.viewer_edge_color,
            edge_width=1.5,
            show_points=False,
            point_color="black",
            point_size=5.0,
            mark_seeds=True,
            seed_array="seed_mask",
            seed_color="red",
            seed_size=12.0,
            fragment_argmax=False,
            phi_vec_name="phi_vec",
            cell_argmax=False,
        )
        if not shown:
            _vtk_show(
                mesh_paths=mesh_paths,
                contours_paths=contour_paths,
                scalars="phi_geodesic",
                screenshot=None,
                offscreen=bool(args.viewer_offscreen),
                show_edges=bool(args.viewer_show_edges),
                edge_color=args.viewer_edge_color,
                edge_width=1.5,
                show_points=False,
                point_color="black",
                point_size=5.0,
                mark_seeds=True,
                seed_array="seed_mask",
                seed_color="red",
                seed_size=12.0,
                fragment_argmax=False,
                phi_vec_name="phi_vec",
                cell_argmax=False,
            )


if __name__ == "__main__":  # pragma: no cover
    main()
