from __future__ import annotations

import argparse
import os

import numpy as np

from .io import load_polydata, write_results, mean_edge_length
from .volume import read_unstructured_grid, extract_surface, polydata_to_VF
from .meshio_support import load_surface_from_mesh_file
from .sources import boundary_vertex_ids, gfps_geodesic_seeds
from .segment import segment_with_activation, contours_polydata


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Geodesics in Heat (potpourri3d) pipeline")
    ap.add_argument("mesh", help="Input .vtp/.vtk polydata file")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--single", type=int, help="Single source vertex id")
    src.add_argument("--seeds", type=int, nargs="+", help="Multiple source vertex ids (space-separated)")
    src.add_argument("--boundary", action="store_true", help="Use boundary vertices as sources")
    src.add_argument("--gfps", type=int, metavar="K", help="Geodesic farthest-point sampling with K seeds")
    ap.add_argument("--time-step", type=float, default=None, help="Diffusion time t; default t≈h² (paper)")
    # Paper defaults: phi only, no activation, no contours unless requested
    ap.add_argument("--activation", action="store_true", help="Enable activation/banding (not in core paper algorithm)")
    ap.add_argument("--delta", type=float, default=None, help="Band width for activation (default ≈ 5h if --activation)")
    ap.add_argument("--bands", type=float, nargs="*", help="Explicit band thresholds (use with --activation)")
    ap.add_argument("--contours", action="store_true", help="Extract φ iso-contours for visualization")
    ap.add_argument("--contours-delta", type=float, default=None, help="Contour spacing Δ (default ≈ 5h if --contours)")
    ap.add_argument("--out-mesh", default="mesh_with_phi.vtp")
    ap.add_argument("--out-contours", default="phi_contours.vtp")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.mesh):
        raise SystemExit(f"Input file not found: {args.mesh}")

    # Load surface: support .vtp/.vtk PolyData directly; fallback to .vtk UnstructuredGrid and .mesh via extraction
    pd = None; V = None; F = None
    try:
        pd, V, F = load_polydata(args.mesh)
    except Exception:
        ext = args.mesh.lower()
        if ext.endswith('.mesh'):
            pd, V, F = load_surface_from_mesh_file(args.mesh)
        else:
            # try unstructured grid -> extract surface
            ug = read_unstructured_grid(args.mesh)
            pd = extract_surface(ug, keep_quads=False, keep_ids=True)
            V, F = polydata_to_VF(pd)
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

    from .heat import HeatGeodesic
    geo = HeatGeodesic(V, F, t=t)
    phi = geo.phi_to_subset(seeds)

    # Mark source vertices on the output for visualization
    try:
        import numpy as _np
        seed_mask = _np.zeros(V.shape[0], dtype=_np.int32)
        seed_mask[_np.asarray(seeds, dtype=_np.intp)] = 1
        from vtk.util import numpy_support as _nps
        arr_seed = _nps.numpy_to_vtk(seed_mask, deep=True)
        arr_seed.SetName("seed_mask")
        pd.GetPointData().AddArray(arr_seed)
    except Exception:
        pass

    labels = None
    if args.activation:
        # Compute labels from the already-computed φ
        if args.bands is not None:
            labels = np.digitize(phi, args.bands)
        else:
            delta = args.delta if args.delta is not None else 5.0 * h
            labels = np.floor(phi / float(delta)).astype(np.int32)

    contours = None
    if args.contours:
        cdelta = args.contours_delta if args.contours_delta is not None else 5.0 * h
        contours = contours_polydata(pd, phi, delta=cdelta)

    write_results(pd, phi, labels=labels, contours_pd=contours, out_mesh=args.out_mesh, out_contours=args.out_contours)


if __name__ == "__main__":
    main()
