from __future__ import annotations

import argparse
import os
import numpy as np
from vtk.util import numpy_support as nps

from .volume import (
    read_unstructured_grid,
    extract_surface,
    polydata_to_VF,
    map_volume_points_to_surface_ids,
    surface_quality_report,
    mean_edge_length_polydata,
    write_polydata,
)
from .io import write_results
from .meshio_support import load_surface_from_mesh_file
from .segment import contours_polydata
from .sources import gfps_geodesic_seeds
from .heat import HeatGeodesic


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Heat-method φ on boundary surface of a volume (tet/hex)")
    ap.add_argument("volume", help="Input .vtu/.vtk UnstructuredGrid (tet/hex)")
    ap.add_argument("--keep-quads", action="store_true", help="Keep polygonal faces (quads) instead of triangulating (default is triangles per paper)")
    ap.add_argument("--no-ids", action="store_true", help="Do not pass through original point/cell ids to surface (default keeps ids)")
    ap.add_argument("--save-surface", default=None, help="Optional path to save the extracted surface .vtp before φ")

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--single-vol", type=int, help="Single source: volume point id (mapped to surface)")
    src.add_argument("--single-surf", type=int, help="Single source: surface vertex id (after extraction)")
    src.add_argument("--seeds-vol", type=int, nargs="+", help="Multiple sources: volume point ids (mapped to surface)")
    src.add_argument("--seeds-surf", type=int, nargs="+", help="Multiple sources: surface vertex ids")
    src.add_argument("--gfps", type=int, metavar="K", help="Geodesic farthest-point sampling on surface with K seeds")
    src.add_argument("--boundary", action="store_true", help="Use all boundary vertices as sources (i.e., entire surface)")

    ap.add_argument("--time-step", type=float, default=None, help="Diffusion time t; default t≈h² on surface (paper)")
    ap.add_argument("--contours", action="store_true", help="Extract φ iso-contours for visualization")
    ap.add_argument("--contours-delta", type=float, default=None, help="Contour spacing Δ (default ≈ 5h if --contours)")
    ap.add_argument("--out-mesh", default="phi_surface.vtp")
    ap.add_argument("--out-contours", default="phi_contours.vtp")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.volume):
        raise SystemExit(f"Input file not found: {args.volume}")

    is_meshio = args.volume.lower().endswith('.mesh')
    ug = None
    if is_meshio:
        # Build a triangulated surface directly from .mesh
        pd, Vtmp, Ftmp = load_surface_from_mesh_file(args.volume)
        # Note: ID passthrough not preserved for .mesh path
    else:
        ug = read_unstructured_grid(args.volume)
        pd = extract_surface(ug, keep_quads=args.keep_quads, keep_ids=(not args.no_ids))

    # Quality report
    rep = surface_quality_report(pd)
    print(f"Surface: points={rep['num_points']} faces={rep['num_faces']} closed={rep['is_closed']} boundary_edges={rep['num_boundary_edges']}")

    # Save the surface before adding φ, if requested
    if args.save_surface:
        write_polydata(pd, args.save_surface)

    # If triangulated, we can convert to V,F; otherwise use polydata directly for h
    if not args.keep_quads:
        V, F = polydata_to_VF(pd)
    else:
        V = None; F = None

    h = mean_edge_length_polydata(pd)
    t = float(args.time_step) if args.time_step is not None else (h * h)

    # Determine seeds on the surface mesh
    if args.single_surf is not None:
        seeds = np.array([int(args.single_surf)], dtype=np.int32)
    elif args.seeds_surf is not None:
        seeds = np.asarray(args.seeds_surf, dtype=np.int32)
    elif args.single_vol is not None:
        # Prefer exact mapping via original ids if available; otherwise use nearest surface vertex
        sid = None
        if not args.no_ids:
            arr = pd.GetPointData().GetArray("vtkOriginalPointIds")
            if arr is None:
                arr = pd.GetPointData().GetArray("origPointId_vol")
            if arr is not None:
                ids = np.array([arr.GetValue(i) for i in range(arr.GetNumberOfTuples())], dtype=np.int64)
                matches = np.flatnonzero(ids == int(args.single_vol))
                if matches.size > 0:
                    sid = int(matches[0])
        if sid is None:
            # Fallback: nearest surface vertex
            if ug is not None:
                vol2surf = map_volume_points_to_surface_ids(ug, pd)
                sid = int(vol2surf[int(args.single_vol)])
            else:
                # .mesh path: read full volume points and map by nearest
                from scipy.spatial import cKDTree
                import meshio
                m = meshio.read(args.volume)
                P = np.asarray(m.points, dtype=float)
                if P.shape[1] == 2:
                    P = np.column_stack([P, np.zeros((P.shape[0],), dtype=float)])
                pts_surf = nps.vtk_to_numpy(pd.GetPoints().GetData())
                tree = cKDTree(pts_surf)
                target = P[int(args.single_vol)]
                _, k = tree.query(target, k=1)
                sid = int(k)
        seeds = np.array([sid], dtype=np.int32)
    elif args.seeds_vol is not None:
        # Map many volume ids to surface ids
        vol_ids = np.asarray(args.seeds_vol, dtype=int)
        mapped = None
        if not args.no_ids:
            arr = pd.GetPointData().GetArray("vtkOriginalPointIds")
            if arr is None:
                arr = pd.GetPointData().GetArray("origPointId_vol")
            if arr is not None:
                ids = np.array([arr.GetValue(i) for i in range(arr.GetNumberOfTuples())], dtype=np.int64)
                # Build map from vol id -> first occurrence surface id
                surf_map = {int(ids[i]): i for i in range(ids.size)}
                mapped = np.array([surf_map.get(int(v), -1) for v in vol_ids], dtype=int)
                # For any -1, fall back below
        if mapped is None or (mapped < 0).any():
            if ug is not None:
                vol2surf = map_volume_points_to_surface_ids(ug, pd)
                miss = (mapped < 0) if mapped is not None else np.ones_like(vol_ids, dtype=bool)
                surf_ids = vol2surf[vol_ids]
                if mapped is None:
                    mapped = surf_ids.astype(int)
                else:
                    mapped[miss] = surf_ids[miss].astype(int)
            else:
                # .mesh path fallback: nearest surface vertex using full volume coordinates
                from scipy.spatial import cKDTree
                import meshio
                m = meshio.read(args.volume)
                P = np.asarray(m.points, dtype=float)
                if P.shape[1] == 2:
                    P = np.column_stack([P, np.zeros((P.shape[0],), dtype=float)])
                pts_surf = nps.vtk_to_numpy(pd.GetPoints().GetData())
                tree = cKDTree(pts_surf)
                targets = P[vol_ids]
                _, k = tree.query(targets, k=1)
                nearest = k.astype(int)
                if mapped is None:
                    mapped = nearest
                else:
                    miss = (mapped < 0)
                    mapped[miss] = nearest[miss]
        seeds = mapped.astype(np.int32)
    elif args.boundary:
        # all surface vertices
        seeds = np.arange(pd.GetNumberOfPoints(), dtype=np.int32)
    else:
        if args.keep_quads:
            # For GFPS we need a triangulated mesh for our helper; create a temp triangulation
            from .io import mean_edge_length  # not needed; just reuse HeatGeodesic w/ temp
            tri = vtk.vtkTriangleFilter(); tri.SetInputData(pd); tri.PassLinesOff(); tri.PassVertsOff(); tri.Update()
            pd_t = tri.GetOutput()
            Vt, Ft = polydata_to_VF(pd_t)
            seeds = gfps_geodesic_seeds(Vt, Ft, int(args.gfps))
        else:
            seeds = gfps_geodesic_seeds(V, F, int(args.gfps))

    # For the heat method we need triangles; if keep_quads, build a triangulated copy for computation
    if args.keep_quads:
        tri = vtk.vtkTriangleFilter(); tri.SetInputData(pd); tri.PassLinesOff(); tri.PassVertsOff(); tri.Update()
        pd_comp = tri.GetOutput()
        Vc, Fc = polydata_to_VF(pd_comp)
        geo = HeatGeodesic(Vc, Fc, t=t)
        phi = geo.phi_to_subset(seeds)
        # Write/visualize on the original (quad) surface; points are identical/order-preserving.
        pd_use = pd
    else:
        geo = HeatGeodesic(V, F, t=t)
        phi = geo.phi_to_subset(seeds)
        pd_use = pd

    # Mark seed vertices for visualization
    try:
        import numpy as _np
        from vtk.util import numpy_support as _nps
        n_pts = pd_use.GetNumberOfPoints()
        seed_mask = _np.zeros(n_pts, dtype=_np.int32)
        sidx = _np.asarray(seeds, dtype=_np.intp)
        seed_mask[sidx] = 1
        arr_seed = _nps.numpy_to_vtk(seed_mask, deep=True)
        arr_seed.SetName("seed_mask")
        pd_use.GetPointData().AddArray(arr_seed)
    except Exception:
        pass

    contours = None
    if args.contours:
        cdelta = args.contours_delta if args.contours_delta is not None else 5.0 * h
        contours = contours_polydata(pd_use, phi, delta=cdelta)
    write_results(pd_use, phi, labels=None, contours_pd=contours, out_mesh=args.out_mesh, out_contours=args.out_contours)


if __name__ == "__main__":
    main()
