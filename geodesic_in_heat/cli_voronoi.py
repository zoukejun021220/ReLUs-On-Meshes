from __future__ import annotations

import argparse
import os
import numpy as np
import vtk

from .io import load_polydata, write_results
from .volume import read_unstructured_grid, extract_surface, polydata_to_VF
from .meshio_support import load_surface_from_mesh_file
from .sources import gfps_geodesic_seeds
from .voronoi import geodesic_distance_matrix, voronoi_labels, bisector_polylines, attach_phi_vector
from .voronoi import subdivide_voronoi_polydata


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Geodesic Voronoi (heat method) via per-vertex distance vectors + argmin")
    ap.add_argument("mesh", help="Input surface (.vtp/.vtk) or volume (.vtu/.vtk) or .mesh (meshio)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--seeds", type=int, nargs="+", help="Seed vertex ids (space-separated)")
    src.add_argument("--gfps", type=int, metavar="K", help="Geodesic farthest-point sampling with K seeds")
    ap.add_argument("--time-step", type=float, default=None, help="Diffusion time t; default t≈h² (paper)")

    # Outputs
    ap.add_argument("--out-mesh", default="mesh_voronoi.vtp", help="Output mesh with 'label' and optional 'phi_vec'")
    ap.add_argument("--write-phi-vec", action="store_true", help="Attach K-component 'phi_vec' to output mesh")
    ap.add_argument("--bisectors", action="store_true", help="Extract and write bisector polylines")
    ap.add_argument("--out-bisectors", default="voronoi_bisectors.vtp")
    ap.add_argument("--sample-points", action="store_true", help="Emit sampled points colored by Voronoi label")
    ap.add_argument("--per-edge", type=int, default=20, help="Sampling resolution per triangle edge (default 20)")
    ap.add_argument("--out-points", default="voronoi_points.vtp")
    ap.add_argument("--subdivide", type=int, default=0, help="Uniformly subdivide faces and output crisp per-vertex labels (no shader)")
    ap.add_argument("--out-subdivided", default="voronoi_subdivided.vtp")
    ap.add_argument("--subdivide-gpu", action="store_true", help="Use CuPy (CUDA) to accelerate subdivision if available")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(args.mesh):
        raise SystemExit(f"Input file not found: {args.mesh}")

    # Load a triangulated surface pd and (V,F)
    pd = None
    V = None
    F = None
    try:
        pd, V, F = load_polydata(args.mesh)
    except Exception:
        ext = args.mesh.lower()
        if ext.endswith('.mesh'):
            pd, V, F = load_surface_from_mesh_file(args.mesh)
        else:
            ug = read_unstructured_grid(args.mesh)
            pd = extract_surface(ug, keep_quads=False, keep_ids=True)
            V, F = polydata_to_VF(pd)

    # Seeds
    if args.seeds is not None:
        seeds = np.asarray(args.seeds, dtype=np.int32)
    else:
        seeds = gfps_geodesic_seeds(V, F, int(args.gfps))

    # Time step
    from .io import mean_edge_length
    h = mean_edge_length(V, F)
    t = float(args.time_step) if args.time_step is not None else (h * h)

    # Compute Phi and labels
    Phi = geodesic_distance_matrix(V, F, seeds, t=t)
    labels = voronoi_labels(Phi)

    # Optionally build a subdivided mesh for crisp Voronoi patches without custom shaders
    if int(args.subdivide) > 0:
        sub_pd = subdivide_voronoi_polydata(
            V, F, Phi,
            per_edge=int(args.subdivide),
            keep_phi_vec=bool(args.write_phi_vec),
            use_gpu=bool(args.subdivide_gpu),
        )
        wsub = vtk.vtkXMLPolyDataWriter(); wsub.SetFileName(args.out_subdivided); wsub.SetInputData(sub_pd); wsub.Write()
        # Also attach arrays to the original pd for completeness
    from vtk.util import numpy_support as nps
    arr_lbl = nps.numpy_to_vtk(labels.astype(np.int32), deep=True)
    arr_lbl.SetName("label")
    pd.GetPointData().AddArray(arr_lbl)
    if args.write_phi_vec:
        attach_phi_vector(pd, Phi, name="phi_vec")

    # Optionally compute bisectors
    bis = None
    if args.bisectors:
        bis = bisector_polylines(pd, Phi)

    # Write mesh + optional bisectors
    write_results(pd, phi=Phi[:, 0], labels=labels, contours_pd=bis, out_mesh=args.out_mesh, out_contours=args.out_bisectors)
    # Note: write_results expects a 'phi' array; we pass Phi[:,0] just to retain a float scalar for viewers.

    # Optional point sampling
    if args.sample_points:
        from .voronoi import sample_points
        pts = sample_points(V, F, Phi, per_edge=int(args.per_edge))
        w = vtk.vtkXMLPolyDataWriter()
        w.SetFileName(args.out_points)
        w.SetInputData(pts)
        w.Write()


if __name__ == "__main__":
    main()
