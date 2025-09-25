from __future__ import annotations

import argparse

from .viewer import visualize


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Scriptable per-fragment interpolation viewer (VTK OpenGL2)")
    ap.add_argument("mesh", help="Input .vtk/.vtp PolyData or .mesh (Medit)")
    ap.add_argument("--arrays", nargs="+", required=True, help="Point array names used as f0..fN in expression")
    ap.add_argument("--expr", required=True, help="Expression over f0.., e.g., 'relu(f0[0])' or 'argmax(f0)'")
    ap.add_argument("--mode", choices=["scalar", "label", "rgb"], default="scalar", help="Output interpretation")
    ap.add_argument("--labels", type=int, default=8, help="Number of labels for label mode")
    ap.add_argument(
        "--warp",
        default="identity",
        choices=("identity", "sqrt", "log", "power"),
        help="Warp interpolated samples before evaluation",
    )
    ap.add_argument(
        "--warp-power",
        type=float,
        default=1.0,
        help="Exponent when --warp=power",
    )
    ap.add_argument("--triangulate", action="store_true", help="Triangulate polygons before rendering")
    ap.add_argument("--offscreen", action="store_true", help="No interactor; just render and save if screenshot is set")
    ap.add_argument("--screenshot", default=None, help="Save PNG screenshot to this path")
    ap.add_argument("--show-edges", action="store_true", help="Wireframe overlay")
    ap.add_argument("--show-points", action="store_true", help="Render mesh vertices as points overlay")
    ap.add_argument("--point-color", default="black", help="Point color name (black/white/red/green/blue)")
    ap.add_argument("--point-size", type=float, default=3.0, help="Point size in pixels")
    ap.add_argument("--mark-seeds", action="store_true", help="Overlay seed (pinned) points")
    ap.add_argument("--seed-array", default=None, help="Point array name used as seed mask (auto-detect if omitted)")
    ap.add_argument("--seed-color", default="red", help="Seed point color")
    ap.add_argument("--seed-size", type=float, default=8.0, help="Seed point size in pixels")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit seed vertex indices")
    ap.add_argument("--show-seams", action="store_true", help="Overlay per-face Voronoi seams")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    visualize(
        args.mesh,
        arrays=args.arrays,
        expr=args.expr,
        mode=args.mode,
        num_labels=args.labels,
        warp_mode=args.warp,
        warp_power=args.warp_power,
        triangulate=args.triangulate,
        offscreen=args.offscreen,
        screenshot=args.screenshot,
        show_edges=args.show_edges,
        show_points=args.show_points,
        point_color=args.point_color,
        point_size=args.point_size,
        mark_seeds=args.mark_seeds,
        seed_array=args.seed_array,
        seed_color=args.seed_color,
        seed_size=args.seed_size,
        seeds=args.seeds,
        show_seams=args.show_seams,
    )


if __name__ == "__main__":
    main()
