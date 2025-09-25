#!/usr/bin/env python3
"""Interactive vertex picker using Open3D's VisualizerWithEditing."""
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import open3d as o3d

try:
    import meshio
except ImportError:  # pragma: no cover - meshio is a repo dependency but we guard regardless
    meshio = None


_NATIVE_EXTENSIONS: tuple[str, ...] = (".obj", ".ply", ".stl", ".off", ".gltf", ".glb")


def _load_mesh(path: str) -> o3d.geometry.TriangleMesh:
    """Load a triangle mesh, falling back to meshio for VTK/VTU/VTP files."""
    ext = os.path.splitext(path)[1].lower()
    if ext in _NATIVE_EXTENSIONS:
        mesh = o3d.io.read_triangle_mesh(path)
        if mesh.is_empty():
            raise ValueError(f"Loaded mesh is empty: {path}")
        return mesh
    if meshio is None:
        raise RuntimeError("meshio is required to import this mesh format; install it or convert to .obj/.ply")
    data = meshio.read(path)
    tri_data = None
    for block in data.cells:
        if block.type == "triangle":
            tri_data = block.data
            break
    if tri_data is None:
        raise ValueError("No triangle cells found; ensure the mesh is triangulated")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(data.points[:, :3])
    mesh.triangles = o3d.utility.Vector3iVector(tri_data.astype(int))
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    return mesh


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pick vertex indices with Open3D")
    ap.add_argument("mesh", help="Path to the mesh (supports OBJ/PLY/STL/OFF, or VTK/VTP/VTU via meshio)")
    ap.add_argument("--window-name", default="Open3D Vertex Picker", help="Window title")
    ap.add_argument("--width", type=int, default=1024, help="Window width in pixels")
    ap.add_argument("--height", type=int, default=768, help="Window height in pixels")
    return ap.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if not os.path.exists(args.mesh):
        print(f"Mesh not found: {args.mesh}", file=sys.stderr)
        return 1
    try:
        mesh = _load_mesh(args.mesh)
    except Exception as exc:  # broad to surface clear message to the caller
        print(f"Failed to load mesh: {exc}", file=sys.stderr)
        return 1

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    print("\nInstructions: hold Shift and left-click to pick vertices; press Q or Esc to exit.\n")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=args.window_name, width=args.width, height=args.height)
    vis.add_geometry(mesh)
    vis.run()
    vis.destroy_window()

    picked = vis.get_picked_points()
    if picked:
        print("Picked vertex indices:", " ".join(str(idx) for idx in picked))
    else:
        print("No vertices were picked.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
