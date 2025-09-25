from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import vtk
from vtk.util import numpy_support as nps


def _vtk_polydata_from_triangles(V: np.ndarray, F: np.ndarray) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(V.astype(np.float64), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)

    ca = vtk.vtkCellArray()
    # slow but robust across VTK versions
    for tri in F.astype(np.int64):
        idl = vtk.vtkIdList()
        idl.SetNumberOfIds(3)
        idl.SetId(0, int(tri[0]))
        idl.SetId(1, int(tri[1]))
        idl.SetId(2, int(tri[2]))
        ca.InsertNextCell(idl)
    pd.SetPolys(ca)
    return pd


def _fix_medit_inline_counts(path: str) -> str | None:
    """If a Medit .mesh uses inline counts (e.g., "Vertices 123"), rewrite to the
    meshio-expected form (header then count on next line) in a temp file.

    Returns the path to the fixed temp file, or None if no change was made.
    """
    try:
        import re
        import tempfile
        from pathlib import Path

        text = Path(path).read_text()
        patterns = [
            r"^(Vertices)\s+(\d+)\s*$",
            r"^(Edges)\s+(\d+)\s*$",
            r"^(Triangles)\s+(\d+)\s*$",
            r"^(Quadrilaterals)\s+(\d+)\s*$",
            r"^(Tetrahedra)\s+(\d+)\s*$",
            r"^(Hexahedra)\s+(\d+)\s*$",
        ]
        changed = False
        for pat in patterns:
            new_text, n = re.subn(pat, r"\1\n\2", text, flags=re.MULTILINE)
            if n:
                changed = True
                text = new_text
        if not changed:
            return None
        fd, tmp = tempfile.mkstemp(suffix=".mesh")
        import os
        os.close(fd)
        Path(tmp).write_text(text)
        return tmp
    except Exception:
        return None


def load_surface_from_mesh_file(path: str, keep_quads: bool = False) -> Tuple[vtk.vtkPolyData, np.ndarray, np.ndarray]:
    """Load a Medit .mesh (or other meshio-supported) file and return a triangulated surface.

    - If the file contains triangles/quads, use them (triangulate quads if keep_quads=False).
    - If the file contains tets/hexes, extract boundary surface and triangulate by default.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        import meshio
    except Exception as e:
        raise RuntimeError("meshio is required to read .mesh files. Install with `pip install meshio`." ) from e

    try:
        m = meshio.read(path)
    except Exception:
        # Some Medit files use inline counts ("Vertices N"), which meshio's
        # reader may not accept. Try a lightweight text rewrite.
        if ext == ".mesh":
            fixed = _fix_medit_inline_counts(path)
            if fixed is not None:
                m = meshio.read(fixed)
            else:
                raise
        else:
            raise
    P = np.asarray(m.points, dtype=np.float64)
    if P.shape[1] == 2:
        P = np.column_stack([P, np.zeros((P.shape[0],), dtype=np.float64)])

    # Build cells dict
    cells_dict = {}
    for cb in m.cells:
        cells_dict.setdefault(cb.type, [])
        cells_dict[cb.type].append(cb.data)
    # concatenate lists
    cells_dict = {k: np.vstack(v) if len(v) > 1 else v[0] for k, v in cells_dict.items()}

    F_tri: np.ndarray | None = None

    # If surface faces exist
    if "triangle" in cells_dict:
        F_tri = np.asarray(cells_dict["triangle"], dtype=np.int64)
    elif "quad" in cells_dict:
        Q = np.asarray(cells_dict["quad"], dtype=np.int64)
        if keep_quads:
            # triangulate anyway for computation; but we return triangles
            F_tri = np.vstack([np.c_[Q[:, 0], Q[:, 1], Q[:, 2]], np.c_[Q[:, 0], Q[:, 2], Q[:, 3]]])
        else:
            F_tri = np.vstack([np.c_[Q[:, 0], Q[:, 1], Q[:, 2]], np.c_[Q[:, 0], Q[:, 2], Q[:, 3]]])

    # Else build boundary from volume cells
    if F_tri is None and ("tetra" in cells_dict or "hexahedron" in cells_dict):
        tris = []
        # Tets → triangle faces
        if "tetra" in cells_dict:
            T = np.asarray(cells_dict["tetra"], dtype=np.int64)
            faces = np.vstack([
                T[:, [0, 1, 2]],
                T[:, [0, 1, 3]],
                T[:, [0, 2, 3]],
                T[:, [1, 2, 3]],
            ])
            # mark duplicates
            faces_sorted = np.sort(faces, axis=1)
            # unique with counts
            from collections import defaultdict
            counts = defaultdict(int)
            for row in faces_sorted:
                counts[tuple(row)] += 1
            boundary_mask = np.array([counts[tuple(row)] == 1 for row in faces_sorted])
            tris.append(faces[boundary_mask])
        # Hexes → quad faces → triangulate
        if "hexahedron" in cells_dict:
            H = np.asarray(cells_dict["hexahedron"], dtype=np.int64)
            # Faces for (0..7) hexahedron (VTK/meshio ordering)
            hex_faces = np.array(
                [
                    [0, 1, 2, 3],  # bottom
                    [4, 5, 6, 7],  # top
                    [0, 1, 5, 4],  # front
                    [1, 2, 6, 5],  # right
                    [2, 3, 7, 6],  # back
                    [3, 0, 4, 7],  # left
                ],
                dtype=np.int64,
            )
            quads = np.concatenate([H[:, f] for f in hex_faces], axis=0)
            quads_sorted = np.sort(quads, axis=1)
            from collections import defaultdict
            counts_q = defaultdict(int)
            for row in quads_sorted:
                counts_q[tuple(row)] += 1
            boundary = np.array([counts_q[tuple(row)] == 1 for row in quads_sorted])
            bq = quads[boundary]
            # triangulate each quad consistently
            tris.append(np.vstack([np.c_[bq[:, 0], bq[:, 1], bq[:, 2]], np.c_[bq[:, 0], bq[:, 2], bq[:, 3]]]))
        if len(tris) == 0:
            raise RuntimeError("No surface triangles could be derived from the volume cells in .mesh file")
        F_tri = np.vstack(tris)

    if F_tri is None:
        raise RuntimeError("Unsupported .mesh content: expected triangle/quad or tetra/hexahedron cells")

    # Compress to used vertices only to ensure a well-formed surface mesh
    used = np.unique(F_tri)
    remap = -np.ones(P.shape[0], dtype=np.int64)
    remap[used] = np.arange(used.size, dtype=np.int64)
    Vc = P[used].astype(np.float64)
    Fc = remap[F_tri].astype(np.int32)
    pd = _vtk_polydata_from_triangles(Vc, Fc)
    # Preserve original point ids from the source file for mapping
    arr = nps.numpy_to_vtk(used.astype(np.int64), deep=True)
    arr.SetName("origPointId_vol")
    pd.GetPointData().AddArray(arr)
    return pd, Vc, Fc
