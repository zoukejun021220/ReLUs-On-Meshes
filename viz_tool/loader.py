from __future__ import annotations

from typing import Tuple

from pathlib import Path

import numpy as np
import vtk
from vtk.util import numpy_support as nps


def _vtk_polydata_from_triangles(V: np.ndarray, F: np.ndarray) -> vtk.vtkPolyData:
    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(V.astype(np.float64), deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    ca = vtk.vtkCellArray()
    for tri in F.astype(np.int64):
        idl = vtk.vtkIdList(); idl.SetNumberOfIds(3)
        idl.SetId(0, int(tri[0])); idl.SetId(1, int(tri[1])); idl.SetId(2, int(tri[2]))
        ca.InsertNextCell(idl)
    pd.SetPolys(ca)
    return pd


def _triangulate_pd(pd: vtk.vtkPolyData) -> vtk.vtkPolyData:
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd)
    tri.PassLinesOff(); tri.PassVertsOff()
    tri.Update()
    return tri.GetOutput()


def _load_polydata_vtk(path: str) -> vtk.vtkPolyData:
    lower = path.lower()
    if lower.endswith(".vtp"):
        r = vtk.vtkXMLPolyDataReader()
    elif lower.endswith(".vtk"):
        r = vtk.vtkPolyDataReader()
    else:
        raise ValueError("Expected .vtp or legacy .vtk for PolyData")
    r.SetFileName(path); r.Update()
    pd = r.GetOutput()
    if pd is None or pd.GetNumberOfPoints() == 0:
        raise ValueError("No PolyData in file")
    return pd



def _load_mesh_medit(path: str) -> vtk.vtkPolyData:
    try:
        import meshio
    except Exception as e:
        raise RuntimeError("meshio is required to read .mesh; pip install meshio") from e
    m = meshio.read(path)
    V = np.asarray(m.points, dtype=float)
    if V.shape[1] == 2:
        V = np.column_stack([V, np.zeros((V.shape[0],), dtype=float)])
    cells = {}
    for cb in m.cells:
        cells.setdefault(cb.type, []).append(cb.data)
    F = None
    if "triangle" in cells:
        tris = cells["triangle"]; F = tris[0] if len(tris) == 1 else np.vstack(tris)
    elif "quad" in cells:
        quads = cells["quad"]; Q = quads[0] if len(quads) == 1 else np.vstack(quads)
        F = np.vstack([np.c_[Q[:, 0], Q[:, 1], Q[:, 2]], np.c_[Q[:, 0], Q[:, 2], Q[:, 3]]])
    else:
        # try boundary of tets/hexes
        tris = []
        if "tetra" in cells:
            T = cells["tetra"]; T = T[0] if len(T) == 1 else np.vstack(T)
            tris.append(T[:, [0, 1, 2]]); tris.append(T[:, [0, 1, 3]]); tris.append(T[:, [0, 2, 3]]); tris.append(T[:, [1, 2, 3]])
        if len(tris) == 0:
            raise RuntimeError("Unsupported .mesh cells; need triangle/quad/tetra")
        F = np.vstack(tris)
    # compress used vertices
    used = np.unique(F)
    remap = -np.ones(V.shape[0], dtype=np.int64); remap[used] = np.arange(used.size)
    Vc = V[used]
    Fc = remap[F].astype(np.int64)
    return _vtk_polydata_from_triangles(Vc, Fc)


def _numpy_to_vtk_array(array: np.ndarray, name: str) -> vtk.vtkDataArray:
    arr = np.ascontiguousarray(array)
    if arr.ndim == 1:
        vtk_arr = nps.numpy_to_vtk(arr, deep=True)
        vtk_arr.SetNumberOfComponents(1)
    else:
        vtk_arr = nps.numpy_to_vtk(arr.reshape(-1, arr.shape[1]), deep=True)
        vtk_arr.SetNumberOfComponents(arr.shape[1])
    vtk_arr.SetName(name)
    return vtk_arr


def _load_polydata_npz(path: str) -> Tuple[vtk.vtkPolyData, np.ndarray, np.ndarray]:
    data = np.load(path)
    if "vertices" not in data or "faces" not in data:
        raise ValueError("NPZ must provide 'vertices' and 'faces'")

    verts = np.asarray(data["vertices"], dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] not in (2, 3):
        raise ValueError("vertices must have shape (N, 2|3)")
    if verts.shape[1] == 2:
        verts = np.column_stack((verts, np.zeros((verts.shape[0],), dtype=np.float64)))

    faces = np.asarray(data["faces"], dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")

    pd = _vtk_polydata_from_triangles(verts, faces)
    n_points = verts.shape[0]
    n_cells = faces.shape[0]

    point_data = pd.GetPointData()
    cell_data = pd.GetCellData()

    for key in data.files:
        if key in {"vertices", "faces"}:
            continue
        arr = np.asarray(data[key])
        if arr.size == 0:
            continue
        if arr.shape[0] == n_points:
            arr2d = arr.reshape(n_points, -1)
            vtk_arr = _numpy_to_vtk_array(arr2d, key)
            point_data.AddArray(vtk_arr)
        elif arr.shape[0] == n_cells:
            arr2d = arr.reshape(n_cells, -1)
            vtk_arr = _numpy_to_vtk_array(arr2d, key)
            cell_data.AddArray(vtk_arr)

    return pd, verts.astype(np.float64), faces.astype(np.int32)


def load_surface(path: str, triangulate: bool = False) -> Tuple[vtk.vtkPolyData, np.ndarray, np.ndarray]:
    """Load a surface mesh to PolyData and return (pd, V, F) with triangles.

    - Supports .vtk/.vtp PolyData and .mesh via meshio.
    - If triangulate is True, run vtkTriangleFilter.
    """
    lower = path.lower()
    if lower.endswith((".vtp", ".vtk")):
        pd = _load_polydata_vtk(path)
    elif lower.endswith(".mesh"):
        pd = _load_mesh_medit(path)
    elif lower.endswith(".npz"):
        pd, V, F = _load_polydata_npz(path)
        return pd, V, F
    else:
        raise ValueError("Use .vtk/.vtp, .mesh, or .npz")
    if triangulate:
        pd = _triangulate_pd(pd)
    V = nps.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
    ca = nps.vtk_to_numpy(pd.GetPolys().GetData())
    if ca.size == 0:
        F = np.zeros((0, 3), dtype=np.int32)
    else:
        # Works for triangles; if quads/polys present, triangulate=True recommended.
        try:
            F = ca.reshape(-1, 4)[:, 1:4].astype(np.int32)
        except Exception:
            # Fallback: triangulate and retry
            pd_t = _triangulate_pd(pd)
            V = nps.vtk_to_numpy(pd_t.GetPoints().GetData()).astype(np.float64)
            ca = nps.vtk_to_numpy(pd_t.GetPolys().GetData())
            F = ca.reshape(-1, 4)[:, 1:4].astype(np.int32)
            pd = pd_t
    return pd, V, F
