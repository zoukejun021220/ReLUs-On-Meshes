from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support as nps


def read_unstructured_grid(path: str) -> vtk.vtkUnstructuredGrid:
    lower = path.lower()
    if lower.endswith(".vtu"):
        r = vtk.vtkXMLUnstructuredGridReader()
        r.SetFileName(path)
        r.Update()
        return r.GetOutput()
    elif lower.endswith(".vtk"):
        r = vtk.vtkUnstructuredGridReader()
        r.SetFileName(path)
        r.Update()
        ug = r.GetOutput()
        if not isinstance(ug, vtk.vtkUnstructuredGrid):
            raise ValueError("Expected an UnstructuredGrid in legacy .vtk")
        return ug
    else:
        raise ValueError("Unsupported file extension; use .vtu or .vtk for unstructured grids")


def write_polydata(pd: vtk.vtkPolyData, path: str) -> None:
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(pd)
    w.Write()


def extract_surface(
    ug: vtk.vtkUnstructuredGrid,
    keep_quads: bool = False,
    keep_ids: bool = True,
) -> vtk.vtkPolyData:
    """Extract boundary surface from a tet/hex volume, clean, and optionally triangulate.

    - keep_quads: keep polygonal faces (e.g., quads from hex); otherwise triangulate
    - keep_ids: pass original point/cell ids for back-mapping
    """
    dataset_for_surface = ug
    if keep_ids:
        idf = vtk.vtkIdFilter()
        idf.SetInputData(ug)
        idf.SetPointIdsArrayName("origPointId_vol")
        idf.SetCellIdsArrayName("origCellId_vol")
        idf.FieldDataOn()
        idf.CellIdsOn()
        idf.PointIdsOn()
        idf.Update()
        dataset_for_surface = idf.GetOutput()

    surf = vtk.vtkDataSetSurfaceFilter()
    surf.SetInputData(dataset_for_surface)
    if keep_ids:
        surf.PassThroughPointIdsOn()
        surf.PassThroughCellIdsOn()
    surf.Update()
    pd = surf.GetOutput()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(pd)
    clean.ConvertStripsToPolysOn()
    clean.PointMergingOn()
    clean.Update()
    pd = clean.GetOutput()

    if not keep_quads:
        tri = vtk.vtkTriangleFilter()
        tri.SetInputData(pd)
        tri.PassLinesOff()
        tri.PassVertsOff()
        tri.Update()
        pd = tri.GetOutput()

    norms = vtk.vtkPolyDataNormals()
    norms.SetInputData(pd)
    norms.ConsistencyOn()
    norms.SplittingOff()
    norms.AutoOrientNormalsOn()
    norms.ComputePointNormalsOff()
    norms.ComputeCellNormalsOn()
    norms.Update()
    return norms.GetOutput()


def surface_quality_report(pd: vtk.vtkPolyData) -> dict:
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.NonManifoldEdgesOn()
    fe.FeatureEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    ne = fe.GetOutput().GetNumberOfCells()
    return {
        "is_closed": ne == 0,
        "num_boundary_edges": int(ne),
        "num_points": int(pd.GetNumberOfPoints()),
        "num_faces": int(pd.GetNumberOfCells()),
    }


def mean_edge_length_polydata(pd: vtk.vtkPolyData) -> float:
    """Compute mean edge length on polydata (triangles or general polygons)."""
    ex = vtk.vtkExtractEdges()
    ex.SetInputData(pd)
    ex.Update()
    edges = ex.GetOutput()
    if edges.GetNumberOfLines() == 0:
        return 0.0
    pts = nps.vtk_to_numpy(edges.GetPoints().GetData())
    lines = nps.vtk_to_numpy(edges.GetLines().GetData())
    # vtk cell array: [n, id0, id1, n, id0, id1, ...] for lines, so n==2 per segment
    conn = lines.reshape(-1, 3)[:, 1:3]
    le = np.linalg.norm(pts[conn[:, 0]] - pts[conn[:, 1]], axis=1)
    return float(le.mean())


def polydata_to_VF(pd: vtk.vtkPolyData):
    V = nps.vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float64)
    ca = nps.vtk_to_numpy(pd.GetPolys().GetData())
    # If polygons not triangulated, caller should triangulate first; else reshape fails
    F = ca.reshape(-1, 4)[:, 1:4].astype(np.int32)
    return V, F


def map_volume_points_to_surface_ids(ug: vtk.vtkUnstructuredGrid, pd_surf: vtk.vtkPolyData) -> np.ndarray:
    """For each volume point, find nearest surface vertex id. Useful to map seed ids."""
    from scipy.spatial import cKDTree

    Vvol = nps.vtk_to_numpy(ug.GetPoints().GetData())
    Vsurf = nps.vtk_to_numpy(pd_surf.GetPoints().GetData())
    tree = cKDTree(Vsurf)
    _, ids = tree.query(Vvol, k=1)
    return ids.astype(np.int32)
