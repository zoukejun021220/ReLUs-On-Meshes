import numpy as np
import vtk
from vtk.util import numpy_support as nps


def load_polydata(path: str):
    lower = path.lower()
    if lower.endswith(".vtp"):
        r = vtk.vtkXMLPolyDataReader()
        r.SetFileName(path)
        r.Update()
        pd = r.GetOutput()
    elif lower.endswith(".vtk"):
        r = vtk.vtkPolyDataReader()
        r.SetFileName(path)
        r.Update()
        pd = r.GetOutput()
    else:
        raise ValueError("Unsupported file extension; use .vtp or .vtk")

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(pd)
    tri.PassLinesOff()
    tri.PassVertsOff()
    tri.Update()
    pd_tri = tri.GetOutput()

    # Guard: if the source .vtk was not PolyData (e.g., UnstructuredGrid), the
    # PolyData reader yields an empty dataset. In that case, signal the caller
    # to try the unstructured-grid path.
    if pd_tri is None or pd_tri.GetNumberOfPoints() == 0 or pd_tri.GetNumberOfPolys() == 0:
        raise ValueError("No PolyData found in file; try unstructured-grid surface extraction")

    V = nps.vtk_to_numpy(pd_tri.GetPoints().GetData()).astype(np.float64)
    ca = nps.vtk_to_numpy(pd_tri.GetPolys().GetData())
    F = ca.reshape(-1, 4)[:, 1:4].astype(np.int32)
    return pd_tri, V, F


def mean_edge_length(V: np.ndarray, F: np.ndarray) -> float:
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E = np.unique(np.sort(E, axis=1), axis=0)
    le = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    return float(le.mean())


def write_results(
    pd: vtk.vtkPolyData,
    phi: np.ndarray,
    labels: np.ndarray | None,
    contours_pd: vtk.vtkPolyData | None,
    out_mesh: str = "mesh_with_phi.vtp",
    out_contours: str = "phi_contours.vtp",
):
    arr_phi = nps.numpy_to_vtk(phi.astype(np.float64), deep=True)
    arr_phi.SetName("phi_geodesic")
    pd.GetPointData().AddArray(arr_phi)
    if labels is not None:
        arr_lbl = nps.numpy_to_vtk(labels.astype(np.int32), deep=True)
        arr_lbl.SetName("band_label")
        pd.GetPointData().AddArray(arr_lbl)

    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(out_mesh)
    w.SetInputData(pd)
    w.Write()

    if contours_pd is not None:
        w2 = vtk.vtkXMLPolyDataWriter()
        w2.SetFileName(out_contours)
        w2.SetInputData(contours_pd)
        w2.Write()
