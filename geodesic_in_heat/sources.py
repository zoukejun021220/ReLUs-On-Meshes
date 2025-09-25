from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support as nps


def boundary_vertex_ids(pd: vtk.vtkPolyData) -> np.ndarray:
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    bpd = fe.GetOutput()

    from scipy.spatial import cKDTree

    V_all = nps.vtk_to_numpy(pd.GetPoints().GetData())
    V_b = nps.vtk_to_numpy(bpd.GetPoints().GetData())
    tree = cKDTree(V_all)
    _, ids = tree.query(V_b, k=1)
    return np.unique(ids).astype(np.int32)


def gfps_geodesic_seeds(V: np.ndarray, F: np.ndarray, K: int, start: int | None = None) -> np.ndarray:
    from .heat import HeatGeodesic

    n = V.shape[0]
    if start is None:
        start = int(np.random.randint(n))
    seeds = [start]
    dmin = np.full(n, np.inf)
    geo = HeatGeodesic(V, F)
    for _ in range(1, int(K)):
        phi = geo.phi_to_subset([seeds[-1]])
        dmin = np.minimum(dmin, phi)
        next_seed = int(np.argmax(dmin))
        seeds.append(next_seed)
    return np.array(seeds, dtype=np.int32)

