from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support as nps

from .io import mean_edge_length
from .heat import HeatGeodesic


def segment_with_activation(
    V: np.ndarray,
    F: np.ndarray,
    seeds: np.ndarray,
    delta: float | None = None,
    bands: list[float] | np.ndarray | None = None,
):
    geo = HeatGeodesic(V, F)
    phi = geo.phi_to_subset(seeds)
    labels = None
    if bands is not None:
        labels = np.digitize(phi, bands)
    else:
        if delta is None:
            h = mean_edge_length(V, F)
            delta = 5.0 * h
        labels = np.floor(phi / float(delta)).astype(np.int32)
    return phi, labels


def contours_polydata(
    pd: vtk.vtkPolyData,
    phi: np.ndarray,
    num_levels: int | None = None,
    delta: float | None = None,
) -> vtk.vtkPolyData:
    arr = nps.numpy_to_vtk(phi.astype(np.float64))
    arr.SetName("phi_geodesic")
    pd.GetPointData().SetScalars(arr)

    if num_levels is None and delta is not None:
        lo, hi = float(phi.min()), float(phi.max())
        num_levels = max(1, int((hi - lo) / float(delta)))

    cf = vtk.vtkContourFilter()
    cf.SetInputData(pd)
    if delta is not None:
        lo, hi = float(phi.min()), float(phi.max())
        cf.GenerateValues(int(num_levels), lo, hi)
    else:
        cf.GenerateValues(int(num_levels or 10), pd.GetPointData().GetScalars().GetRange())
    cf.Update()
    return cf.GetOutput()

