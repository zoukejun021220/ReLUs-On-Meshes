"""Utilities for extracting geodesic fronts (iso-distance contours)."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import vtk
from vtk.util import numpy_support as nps


def determine_front_levels(
    phi: np.ndarray,
    *,
    levels: Sequence[float] | None = None,
    spacing: float | None = None,
    num_levels: int | None = None,
    max_distance: float | None = None,
    include_zero: bool = False,
    start_distance: float | None = None,
) -> np.ndarray:
    """Compute distance values at which to extract front iso-curves.

    Parameters
    ----------
    phi:
        Per-vertex geodesic distances.
    levels:
        Explicit front distances. If provided, overrides spacing/num_levels.
    spacing:
        Desired spacing between consecutive fronts (ignored if `levels` given).
    num_levels:
        Number of fronts to generate when spacing is not provided. If both
        spacing and num_levels are omitted, defaults to 10 evenly spaced levels.
    max_distance:
        Optional cap on the maximum distance to visualize.
    include_zero:
        If True, include 0 in the output (front through the seed).
    start_distance:
        Optional lower bound for the first contour (defaults to the first
        positive distance in `phi`).
    """
    phi = np.asarray(phi, dtype=np.float64).ravel()
    if phi.size == 0:
        return np.asarray([], dtype=np.float64)

    if levels is not None:
        vals = np.asarray(list(levels), dtype=np.float64)
    else:
        lo = float(np.min(phi))
        hi = float(np.max(phi))
        if max_distance is not None:
            hi = min(hi, float(max_distance))
        if start_distance is not None:
            start = float(start_distance)
        else:
            # Skip distances extremely close to the minimum (typically zero at the seed)
            mask = phi > lo + 1e-8
            if not np.any(mask):
                start = float(lo)
            else:
                start = float(phi[mask].min())
        if spacing is not None and spacing > 0:
            vals = np.arange(start, hi + 1e-8, float(spacing), dtype=np.float64)
        else:
            n = int(num_levels) if num_levels is not None else 10
            if n <= 0:
                return np.asarray([0.0], dtype=np.float64) if include_zero else np.asarray([], dtype=np.float64)
            vals = np.linspace(start, hi, n + 1, dtype=np.float64)[1:]
    if max_distance is not None:
        vals = vals[vals <= float(max_distance) + 1e-8]
    vals = np.unique(vals)
    if not include_zero:
        vals = vals[vals > 1e-12]
    else:
        vals = np.concatenate(([0.0], vals[vals > 1e-12]))
    return vals.astype(np.float64)


def fronts_polydata(
    pd: vtk.vtkPolyData,
    phi: np.ndarray,
    levels: Iterable[float],
) -> vtk.vtkPolyData | None:
    """Extract front polylines for the provided levels.

    Returns a vtkPolyData containing polylines; None if `levels` is empty.
    """
    lvls = [float(v) for v in levels if np.isfinite(v)]
    if len(lvls) == 0:
        return None

    phi_arr = nps.numpy_to_vtk(np.asarray(phi, dtype=np.float64), deep=True)
    phi_arr.SetName("phi_geodesic")

    # Work on a shallow copy to avoid mutating caller's polydata scalars.
    pd_use = vtk.vtkPolyData()
    pd_use.ShallowCopy(pd)
    pd_use.GetPointData().SetScalars(phi_arr)

    cf = vtk.vtkContourFilter()
    cf.SetInputData(pd_use)
    for idx, val in enumerate(lvls):
        cf.SetValue(idx, val)
    cf.Update()
    return cf.GetOutput()

