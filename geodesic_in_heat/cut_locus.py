"""Cut-locus extraction utilities for geodesic-in-heat distance fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import vtk
from vtk.util import numpy_support as nps


@dataclass(slots=True)
class CutLocusResult:
    """Container for cut-locus extraction results."""

    lines: vtk.vtkPolyData
    method: str
    threshold: float
    num_components: int


def _face_gradients(V: np.ndarray, F: np.ndarray, phi: np.ndarray, *, eps: float = 1e-14) -> np.ndarray:
    """Compute per-face gradients of the piecewise-linear scalar field phi.

    Gradient formula follows the heat method paper (Sec. 3.2.1).
    """
    phi = np.asarray(phi, dtype=np.float64).ravel()
    grads = np.zeros((F.shape[0], 3), dtype=np.float64)
    for f, (i, j, k) in enumerate(F):
        v0, v1, v2 = V[i], V[j], V[k]
        Nraw = np.cross(v1 - v0, v2 - v0)
        nrm = np.linalg.norm(Nraw)
        if nrm < eps:
            continue
        n_hat = Nraw / nrm
        coeff = 1.0 / nrm  # = 1 / (2 * area)
        e1 = v2 - v1  # edge opposite v0
        e2 = v0 - v2  # edge opposite v1
        e3 = v1 - v0  # edge opposite v2
        grads[f] = coeff * (
            phi[i] * np.cross(n_hat, e1)
            + phi[j] * np.cross(n_hat, e2)
            + phi[k] * np.cross(n_hat, e3)
        )
    return grads


def _edge_adjacency(F: np.ndarray) -> dict[tuple[int, int], list[int]]:
    from collections import defaultdict

    adj: dict[tuple[int, int], list[int]] = defaultdict(list)
    for f, (i, j, k) in enumerate(F):
        for a, b in ((i, j), (j, k), (k, i)):
            key = (a, b) if a < b else (b, a)
            adj[key].append(f)
    return adj


def _connectivity_with_regions(lines: vtk.vtkPolyData) -> tuple[vtk.vtkPolyData, int]:
    cf = vtk.vtkPolyDataConnectivityFilter()
    cf.SetInputData(lines)
    cf.SetExtractionModeToAllRegions()
    cf.ColorRegionsOn()
    cf.Update()
    out = vtk.vtkPolyData()
    out.ShallowCopy(cf.GetOutput())
    return out, int(cf.GetNumberOfExtractedRegions())


def cut_locus_by_gradient_jump(
    pd: vtk.vtkPolyData,
    V: np.ndarray,
    F: np.ndarray,
    phi: np.ndarray,
    *,
    top_percent: float = 5.0,
    project_on_edge: bool = True,
    seeds: Sequence[int] | None = None,
    exclude_radius_multiplier: float = 0.0,
    min_component_length_multiplier: float = 0.0,
) -> CutLocusResult:
    """Extract cut-locus polylines using gradient jumps across interior edges.

    Parameters
    ----------
    pd : vtkPolyData
        Input surface (triangulated) for point positions.
    V, F : array-like
        Geometry and connectivity matching pd (V shape (n,3), F shape (m,3)).
    phi : array-like
        Geodesic distance per vertex.
    top_percent : float, default 5.0
        Keep the top X percent of edges ranked by jump magnitude.
    project_on_edge : bool, default True
        If True, compare gradient components projected along the edge direction;
        otherwise use full vector differences.
    """
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    phi = np.asarray(phi, dtype=np.float64).ravel()

    grads = _face_gradients(V, F, phi)
    adj = _edge_adjacency(F)

    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(V, deep=True))
    lines = vtk.vtkCellArray()

    # Mean edge length for masking/thresholds
    edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    if edges.size > 0:
        edges = np.unique(np.sort(edges, axis=1), axis=0)
    edge_lengths = (
        np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
        if edges.size > 0
        else np.array([], dtype=np.float64)
    )
    h = float(edge_lengths.mean()) if edge_lengths.size > 0 else 0.0

    mask = np.ones(phi.shape[0], dtype=bool)
    exclude_multiplier = float(exclude_radius_multiplier)
    if seeds is not None and len(seeds) > 0 and exclude_multiplier > 0.0 and h > 0.0:
        rmin = exclude_multiplier * h
        phi_max = float(np.max(phi))
        for s in seeds:
            idx = int(s)
            if 0 <= idx < phi.size:
                cap = min(phi[idx] + rmin, phi_max)
                mask &= phi >= cap
        if not np.any(mask):
            mask[:] = True

    edge_records: list[tuple[int, int, float]] = []
    for (a, b), flist in adj.items():
        if len(flist) != 2:
            continue  # skip boundary/non-manifold edges
        if not (mask[int(a)] and mask[int(b)]):
            continue
        f1, f2 = flist
        g1, g2 = grads[f1], grads[f2]
        if project_on_edge:
            edge_vec = V[b] - V[a]
            length = np.linalg.norm(edge_vec)
            if length > 0:
                tangent = edge_vec / length
                score = abs(float(np.dot(g1 - g2, tangent)))
            else:
                score = 0.0
        else:
            score = float(np.linalg.norm(g1 - g2))
        edge_records.append((int(a), int(b), score))

    out = vtk.vtkPolyData()
    out.SetPoints(pts)

    if not edge_records:
        out.SetLines(lines)
        return CutLocusResult(lines=out, method="gradient_jump", threshold=float("nan"), num_components=0)

    scores = np.asarray([r[2] for r in edge_records], dtype=np.float64)
    pct = float(np.clip(top_percent, 0.0, 100.0))
    thresh = float(np.percentile(scores, 100.0 - pct)) if pct < 100.0 else float(np.min(scores))
    if np.isfinite(thresh) and thresh <= 0.0:
        thresh = float(np.min(scores[scores > 0])) if np.any(scores > 0) else float(np.min(scores))

    arr_jump = vtk.vtkDoubleArray()
    arr_jump.SetName("jump")
    for a, b, s in edge_records:
        if s < thresh:
            continue
        lines.InsertNextCell(2)
        lines.InsertCellPoint(int(a))
        lines.InsertCellPoint(int(b))
        arr_jump.InsertNextValue(float(s))

    out.SetLines(lines)
    if arr_jump.GetNumberOfTuples() > 0:
        out.GetCellData().AddArray(arr_jump)
        out.GetCellData().SetActiveScalars("jump")

    if lines.GetNumberOfCells() == 0:
        return CutLocusResult(lines=out, method="gradient_jump", threshold=thresh, num_components=0)

    length_multiplier = max(float(min_component_length_multiplier), 0.0)
    length_threshold = (length_multiplier * h) if (length_multiplier > 0.0 and h > 0.0) else 0.0

    filtered = out
    if length_threshold > 0.0:
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputData(out)
        conn.SetExtractionModeToAllRegions()
        conn.ColorRegionsOn()
        conn.Update()
        n_regions = int(conn.GetNumberOfExtractedRegions())

        app = vtk.vtkAppendPolyData()
        kept = 0
        for rid in range(n_regions):
            conn.SetExtractionModeToSpecifiedRegions()
            conn.InitializeSpecifiedRegionList()
            conn.AddSpecifiedRegion(rid)
            conn.Update()
            sub = vtk.vtkPolyData()
            sub.ShallowCopy(conn.GetOutput())
            if _polyline_total_length(sub) >= length_threshold:
                app.AddInputData(sub)
                kept += 1
        if kept == 0:
            empty = vtk.vtkPolyData()
            empty.SetPoints(pd.GetPoints())
            empty.SetLines(vtk.vtkCellArray())
            return CutLocusResult(lines=empty, method="gradient_jump", threshold=thresh, num_components=0)
        app.Update()
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputConnection(app.GetOutputPort())
        cleaner.Update()
        filtered = vtk.vtkPolyData()
        filtered.ShallowCopy(cleaner.GetOutput())

    if filtered.GetNumberOfCells() == 0:
        return CutLocusResult(lines=filtered, method="gradient_jump", threshold=thresh, num_components=0)

    lines_regions, n_comp = _connectivity_with_regions(filtered)
    return CutLocusResult(lines=lines_regions, method="gradient_jump", threshold=thresh, num_components=n_comp)


def _cotan_laplacian(
    V: np.ndarray,
    F: np.ndarray,
) -> tuple["_MatrixLike", np.ndarray]:
    """Assemble the cotangent Laplacian (sparse if SciPy is available) and vertex areas."""

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    n = int(V.shape[0])

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    diag = np.zeros(n, dtype=np.float64)
    area = np.zeros(n, dtype=np.float64)

    for (i, j, k) in F:
        v0, v1, v2 = V[i], V[j], V[k]
        N = np.cross(v1 - v0, v2 - v0)
        twice_area = np.linalg.norm(N)
        if twice_area == 0.0:
            continue
        e0 = v1 - v2
        e1 = v2 - v0
        e2 = v0 - v1
        cot_i = float(np.dot(e1, e2) / twice_area)
        cot_j = float(np.dot(e2, e0) / twice_area)
        cot_k = float(np.dot(e0, e1) / twice_area)
        for (a, b, w) in ((i, j, cot_k), (j, k, cot_i), (k, i, cot_j)):
            rows.append(a)
            cols.append(b)
            data.append(-w)
            rows.append(b)
            cols.append(a)
            data.append(-w)
            diag[a] += w
            diag[b] += w
        face_area = 0.5 * twice_area
        share = face_area / 3.0
        area[i] += share
        area[j] += share
        area[k] += share

    rows.extend(range(n))
    cols.extend(range(n))
    data.extend(diag.tolist())

    try:
        from scipy.sparse import coo_matrix  # type: ignore

        L = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    except Exception:
        L = np.zeros((n, n), dtype=np.float64)
        for r, c, w in zip(rows, cols, data):
            L[r, c] += w
    return L, area


def cut_locus_by_laplacian(
    pd: vtk.vtkPolyData,
    V: np.ndarray,
    F: np.ndarray,
    phi: np.ndarray,
    *,
    top_percent: float = 5.0,
    seeds: Sequence[int] | None = None,
    exclude_radius_multiplier: float = 0.0,
    min_component_length_multiplier: float = 0.0,
) -> CutLocusResult:
    """Extract cut-locus polylines by thresholding |Δφ| (cotangent Laplacian).

    Parameters
    ----------
    pd:
        Triangulated surface whose points correspond to ``V``.
    V, F:
        Vertex positions and triangle indices.
    phi:
        Geodesic distance per vertex.
    top_percent:
        Keeps the top ``top_percent`` fraction of |Δφ| samples by magnitude.
    seeds:
        Seed vertex ids; used to suppress a small geodesic ball around each
        seed prior to thresholding. Pass ``None`` to disable masking.
    exclude_radius_multiplier:
        Radius multiplier (× mean edge length) for the excluded region around
        each seed. Values <= 0 disable masking.
    min_component_length_multiplier:
        Discard connected components shorter than this multiplier times the
        mean edge length. Set to 0 to disable filtering.
    """

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int32)
    phi = np.asarray(phi, dtype=np.float64).ravel()

    L, area = _cotan_laplacian(V, F)
    lap = L.dot(phi) if hasattr(L, "dot") else L @ phi
    mag = np.abs(lap / (area + 1e-16))

    mask = np.ones_like(mag, dtype=bool)
    exclude_multiplier = float(exclude_radius_multiplier)
    if seeds is not None and len(seeds) > 0 and exclude_multiplier > 0.0:
        edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        if edges.size > 0:
            edges = np.unique(np.sort(edges, axis=1), axis=0)
        if edges.size > 0:
            edge_lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
            h = float(edge_lengths.mean()) if edge_lengths.size > 0 else 0.0
        else:
            h = 0.0
        if h > 0.0:
            rmin = exclude_multiplier * h
            phi_max = float(np.max(phi))
            for s in seeds:
                idx = int(s)
                if 0 <= idx < phi.size:
                    cap = min(phi[idx] + rmin, phi_max)
                    mask &= phi >= cap
            if not np.any(mask):
                mask[:] = True

    mag_masked = np.where(mask, mag, -np.inf)
    valid = mag_masked > -np.inf
    if not np.any(valid):
        empty = vtk.vtkPolyData()
        empty.SetPoints(pd.GetPoints())
        empty.SetLines(vtk.vtkCellArray())
        return CutLocusResult(lines=empty, method="laplacian", threshold=float("nan"), num_components=0)

    pct = float(np.clip(top_percent, 0.0, 100.0))
    threshold = (
        float(np.percentile(mag_masked[valid], 100.0 - pct))
        if pct < 100.0
        else float(np.min(mag_masked[valid]))
    )

    arr_mag = nps.numpy_to_vtk(mag_masked.astype(np.float64), deep=True)
    arr_mag.SetName("abs_laplacian")

    pd_use = vtk.vtkPolyData()
    pd_use.ShallowCopy(pd)
    pd_use.GetPointData().AddArray(arr_mag)
    pd_use.GetPointData().SetActiveScalars("abs_laplacian")

    cf = vtk.vtkContourFilter()
    cf.SetInputData(pd_use)
    cf.SetValue(0, threshold)
    cf.Update()
    lines = vtk.vtkPolyData()
    lines.ShallowCopy(cf.GetOutput())

    if lines.GetNumberOfCells() == 0:
        return CutLocusResult(lines=lines, method="laplacian", threshold=threshold, num_components=0)

    length_multiplier = max(float(min_component_length_multiplier), 0.0)
    length_threshold = 0.0
    if length_multiplier > 0.0:
        edges = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        if edges.size > 0:
            edges = np.unique(np.sort(edges, axis=1), axis=0)
        if edges.size > 0:
            edge_lengths = np.linalg.norm(V[edges[:, 0]] - V[edges[:, 1]], axis=1)
            mean_edge = float(edge_lengths.mean()) if edge_lengths.size > 0 else 0.0
            if mean_edge > 0.0:
                length_threshold = length_multiplier * mean_edge

    if length_threshold > 0.0:
        conn = vtk.vtkPolyDataConnectivityFilter()
        conn.SetInputData(lines)
        conn.SetExtractionModeToAllRegions()
        conn.ColorRegionsOn()
        conn.Update()
        n_regions = int(conn.GetNumberOfExtractedRegions())

        app = vtk.vtkAppendPolyData()
        kept = 0
        for rid in range(n_regions):
            conn.SetExtractionModeToSpecifiedRegions()
            conn.InitializeSpecifiedRegionList()
            conn.AddSpecifiedRegion(rid)
            conn.Update()
            sub = vtk.vtkPolyData()
            sub.ShallowCopy(conn.GetOutput())
            if _polyline_total_length(sub) >= length_threshold:
                app.AddInputData(sub)
                kept += 1
        if kept == 0:
            empty = vtk.vtkPolyData()
            empty.SetPoints(pd.GetPoints())
            empty.SetLines(vtk.vtkCellArray())
            return CutLocusResult(lines=empty, method="laplacian", threshold=threshold, num_components=0)
        app.Update()
        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(app.GetOutputPort())
        clean.Update()
        lines = vtk.vtkPolyData()
        lines.ShallowCopy(clean.GetOutput())

    if lines.GetNumberOfCells() == 0:
        return CutLocusResult(lines=lines, method="laplacian", threshold=threshold, num_components=0)

    lines_regions, n_comp = _connectivity_with_regions(lines)
    return CutLocusResult(lines=lines_regions, method="laplacian", threshold=threshold, num_components=n_comp)


def _polyline_total_length(lines: vtk.vtkPolyData) -> float:
    if lines.GetNumberOfLines() == 0:
        return 0.0
    pts_obj = lines.GetPoints()
    if pts_obj is None or pts_obj.GetNumberOfPoints() == 0:
        return 0.0
    pts = nps.vtk_to_numpy(pts_obj.GetData())
    cell_array = lines.GetLines()
    cell_array.InitTraversal()
    ids = vtk.vtkIdList()
    total = 0.0
    while cell_array.GetNextCell(ids):
        if ids.GetNumberOfIds() < 2:
            continue
        for i in range(ids.GetNumberOfIds() - 1):
            a = int(ids.GetId(i))
            b = int(ids.GetId(i + 1))
            total += float(np.linalg.norm(pts[a] - pts[b]))
    return total


_MatrixLike = np.ndarray  # type alias for typing friendliness
