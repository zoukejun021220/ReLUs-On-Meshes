from __future__ import annotations

import numpy as np
import vtk
from vtk.util import numpy_support as nps

from .heat import HeatGeodesic


def geodesic_distance_matrix(
    V: np.ndarray,
    F: np.ndarray,
    seeds: np.ndarray | list[int],
    t: float | None = None,
) -> np.ndarray:
    """Compute per-seed geodesic distances (n x K) via the heat method.

    - V: (n,3) vertices
    - F: (m,3) triangle indices
    - seeds: iterable of K vertex indices
    - t: diffusion time (if None, solver default will be used)
    Returns Phi with shape (n, K).
    """
    seeds = np.asarray(seeds, dtype=np.int32).ravel()
    if seeds.size == 0:
        raise ValueError("seeds must contain at least one vertex id")
    n = int(V.shape[0])
    K = int(seeds.size)
    geo = HeatGeodesic(V, F, t=t)
    Phi = np.empty((n, K), dtype=np.float64)
    for k, s in enumerate(seeds):
        Phi[:, k] = geo.phi_to_subset([int(s)])
    return Phi


def voronoi_labels(Phi: np.ndarray) -> np.ndarray:
    """Argmin over columns for each row; returns (n,) int labels in [0..K-1]."""
    if Phi.ndim != 2:
        raise ValueError("Phi must be (n, K)")
    return np.argmin(Phi, axis=1).astype(np.int32)


def attach_phi_vector(pd: vtk.vtkPolyData, Phi: np.ndarray, name: str = "phi_vec") -> None:
    """Attach an n×K array as a K-component point array to pd for GPU interpolation."""
    n_pts = pd.GetNumberOfPoints()
    if Phi.shape[0] != n_pts:
        raise ValueError(f"Phi rows ({Phi.shape[0]}) must equal number of points ({n_pts})")
    arr = nps.numpy_to_vtk(Phi.astype(np.float64), deep=True)
    arr.SetName(name)
    pd.GetPointData().AddArray(arr)


def bisector_polylines(pd: vtk.vtkPolyData, Phi: np.ndarray, dominance_filter: bool = False) -> vtk.vtkPolyData:
    """Extract pairwise bisector polylines from Phi (n×K) on pd.

    - For each pair (i<j), contour W_ij = Phi[:,i]-Phi[:,j] at 0.
    - If dominance_filter=True, remove segments dominated by other seeds (optional; expensive).
    """
    if Phi.ndim != 2:
        raise ValueError("Phi must be (n, K)")
    n, K = Phi.shape
    if pd.GetNumberOfPoints() != n:
        raise ValueError("pd points must match Phi rows")
    app = vtk.vtkAppendPolyData()
    for i in range(K):
        for j in range(i + 1, K):
            Wij = Phi[:, i] - Phi[:, j]
            arr = nps.numpy_to_vtk(Wij.astype(np.float64), deep=True)
            arr.SetName("W")
            pd.GetPointData().SetScalars(arr)
            cf = vtk.vtkContourFilter()
            cf.SetInputData(pd)
            cf.SetValue(0, 0.0)
            cf.Update()
            out = cf.GetOutput()
            if dominance_filter and K > 2 and out.GetNumberOfPoints() > 0:
                # Filter segments where some other channel k has Phi_k < Phi_i and Phi_k < Phi_j
                # Sample Phi at polyline points via nearest vertex (cheap & robust)
                lines = out
                pts = nps.vtk_to_numpy(lines.GetPoints().GetData())
                verts = nps.vtk_to_numpy(pd.GetPoints().GetData())
                # nearest vertex indices
                from scipy.spatial import cKDTree
                tree = cKDTree(verts)
                _, idx = tree.query(pts, k=1)
                keep_mask = np.ones(idx.size, dtype=bool)
                for k in range(K):
                    if k == i or k == j:
                        continue
                    keep_mask &= (Phi[idx, k] >= Phi[idx, i]) | (Phi[idx, k] >= Phi[idx, j])
                # Build filtered polydata with kept points (approximate; retains topology if mask is all True)
                if not keep_mask.all():
                    sel = vtk.vtkSelectionNode()
                    sel.SetFieldType(vtk.vtkSelectionNode.POINT)
                    sel.SetContentType(vtk.vtkSelectionNode.INDICES)
                    ids = vtk.vtkIdTypeArray()
                    for t, ok in enumerate(keep_mask):
                        if ok:
                            ids.InsertNextValue(t)
                    sel.SetSelectionList(ids)
                    selection = vtk.vtkSelection()
                    selection.AddNode(sel)
                    es = vtk.vtkExtractSelection()
                    es.SetInputData(0, lines)
                    es.SetInputData(1, selection)
                    es.Update()
                    gf = vtk.vtkGeometryFilter()
                    gf.SetInputData(es.GetOutput())
                    gf.Update()
                    app.AddInputData(gf.GetOutput())
                else:
                    app.AddInputData(out)
            else:
                app.AddInputData(out)
    app.Update()
    return app.GetOutput()


def sample_points(V: np.ndarray, F: np.ndarray, Phi: np.ndarray, per_edge: int = 20) -> vtk.vtkPolyData:
    """Sample a barycentric grid in each face, interpolate Phi, and emit colored points by argmin label.

    - per_edge: grid resolution; total samples per triangle ≈ (per_edge+1)(per_edge+2)/2
    """
    if Phi.shape[0] != V.shape[0]:
        raise ValueError("Phi rows must equal number of vertices")
    K = Phi.shape[1]
    pts = []
    labels = []
    for tri in F:
        a, b, c = V[tri]
        Phi_tri = Phi[tri]  # (3, K)
        for i in range(per_edge + 1):
            for j in range(per_edge + 1 - i):
                k = per_edge - i - j
                lam = np.array([i, j, k], dtype=np.float64)
                lam /= float(per_edge)
                p = lam[0] * a + lam[1] * b + lam[2] * c
                d = lam @ Phi_tri  # (K,)
                lbl = int(np.argmin(d))
                pts.append(p)
                labels.append(lbl)
    P = np.asarray(pts, dtype=np.float64)
    L = np.asarray(labels, dtype=np.int32)
    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(nps.numpy_to_vtk(P, deep=True))
    vtp = vtk.vtkPolyData()
    vtp.SetPoints(vtk_pts)
    ca = vtk.vtkCellArray()
    for i in range(P.shape[0]):
        ca.InsertNextCell(1)
        ca.InsertCellPoint(i)
    vtp.SetVerts(ca)
    arr = nps.numpy_to_vtk(L, deep=True)
    arr.SetName("label")
    vtp.GetPointData().AddArray(arr)
    vtp.GetPointData().SetActiveScalars("label")
    return vtp


def categorical_lookup_table(num_labels: int) -> vtk.vtkLookupTable:
    """Generate a basic categorical LUT; deterministic colors for testing/preview."""
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(max(2, int(num_labels)))
    lut.Build()
    rng = np.random.RandomState(1234)
    colors = rng.rand(max(2, int(num_labels)), 3)
    for i in range(max(2, int(num_labels))):
        r, g, b = colors[i]
        lut.SetTableValue(i, float(r), float(g), float(b), 1.0)
    return lut


def _barycentric_grid(per_edge: int):
    pe = int(per_edge)
    lams = []  # (S,3)
    for i in range(pe + 1):
        for j in range(pe + 1 - i):
            k = pe - i - j
            lams.append([i, j, k])
    lams = np.asarray(lams, dtype=np.float64) / float(pe)
    # connectivity per reference triangle grid (two tris per square)
    # index helper: row i has (pe-i+1) items, starting at off[i]
    row_counts = [pe - i + 1 for i in range(pe + 1)]
    off = [0]
    for i in range(pe):
        off.append(off[-1] + row_counts[i])
    cells = []
    def ID(ii: int, jj: int) -> int:
        return off[ii] + jj
    for i in range(pe):
        for j in range(pe - i):
            v00 = ID(i, j)
            v10 = ID(i + 1, j)
            v01 = ID(i, j + 1)
            v11 = ID(i + 1, j + 1)
            cells.append([v00, v10, v01])
            if j + 1 <= pe - (i + 1):
                cells.append([v10, v11, v01])
    cells = np.asarray(cells, dtype=np.int64)
    return lams, cells, np.asarray(off, dtype=np.int64)


def subdivide_voronoi_polydata(
    V: np.ndarray,
    F: np.ndarray,
    Phi: np.ndarray,
    per_edge: int = 20,
    keep_phi_vec: bool = False,
    use_gpu: bool = False,
    block_faces: int = 2048,
) -> vtk.vtkPolyData:
    """Uniformly subdivide each triangle into a barycentric grid and color by
    Voronoi label computed from interpolated distances.

    - V: (n,3) vertices
    - F: (m,3) triangle indices
    - Phi: (n,K) per-vertex distances to K seeds
    - per_edge: number of subdivisions along each edge
    Returns a new vtkPolyData with triangles and point array 'label'. If
    keep_phi_vec=True, also adds a K-component 'phi_vec' at points.
    """
    if Phi.shape[0] != V.shape[0]:
        raise ValueError("Phi rows must equal number of vertices")
    K = int(Phi.shape[1])
    n = int(V.shape[0])
    m = int(F.shape[0])
    pe = int(per_edge)
    if pe < 1:
        raise ValueError("per_edge must be >= 1")

    # Build reference grid once
    lam_ref, cells_ref, _ = _barycentric_grid(pe)
    S = lam_ref.shape[0]  # samples per face
    T = cells_ref.shape[0]  # tris per face

    pts = []
    labels = []
    phi_acc = [] if keep_phi_vec else None
    conns = []

    faces = F.astype(np.int64)
    Nf = faces.shape[0]

    # Optional GPU path (CuPy)
    xp = np
    if use_gpu:
        try:
            import cupy as cp  # type: ignore
            xp = cp
        except Exception:
            xp = np
            use_gpu = False

    # Process in face blocks to bound memory
    import warnings
    for start in range(0, Nf, int(block_faces)):
        end = min(Nf, start + int(block_faces))
        Fblk = faces[start:end]  # (B,3)
        B = Fblk.shape[0]
        Vtri = V[Fblk]  # (B,3,3)
        Ptri = Phi[Fblk]  # (B,3,K)

        if use_gpu:
            try:
                lam = xp.asarray(lam_ref)  # (S,3)
                Vx = xp.asarray(Vtri)
                Px = xp.asarray(Ptri)
                # points: (B,S,3) = (B,3,3) @ (S,3)^T
                Pblk = xp.einsum('bij,sj->bsi', Vx, lam)
                # distances: (B,S,K) = (B,3,K) and (S,3)
                Dblk = xp.einsum('bik,sj->bsi', Px.transpose(0,2,1), lam)  # Px (B,K,3) contracted with lam (S,3)
                Lblk = xp.argmin(Dblk, axis=2).astype(xp.int32)  # (B,S)
                if xp is not np:
                    Pblk = xp.asnumpy(Pblk)
                    Lblk = xp.asnumpy(Lblk)
            except Exception as e:
                warnings.warn(f"CuPy GPU fallback to CPU: {e}")
                use_gpu = False
                lam = lam_ref
                Pblk = np.einsum('bij,sj->bsi', Vtri, lam)
                Dblk = np.einsum('bkj,sj->bsk', Ptri.transpose(0,2,1), lam)
                Lblk = np.argmin(Dblk, axis=2).astype(np.int32)
        else:
            lam = lam_ref  # (S,3)
            # points
            Pblk = np.einsum('bij,sj->bsi', Vtri, lam)
            # distances (B,S,K): D[b,s,:] = lam[s] @ Ptri[b]
            Dblk = np.einsum('bkj,sj->bsk', Ptri.transpose(0,2,1), lam)
            Lblk = np.argmin(Dblk, axis=2).astype(np.int32)

        base = len(pts)
        pts.append(Pblk.reshape(-1, 3))
        labels.append(Lblk.reshape(-1))
        if keep_phi_vec:
            if use_gpu:
                try:
                    # Try to compute on device then bring to host
                    Px = xp.asarray(Ptri)
                    lam = xp.asarray(lam_ref)
                    Dg = xp.einsum('bik,sj->bsi', Px.transpose(0,2,1), lam)
                    Dnp = xp.asnumpy(Dg)
                except Exception:
                    Dnp = np.einsum('bkj,sj->bsk', Ptri.transpose(0,2,1), lam_ref)
                phi_acc.append(Dnp.reshape(-1, Dnp.shape[2]))
            else:
                phi_acc.append(Dblk.reshape(-1, Dblk.shape[2]))

        # Connectivity for this block
        # For each face b in 0..B-1, add cells_ref + b*S + base
        for b in range(B):
            off = base + b * S
            conns.append((cells_ref + off).astype(np.int64))

    P = np.concatenate(pts, axis=0)
    L = np.concatenate(labels, axis=0)
    C = np.concatenate(conns, axis=0)

    vtk_pts = vtk.vtkPoints()
    vtk_pts.SetData(nps.numpy_to_vtk(P, deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_pts)

    ca = vtk.vtkCellArray()
    for tri in C:
        idl = vtk.vtkIdList(); idl.SetNumberOfIds(3)
        idl.SetId(0, int(tri[0])); idl.SetId(1, int(tri[1])); idl.SetId(2, int(tri[2]))
        ca.InsertNextCell(idl)
    pd.SetPolys(ca)

    arr_lbl = nps.numpy_to_vtk(L, deep=True)
    arr_lbl.SetName("label")
    pd.GetPointData().AddArray(arr_lbl)
    pd.GetPointData().SetActiveScalars("label")

    if keep_phi_vec:
        arr_phi = nps.numpy_to_vtk(np.concatenate(phi_acc, axis=0).astype(np.float64), deep=True)
        arr_phi.SetName("phi_vec")
        pd.GetPointData().AddArray(arr_phi)

    return pd
