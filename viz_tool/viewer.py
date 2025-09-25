from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import vtk
from vtk.util import numpy_support as nps

from .loader import load_surface
from .expr import to_glsl
from .shader import build_varyings, build_impl


def _read_seed_file(path: str, default_label: str) -> Tuple[List[int], List[str]]:
    indices: List[int] = []
    labels: List[str] = []
    try:
        with open(path, "r", encoding="utf8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                for comment in ("//", "#"):
                    if comment in line:
                        line = line.split(comment, 1)[0].strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                if not parts:
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                label = " ".join(parts[1:]) if len(parts) > 1 else default_label
                indices.append(idx)
                labels.append(label if label else default_label)
    except FileNotFoundError:
        raise
    return indices, labels


def _choose_out_var() -> str:
    try:
        vmaj = int(vtk.vtkVersion.GetVTKMajorVersion())
        vmin = int(vtk.vtkVersion.GetVTKMinorVersion())
    except Exception:
        vmaj, vmin = 9, 1
    return "fragOutput0" if (vmaj > 9 or (vmaj == 9 and vmin >= 1)) else "gl_FragData[0]"


def _seam_polydata_from_fields(
    V: np.ndarray,
    F: np.ndarray,
    field: np.ndarray,
    eps: float = 1e-12,
) -> vtk.vtkPolyData:
    field = np.asarray(field)
    if field.ndim == 1:
        field = field.reshape(-1, 1)
    if field.ndim != 2:
        return vtk.vtkPolyData()
    if F.size == 0:
        return vtk.vtkPolyData()

    T = F.shape[0]
    C = field.shape[1]
    if C < 2:
        return vtk.vtkPolyData()

    field_faces = field[F]
    face_means = field_faces.mean(axis=1)
    idx = np.argpartition(face_means, kth=1, axis=1)[:, :2]
    first = face_means[np.arange(T), idx[:, 0]]
    second = face_means[np.arange(T), idx[:, 1]]
    swap = second < first
    i = idx[:, 0].copy()
    j = idx[:, 1].copy()
    i[swap], j[swap] = j[swap], i[swap]

    gi = np.take_along_axis(field_faces, i[:, None, None], axis=2).squeeze(2)
    gj = np.take_along_axis(field_faces, j[:, None, None], axis=2).squeeze(2)
    g = gi - gj

    edges = ((0, 1), (1, 2), (2, 0))
    segments: list[np.ndarray] = []
    seg_lengths: list[int] = []
    for t in range(T):
        pts: list[np.ndarray] = []
        for a, b in edges:
            ga = float(g[t, a])
            gb = float(g[t, b])
            if abs(ga) <= eps and abs(gb) <= eps:
                continue
            if ga * gb <= 0.0:
                denom = ga - gb
                if abs(denom) <= eps:
                    continue
                ta = ga / denom
                if -eps <= ta <= 1.0 + eps:
                    ta_clamped = min(max(ta, 0.0), 1.0)
                    Pa = (1.0 - ta_clamped) * V[F[t, a]] + ta_clamped * V[F[t, b]]
                    pts.append(Pa.astype(np.float64))
        if len(pts) == 2:
            segments.extend(pts)
            seg_lengths.append(2)
        elif len(pts) == 3:
            d01 = np.linalg.norm(pts[0] - pts[1])
            d12 = np.linalg.norm(pts[1] - pts[2])
            d20 = np.linalg.norm(pts[2] - pts[0])
            if d01 >= d12 and d01 >= d20:
                pair = (pts[0], pts[1])
            elif d12 >= d20:
                pair = (pts[1], pts[2])
            else:
                pair = (pts[2], pts[0])
            segments.extend(pair)
            seg_lengths.append(2)

    poly = vtk.vtkPolyData()
    if not segments:
        return poly

    seg_points = np.asarray(segments, dtype=np.float64)
    pts = vtk.vtkPoints()
    pts.SetData(nps.numpy_to_vtk(seg_points, deep=True))
    poly.SetPoints(pts)

    cells = vtk.vtkCellArray()
    base = 0
    for n in seg_lengths:
        cells.InsertNextCell(n)
        for k in range(n):
            cells.InsertCellPoint(base + k)
        base += n
    poly.SetLines(cells)
    return poly


def visualize(
    path: str,
    arrays: List[str],
    expr: str,
    mode: str = "scalar",
    num_labels: int | None = None,
    warp_mode: str = "identity",
    warp_power: float = 1.0,
    triangulate: bool = False,
    offscreen: bool = False,
    screenshot: str | None = None,
    show_edges: bool = False,
    show_points: bool = False,
    point_color: str = "black",
    point_size: float = 3.0,
    mark_seeds: bool = False,
    seed_array: str | None = None,
    seed_color: str = "red",
    seed_size: float = 8.0,
    seeds: List[int] | None = None,
    seed_file: str | None = None,
    seed_label: str = "seed",
    show_seams: bool = False,
) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    pd, V, F = load_surface(path, triangulate=triangulate)
    mark_seeds = mark_seeds or bool(seed_file) or bool(seeds)

    # Mapper and actor
    mapper = vtk.vtkOpenGLPolyDataMapper()
    mapper.SetInputData(pd)
    mapper.ScalarVisibilityOff()

    comps: list[int] = []
    component_names: list[list[str]] = []
    n_pts = pd.GetNumberOfPoints()

    for i, nm in enumerate(arrays):
        arr = pd.GetPointData().GetArray(nm)
        if arr is None:
            raise RuntimeError(f"Point array not found: {nm}")
        comp_total = int(arr.GetNumberOfComponents())
        comps.append(comp_total)
        if comp_total == 0:
            component_names.append([])
            continue
        data = nps.vtk_to_numpy(arr).reshape(n_pts, comp_total)
        names: list[str] = []
        for j in range(comp_total):
            comp_data = data[:, j].astype(np.float32)
            vtk_arr = nps.numpy_to_vtk(comp_data, deep=True)
            vtk_arr.SetNumberOfComponents(1)
            comp_name = f"__viz_comp_f{i}_{j}"
            vtk_arr.SetName(comp_name)
            pd.GetPointData().AddArray(vtk_arr)
            mapper.MapDataArrayToVertexAttribute(
                f"attr_f{i}_{j}",
                comp_name,
                vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                -1,
            )
            names.append(comp_name)
        component_names.append(names)

    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.LightingOff()
    prop.SetInterpolationToGouraud()
    prop.SetAmbient(1.0)
    prop.SetDiffuse(0.0)
    prop.SetSpecular(0.0)

    # Build shader code
    vdec, vimpl, fdec = build_varyings(
        arrays,
        comps,
        warp_mode=warp_mode,
        warp_power=warp_power,
    )
    array_sizes = {f"f{i}": comps[i] for i in range(len(arrays))}
    expr_glsl = to_glsl(expr, array_sizes)
    for i, total in enumerate(comps):
        if total > 4:
            expr_glsl = expr_glsl.replace(f"argmin(f{i})", f"f{i}_argmin()")
            expr_glsl = expr_glsl.replace(f"argmax(f{i})", f"f{i}_argmax()")

    # Estimate scalar range for mapping if needed (use first array as proxy)
    datamin = datamax = None
    if mode == "scalar":
        try:
            arr0 = pd.GetPointData().GetArray(arrays[0])
            vals = nps.vtk_to_numpy(arr0)
            datamin = float(np.nanmin(vals))
            datamax = float(np.nanmax(vals))
        except Exception:
            datamin, datamax = 0.0, 1.0

    max_labels = max(comps, default=1)
    auto_labels = num_labels if num_labels is not None else max(1, max_labels)

    fimpl_body = build_impl(expr_glsl, mode, datamin, datamax, auto_labels)
    out_var = _choose_out_var()

    sp = actor.GetShaderProperty()
    sp.AddVertexShaderReplacement("//VTK::Color::Dec", True, vdec, False)
    sp.AddVertexShaderReplacement("//VTK::Color::Impl", True, vimpl, False)
    sp.AddFragmentShaderReplacement("//VTK::Color::Dec", True, fdec, False)
    sp.AddFragmentShaderReplacement(
        "//VTK::Color::Impl",
        True,
        fimpl_body + f"{out_var} = vec4(outColor, 1.0); return;\n",
        False,
    )

    # Render window
    ren = vtk.vtkRenderer(); ren.SetBackground(1, 1, 1)
    ren.AddActor(actor)
    if show_edges:
        edge_mapper = vtk.vtkPolyDataMapper(); edge_mapper.SetInputData(pd)
        edge_mapper.ScalarVisibilityOff()
        edge_actor = vtk.vtkActor(); edge_actor.SetMapper(edge_mapper)
        edge_actor.GetProperty().SetRepresentationToWireframe()
        prop_edge = edge_actor.GetProperty()
        prop_edge.SetColor(0.0, 0.0, 0.0)
        prop_edge.SetLineWidth(1.0)
        prop_edge.LightingOff()
        prop_edge.SetAmbient(1.0)
        prop_edge.SetDiffuse(0.0)
        prop_edge.SetSpecular(0.0)
        prop_edge.SetInterpolationToFlat()
        prop_edge.RenderLinesAsTubesOn()
        ren.AddActor(edge_actor)
    if show_points:
        vgf = vtk.vtkVertexGlyphFilter(); vgf.SetInputData(pd); vgf.Update()
        pmapper = vtk.vtkPolyDataMapper(); pmapper.SetInputData(vgf.GetOutput())
        pactor = vtk.vtkActor(); pactor.SetMapper(pmapper)
        # parse color
        col = (0.0, 0.0, 0.0)
        name = (point_color or "").strip().lower()
        if name in ("white", "w"):
            col = (1.0, 1.0, 1.0)
        elif name in ("red", "r"):
            col = (1.0, 0.0, 0.0)
        elif name in ("green", "g"):
            col = (0.0, 1.0, 0.0)
        elif name in ("blue", "b"):
            col = (0.0, 0.0, 1.0)
        pactor.GetProperty().SetColor(*col)
        pactor.GetProperty().SetPointSize(float(point_size))
        pactor.GetProperty().LightingOff()
        ren.AddActor(pactor)

    rw = vtk.vtkRenderWindow(); rw.SetSize(1024, 768)
    rw.SetMultiSamples(8)
    rw.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(rw)

    # Seed overlay
    if mark_seeds:
        total_pts = pd.GetNumberOfPoints()
        seed_entries: dict[int, str] = {}

        def add_seed(idx: int, label: str) -> None:
            if 0 <= idx < total_pts and idx not in seed_entries:
                seed_entries[idx] = label if label else seed_label

        if seeds:
            for val in seeds:
                try:
                    add_seed(int(val), seed_label)
                except (TypeError, ValueError):
                    continue

        if seed_file:
            try:
                file_indices, file_labels = _read_seed_file(seed_file, seed_label)
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Seed file not found: {seed_file}") from exc
            for idx, lbl in zip(file_indices, file_labels):
                add_seed(idx, lbl)

        if not seed_entries:
            candidates = [seed_array] if seed_array else ["seed_mask", "is_seed", "source_mask", "seeds", "seed_id"]
            for nm in candidates:
                if not nm:
                    continue
                arr = pd.GetPointData().GetArray(nm)
                if arr is None:
                    continue
                vals = nps.vtk_to_numpy(arr)
                try:
                    mask = vals.astype(bool)
                except Exception:
                    mask = vals > 0
                for idx, flag in enumerate(mask):
                    if flag:
                        add_seed(int(idx), seed_label)
                if seed_entries:
                    break

        if seed_entries:
            ordered_idx = list(seed_entries.keys())
            ordered_labels = [seed_entries[i] for i in ordered_idx]

            vpts = vtk.vtkPoints()
            for i in ordered_idx:
                vpts.InsertNextPoint(pd.GetPoint(i))
            spd = vtk.vtkPolyData(); spd.SetPoints(vpts)
            ca = vtk.vtkCellArray()
            for i in range(len(ordered_idx)):
                ca.InsertNextCell(1); ca.InsertCellPoint(i)
            spd.SetVerts(ca)

            label_array = vtk.vtkStringArray(); label_array.SetName("seed_labels")
            for lbl in ordered_labels:
                label_array.InsertNextValue(lbl if lbl else seed_label)
            spd.GetPointData().AddArray(label_array)

            smap = vtk.vtkPolyDataMapper(); smap.SetInputData(spd)
            sact = vtk.vtkActor(); sact.SetMapper(smap)
            scol = (1.0, 0.2, 0.2)
            name = (seed_color or "").strip().lower()
            if name in ("black", "k"):
                scol = (0.0, 0.0, 0.0)
            elif name in ("white", "w"):
                scol = (1.0, 1.0, 1.0)
            elif name in ("green", "g"):
                scol = (0.0, 0.8, 0.2)
            elif name in ("blue", "b"):
                scol = (0.2, 0.4, 1.0)
            sact.GetProperty().SetColor(*scol)
            sact.GetProperty().SetPointSize(float(seed_size))
            sact.GetProperty().LightingOff()
            ren.AddActor(sact)

            if any(lbl for lbl in ordered_labels):
                label_mapper = vtk.vtkLabeledDataMapper()
                label_mapper.SetInputData(spd)
                label_mapper.SetLabelModeToLabelFieldData()
                label_mapper.SetFieldDataName("seed_labels")
                text_prop = label_mapper.GetLabelTextProperty()
                text_prop.SetColor(*scol)
                text_prop.SetBold(True)
                label_actor = vtk.vtkActor2D(); label_actor.SetMapper(label_mapper)
                ren.AddActor(label_actor)

    if show_seams and arrays:
        arr = pd.GetPointData().GetArray(arrays[0])
        if arr is not None:
            try:
                field_np = nps.vtk_to_numpy(arr).reshape(n_pts, -1)
                seam_pd = _seam_polydata_from_fields(V, F, field_np)
                if seam_pd.GetNumberOfPoints() > 0:
                    seam_mapper = vtk.vtkPolyDataMapper(); seam_mapper.SetInputData(seam_pd)
                    seam_actor = vtk.vtkActor(); seam_actor.SetMapper(seam_mapper)
                    seam_prop = seam_actor.GetProperty()
                    seam_prop.SetColor(0.0, 0.0, 0.0)
                    seam_prop.SetLineWidth(2.0)
                    seam_prop.LightingOff()
                    ren.AddActor(seam_actor)
            except Exception:
                pass

    rw.Render()
    if screenshot:
        w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(rw); w2i.Update()
        wr = vtk.vtkPNGWriter(); wr.SetFileName(screenshot); wr.SetInputData(w2i.GetOutput()); wr.Write()
    if not offscreen:
        style = vtk.vtkInteractorStyleTrackballCamera()
        iren.SetInteractorStyle(style)
        iren.Initialize(); iren.Start()
