from __future__ import annotations

import argparse
import os
from typing import Iterable

import vtk
from vtk.util import numpy_support as nps


def _read_polydata(path: str) -> vtk.vtkPolyData:
    lower = path.lower()
    if lower.endswith(".vtp"):
        r = vtk.vtkXMLPolyDataReader()
    elif lower.endswith(".vtk"):
        r = vtk.vtkPolyDataReader()
    else:
        raise ValueError("Provide a .vtp or legacy .vtk PolyData file")
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def _install_argmax_shader(
    actor,
    num_comp: int,
    attr_name: str = "phiAttr",
    warp_mode: str = "identity",
    warp_power: float = 1.0,
) -> None:
    import vtk

    d = int(num_comp)
    if d < 1:
        raise ValueError("fragment argmax shader requires at least one component; got %d" % d)

    warp_mode_norm = (warp_mode or "identity").strip().lower()
    if warp_mode_norm not in {"identity", "sqrt", "log", "power"}:
        raise ValueError(f"Unsupported warp mode '{warp_mode}'.")
    if warp_mode_norm == "power":
        if warp_power <= 0.0:
            raise ValueError("warp_power must be positive when warp_mode='power'.")
        warp_power = float(warp_power)

    const_decl = f"const int PHI_DIM = {d};\n"
    # Declare one vertex attribute per component and forward to fragment stage via array varying
    lines_dec = [const_decl]
    for i in range(d):
        lines_dec.append(f"in float {attr_name}{i};")
    lines_dec.append("out float v_phi[PHI_DIM];")
    vdec = "\n".join(lines_dec) + "\n"
    vimpl = "\n".join([f"v_phi[{i}] = {attr_name}{i};" for i in range(d)]) + "\n"

    warp_funcs = {
        "identity": "float warp_value(float v) { return v; }\n",
        "sqrt": "float warp_value(float v) { return sqrt(max(v, 0.0)); }\n",
        "log": "float warp_value(float v) { return log(max(v, 1e-6)); }\n",
    }
    if warp_mode_norm == "power":
        power_str = ("%.8f" % warp_power).rstrip("0").rstrip(".")
        if not power_str:
            power_str = "0.0"
        warp_funcs["power"] = (
            f"float warp_value(float v) {{ return pow(max(v, 0.0), {power_str}); }}\n"
        )
    warp_decl = warp_funcs[warp_mode_norm]

    # Fragment: declare varying, helpers, and compute argmin with warp_value applied post-interpolation
    fdec = (
        const_decl
        + "in float v_phi[PHI_DIM];\n"
        + "// Define fallbacks so compilation succeeds regardless of lighting blocks\n"
        + "const vec3 ambientColor = vec3(0.0);\n"
        + "const vec3 diffuseColor = vec3(0.0);\n"
        + "const float opacity = 1.0;\n"
        + warp_decl
        + "vec3 hsv2rgb(vec3 c){\n"
        + "  vec3 rgb = clamp(abs(mod(c.x*6.0 + vec3(0.0,4.0,2.0), 6.0)-3.0)-1.0, 0.0, 1.0);\n"
        + "  return c.z * mix(vec3(1.0), rgb, c.y);\n"
        + "}\n"
        + "vec3 label_color(int lbl){\n"
        + "  float h = fract(float(lbl) * 0.61803398875);\n"
        + "  return hsv2rgb(vec3(h, 0.65, 0.95));\n"
        + "}\n"
    )
    fimpl_core_lines = [
        "int lbl = 0;",
        "float best = warp_value(v_phi[0]);",
        "for (int i = 1; i < PHI_DIM; ++i) {",
        "  float cand = warp_value(v_phi[i]);",
        "  if (cand < best) { best = cand; lbl = i; }",
        "}",
        "vec3 col = label_color(lbl);",
    ]
    fimpl = "\n".join(fimpl_core_lines) + "\n"

    # Choose a compatible fragment output variable across VTK versions
    # - VTK 9.1+ uses 'fragOutput0'
    # - Older VTK often uses 'gl_FragData[0]' (or gl_FragColor)
    try:
        vmaj = int(vtk.vtkVersion.GetVTKMajorVersion())
        vmin = int(vtk.vtkVersion.GetVTKMinorVersion())
    except Exception:
        vmaj, vmin = 9, 1
    if vmaj > 9 or (vmaj == 9 and vmin >= 1):
        out_var = "fragOutput0"
    else:
        out_var = "gl_FragData[0]"

    sp = actor.GetShaderProperty()
    # Declarations & assignment in vertex shader color sections to ensure availability
    sp.AddVertexShaderReplacement("//VTK::Color::Dec", True, vdec, False)
    sp.AddVertexShaderReplacement("//VTK::Color::Impl", True, vimpl, False)
    # In fragment, declare our varying early and emit final color immediately
    sp.AddFragmentShaderReplacement("//VTK::Color::Dec", True, fdec, False)
    sp.AddFragmentShaderReplacement("//VTK::Color::Impl", True, fimpl + f"{out_var} = vec4(col, 1.0); return;\n", False)


def _try_pyvista_show(
    mesh_paths: list[str],
    contours_paths: list[str] | None,
    scalars: str,
    screenshot: str | None,
    offscreen: bool,
    show_edges: bool,
    edge_color: str,
    edge_width: float,
    show_points: bool,
    point_color: str,
    point_size: float,
    mark_seeds: bool,
    seed_array: str | None,
    seed_color: str,
    seed_size: float,
    fragment_argmax: bool,
    phi_vec_name: str,
    cell_argmax: bool,
    fragment_warp_mode: str,
    fragment_warp_power: float,
) -> bool:
    try:
        import pyvista as pv
    except Exception:
        return False
    try:
        if offscreen:
            try:
                # Prefer theme flag; keep xvfb as a best-effort fallback
                pv.global_theme.off_screen = True
                pv.start_xvfb()
            except Exception:
                pass
        pl = pv.Plotter()
        pl.set_background("white")
        any_loaded = False
        for i, mp in enumerate(mesh_paths):
            if not os.path.exists(mp):
                print(f"Mesh not found: {mp}")
                continue
            mesh = pv.read(mp)
            any_loaded = True
            # Regular scalar coloring vs. per-fragment argmax vs. per-cell fallback
            if cell_argmax:
                import numpy as _np
                from vtk.util import numpy_support as _nps
                pd = None
                try:
                    import vtk as _vtk
                    rdr = _vtk.vtkXMLPolyDataReader(); rdr.SetFileName(mp); rdr.Update(); pd = rdr.GetOutput()
                except Exception:
                    pass
                if pd is None:
                    raise RuntimeError("Failed to load polydata for cell-argmax")
                arr = pd.GetPointData().GetArray(phi_vec_name)
                if arr is None:
                    raise RuntimeError(f"'{phi_vec_name}' not found for cell-argmax")
                Phi = _nps.vtk_to_numpy(arr)  # (n,K)
                ca = _nps.vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1,4)[:,1:4]
                cent = Phi[ca].mean(axis=1)  # (m,K)
                labels = _np.argmin(cent, axis=1).astype(_np.int32)
                mesh.cell_data['label_cell'] = labels
                pl.add_mesh(mesh, scalars='label_cell', cmap='glasbey', show_scalar_bar=True, smooth_shading=False)
                if show_edges:
                    pl.add_mesh(mesh, style='wireframe', color=edge_color, line_width=edge_width, lighting=False)
            elif fragment_argmax:
                if phi_vec_name not in mesh.point_data:
                    raise RuntimeError(f"fragment-argmax requested but '{phi_vec_name}' not found in point_data.")
                # Build a native VTK actor + OpenGL mapper to access attribute mapping API
                import vtk
                rdr = vtk.vtkXMLPolyDataReader(); rdr.SetFileName(mp); rdr.Update()
                pd = rdr.GetOutput()
                mapper = vtk.vtkOpenGLPolyDataMapper()
                mapper.SetInputData(pd)
                mapper.ScalarVisibilityOff()
                actor = vtk.vtkActor(); actor.SetMapper(mapper)
                # Map point-data array components individually to vertex attributes phiAttr0..phiAttr{d-1}
                K = pd.GetPointData().GetArray(phi_vec_name).GetNumberOfComponents()
                for i_c in range(K):
                    mapper.MapDataArrayToVertexAttribute(f"phiAttr{i_c}", phi_vec_name, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, i_c)
                _install_argmax_shader(
                    actor,
                    K,
                    attr_name="phiAttr",
                    warp_mode=fragment_warp_mode,
                    warp_power=fragment_warp_power,
                )
                # Flat, unlit shading to avoid diffuse lighting variations
                prop = actor.GetProperty(); prop.LightingOff(); prop.SetInterpolationToFlat(); prop.SetDiffuse(0.0); prop.SetAmbient(1.0); prop.SetSpecular(0.0)
                pl.add_actor(actor)
                # Separate wireframe overlay so edges render on top
                if show_edges:
                    pl.add_mesh(mesh, style="wireframe", color=edge_color, line_width=edge_width, lighting=False)
            else:
                kwargs = dict(smooth_shading=True)
                if scalars in mesh.point_data:
                    kwargs.update(dict(scalars=scalars, show_scalar_bar=(i == 0)))
                if show_edges:
                    kwargs.update(dict(show_edges=True, edge_color=edge_color, line_width=edge_width))
                pl.add_mesh(mesh, **kwargs)
            if show_points:
                pl.add_points(mesh.points, color=point_color, point_size=point_size, render_points_as_spheres=True)
            # Mark seeds if present
            if mark_seeds:
                seed_names = [seed_array] if seed_array else [
                    "seed_mask", "is_seed", "source_mask", "seeds", "seed_id"
                ]
                name = next((nm for nm in seed_names if nm and (nm in mesh.point_data)), None)
                if name:
                    mask = mesh.point_data[name]
                    try:
                        sel = mask.astype(bool)
                    except Exception:
                        sel = mask > 0
                    pts = mesh.points[sel]
                    if pts.size > 0:
                        pl.add_points(pts, color=seed_color, point_size=seed_size, render_points_as_spheres=True)
        if not any_loaded:
            raise FileNotFoundError("None of the mesh paths could be loaded.")
        if contours_paths:
            for cp in contours_paths:
                if cp and os.path.exists(cp):
                    try:
                        lines = pv.read(cp)
                        pl.add_mesh(lines, color="white", line_width=2)
                    except Exception:
                        pass
        pl.show(screenshot=screenshot)
        return True
    except Exception as e:
        # Fallback to pure VTK path in calling code
        print(f"[viewer] PyVista path failed: {e}")
        return False


def _parse_color(color: str) -> tuple[float, float, float]:
    name = (color or "").strip().lower()
    if name in ("black", "k"):
        return (0.0, 0.0, 0.0)
    if name in ("white", "w"):
        return (1.0, 1.0, 1.0)
    if name.startswith("#") and len(name) == 7:
        r = int(name[1:3], 16) / 255.0
        g = int(name[3:5], 16) / 255.0
        b = int(name[5:7], 16) / 255.0
        return (r, g, b)
    # default
    return (0.0, 0.0, 0.0)


def _vtk_show(
    mesh_paths: list[str],
    contours_paths: list[str] | None,
    scalars: str,
    screenshot: str | None,
    offscreen: bool,
    show_edges: bool,
    edge_color: str,
    edge_width: float,
    show_points: bool,
    point_color: str,
    point_size: float,
    mark_seeds: bool,
    seed_array: str | None,
    seed_color: str,
    seed_size: float,
    fragment_argmax: bool,
    phi_vec_name: str,
    cell_argmax: bool,
    fragment_warp_mode: str,
    fragment_warp_power: float,
) -> None:
    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    any_loaded = False
    for i, mp in enumerate(mesh_paths):
        if not os.path.exists(mp):
            print(f"Mesh not found: {mp}")
            continue
        pd = _read_polydata(mp)
        any_loaded = True
        if cell_argmax:
            arr = pd.GetPointData().GetArray(phi_vec_name)
            if arr is None:
                raise RuntimeError(f"cell-argmax requested but '{phi_vec_name}' not found on {mp}")
            from vtk.util import numpy_support as _nps
            import numpy as _np
            Phi = _nps.vtk_to_numpy(arr)
            ca = _nps.vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1,4)[:,1:4]
            labels = _np.argmin(Phi[ca].mean(axis=1), axis=1).astype(_np.int32)
            lbl = _nps.numpy_to_vtk(labels, deep=True); lbl.SetName('label_cell')
            pd.GetCellData().AddArray(lbl)
            mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(pd)
            mapper.SetScalarModeToUseCellData(); mapper.SelectColorArray('label_cell'); mapper.SetScalarVisibility(True)
            actor = vtk.vtkActor(); actor.SetMapper(mapper)
        elif fragment_argmax:
            arr = pd.GetPointData().GetArray(phi_vec_name)
            if arr is None:
                raise RuntimeError(f"fragment-argmax requested but '{phi_vec_name}' not found on {mp}")
            K = int(arr.GetNumberOfComponents())
            mapper = vtk.vtkOpenGLPolyDataMapper()
            mapper.SetInputData(pd)
            mapper.ScalarVisibilityOff()
            # Map components individually to attributes phiAttr0..phiAttr{d-1}
            for i_c in range(K):
                mapper.MapDataArrayToVertexAttribute(f"phiAttr{i_c}", phi_vec_name, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, i_c)
            actor = vtk.vtkActor(); actor.SetMapper(mapper)
            _install_argmax_shader(
                actor,
                K,
                attr_name="phiAttr",
                warp_mode=fragment_warp_mode,
                warp_power=fragment_warp_power,
            )
            # Flat, unlit shading for crisp patches
            prop = actor.GetProperty()
            prop.LightingOff(); prop.SetInterpolationToFlat(); prop.SetDiffuse(0.0); prop.SetAmbient(1.0); prop.SetSpecular(0.0)
        else:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pd)
            mapper.SetScalarModeToUsePointData()
            mapper.SetColorModeToMapScalars()
            if scalars:
                arr = pd.GetPointData().GetArray(scalars)
                if arr is not None:
                    pd.GetPointData().SetActiveScalars(scalars)
            if pd.GetPointData().GetScalars() is not None:
                mapper.SetScalarRange(pd.GetPointData().GetScalars().GetRange())
                mapper.SetScalarVisibility(True)
        if not fragment_argmax:
            actor = vtk.vtkActor(); actor.SetMapper(mapper)
        ren.AddActor(actor)

        # Separate wireframe overlay to ensure edges are visible with custom shader
        if show_edges:
            edge_mapper = vtk.vtkPolyDataMapper(); edge_mapper.SetInputData(pd)
            edge_actor = vtk.vtkActor(); edge_actor.SetMapper(edge_mapper)
            edge_actor.GetProperty().SetRepresentationToWireframe()
            ec = _parse_color(edge_color)
            edge_actor.GetProperty().SetColor(*ec)
            edge_actor.GetProperty().SetLineWidth(float(edge_width))
            edge_actor.GetProperty().LightingOff()
            ren.AddActor(edge_actor)

        if show_points:
            vgf = vtk.vtkVertexGlyphFilter()
            vgf.SetInputData(pd)
            vgf.Update()
            pmapper = vtk.vtkPolyDataMapper()
            pmapper.SetInputData(vgf.GetOutput())
            pactor = vtk.vtkActor()
            pactor.SetMapper(pmapper)
            pc = _parse_color(point_color)
            pactor.GetProperty().SetColor(*pc)
            pactor.GetProperty().SetPointSize(float(point_size))
            ren.AddActor(pactor)

        # Seed markers
        if mark_seeds:
            seed_names = [seed_array] if seed_array else [
                "seed_mask", "is_seed", "source_mask", "seeds", "seed_id"
            ]
            name = None
            for nm in seed_names:
                if nm:
                    arr = pd.GetPointData().GetArray(nm)
                    if arr is not None:
                        name = nm
                        break
            if name is not None:
                arr = pd.GetPointData().GetArray(name)
                vals = nps.vtk_to_numpy(arr)
                try:
                    sel = vals.astype(bool)
                except Exception:
                    sel = vals > 0
                idx = (sel.nonzero()[0]).astype(int)
                if idx.size > 0:
                    pts = vtk.vtkPoints()
                    pts.SetNumberOfPoints(idx.size)
                    orig_pts = pd.GetPoints()
                    for j, ii in enumerate(idx):
                        pts.SetPoint(j, orig_pts.GetPoint(int(ii)))
                    vpoly = vtk.vtkPolyData()
                    vpoly.SetPoints(pts)
                    ca = vtk.vtkCellArray()
                    for j in range(idx.size):
                        ca.InsertNextCell(1)
                        ca.InsertCellPoint(j)
                    vpoly.SetVerts(ca)
                    pmapper2 = vtk.vtkPolyDataMapper()
                    pmapper2.SetInputData(vpoly)
                    pactor2 = vtk.vtkActor()
                    pactor2.SetMapper(pmapper2)
                    sc = _parse_color(seed_color)
                    pactor2.GetProperty().SetColor(*sc)
                    pactor2.GetProperty().SetPointSize(float(seed_size))
                    ren.AddActor(pactor2)

    # Optional contours overlay
    if contours_paths:
        for cp in contours_paths:
            if cp and os.path.exists(cp):
                cr = _read_polydata(cp)
                cmapper = vtk.vtkPolyDataMapper()
                cmapper.SetInputData(cr)
                cact = vtk.vtkActor()
                cact.SetMapper(cmapper)
                cact.GetProperty().SetColor(1.0, 1.0, 1.0)
                cact.GetProperty().SetLineWidth(2.0)
                ren.AddActor(cact)

    if not any_loaded:
        raise FileNotFoundError("None of the mesh paths could be loaded.")

    rw = vtk.vtkRenderWindow()
    # Use a larger default window for clearer screenshots
    try:
        rw.SetSize(1024, 768)
    except Exception:
        pass
    rw.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(rw)

    rw.Render()
    if screenshot:
        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(rw)
        w2i.Update()
        wr = vtk.vtkPNGWriter()
        wr.SetFileName(screenshot)
        wr.SetInputData(w2i.GetOutput())
        wr.Write()
    if not offscreen:
        iren.Initialize()
        iren.Start()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="One-line viewer: show phi_geodesic with optional edges/points overlays")
    ap.add_argument("mesh", nargs="+", help="One or more .vtp/.vtk meshes with point array phi_geodesic")
    ap.add_argument("--contours", nargs="*", default=None, help="Optional contours .vtp/.vtk files to overlay")
    ap.add_argument("--scalars", default="phi_geodesic", help="Scalar array name to color by (default phi_geodesic)")
    ap.add_argument("--show-edges", action="store_true", help="Draw mesh edges overlay")
    ap.add_argument("--edge-color", default="black", help="Edge color (name or #RRGGBB)")
    ap.add_argument("--edge-width", type=float, default=1.0, help="Edge line width")
    ap.add_argument("--show-points", action="store_true", help="Draw mesh vertices as points")
    ap.add_argument("--point-color", default="black", help="Point color (name or #RRGGBB)")
    ap.add_argument("--point-size", type=float, default=4.0, help="Point size in pixels")
    ap.add_argument("--screenshot", default=None, help="Save a PNG instead of opening a window")
    ap.add_argument("--offscreen", action="store_true", help="Use offscreen rendering if available")
    ap.add_argument("--mark-seeds", action="store_true", help="Highlight pinned/source vertices if seed array present")
    ap.add_argument("--seed-array", default=None, help="Seed array name (default: auto-detect seed_mask/is_seed/...)")
    ap.add_argument("--seed-color", default="#ff3333", help="Seed color (name or #RRGGBB)")
    ap.add_argument("--seed-size", type=float, default=10.0, help="Seed point size in pixels")
    ap.add_argument("--fragment-argmax", action="store_true", help="Per-fragment argmax(-phi_vec) shading; requires point array 'phi_vec'")
    ap.add_argument("--phi-vec-name", default="phi_vec", help="Name of K-component distance vector array (default phi_vec)")
    ap.add_argument("--cell-argmax", action="store_true", help="Color per-triangle by argmin of phi_vec at triangle centroid (robust fallback)")
    ap.add_argument(
        "--fragment-warp",
        default="identity",
        choices=("identity", "sqrt", "log", "power"),
        help="Warp interpolated phi before argmax (identity/sqrt/log/power; default identity)",
    )
    ap.add_argument(
        "--fragment-warp-power",
        type=float,
        default=1.0,
        help="Exponent used when --fragment-warp=power (default 1.0)",
    )
    ap.add_argument("--force-vtk", action="store_true", help="Bypass PyVista and render with pure VTK pipeline")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Try pyvista unless force-vtk
    ok = False
    if not args.force_vtk:
        ok = _try_pyvista_show(
            args.mesh,
            args.contours,
            args.scalars,
            args.screenshot,
            args.offscreen,
            args.show_edges,
            args.edge_color,
            args.edge_width,
            args.show_points,
            args.point_color,
            args.point_size,
            args.mark_seeds,
            args.seed_array,
            args.seed_color,
            args.seed_size,
            args.fragment_argmax,
            args.phi_vec_name,
            args.cell_argmax,
            args.fragment_warp,
            args.fragment_warp_power,
        )
    if args.force_vtk or not ok:
        _vtk_show(
            args.mesh,
            args.contours,
            args.scalars,
            args.screenshot,
            args.offscreen,
            args.show_edges,
            args.edge_color,
            args.edge_width,
            args.show_points,
            args.point_color,
            args.point_size,
            args.mark_seeds,
            args.seed_array,
            args.seed_color,
            args.seed_size,
            args.fragment_argmax,
            args.phi_vec_name,
            args.cell_argmax,
            args.fragment_warp,
            args.fragment_warp_power,
        )


if __name__ == "__main__":
    main()
