#!/usr/bin/env python3
"""
Visualize Voronoi segmentation results produced by cli_voronoi.

Examples
  python scripts/visualize_voronoi.py --mesh voronoi.vtp --bisectors bisectors.vtp --scalars label
  python scripts/visualize_voronoi.py --mesh voronoi.vtp --scalars phi_geodesic --screenshot voronoi.png
"""

from __future__ import annotations

import argparse
import os


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Visualize Voronoi (label) or continuous phi_geodesic with optional bisectors overlay")
    ap.add_argument("--mesh", required=True, help="Path to .vtp mesh with arrays like 'label' or 'phi_geodesic'")
    ap.add_argument("--bisectors", default=None, help="Optional .vtp polylines (e.g., from --bisectors)")
    ap.add_argument("--scalars", default="label", help="Array to color by: 'label' or 'phi_geodesic'")
    ap.add_argument("--show-edges", action="store_true", help="Overlay triangle edges")
    ap.add_argument("--show-points", action="store_true", help="Overlay mesh vertices as points")
    ap.add_argument("--point-color", default="black", help="Point color (name or #RRGGBB)")
    ap.add_argument("--point-size", type=float, default=4.0, help="Point size in pixels")
    ap.add_argument("--screenshot", default=None, help="Optional path to save a PNG instead of showing a window")
    ap.add_argument("--fragment-argmax", action="store_true", help="Per-fragment argmax(-phi_vec) shading; requires point array 'phi_vec'")
    ap.add_argument("--phi-vec-name", default="phi_vec", help="Name of K-component distance vector array (default phi_vec)")
    return ap.parse_args(argv)


def _install_argmax_shader(actor, num_comp: int, attr_name: str = "phiAttr") -> None:
    import vtk
    d = int(num_comp)
    if d < 1 or d > 4:
        raise ValueError("fragment argmax shader supports 1..4 components; got %d" % d)
    lines_dec = []
    for i in range(d):
        lines_dec.append(f"in float {attr_name}{i};")
    for i in range(d, 4):
        lines_dec.append(f"const float {attr_name}{i} = 1e20;")
    lines_dec.append("out vec4 v_phi;")
    vdec = "\n".join(lines_dec) + "\n"
    vimpl = "v_phi = vec4(%s);\n" % ",".join([f"{attr_name}{i}" for i in range(4)])
    fdec = (
        "in vec4 v_phi;\n"
        "vec3 hsv2rgb(vec3 c){\n"
        "  vec3 rgb = clamp(abs(mod(c.x*6.0 + vec3(0.0,4.0,2.0), 6.0)-3.0)-1.0, 0.0, 1.0);\n"
        "  return c.z * mix(vec3(1.0), rgb, c.y);\n"
        "}\n"
    )
    if d == 1:
        fimpl_core = "int lbl = 0; float m = v_phi.x;\n"
    elif d == 2:
        fimpl_core = (
            "int lbl = 0; float m = v_phi.x;\n"
            "if (v_phi.y < m) { m = v_phi.y; lbl = 1; }\n"
        )
    elif d == 3:
        fimpl_core = (
            "int lbl = 0; float m = v_phi.x;\n"
            "if (v_phi.y < m) { m = v_phi.y; lbl = 1; }\n"
            "if (v_phi.z < m) { m = v_phi.z; lbl = 2; }\n"
        )
    else:
        fimpl_core = (
            "int lbl = 0; float m = v_phi.x;\n"
            "if (v_phi.y < m) { m = v_phi.y; lbl = 1; }\n"
            "if (v_phi.z < m) { m = v_phi.z; lbl = 2; }\n"
            "if (v_phi.w < m) { m = v_phi.w; lbl = 3; }\n"
        )
    fimpl_tail = (
        "vec3 palette[4] = vec3[](\n"
        "  vec3(0.90,0.20,0.25),\n"
        "  vec3(0.25,0.85,0.30),\n"
        "  vec3(0.20,0.45,0.95),\n"
        "  vec3(0.95,0.85,0.25)\n"
        ");\n"
        "vec3 col = palette[lbl % 4];\n"
        "diffuseColor = col; ambientColor = col;\n"
    )
    fimpl = fimpl_core + fimpl_tail
    sp = actor.GetShaderProperty()
    sp.AddVertexShaderReplacement("//VTK::Color::Dec", True, vdec, False)
    sp.AddVertexShaderReplacement("//VTK::Color::Impl", True, vimpl, False)
    sp.AddFragmentShaderReplacement("//VTK::Normal::Dec", True, fdec, False)
    sp.AddFragmentShaderReplacement("//VTK::Light::Impl", True, fimpl + "fragOutput0 = vec4(col, 1.0);\n", False)


def _pyvista_show(mesh_path: str, bisectors_path: str | None, scalars: str, show_edges: bool, show_points: bool, point_color: str, point_size: float, screenshot: str | None, fragment_argmax: bool, phi_vec_name: str) -> bool:
    try:
        import pyvista as pv
    except Exception:
        return False
    if screenshot:
        try:
            pv.start_xvfb()
        except Exception:
            pass
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)
    mesh = pv.read(mesh_path)
    pl = pv.Plotter()
    pl.set_background("white")

    if fragment_argmax:
        if phi_vec_name not in mesh.point_data:
            raise RuntimeError(f"fragment-argmax requested but '{phi_vec_name}' not found in point_data.")
        import vtk
        rdr = vtk.vtkXMLPolyDataReader(); rdr.SetFileName(mesh_path); rdr.Update()
        pd = rdr.GetOutput()
        mapper = vtk.vtkOpenGLPolyDataMapper(); mapper.SetInputData(pd); mapper.ScalarVisibilityOff()
        K = pd.GetPointData().GetArray(phi_vec_name).GetNumberOfComponents()
        d_use = min(4, K)
        for i in range(d_use):
            mapper.MapDataArrayToVertexAttribute(f"phiAttr{i}", phi_vec_name, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, i)
        actor = vtk.vtkActor(); actor.SetMapper(mapper)
        _install_argmax_shader(actor, d_use, attr_name="phiAttr")
        if show_edges:
            actor.GetProperty().EdgeVisibilityOn(); actor.GetProperty().SetEdgeColor(0,0,0); actor.GetProperty().SetLineWidth(1.0)
        if show_points:
            pl.add_points(mesh.points, color=point_color, point_size=point_size, render_points_as_spheres=True)
        pl.add_actor(actor)
    else:
        kwargs = dict(smooth_shading=True, show_scalar_bar=True)
        if scalars not in mesh.point_data:
            raise RuntimeError(f"Point array '{scalars}' not found. Available: {list(mesh.point_data.keys())}")
        if scalars == "label":
            kwargs.update(dict(cmap="glasbey"))
        if show_edges:
            kwargs.update(dict(show_edges=True, edge_color="black", line_width=1.0))
        pl.add_mesh(mesh, scalars=scalars, **kwargs)
        if show_points:
            pl.add_points(mesh.points, color=point_color, point_size=point_size, render_points_as_spheres=True)
    if bisectors_path and os.path.exists(bisectors_path):
        try:
            lines = pv.read(bisectors_path)
            pl.add_mesh(lines, color="white", line_width=2)
        except Exception:
            pass
    pl.show(screenshot=screenshot)
    return True


def _vtk_show(mesh_path: str, bisectors_path: str | None, scalars: str, show_edges: bool, show_points: bool, point_color: str, point_size: float, screenshot: str | None, fragment_argmax: bool, phi_vec_name: str) -> None:
    import vtk
    from vtk.util import numpy_support as nps

    def read_polydata(p: str) -> vtk.vtkPolyData:
        r = vtk.vtkXMLPolyDataReader()
        r.SetFileName(p)
        r.Update()
        return r.GetOutput()

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(mesh_path)
    pd = read_polydata(mesh_path)

    if fragment_argmax:
        mapper = vtk.vtkOpenGLPolyDataMapper(); mapper.SetInputData(pd); mapper.ScalarVisibilityOff()
        arr = pd.GetPointData().GetArray(phi_vec_name)
        if arr is None:
            names = [pd.GetPointData().GetArrayName(i) for i in range(pd.GetPointData().GetNumberOfArrays())]
            raise RuntimeError(f"fragment-argmax requested but '{phi_vec_name}' not found. Available: {names}")
        mapper.MapDataArrayToVertexAttribute("phiAttr", phi_vec_name, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, -1)
        actor = vtk.vtkActor(); actor.SetMapper(mapper)
        _install_argmax_shader(actor, min(4, arr.GetNumberOfComponents()), attr_name="phiAttr")
    else:
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(pd); mapper.SetScalarModeToUsePointData()
        if scalars:
            arr = pd.GetPointData().GetArray(scalars)
            if arr is None:
                raise RuntimeError(f"Point array '{scalars}' not found. Available: {[pd.GetPointData().GetArrayName(i) for i in range(pd.GetPointData().GetNumberOfArrays())]}")
            pd.GetPointData().SetActiveScalars(scalars)
        if pd.GetPointData().GetScalars() is not None:
            mapper.SetScalarRange(pd.GetPointData().GetScalars().GetRange())
            mapper.SetScalarVisibility(True)
        actor = vtk.vtkActor(); actor.SetMapper(mapper)
    if show_edges:
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(0, 0, 0)
        actor.GetProperty().SetLineWidth(1.0)

    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    ren.AddActor(actor)

    if show_points:
        vgf = vtk.vtkVertexGlyphFilter(); vgf.SetInputData(pd); vgf.Update()
        pmapper = vtk.vtkPolyDataMapper(); pmapper.SetInputData(vgf.GetOutput())
        pactor = vtk.vtkActor(); pactor.SetMapper(pmapper)
        # Simple named colors
        colors = {"black": (0,0,0), "white": (1,1,1)}
        pcolor = colors.get(point_color.lower(), (0,0,0))
        pactor.GetProperty().SetColor(*pcolor)
        pactor.GetProperty().SetPointSize(float(point_size))
        ren.AddActor(pactor)

    if bisectors_path and os.path.exists(bisectors_path):
        try:
            pd_lines = read_polydata(bisectors_path)
            cmapper = vtk.vtkPolyDataMapper(); cmapper.SetInputData(pd_lines)
            cact = vtk.vtkActor(); cact.SetMapper(cmapper)
            cact.GetProperty().SetColor(1.0, 1.0, 1.0)
            cact.GetProperty().SetLineWidth(2.0)
            ren.AddActor(cact)
        except Exception:
            pass

    rw = vtk.vtkRenderWindow(); rw.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor(); iren.SetRenderWindow(rw)
    rw.Render()
    if screenshot:
        w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(rw); w2i.Update()
        wr = vtk.vtkPNGWriter(); wr.SetFileName(screenshot); wr.SetInputData(w2i.GetOutput()); wr.Write()
    else:
        iren.Initialize(); iren.Start()


def main(argv=None):
    args = parse_args(argv)
    ok = _pyvista_show(args.mesh, args.bisectors, args.scalars, args.show_edges, args.show_points, args.point_color, args.point_size, args.screenshot, args.fragment_argmax, args.phi_vec_name)
    if not ok:
        _vtk_show(args.mesh, args.bisectors, args.scalars, args.show_edges, args.show_points, args.point_color, args.point_size, args.screenshot, args.fragment_argmax, args.phi_vec_name)


if __name__ == "__main__":
    main()
