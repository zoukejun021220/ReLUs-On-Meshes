import unittest

import numpy as np
import vtk
from vtk.util import numpy_support as nps

from geodesic_in_heat.cut_locus import (
    cut_locus_by_gradient_jump,
    cut_locus_by_laplacian,
)


def _make_square_mesh():
    V = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=float,
    )
    F = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
        ],
        dtype=int,
    )
    pts = vtk.vtkPoints(); pts.SetData(nps.numpy_to_vtk(V, deep=True))
    pd = vtk.vtkPolyData(); pd.SetPoints(pts)
    ca = vtk.vtkCellArray()
    for tri in F:
        idl = vtk.vtkIdList(); idl.SetNumberOfIds(3)
        idl.SetId(0, int(tri[0])); idl.SetId(1, int(tri[1])); idl.SetId(2, int(tri[2]))
        ca.InsertNextCell(idl)
    pd.SetPolys(ca)
    return pd, V, F


class TestCutLocus(unittest.TestCase):
    def setUp(self):
        self.pd, self.V, self.F = _make_square_mesh()

    def test_gradient_jump_cut_locus(self):
        phi = np.array([0.0, 1.0, 1.5, 0.2], dtype=float)
        res = cut_locus_by_gradient_jump(
            self.pd,
            self.V,
            self.F,
            phi,
            top_percent=100.0,
        )
        self.assertGreaterEqual(res.lines.GetNumberOfCells(), 1)
        self.assertGreaterEqual(res.num_components, 1)
        jump = res.lines.GetCellData().GetArray("jump")
        self.assertIsNotNone(jump)

    def test_gradient_jump_cut_locus_with_filters(self):
        phi = np.array([0.0, 1.0, 1.5, 0.2], dtype=float)
        res = cut_locus_by_gradient_jump(
            self.pd,
            self.V,
            self.F,
            phi,
            top_percent=100.0,
            seeds=[0],
            exclude_radius_multiplier=1.0,
            min_component_length_multiplier=100.0,
        )
        self.assertEqual(res.lines.GetNumberOfCells(), 0)
        self.assertEqual(res.num_components, 0)

    def test_laplacian_cut_locus(self):
        phi = np.array([0.0, 0.5, 1.0, 0.3], dtype=float)
        res = cut_locus_by_laplacian(
            self.pd,
            self.V,
            self.F,
            phi,
            top_percent=50.0,
        )
        self.assertIsNotNone(res.lines)
        # If the contour is empty, num_components is zero; otherwise positive.
        if res.lines.GetNumberOfCells() > 0:
            self.assertGreaterEqual(res.num_components, 1)
        else:
            self.assertEqual(res.num_components, 0)

    def test_laplacian_cut_locus_with_filters(self):
        phi = np.array([0.0, 0.5, 1.0, 0.3], dtype=float)
        res = cut_locus_by_laplacian(
            self.pd,
            self.V,
            self.F,
            phi,
            top_percent=100.0,
            seeds=[0],
            exclude_radius_multiplier=1.0,
            min_component_length_multiplier=100.0,
        )
        self.assertIsNotNone(res.lines)
        self.assertEqual(res.num_components, 0)
        self.assertEqual(res.lines.GetNumberOfCells(), 0)


if __name__ == "__main__":
    unittest.main()
