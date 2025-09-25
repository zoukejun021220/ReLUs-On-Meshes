import unittest

import numpy as np
import vtk
from vtk.util import numpy_support as nps

from geodesic_in_heat.fronts import determine_front_levels, fronts_polydata


class TestFrontUtilities(unittest.TestCase):
    def test_determine_front_levels_spacing(self):
        phi = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        levels = determine_front_levels(phi, spacing=0.5, include_zero=False)
        self.assertTrue(np.allclose(levels[:3], np.array([0.5, 1.0, 1.5])))
        self.assertTrue(levels.max() <= 2.0 + 1e-8)

    def test_fronts_polydata(self):
        # Simple triangle mesh
        pts = vtk.vtkPoints()
        pts.SetData(nps.numpy_to_vtk(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float), deep=True))
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        ca = vtk.vtkCellArray()
        idl = vtk.vtkIdList()
        idl.SetNumberOfIds(3)
        for i, idx in enumerate((0, 1, 2)):
            idl.SetId(i, idx)
        ca.InsertNextCell(idl)
        pd.SetPolys(ca)

        phi = np.array([0.0, 1.0, 1.5], dtype=float)
        levels = [1.0]
        fronts = fronts_polydata(pd, phi, levels)
        self.assertIsNotNone(fronts)
        self.assertGreater(fronts.GetNumberOfPoints(), 0)
        self.assertGreater(fronts.GetNumberOfLines(), 0)


if __name__ == "__main__":
    unittest.main()
