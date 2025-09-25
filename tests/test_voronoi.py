import os
import unittest

import numpy as np

from geodesic_in_heat.io import load_polydata
from geodesic_in_heat.sources import gfps_geodesic_seeds
from geodesic_in_heat.voronoi import (
    geodesic_distance_matrix,
    voronoi_labels,
    bisector_polylines,
    sample_points,
)


DATA_MESH = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "bunny_phi.vtp"))


class TestVoronoiPipeline(unittest.TestCase):
    def setUp(self):
        if not os.path.exists(DATA_MESH):
            self.skipTest(f"Test mesh not found: {DATA_MESH}")
        self.pd, self.V, self.F = load_polydata(DATA_MESH)

    def test_distance_matrix_and_labels(self):
        seeds = gfps_geodesic_seeds(self.V, self.F, K=4)
        Phi = geodesic_distance_matrix(self.V, self.F, seeds)
        self.assertEqual(Phi.shape[0], self.V.shape[0])
        self.assertEqual(Phi.shape[1], 4)
        self.assertTrue(np.isfinite(Phi).all())

        labels = voronoi_labels(Phi)
        self.assertEqual(labels.shape, (self.V.shape[0],))
        self.assertTrue((labels >= 0).all() and (labels < 4).all())
        # Expect at least two regions
        self.assertGreaterEqual(np.unique(labels).size, 2)

    def test_bisectors_and_sampling(self):
        seeds = gfps_geodesic_seeds(self.V, self.F, K=3)
        Phi = geodesic_distance_matrix(self.V, self.F, seeds)
        bis = bisector_polylines(self.pd, Phi)
        self.assertGreaterEqual(bis.GetNumberOfPoints(), 1)
        self.assertGreaterEqual(bis.GetNumberOfLines(), 1)

        pts = sample_points(self.V, self.F, Phi, per_edge=5)
        self.assertGreater(pts.GetNumberOfPoints(), 0)
        self.assertIsNotNone(pts.GetPointData().GetArray("label"))

    def test_cli_end_to_end(self):
        import subprocess, sys, tempfile
        with tempfile.TemporaryDirectory() as td:
            out_mesh = os.path.join(td, "bunny_voronoi.vtp")
            out_bis = os.path.join(td, "bunny_bisectors.vtp")
            out_pts = os.path.join(td, "bunny_points.vtp")
            cmd = [
                sys.executable, "-m", "geodesic_in_heat.cli_voronoi", DATA_MESH,
                "--gfps", "4",
                "--bisectors", "--out-mesh", out_mesh, "--out-bisectors", out_bis,
                "--sample-points", "--per-edge", "5", "--out-points", out_pts,
                "--write-phi-vec",
            ]
            subprocess.check_call(cmd)
            self.assertTrue(os.path.exists(out_mesh))
            self.assertTrue(os.path.exists(out_bis))
            self.assertTrue(os.path.exists(out_pts))

            # Verify arrays exist
            import vtk
            rdr = vtk.vtkXMLPolyDataReader(); rdr.SetFileName(out_mesh); rdr.Update()
            pd2 = rdr.GetOutput()
            self.assertIsNotNone(pd2.GetPointData().GetArray("label"))
            self.assertIsNotNone(pd2.GetPointData().GetArray("phi_vec"))


if __name__ == "__main__":
    unittest.main()

