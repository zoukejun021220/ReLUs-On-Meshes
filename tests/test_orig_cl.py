import tempfile
import unittest

import numpy as np
import vtk

from orig_cl.mesh import SurfaceMesh, to_vtk_polydata
from orig_cl.heat import HeatSolver
from orig_cl.topology import build_edge_topology
from orig_cl.channels import pairwise_face_geometry, edge_pairwise_metrics, soft_assignments
from orig_cl.losses import (
    intrinsic_alignment_loss,
    channel_smoothness,
    fan_regularizer,
    triple_junction_penalty,
)
from orig_cl.cut_locus import edge_scores, select_active_edges
from orig_cl.pipeline import OriginalMeshPipeline, PipelineConfig, RingSeeds
from orig_cl.optimizer import optimize_channels


def make_unit_triangle_mesh() -> SurfaceMesh:
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )
    return SurfaceMesh(verts, faces)


class TestHeatSolver(unittest.TestCase):
    def test_single_source_distance(self) -> None:
        mesh = make_unit_triangle_mesh()
        solver = HeatSolver(mesh, blend_lambda=0.0, mu_conn=0.0)
        phi = solver.single_source_distance(0)
        self.assertEqual(phi.shape, (mesh.vertices.shape[0],))
        self.assertTrue(np.isfinite(phi).all())
        self.assertLess(abs(phi[0]), 1e-8)
        self.assertTrue(np.all(phi >= -1e-6))

    def test_pairwise_metrics_do_not_nan(self) -> None:
        mesh = make_unit_triangle_mesh()
        solver = HeatSolver(mesh, blend_lambda=0.0, mu_conn=0.0)
        D = np.stack([
            solver.single_source_distance(0),
            solver.single_source_distance(1),
        ], axis=1)
        f = D.copy()
        geom = pairwise_face_geometry(f, mesh.faces, solver.face_gradients, mesh.vertices)
        topology = build_edge_topology(mesh.vertices, mesh.faces)
        metrics = edge_pairwise_metrics(f, geom, topology, mesh.vertices)
        self.assertTrue(np.isfinite(metrics.phi).all())
        self.assertTrue(np.isfinite(metrics.gamma_grad_hat).all())
        w_e = np.ones_like(metrics.phi)
        self.assertGreaterEqual(intrinsic_alignment_loss(w_e, metrics.m), 0.0)
        self.assertGreaterEqual(channel_smoothness(f, topology), 0.0)
        s_beta = soft_assignments(f, beta=5.0)
        angles = np.array([0.0, np.pi])
        self.assertGreaterEqual(fan_regularizer(s_beta, angles, topology), 0.0)
        eta_hat = np.ones(topology.edges.shape[0])
        self.assertGreaterEqual(triple_junction_penalty(s_beta, mesh.faces, eta_hat, topology), 0.0)
        scores = edge_scores(metrics.phi, eta_hat, solver.edge_reliability_weights, topology)
        active = select_active_edges(scores, topology)
        self.assertTrue(active.size <= topology.edges.shape[0])

    def test_optimizer_single_step(self) -> None:
        mesh = make_unit_triangle_mesh()
        pd = to_vtk_polydata(mesh)
        with tempfile.NamedTemporaryFile(suffix=".vtp") as tmp:
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(tmp.name)
            writer.SetInputData(pd)
            writer.Write()
            pipeline = OriginalMeshPipeline(tmp.name, PipelineConfig())
            d_s = pipeline.single_source_distance(0)
            seeds = RingSeeds(indices=np.array([0, 1], dtype=np.int32), angles=np.array([0.0, np.pi]))
            radius = 1.0
            f_opt, losses = optimize_channels(
                pipeline,
                seeds,
                d_s,
                alpha=1.0,
                beta=5.0,
                radius=radius,
                maxiter=1,
            )
            self.assertEqual(f_opt.shape[0], mesh.vertices.shape[0])
            self.assertIn("alignment", losses)


if __name__ == "__main__":
    unittest.main()
