import unittest

import vtk

from geodesic_in_heat import view


class TestViewShader(unittest.TestCase):
    def test_install_argmax_shader_accepts_many_components(self):
        actor = vtk.vtkActor()
        view._install_argmax_shader(actor, num_comp=6, warp_mode="power", warp_power=0.5)
        sp = actor.GetShaderProperty()
        self.assertEqual(sp.GetNumberOfShaderReplacements(), 4)

    def test_install_argmax_shader_rejects_invalid_warp(self):
        actor = vtk.vtkActor()
        with self.assertRaises(ValueError):
            view._install_argmax_shader(actor, num_comp=3, warp_mode="invalid")

    def test_install_argmax_shader_rejects_nonpositive_power(self):
        actor = vtk.vtkActor()
        with self.assertRaises(ValueError):
            view._install_argmax_shader(actor, num_comp=3, warp_mode="power", warp_power=0.0)


if __name__ == "__main__":
    unittest.main()
