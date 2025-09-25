import unittest

from viz_tool import shader


class TestVizToolShader(unittest.TestCase):
    def test_build_varyings_power_mode_injects_warp(self):
        vdec, vimpl, fdec = shader.build_varyings(
            ["phi"],
            [6],
            warp_mode="power",
            warp_power=0.5,
        )
        self.assertIn("warp_value", fdec)
        self.assertIn("pow(max(v, 0.0)", fdec)
        self.assertIn("v_f0_4", fdec)
        self.assertTrue(vdec)
        self.assertTrue(vimpl)

    def test_build_varyings_invalid_mode(self):
        with self.assertRaises(ValueError):
            shader.build_varyings(["phi"], [3], warp_mode="invalid")


if __name__ == "__main__":
    unittest.main()
