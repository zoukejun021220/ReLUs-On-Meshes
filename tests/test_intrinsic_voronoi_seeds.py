from __future__ import annotations

import unittest

import numpy as np
import torch

from intrinsic_voronoi.mesh import load_mesh, precompute_geometry
from intrinsic_voronoi.trainer import TrainingConfig, VoronoiTrainer


class VoronoiSeedPinningTest(unittest.TestCase):
    def test_seed_channels_remain_zero(self) -> None:
        vertices = np.array(
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
            dtype=np.int64,
        )

        mesh = load_mesh(vertices, faces)
        seeds = [0, 1, 2]
        device = torch.device("cpu")

        precomp = precompute_geometry(mesh, seeds, device=device)
        config = TrainingConfig(
            warmup_steps=0,
            main_steps=1,
            refine_steps=0,
            max_steps=1,
            checkpoint_interval=None,
            log_interval=10,
        )

        trainer = VoronoiTrainer(mesh, seeds, precomp=precomp, device=device, config=config)
        trainer.setup()

        with torch.no_grad():
            trainer.field_param[seeds[0], 0] = 5.0
            trainer.field_param[seeds[1], 1] = -3.0
            trainer.field_param[seeds[2], 2] = 1.0

        trainer.train()

        with torch.no_grad():
            seed_rows = torch.tensor(seeds, device=device, dtype=torch.long)
            channel_indices = torch.arange(len(seeds), device=device, dtype=torch.long)
            pinned = trainer.field_param[seed_rows, channel_indices]
            expected = torch.tensor([5.0, -3.0, 1.0], device=device)
            self.assertTrue(torch.allclose(pinned, expected))


if __name__ == "__main__":
    unittest.main()
