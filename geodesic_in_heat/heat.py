from __future__ import annotations

import numpy as np


class HeatGeodesic:
    def __init__(self, V: np.ndarray, F: np.ndarray, t: float | None = None):
        import potpourri3d as pp3d

        self.n_verts = int(V.shape[0])
        if t is None:
            self.solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
        else:
            try:
                self.solver = pp3d.MeshHeatMethodDistanceSolver(V, F, t)
            except TypeError:
                self.solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
                # Support both snake_case and camelCase depending on version
                if hasattr(self.solver, "set_time_step"):
                    self.solver.set_time_step(t)
                elif hasattr(self.solver, "setTimeStep"):
                    self.solver.setTimeStep(t)
                else:
                    raise RuntimeError("potpourri3d solver missing time-step setter")

    def phi_to_subset(self, seeds_idx: np.ndarray | list[int]) -> np.ndarray:
        seeds = np.asarray(seeds_idx, dtype=np.int32).ravel()
        if seeds.size == 0:
            raise ValueError("seeds_idx must contain at least one index")
        if np.any(seeds < 0) or np.any(seeds >= self.n_verts):
            raise IndexError("seed index out of range for this mesh")

        # Some versions of potpourri3d only accept a single int; fall back to min over single-source distances
        if seeds.size == 1:
            return self.solver.compute_distance(int(seeds[0]))

        seeds = np.unique(seeds)
        phi = None
        for s in seeds:
            d = self.solver.compute_distance(int(s))
            if phi is None:
                phi = d
            else:
                phi = np.minimum(phi, d)
        return phi
