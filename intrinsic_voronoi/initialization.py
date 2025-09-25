"""Field initialization strategies."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .mesh import MeshData, PrecomputedGeometry


@dataclass
class InitializationConfig:
    method: str = "dijkstra"  # or "heat"
    use_heat_method: bool = False


def initialize_fields(
    mesh: MeshData,
    seeds: Sequence[int],
    precomp: PrecomputedGeometry,
    *,
    device: torch.device,
    method: str = "dijkstra",
) -> torch.Tensor:
    """Return initial field tensor of shape (num_vertices, num_channels)."""
    if method not in {"dijkstra", "heat"}:
        raise ValueError("Initialization method must be 'dijkstra' or 'heat'")

    if method == "heat":
        return _heat_method_init(mesh, seeds, precomp, device=device)

    return _dijkstra_init(mesh, seeds, precomp, device=device)


def _build_adjacency(precomp: PrecomputedGeometry) -> List[List[tuple[int, float]]]:
    edges = precomp.edge_indices.cpu().numpy()
    lengths = precomp.edge_lengths.cpu().numpy()
    num_vertices = precomp.vertices.shape[0]
    adjacency: List[List[tuple[int, float]]] = [[] for _ in range(num_vertices)]
    for (u, v), w in zip(edges, lengths):
        weight = float(w)
        adjacency[int(u)].append((int(v), weight))
        adjacency[int(v)].append((int(u), weight))
    return adjacency


def _dijkstra_init(
    mesh: MeshData,
    seeds: Sequence[int],
    precomp: PrecomputedGeometry,
    *,
    device: torch.device,
) -> torch.Tensor:
    adjacency = _build_adjacency(precomp)
    num_vertices = mesh.vertices.shape[0]
    num_channels = len(seeds)
    distances = np.full((num_channels, num_vertices), np.inf, dtype=np.float64)

    for channel, seed in enumerate(seeds):
        dist = distances[channel]
        seed = int(seed)
        dist[seed] = 0.0
        queue: List[tuple[float, int]] = [(0.0, seed)]
        while queue:
            current, vertex = heapq.heappop(queue)
            if current > dist[vertex]:
                continue
            for neighbor, weight in adjacency[vertex]:
                new_dist = current + weight
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    heapq.heappush(queue, (new_dist, neighbor))

    field = torch.from_numpy(distances.T).to(device=device, dtype=torch.float32)
    return field


def _heat_method_init(
    mesh: MeshData,
    seeds: Sequence[int],
    precomp: PrecomputedGeometry,
    *,
    device: torch.device,
) -> torch.Tensor:
    try:
        from geodesic_in_heat.heat import HeatGeodesic
    except ImportError:
        warnings.warn(
            "geodesic_in_heat.HeatGeodesic unavailable; falling back to SciPy heat initialization",
            stacklevel=2,
        )
        try:
            return _heat_method_init_scipy(mesh, seeds, precomp, device=device)
        except ImportError:
            warnings.warn(
                "SciPy heat initialization unavailable; falling back to Dijkstra initialization",
                stacklevel=2,
            )
            return _dijkstra_init(mesh, seeds, precomp, device=device)

    vertices = precomp.vertices.detach().cpu().numpy()
    faces = precomp.faces.detach().cpu().numpy()
    num_vertices = vertices.shape[0]
    num_channels = len(seeds)

    if num_channels == 0:
        raise ValueError("At least one seed index is required for heat initialization")

    try:
        solver = HeatGeodesic(vertices, faces)
    except Exception as exc:
        warnings.warn(
            f"Failed to construct HeatGeodesic solver ({exc!r}); falling back to SciPy heat initialization",
            stacklevel=2,
        )
        try:
            return _heat_method_init_scipy(mesh, seeds, precomp, device=device)
        except ImportError:
            warnings.warn(
                "SciPy heat initialization unavailable; falling back to Dijkstra initialization",
                stacklevel=2,
            )
            return _dijkstra_init(mesh, seeds, precomp, device=device)

    field = np.zeros((num_vertices, num_channels), dtype=np.float64)
    for channel, seed in enumerate(seeds):
        dist = solver.phi_to_subset([int(seed)])
        if dist.shape[0] != num_vertices:
            raise ValueError("HeatGeodesic returned distance array with unexpected shape")
        dist = dist - dist[int(seed)]
        field[:, channel] = dist

    return torch.from_numpy(field).to(device=device, dtype=torch.float32)


def _heat_method_init_scipy(
    mesh: MeshData,
    seeds: Sequence[int],
    precomp: PrecomputedGeometry,
    *,
    device: torch.device,
) -> torch.Tensor:
    import scipy.sparse  # type: ignore
    import scipy.sparse.linalg  # type: ignore

    vertices = precomp.vertices.cpu().numpy()
    faces = precomp.faces.cpu().numpy()
    num_vertices = vertices.shape[0]
    num_channels = len(seeds)

    cot_weights = _cotangent_weights(vertices, faces)
    mass_diag = _vertex_areas(vertices, faces)

    S = scipy.sparse.csr_matrix((num_vertices, num_vertices))
    # Assemble stiffness matrix from cotangent weights
    I = []
    J = []
    V = []
    for (i, j), w in cot_weights.items():
        I.append(i)
        J.append(j)
        V.append(-w)
        I.append(j)
        J.append(i)
        V.append(-w)
        I.append(i)
        J.append(i)
        V.append(w)
        I.append(j)
        J.append(j)
        V.append(w)
    S = scipy.sparse.csr_matrix((V, (I, J)), shape=(num_vertices, num_vertices))
    M = scipy.sparse.diags(mass_diag)

    t = precomp.mean_edge_length ** 2
    lhs = M + t * S

    field = np.zeros((num_vertices, num_channels), dtype=np.float64)

    for channel, seed in enumerate(seeds):
        rhs = np.zeros(num_vertices, dtype=np.float64)
        rhs[int(seed)] = mass_diag[int(seed)]
        u = scipy.sparse.linalg.spsolve(lhs, rhs)
        grad = _face_gradients_from_scalar(u, vertices, faces)
        div = _discrete_divergence(grad, vertices, faces)
        phi = scipy.sparse.linalg.spsolve(S, div)
        phi -= phi[int(seed)]
        field[:, channel] = phi

    return torch.from_numpy(field).to(device=device, dtype=torch.float32)


def _cotangent_weights(vertices: np.ndarray, faces: np.ndarray) -> Dict[Tuple[int, int], float]:
    from collections import defaultdict

    weights = defaultdict(float)
    for tri in faces:
        pts = vertices[tri]
        for offset in range(3):
            i = tri[offset]
            j = tri[(offset + 1) % 3]
            k = tri[(offset + 2) % 3]
            vi = vertices[i]
            vj = vertices[j]
            vk = vertices[k]
            u = vi - vk
            v = vj - vk
            cot = np.dot(u, v) / np.linalg.norm(np.cross(u, v))
            weights[tuple(sorted((i, j)))] += 0.5 * cot
    return weights


def _vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    areas = np.zeros(vertices.shape[0], dtype=np.float64)
    for tri in faces:
        pts = vertices[tri]
        area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        for idx in tri:
            areas[idx] += area / 3.0
    return areas


def _face_gradients_from_scalar(values: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    grads = np.zeros((faces.shape[0], 3), dtype=np.float64)
    for face_idx, (a, b, c) in enumerate(faces):
        x0, x1, x2 = vertices[[a, b, c]]
        e0 = x1 - x0
        e1 = x2 - x0
        g = np.array([[np.dot(e0, e0), np.dot(e0, e1)], [np.dot(e1, e0), np.dot(e1, e1)]])
        det = np.linalg.det(g)
        if det < 1e-14 * g.trace():
            g += np.eye(2) * 1e-12
        g_inv = np.linalg.inv(g)
        b_vec = np.array([values[b] - values[a], values[c] - values[a]])
        coeff = g_inv @ b_vec
        grads[face_idx] = coeff[0] * e0 + coeff[1] * e1
    return grads


def _discrete_divergence(gradients: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    div = np.zeros(vertices.shape[0], dtype=np.float64)
    for grad, (a, b, c) in zip(gradients, faces):
        pts = vertices[[a, b, c]]
        area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        for idx in (a, b, c):
            div[idx] += np.dot(grad, (vertices[idx] - pts.mean(axis=0))) / 3.0 * area
    return div
