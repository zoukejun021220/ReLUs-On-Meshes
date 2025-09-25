"""Mesh loading and geometry precomputation utilities."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch


@dataclass
class MeshData:
    """Container for raw mesh connectivity and vertex positions."""

    vertices: np.ndarray  # (n, 3)
    faces: np.ndarray  # (m, 3)

    def to_torch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        verts = torch.from_numpy(self.vertices).to(device)
        faces = torch.from_numpy(self.faces).to(device)
        return verts, faces


@dataclass
class EdgeTopology:
    edges: np.ndarray  # (e, 2)
    face_left: np.ndarray  # (e,) indices into faces or -1
    face_right: np.ndarray  # (e,)
    face_edges: np.ndarray  # (m, 3) index of each face's three edges


@dataclass
class PrecomputedGeometry:
    vertices: torch.Tensor  # (n, 3) float64
    faces: torch.Tensor  # (m, 3) long
    edge_indices: torch.Tensor  # (e, 2) long
    edge_faces: torch.Tensor  # (e, 2) long
    face_edges: torch.Tensor  # (m, 3) long
    edge_lengths: torch.Tensor  # (e,) float64
    face_areas: torch.Tensor  # (m,) float64
    face_normals: torch.Tensor  # (m, 3) float64
    edge_unit_vectors: torch.Tensor  # (e, 3) float64
    edge_dihedral_cos: torch.Tensor  # (e,) float64
    edge_dihedral_sin: torch.Tensor  # (e,) float64
    edge_vectors: torch.Tensor  # (m, 2, 3) float64 (E0, E1)
    gram_matrices: torch.Tensor  # (m, 2, 2) float64
    gram_inv: torch.Tensor  # (m, 2, 2) float64
    mean_edge_length: float
    total_area: float
    scale_factor: float
    original_mean_edge: float
    seed_rings: List[np.ndarray]
    seed_radius: float


PathLike = Union[str, Path]


def load_mesh(vertices: np.ndarray, faces: np.ndarray) -> MeshData:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.int64)
    return MeshData(vertices=vertices, faces=faces)


def load_surface_mesh(path: PathLike) -> MeshData:
    """Load a surface mesh (or extract surface from volume) from VTK/VTU/PLY/etc."""

    try:
        import pyvista as pv  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyVista is required to load VTK meshes. Install with `pip install pyvista`."
        ) from exc

    mesh = pv.read(str(path))
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    mesh = mesh.triangulate()

    faces_arr = mesh.faces.reshape(-1, 4)
    faces = faces_arr[:, 1:4].astype(np.int64)
    vertices = mesh.points.astype(np.float64)

    return load_mesh(vertices, faces)


def _build_edge_topology(faces: np.ndarray) -> EdgeTopology:
    edges: List[Tuple[int, int]] = []
    face_left: List[int] = []
    face_right: List[int] = []
    face_edges = np.empty((faces.shape[0], 3), dtype=np.int64)
    edge_map: Dict[Tuple[int, int], int] = {}

    for face_idx, (a, b, c) in enumerate(faces):
        tri_verts = [a, b, c]
        tri_pairs = [(a, b), (b, c), (c, a)]
        for local_idx, (u, v) in enumerate(tri_pairs):
            if u > v:
                u, v = v, u
            key = (int(u), int(v))
            edge_idx = edge_map.get(key)
            if edge_idx is None:
                edge_idx = len(edges)
                edges.append(key)
                face_left.append(face_idx)
                face_right.append(-1)
                edge_map[key] = edge_idx
            else:
                if face_right[edge_idx] != -1:
                    # Duplicate edge; skip but keep first two incidences.
                    pass
                else:
                    face_right[edge_idx] = face_idx
            face_edges[face_idx, local_idx] = edge_idx

    return EdgeTopology(
        edges=np.asarray(edges, dtype=np.int64),
        face_left=np.asarray(face_left, dtype=np.int64),
        face_right=np.asarray(face_right, dtype=np.int64),
        face_edges=face_edges,
    )


def _edge_lengths(vertices: np.ndarray, edges: np.ndarray) -> np.ndarray:
    diff = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    return np.linalg.norm(diff, axis=1)


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e0 = v1 - v0
    e1 = v2 - v0
    cross = np.cross(e0, e1)
    double_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * double_area
    eps_n = 1e-15
    normals = cross / (double_area[:, None] + eps_n)
    g00 = np.einsum("ij,ij->i", e0, e0)
    g01 = np.einsum("ij,ij->i", e0, e1)
    g11 = np.einsum("ij,ij->i", e1, e1)
    gram = np.stack(
        [
            np.stack([g00, g01], axis=1),
            np.stack([g01, g11], axis=1),
        ],
        axis=1,
    )  # (m, 2, 2)
    gram_inv = _stable_inverse_gram(gram)
    edge_vectors = np.stack([e0, e1], axis=1)
    return edge_vectors, gram, gram_inv, normals, areas


def _stable_inverse_gram(gram: np.ndarray) -> np.ndarray:
    g00 = gram[:, 0, 0]
    g01 = gram[:, 0, 1]
    g10 = gram[:, 1, 0]
    g11 = gram[:, 1, 1]
    det = g00 * g11 - g01 * g10
    diag_sum = g00 + g11
    delta_det = 1e-14 * diag_sum
    delta_tik = np.maximum(delta_det, 1e-12)
    use_tik = det < delta_det

    inv = np.empty_like(gram)
    # Exact inverse where stable
    det_safe = det.copy()
    det_safe[use_tik] = 1.0  # temporary placeholder
    inv_exact = np.empty_like(gram)
    inv_exact[:, 0, 0] = g11
    inv_exact[:, 0, 1] = -g01
    inv_exact[:, 1, 0] = -g10
    inv_exact[:, 1, 1] = g00
    inv_exact = inv_exact / det_safe[:, None, None]

    # Tikhonov regularization
    gram_tik = gram + np.eye(2, dtype=gram.dtype)[None, :, :] * delta_tik[:, None, None]
    inv_tik = np.linalg.inv(gram_tik)

    inv[~use_tik] = inv_exact[~use_tik]
    inv[use_tik] = inv_tik[use_tik]
    return inv


def precompute_geometry(
    mesh: MeshData,
    seeds: Sequence[int],
    *,
    device: torch.device,
    seed_radius: float = 2.0,
) -> PrecomputedGeometry:
    if not seeds:
        raise ValueError("At least one seed index is required")

    vertices = mesh.vertices.copy()
    faces = mesh.faces

    edge_topology = _build_edge_topology(faces)
    edge_lengths = _edge_lengths(vertices, edge_topology.edges)
    original_mean_edge = float(edge_lengths.mean())
    if original_mean_edge <= 0:
        raise ValueError("Mean edge length must be positive")

    scale_factor = 1.0 / original_mean_edge
    vertices_scaled = vertices * scale_factor
    edge_lengths_scaled = edge_lengths * scale_factor
    mean_edge_length = float(edge_lengths_scaled.mean())

    edge_vectors, gram, gram_inv, normals, areas = _face_geometry(vertices_scaled, faces)

    edge_vec = vertices_scaled[edge_topology.edges[:, 1]] - vertices_scaled[edge_topology.edges[:, 0]]
    edge_len_safe = edge_lengths_scaled[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        edge_unit = edge_vec / np.maximum(edge_len_safe, 1e-12)

    num_edges = edge_topology.edges.shape[0]
    edge_dihedral_cos = np.ones(num_edges, dtype=np.float64)
    edge_dihedral_sin = np.zeros(num_edges, dtype=np.float64)
    left_faces = edge_topology.face_left
    right_faces = edge_topology.face_right
    interior_mask = right_faces >= 0
    if np.any(interior_mask):
        n_left = normals[left_faces[interior_mask]]
        n_right = normals[right_faces[interior_mask]]
        cos_theta = np.clip(np.einsum("ij,ij->i", n_left, n_right), -1.0, 1.0)
        cross_nr = np.cross(n_left, n_right)
        sin_mag = np.linalg.norm(cross_nr, axis=1)
        edge_dir = edge_unit[interior_mask]
        axis_dot = np.einsum("ij,ij->i", cross_nr, edge_dir)
        sign = np.where(sin_mag > 1e-12, np.sign(axis_dot), 1.0)
        sin_theta = sin_mag * sign
        edge_dihedral_cos[interior_mask] = cos_theta
        edge_dihedral_sin[interior_mask] = sin_theta

    total_area = float(areas.sum())

    seed_rings = compute_seed_rings_from_topology(
        edge_topology.edges,
        edge_lengths_scaled,
        len(vertices_scaled),
        seeds,
        radius=seed_radius * mean_edge_length,
    )

    vertices_t = torch.from_numpy(vertices_scaled).to(device=device, dtype=torch.float64)
    faces_t = torch.from_numpy(faces).to(device=device, dtype=torch.long)
    edge_idx_t = torch.from_numpy(edge_topology.edges).to(device=device, dtype=torch.long)
    edge_faces = np.stack([edge_topology.face_left, edge_topology.face_right], axis=1)
    edge_faces_t = torch.from_numpy(edge_faces).to(device=device, dtype=torch.long)
    face_edges_t = torch.from_numpy(edge_topology.face_edges).to(device=device, dtype=torch.long)
    edge_lengths_t = torch.from_numpy(edge_lengths_scaled).to(device=device, dtype=torch.float64)
    face_areas_t = torch.from_numpy(areas).to(device=device, dtype=torch.float64)
    normals_t = torch.from_numpy(normals).to(device=device, dtype=torch.float64)
    edge_unit_t = torch.from_numpy(edge_unit).to(device=device, dtype=torch.float64)
    edge_cos_t = torch.from_numpy(edge_dihedral_cos).to(device=device, dtype=torch.float64)
    edge_sin_t = torch.from_numpy(edge_dihedral_sin).to(device=device, dtype=torch.float64)
    edge_vectors_t = torch.from_numpy(edge_vectors).to(device=device, dtype=torch.float64)
    gram_t = torch.from_numpy(gram).to(device=device, dtype=torch.float64)
    gram_inv_t = torch.from_numpy(gram_inv).to(device=device, dtype=torch.float64)

    return PrecomputedGeometry(
        vertices=vertices_t,
        faces=faces_t,
        edge_indices=edge_idx_t,
        edge_faces=edge_faces_t,
        face_edges=face_edges_t,
        edge_lengths=edge_lengths_t,
        face_areas=face_areas_t,
        face_normals=normals_t,
        edge_unit_vectors=edge_unit_t,
        edge_dihedral_cos=edge_cos_t,
        edge_dihedral_sin=edge_sin_t,
        edge_vectors=edge_vectors_t,
        gram_matrices=gram_t,
        gram_inv=gram_inv_t,
        mean_edge_length=mean_edge_length,
        total_area=total_area,
        scale_factor=scale_factor,
        original_mean_edge=original_mean_edge,
        seed_rings=seed_rings,
        seed_radius=seed_radius * mean_edge_length,
    )


def compute_seed_rings(mesh: MeshData, seeds: Sequence[int], radius: float) -> List[np.ndarray]:
    edge_topology = _build_edge_topology(mesh.faces)
    edge_lengths = _edge_lengths(mesh.vertices, edge_topology.edges)
    return compute_seed_rings_from_topology(
        edge_topology.edges,
        edge_lengths,
        mesh.vertices.shape[0],
        seeds,
        radius=radius,
    )


def compute_seed_rings_from_topology(
    edges: np.ndarray,
    edge_lengths: np.ndarray,
    num_vertices: int,
    seeds: Sequence[int],
    *,
    radius: float,
) -> List[np.ndarray]:
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(num_vertices)]
    for (u, v), w in zip(edges, edge_lengths):
        weight = float(w)
        adjacency[u].append((v, weight))
        adjacency[v].append((u, weight))

    rings: List[np.ndarray] = []
    for seed in seeds:
        distances = {int(seed): 0.0}
        visited: Dict[int, float] = {}
        queue: List[Tuple[float, int]] = [(0.0, int(seed))]
        members: List[int] = []
        while queue:
            dist, vertex = heapq.heappop(queue)
            if dist > radius:
                continue
            if vertex in visited and visited[vertex] <= dist:
                continue
            visited[vertex] = dist
            members.append(vertex)
            for neighbor, weight in adjacency[vertex]:
                new_dist = dist + weight
                if new_dist <= radius and (neighbor not in visited or new_dist < visited[neighbor]):
                    heapq.heappush(queue, (new_dist, neighbor))
        rings.append(np.asarray(sorted(set(members)), dtype=np.int64))
    return rings
