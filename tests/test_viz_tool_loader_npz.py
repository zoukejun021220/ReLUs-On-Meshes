from __future__ import annotations

import numpy as np
from vtk.util import numpy_support as nps

from viz_tool.loader import load_surface


def test_load_surface_npz(tmp_path):
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int32)
    field = np.linspace(0.0, 1.0, vertices.shape[0], dtype=np.float32)
    labels = np.array([0, 1, 2, 3], dtype=np.int32)
    boundary = np.array([1, 0], dtype=np.int8)

    npz_path = tmp_path / "mesh.npz"
    np.savez(
        npz_path,
        vertices=vertices,
        faces=faces,
        field=field,
        labels=labels,
        boundary_edges=boundary,
    )

    pd, V_loaded, F_loaded = load_surface(str(npz_path))

    assert V_loaded.shape == (4, 3)
    assert np.allclose(V_loaded, vertices)
    assert np.array_equal(F_loaded, faces)

    field_arr = pd.GetPointData().GetArray("field")
    assert field_arr is not None
    np.testing.assert_allclose(nps.vtk_to_numpy(field_arr), field)

    label_arr = pd.GetPointData().GetArray("labels")
    assert label_arr is not None
    np.testing.assert_array_equal(nps.vtk_to_numpy(label_arr), labels)

    boundary_arr = pd.GetCellData().GetArray("boundary_edges")
    assert boundary_arr is not None
    np.testing.assert_array_equal(nps.vtk_to_numpy(boundary_arr), boundary)
