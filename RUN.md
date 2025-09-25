# Running the Intrinsic Voronoi Pipeline

## 1. Environment Setup
- Create and activate a virtual environment (recommended):
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```
- Install core dependencies (PyTorch build may vary by platform; adjust the pip command accordingly):
  ```bash
  pip install -r requirements.txt
  pip install pyvista
  ```

## 2. Prepare Inputs
- Mesh: Provide either a `.npz` file containing `vertices` `(N,3)` and `faces` `(F,3)` arrays, or a surface/volume mesh stored as `.vtk`, `.vtp`, `.vtu`, `.ply`, `.obj`, or `.stl`. Volume meshes are automatically converted to their boundary surface.
- Seeds: Create a plain-text file with one vertex index per line (0-based).

## 3. Train Geodesic Distance Fields
Run the trainer from the repository root:
```bash
python -m intrinsic_voronoi.cli_train path/to/mesh.vtu seeds.txt --output results.npz
```
Optional flags:
- `--device cpu` (or `cuda` if PyTorch with CUDA is installed)
- `--config config.json` to override `TrainingConfig` values. Example JSON fragment:
  ```json
  {
    "max_steps": 20000,
    "gate_config": {"beta_edge": 8.0, "tau0": 0.15}
  }
  ```
- `--checkpoint-dir checkpoints/` to control where 2500-step checkpoints are written (defaults to `<output>_checkpoints/`)
- `--checkpoint-interval 4000` to change the checkpoint cadence or `0` to disable
  - Each checkpoint emits both a `.pt` training state and a `.npz` snapshot with the current field plus mesh geometry/labels for quick inspection.
- `--init-method dijkstra` to revert to graph-distance initialization (default `heat` per seed)
- Override `warmup_steps`, `main_steps`, `refine_steps`, or the per-phase learning rates (`lr_warmup`, `lr_main`, `lr_refine`) in a JSON config to tune the warmup → main → refine schedule (defaults: 5k / 35k / 10k steps).
- When using gradient-alignment straightness (`"interface_loss": "grad_align"`), adjust `grad_align_beta_start` / `grad_align_beta_end` in the JSON to control the scheduled `beta_edge` ramp (defaults follow the warmup→refine beta schedule).

## 4. Outputs
The CLI writes `results.npz` containing:
- `field`: per-vertex geodesic channels `(N, C)`
- `labels`: Voronoi region indices `(N,)`
- `distances`: minimum distance per vertex `(N,)`
- `boundary_edges`: boolean mask for Voronoi seams `(E,)`
- `vertices`, `faces`, `scale_factor`, `original_mean_edge` for provenance

## 5. Visualization (optional)
Use any VTK-compatible viewer (e.g., PyVista, ParaView) to inspect the mesh and seam mask. For PyVista quick-look:
```python
import numpy as np
import pyvista as pv

data = np.load("results.npz")
mesh = pv.PolyData(data["vertices"], faces=np.hstack((np.full((data["faces"].shape[0], 1), 3), data["faces"])).ravel())
mesh["labels"] = data["labels"]
mesh.plot(show_edges=True)
```
