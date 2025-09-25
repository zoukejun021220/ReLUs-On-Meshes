# Repository Guidelines

## Project Structure & Module Organization
- Core Python package: `geodesic_in_heat/`. Keep algorithms in `heat.py`, `volume.py`, `segment.py`, `sources.py`; re-export public symbols from `__init__.py`.
- CLI entry points stay in `cli*.py` modules; avoid mixing surface and volume logic.
- Tests live in `tests/` mirroring module names (`tests/test_voronoi.py` is the template). Store fixtures next to the tests that use them.
- Mesh samples (`*.vtp`, `volume.vtu`) at the repo root support manual runs only. Leave `HexaLab/` assets untouched unless you are updating that toolkit.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` prepares a local virtualenv.
- `python -m unittest -v` runs the suite; narrow scope with `python -m unittest tests/test_voronoi.py -v`.
- `python -m geodesic_in_heat.cli bunny_phi.vtp --single 0 --contours --out-mesh out.vtp` runs the canonical surface solve. Write artefacts under a disposable path.
- `python -m geodesic_in_heat.cli_volume volume.vtu --gfps 8 --out-mesh phi_surface.vtp` exercises the volume-to-surface flow.
- `python -m geodesic_in_heat.view out.vtp --contours phi_contours.vtp --show-edges` visualizes outputs for quick inspections.

## Coding Style & Naming Conventions
- PEP 8, 4-space indent, descriptive names, and targeted type hints. Keep module-level constants uppercase.
- Functions and module attributes use `snake_case`; classes like `HeatGeodesic` use `UpperCamelCase`; CLI modules stay prefixed `cli_`.
- Raise exceptions for failures; CLI scripts rely on `argparse` and exit cleanly.

## Testing Guidelines
- Extend `unittest.TestCase`; name files `test_*.py` and methods `test_*`.
- Prefer deterministic seeds for sources, and write temporary files to per-test temp directories.
- Cover new algorithms or flags with regression tests and assert expected VTK arrays.

## Commit & Pull Request Guidelines
- Write imperative commits with optional scope tags (e.g., `[cli] add --gfps option`). Only commit relevant sources—not generated meshes or screenshots.
- Pull requests describe motivation, list reproduction commands, call out produced arrays, and link related issues.
- Run formatting and `python -m unittest -v` before requesting review; document platform-specific setup needs.

## Security & Configuration Tips
- Dependencies are pinned in `requirements.txt`; note any OS packages required for `vtk` or `potpourri3d`.
- Keep credentials out of history. Use temp locations for artefacts and clean them before merging.
