#!/usr/bin/env python3
"""
Batch-convert OFF surface meshes to Wavefront OBJ (.obj).

Usage:
  python scripts/off_to_obj.py /path/to/dir [--recursive]

Notes:
  - Requires meshio
  - Writes <basename>.obj next to each <basename>.off
  - Skips zero-length and non-.off files
"""

from __future__ import annotations

import argparse
from pathlib import Path


def find_off_files(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        return [p for p in root.rglob("*.off") if p.is_file() and p.stat().st_size > 0]
    return [p for p in root.glob("*.off") if p.is_file() and p.stat().st_size > 0]


def convert_off_to_obj(path: Path) -> Path:
    try:
        import meshio
    except Exception as e:
        raise SystemExit(
            "meshio is required. Install with `pip install -r requirements.txt` or `pip install meshio`."
        ) from e

    mesh = meshio.read(str(path))
    out_path = path.with_suffix(".obj")
    meshio.write(str(out_path), mesh, file_format="obj")
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert all .off meshes in a folder to .obj")
    ap.add_argument("folder", help="Folder containing .off files")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    args = ap.parse_args(argv)

    root = Path(args.folder).expanduser()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")

    offs = find_off_files(root, args.recursive)
    if not offs:
        print("No .off files found.")
        return

    print(f"Converting {len(offs)} OFF file(s) in {root} → .obj ...")
    ok = 0
    for p in offs:
        try:
            outp = convert_off_to_obj(p)
            print(f"✔ {p.name} → {outp.name}")
            ok += 1
        except Exception as e:
            print(f"✖ {p.name}: {e}")
    print(f"Done. {ok}/{len(offs)} converted.")


if __name__ == "__main__":
    main()

