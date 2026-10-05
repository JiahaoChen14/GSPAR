#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Verify that two voxel maps use the same index-to-world lattice."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GSPAR.mapping.voxel_map_cli import resolve_existing_path


REQUIRED_KEYS = ("voxel_size", "origin", "voxel_dims")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查两个 voxel map 是否使用完全相同的网格。"
    )
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    return parser.parse_args()


def _load_grid(path: Path) -> dict[str, object]:
    path = resolve_existing_path(path, default_suffix=".npz")
    if not path.is_file():
        raise FileNotFoundError(f"Voxel map not found: {path}")
    with np.load(path) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"{path} missing grid keys: {missing}")
        return {
            "voxel_size": np.asarray(data["voxel_size"], dtype=np.float32).reshape(1),
            "origin": np.asarray(data["origin"], dtype=np.float32).reshape(3),
            "voxel_dims": np.asarray(data["voxel_dims"], dtype=np.int32).reshape(3),
            "map_from_training_world": (
                np.asarray(data["map_from_training_world"], dtype=np.float32).reshape(4, 4)
                if "map_from_training_world" in data
                else np.eye(4, dtype=np.float32)
            ),
        }


def main() -> None:
    args = parse_args()
    first = _load_grid(args.first)
    second = _load_grid(args.second)
    aligned = bool(
        np.array_equal(first["voxel_size"], second["voxel_size"])
        and np.array_equal(first["origin"], second["origin"])
        and np.array_equal(first["voxel_dims"], second["voxel_dims"])
        and np.allclose(
            first["map_from_training_world"],
            second["map_from_training_world"],
            atol=1e-6,
        )
    )
    print("Voxel grids aligned: yes" if aligned else "Voxel grids aligned: no")
    if not aligned:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
