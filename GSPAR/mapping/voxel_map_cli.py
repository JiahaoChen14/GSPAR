#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Path helpers shared by voxel-map scripts."""

from __future__ import annotations

from pathlib import Path


def normalize_output_prefix(path: str | Path) -> Path:
    """Normalize ``xxx.npz`` or ``xxx.json`` output paths to the prefix ``xxx``."""

    path = Path(path).expanduser()
    return path.with_suffix("") if path.suffix.lower() in {".npz", ".json"} else path


def resolve_existing_path(path: str | Path, *, default_suffix: str | None = None) -> Path:
    """Resolve a complete input filename or a prefix without its default suffix."""

    path = Path(path).expanduser()
    if default_suffix is not None and path.suffix == "":
        candidate = path.with_suffix(default_suffix)
        if candidate.is_file():
            return candidate.resolve()
    return path.resolve()


def resolve_existing_optional_path(path: str | Path | None, *, default_suffix: str | None = None) -> Path | None:
    """Resolve an optional input path while preserving ``None``."""

    return None if path is None else resolve_existing_path(path, default_suffix=default_suffix)
