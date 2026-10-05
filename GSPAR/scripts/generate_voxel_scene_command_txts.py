#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate voxel-map processing commands for each scene in ``dataset.txt``."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NVBLOX_VOXEL_SIZE = "0.05"
NVBLOX_TRUNCATION_DISTANCE = "0.20"
NVBLOX_MAX_ESDF_DISTANCE = "5.0"
GRID_BLOCK_SIZE = "8"
GRID_SCENE_PADDING = "0.02"
PYTHON_COMMAND = ["python"]
DEFAULT_DATASET_FILE = Path("configs/dataset.txt")
DEFAULT_OUTPUT_DIR = Path("GSPAR/voxel_command_txts")


def clean_value(text: str) -> str | None:
    """Treat empty strings and ``none`` as missing values."""

    text = text.strip()
    return None if text == "" or text.lower() == "none" else text


def split_scenes(text: str) -> list[str]:
    """Split a comma-separated scene list."""

    return [item.strip() for item in text.split(",") if item.strip()]


def read_fields(
    lines: list[tuple[int, str]],
    start: int,
    base_indent: int,
) -> tuple[dict, dict[str, dict], int]:
    """Read fields, including grouped overrides of the form ``scenes: a, b:``."""

    fields: dict = {}
    scene_overrides: dict[str, dict] = {}
    i = start
    while i < len(lines):
        indent, text = lines[i]
        if indent < base_indent:
            break
        if indent > base_indent:
            i += 1
            continue
        if ":" not in text:
            raise ValueError(f"Field is missing a colon: {text}")

        key, value = text.split(":", 1)
        key = key.strip()
        raw_value = value.strip()

        # dataset.txt uses this compact form to override fields for a scene group.
        if key == "scenes" and raw_value.endswith(":"):
            scenes = split_scenes(raw_value[:-1])
            override, nested_overrides, i = read_fields(lines, i + 1, base_indent + 1)
            if nested_overrides:
                raise ValueError("A grouped scenes override cannot contain another scenes group")
            for scene in scenes:
                scene_overrides[scene] = merge(scene_overrides.get(scene, {}), override)
            continue

        value = clean_value(value)
        if value is not None:
            fields[key] = value
            i += 1
            continue

        child, nested_overrides, i = read_fields(lines, i + 1, base_indent + 1)
        if nested_overrides:
            raise ValueError(f"Field {key!r} cannot contain a grouped scenes override")
        fields[key] = child or None

    return fields, scene_overrides, i


def merge(base: dict, override: dict) -> dict:
    """Merge default fields with overrides for one scene."""

    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result


def format_scene(value, scene: str):
    """Replace ``{scene}`` placeholders with the current scene name."""

    if isinstance(value, str):
        return value.format(scene=scene)
    if isinstance(value, dict):
        return {key: format_scene(item, scene) for key, item in value.items()}
    return value


def load_records(dataset_file: Path) -> list[dict]:
    """Expand ``dataset.txt`` into one record per scene."""

    raw_lines = []
    for raw in dataset_file.read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.lstrip().startswith("#"):
            raw = raw.replace("：", ":")
            prefix = raw[: len(raw) - len(raw.lstrip())]
            expanded_prefix = prefix.expandtabs(4)
            if len(expanded_prefix) % 4:
                raise ValueError("Each dataset.txt indentation level must use one tab or four spaces")
            raw_lines.append((len(expanded_prefix) // 4, raw.strip()))

    records: list[dict] = []
    i = 0
    while i < len(raw_lines):
        indent, text = raw_lines[i]
        if indent != 0 or not text.endswith(":"):
            i += 1
            continue

        dataset = text[:-1]
        i += 1
        overrides: dict[str, dict] = {}
        format_blocks: list[tuple[dict, dict[str, dict]]] = []

        while i < len(raw_lines) and raw_lines[i][0] > 0:
            indent, text = raw_lines[i]
            if indent != 1 or not text.endswith(":"):
                i += 1
                continue

            name = text[:-1].strip()
            fields, grouped_overrides, i = read_fields(raw_lines, i + 1, 2)
            if name.startswith("format"):
                format_blocks.append((fields, grouped_overrides))
            else:
                selector = name.split(":", 1)[1] if name.startswith("scenes:") else name
                for scene in split_scenes(selector):
                    overrides[scene] = merge(overrides.get(scene, {}), fields)

        for fields, grouped_overrides in format_blocks:
            if not isinstance(fields.get("scenes"), str):
                raise ValueError(f"Format block for dataset {dataset!r} is missing the scenes field")
            scenes = split_scenes(fields["scenes"])
            # Grouped configuration may override listed scenes or add new ones.
            # For example, tanks_and_temples lists m60/panther only in its indoor group.
            scenes.extend(scene for scene in grouped_overrides if scene not in scenes)
            base_fields = {key: value for key, value in fields.items() if key != "scenes"}
            for scene in scenes:
                scene_fields = merge(base_fields, grouped_overrides.get(scene, {}))
                scene_fields = merge(scene_fields, overrides.get(scene, {}))
                scene_fields = format_scene(scene_fields, scene)
                records.append({"dataset": dataset, "scene": scene, "fields": scene_fields})

    return records


def cmd(parts: list) -> str:
    """Format an argument list as a readable multiline shell command."""

    quoted = [shlex.quote(str(part)) for part in parts]
    first = len(PYTHON_COMMAND) + 1 if quoted[:len(PYTHON_COMMAND)] == PYTHON_COMMAND else 1
    if first > 1 and len(quoted) > first and quoted[len(PYTHON_COMMAND)] == "-m":
        # Keep ``python -m package.module`` together before command arguments.
        first += 1
    lines = [" ".join(quoted[:first])]
    index = first
    while index < len(quoted):
        end = index + 1
        if str(parts[index]).startswith("--"):
            while end < len(quoted) and not str(parts[end]).startswith("--"):
                end += 1
        lines.append("  " + " ".join(quoted[index:end]))
        index = end
    return " \\\n".join(lines)


def configured_path(value: str | Path) -> Path:
    """Use a path from ``dataset.txt`` without remapping or checking it."""

    return Path(value)


def ground_truth_mesh_path(value, dataset: str, scene: str) -> Path | None:
    """Resolve a ground-truth mesh, preferring ``mesh_aligned*`` in directories."""

    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("mesh") or value.get("path")
    if not isinstance(value, (str, Path)):
        raise ValueError(f"The gt value for {dataset}/{scene} must be a mesh path, directory, or mesh/path field")

    path = configured_path(value)
    if path.suffix:
        return path
    if not path.is_dir():
        raise ValueError(f"The GT directory for {dataset}/{scene} does not exist and no mesh file is configured: {path}")

    supported = {".ply", ".obj", ".stl", ".off", ".gltf", ".glb"}
    candidates = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in supported
    )
    aligned = [candidate for candidate in candidates if candidate.stem.startswith("mesh_aligned")]
    if aligned:
        return aligned[0]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"The GT directory for {dataset}/{scene} must contain one mesh or a mesh_aligned* file: {path}"
    )


def gspar_output_path(value: str | Path) -> Path:
    """Resolve relative output paths under GSPAR and preserve absolute paths."""

    path = Path(value)
    if path.is_absolute() or path.parts[:1] == ("GSPAR",):
        return path
    return Path("GSPAR") / path


def add_section(lines: list[str], name: str, title: str, commands: list[str]) -> None:
    """Add a shell-comment heading while preserving the machine-readable command name."""

    if commands:
        lines.append(f"# command: {name} - {title}")
        lines.extend(commands)
        lines.append("")


def scene_text(record: dict) -> str:
    """Generate mapping, validation, and GUI commands for one scene."""

    dataset = record["dataset"]
    scene = record["scene"]
    f = record["fields"]

    output = gspar_output_path(f["output"])
    checkpoint = configured_path(f["3dgs model"])
    training_config = configured_path(f["training_config"])
    images = configured_path(f["images"]) if f.get("images") else None
    camera_sparse = configured_path(f["sparse"]) if f.get("sparse") else None
    voxel_size = f.get("voxel size") or NVBLOX_VOXEL_SIZE
    source_voxel_size = f.get("source voxel size") or voxel_size
    target_voxel_size = f.get("target voxel size") or voxel_size
    scene_type = f.get("scene type")
    sparse_prefix = output / "voxel_map"
    nvblox = f.get("nvblox version")
    has_tsdf = isinstance(nvblox, dict) and nvblox.get("sparse") and nvblox.get("depth")
    nvblox_sparse = configured_path(nvblox["sparse"]) if has_tsdf else None
    depth_dir = configured_path(nvblox["depth"]) if has_tsdf else None
    tsdf_dir = output / "sdf"
    tsdf_prefix = output / "voxel_map_tsdf"
    gt_mesh = ground_truth_mesh_path(f.get("gt"), dataset, scene)
    has_gt = gt_mesh is not None
    map_frame_transform = output / "map_frame_transform.json" if has_gt else None
    gt_prefix = output / "voxel_map_gt"

    lines = [
        f"# Dataset: {dataset}",
        f"# Scene: {scene}",
        f"# 3DGS: {checkpoint}",
        f"# RGB: {images or '(not configured)'}",
        f"# Camera intrinsics and poses (COLMAP sparse): {camera_sparse or nvblox_sparse or '(not configured)'}",
        f"# Depth maps: {depth_dir or '(not configured)'}",
    ]
    if has_gt:
        lines.append(f"# GT mesh: {gt_mesh}")
    lines += [
        f"# Training configuration: {training_config}",
        f"# Output directory: {output}",
        (
            "# Note: The GT mesh defines occupied space; nvblox observed-free voxels only support free/unknown labels and endpoint sampling."
            if has_gt
            else "# Note: nvblox reads only depth maps and COLMAP sparse data; the training configuration and GUI use RGB images."
        ),
        "",
    ]
    step = 1

    if has_gt:
        add_section(
            lines,
            "estimate_gt_axis_alignment",
            f"{step}. Estimate shared Manhattan-axis alignment from the GT mesh",
            [cmd([
                *PYTHON_COMMAND, "-m", "GSPAR.mapping.ground_truth.estimate_axis_alignment",
                "--mesh", gt_mesh,
                "--output", map_frame_transform,
            ])],
        )
        step += 1

    sparse_parts = [
        *PYTHON_COMMAND, "GSPAR/mapping/gaussian_occupancy/build_sparse_voxel_map.py",
        "--checkpoint", checkpoint,
        "--output-prefix", sparse_prefix,
        "--voxel-size", target_voxel_size,
        "--block-size", GRID_BLOCK_SIZE,
        "--scene-padding", GRID_SCENE_PADDING,
    ]
    if map_frame_transform is not None:
        sparse_parts += ["--map-frame-transform", map_frame_transform]
    if scene_type:
        sparse_parts += ["--scene-type", scene_type]
    add_section(
        lines,
        "build_sparse_voxel_map",
        f"{step}. Convert 3DGS to voxels",
        [cmd(sparse_parts)],
    )
    step += 1

    if has_tsdf:
        explicit_max_depth = (
            f.get("max_depth")
            or f.get("max depth")
            or f.get("nvblox max_depth")
            or f.get("nvblox max depth")
        )
        nvblox_parts = [
            "./GSPAR/nvblox_gpu_sdf/build/nvblox_gpu_sdf_offline",
            "--colmap_sparse", nvblox_sparse,
            "--depth_dir", depth_dir,
            "--output_dir", tsdf_dir,
            "--voxel_size", source_voxel_size,
            "--truncation_distance", NVBLOX_TRUNCATION_DISTANCE,
            "--max_esdf_distance", NVBLOX_MAX_ESDF_DISTANCE,
        ]
        if explicit_max_depth is not None:
            nvblox_parts += ["--max_depth", explicit_max_depth]
        add_section(
            lines,
            "nvblox_gpu_sdf_offline",
            f"{step}. Generate TSDF/ESDF with nvblox from depth maps and camera poses",
            [cmd(nvblox_parts)],
        )
        step += 1

    if has_tsdf:
        tsdf_parts = [
            *PYTHON_COMMAND, "GSPAR/mapping/build_nvblox_tsdf_voxel_map.py",
            "--tsdf-ply", tsdf_dir / "tsdf_observed.ply",
            "--output-prefix", tsdf_prefix,
            "--grid-checkpoint", checkpoint,
            "--training-config", training_config,
            "--source-voxel-size", source_voxel_size,
            "--voxel-size", target_voxel_size,
            "--block-size", GRID_BLOCK_SIZE,
            "--scene-padding", GRID_SCENE_PADDING,
        ]
        if map_frame_transform is not None:
            tsdf_parts += ["--map-frame-transform", map_frame_transform]
        add_section(
            lines,
            "build_nvblox_tsdf_voxel_map",
            f"{step}. Convert TSDF to voxels",
            [cmd(tsdf_parts)],
        )
        step += 1

    if has_gt:
        gt_parts = [
            *PYTHON_COMMAND, "-m", "GSPAR.mapping.ground_truth.mesh_to_voxel",
            "--mesh", gt_mesh,
            "--grid-map", sparse_prefix.with_suffix(".npz"),
            "--output-prefix", gt_prefix,
            "--map-frame-transform", map_frame_transform,
            # nvblox zero-level meshes use the same shared mesh-vertex
            # voxelizer, making geometry metrics independent of TSDF bands.
            "--surface-mode", "vertices",
        ]
        if has_tsdf:
            gt_parts += ["--free-space-map", tsdf_prefix.with_suffix(".npz")]
        add_section(
            lines,
            "build_ground_truth_mesh_voxel_map",
            f"{step}. Convert the GT mesh to an independent ground-truth voxel reference",
            [cmd(gt_parts)],
        )
        step += 1

    if has_tsdf or has_gt:
        alignment_commands = []
        if has_tsdf:
            alignment_commands.append(cmd([
                *PYTHON_COMMAND, "GSPAR/mapping/verify_voxel_grid_alignment.py",
                "--first", sparse_prefix.with_suffix(".npz"),
                "--second", tsdf_prefix.with_suffix(".npz"),
            ]))
        if has_gt:
            alignment_commands.append(cmd([
                *PYTHON_COMMAND, "GSPAR/mapping/verify_voxel_grid_alignment.py",
                "--first", sparse_prefix.with_suffix(".npz"),
                "--second", gt_prefix.with_suffix(".npz"),
            ]))
        add_section(
            lines,
            "verify_voxel_grid_alignment",
            (
                f"{step}. Verify the shared grid for 3DGS, TSDF, and GT voxels"
                if has_gt and has_tsdf
                else f"{step}. Verify 3DGS and GT voxel-grid alignment"
                if has_gt
                else f"{step}. Verify 3DGS and TSDF voxel-grid alignment"
            ),
            alignment_commands,
        )
        step += 1

    add_section(
        lines,
        "gui",
        (
            f"{step}. Launch the GUI and compare 3DGS with the GT voxel map in Occupancy Overlay"
            if has_gt
            else f"{step}. Launch the GUI to toggle, overlay, and compare both voxel maps"
        ),
        [cmd([
            *PYTHON_COMMAND, "GSPAR/gui/launchViewerOverlay.py",
            "--training-config-path", training_config,
            "--checkpoint-path", checkpoint,
            "--voxel-map", sparse_prefix,
            *(
                ["--extra-voxel-map", gt_prefix]
                if has_gt
                else ["--extra-voxel-map", tsdf_prefix]
                if has_tsdf
                else []
            ),
        ])],
    )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    """Run the command-line entry point."""

    os.chdir(PROJECT_ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    records = load_records(args.dataset_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        scene_dir = args.output_dir / record["dataset"]
        scene_dir.mkdir(parents=True, exist_ok=True)
        (scene_dir / f"{record['scene']}.txt").write_text(
            scene_text(record),
            encoding="utf-8",
        )

    print(f"Wrote {len(records)} scene command files to {args.output_dir}")


if __name__ == "__main__":
    main()
