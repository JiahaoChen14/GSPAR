"""Read scene configuration and resolve dataset paths."""

from __future__ import annotations
from pathlib import Path


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
            raise ValueError(f"字段缺少冒号: {text}")

        key, value = text.split(":", 1)
        key = key.strip()
        raw_value = value.strip()

        # dataset.txt uses this compact form to override fields for a scene group.
        if key == "scenes" and raw_value.endswith(":"):
            scenes = split_scenes(raw_value[:-1])
            override, nested_overrides, i = read_fields(lines, i + 1, base_indent + 1)
            if nested_overrides:
                raise ValueError("scenes 分组覆盖中不能再嵌套 scenes 分组")
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
            raise ValueError(f"字段 {key!r} 中不能使用 scenes 分组覆盖")
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
                raise ValueError("dataset.txt 每一级必须使用一个 tab 或四个空格缩进")
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
                raise ValueError(f"数据集 {dataset!r} 的 format 缺少 scenes 字段")
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
        raise ValueError(f"{dataset}/{scene} 的 gt 必须是 mesh 路径、目录或包含 mesh/path 的字段")

    path = configured_path(value)
    if path.suffix:
        return path
    if not path.is_dir():
        raise ValueError(f"{dataset}/{scene} 的 GT 目录不存在，且没有配置明确 mesh 文件: {path}")

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
        f"{dataset}/{scene} 的 GT 目录需要唯一 mesh，或文件名以 mesh_aligned 开头: {path}"
    )


def gspar_output_path(value: str | Path) -> Path:
    """Resolve relative output paths under GSPAR and preserve absolute paths."""

    path = Path(value)
    if path.is_absolute() or path.parts[:1] == ("GSPAR",):
        return path
    return Path("GSPAR") / path
