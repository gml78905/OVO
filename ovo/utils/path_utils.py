from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("/ws/data/OVO")
_DATA_PREFIXES = ("data", "./data")


def get_data_root() -> Path:
    return Path(os.environ.get("OVO_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser()


def get_input_root() -> Path:
    return get_data_root() / "input"


def get_datasets_root() -> Path:
    return get_input_root() / "Datasets"


def get_output_root() -> Path:
    return get_data_root() / "output"


def get_working_root() -> Path:
    return get_data_root() / "working"


def get_configs_root() -> Path:
    return get_working_root() / "configs"


def get_working_output_root() -> Path:
    return get_working_root() / "output"


def is_data_path(value: str | Path) -> bool:
    normalized = str(value).replace("\\", "/")
    return normalized in _DATA_PREFIXES or any(
        normalized.startswith(prefix + "/") for prefix in _DATA_PREFIXES
    )


def resolve_data_path(value: str | Path, *, data_root: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    normalized = str(value).replace("\\", "/")
    if normalized == "data":
        suffix = ""
    elif normalized.startswith("data/"):
        suffix = normalized[len("data/") :]
    elif normalized == "./data":
        suffix = ""
    elif normalized.startswith("./data/"):
        suffix = normalized[len("./data/") :]
    else:
        return path

    root = data_root or get_data_root()
    return root / suffix


def remap_data_paths(value: Any, *, data_root: Path | None = None) -> Any:
    if isinstance(value, dict):
        return {
            key: remap_data_paths(item, data_root=data_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [remap_data_paths(item, data_root=data_root) for item in value]
    if isinstance(value, tuple):
        return tuple(remap_data_paths(item, data_root=data_root) for item in value)
    if isinstance(value, str) and is_data_path(value):
        return str(resolve_data_path(value, data_root=data_root))
    return value
