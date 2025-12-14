from __future__ import annotations

from pathlib import Path

import hydra


def resolve_path(path_like: str | Path) -> Path:
    path_obj = Path(path_like)
    if path_obj.is_absolute():
        return path_obj
    try:
        base = Path(hydra.utils.get_original_cwd())
    except Exception:
        base = Path.cwd()
    return (base / path_obj).expanduser().resolve()
