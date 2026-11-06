"""Config loading helpers."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "params.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(p: str | Path) -> Path:
    """Resolve a path from the config relative to the project root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def rel_or_abs(p: str | Path) -> str:
    """Store paths relative to the project root when possible (portable DB rows)."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def get_camera(cfg: dict, camera_id: str | None = None) -> dict:
    cams = cfg.get("cameras") or []
    if not cams:
        raise KeyError("No cameras defined in params.yaml")
    if camera_id is None:
        return cams[0]
    for c in cams:
        if c["id"] == camera_id:
            return c
    raise KeyError(f"Camera '{camera_id}' not found in params.yaml (have: {[c['id'] for c in cams]})")
