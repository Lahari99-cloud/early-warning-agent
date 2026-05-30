"""
Versioned artifact registry helpers.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from config import ACTIVE_MODEL_VERSION, ARTIFACT_ROOT


LEGACY_MODEL_PATH = "model/xgboost_model.json"
LEGACY_METADATA_PATH = "model/artifact_metadata.json"
LEGACY_FEATURE_COLUMNS_PATH = "model/feature_columns.pkl"
LEGACY_ENCODERS_PATH = "model/label_encoders.pkl"


def artifact_dir(version: str | None = None) -> Path:
    """Return the directory for a model artifact version."""
    return Path(ARTIFACT_ROOT) / (version or ACTIVE_MODEL_VERSION)


def artifact_path(name: str, version: str | None = None) -> Path:
    """Return a named artifact path for a model version."""
    return artifact_dir(version) / name


def active_version_path() -> Path:
    """Return the active-version pointer path."""
    return Path(ARTIFACT_ROOT) / "active_version.txt"


def read_active_version(default: str = ACTIVE_MODEL_VERSION) -> str:
    """Read the active model version pointer, falling back to config."""
    path = active_version_path()
    if not path.exists():
        return default
    value = path.read_text(encoding="utf-8").strip()
    return value or default


def ensure_artifact_dir(version: str | None = None) -> Path:
    """Create and return the artifact directory for a version."""
    path = artifact_dir(version)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: str | Path, content: str) -> None:
    """Write text by replacing the target with a completed temporary file."""
    target = Path(path)
    if target.parent:
        target.parent.mkdir(parents=True, exist_ok=True)
    _preflight_writable(target)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=target.suffix,
        dir=target.parent or Path("."),
        delete=False,
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        tmp_path.replace(target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON by replacing the target with a completed temporary file."""
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def read_json(path: str | Path) -> Any:
    """Read JSON from disk."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _preflight_writable(path: Path) -> None:
    """Fail fast when a target path or directory is locked or not writable."""
    if path.exists():
        with path.open("r+b"):
            pass
    else:
        with path.open("xb"):
            pass
        path.unlink()
