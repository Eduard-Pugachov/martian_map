"""Load config.yaml and resolve repo-relative paths."""

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path = REPO_ROOT / "config.yaml") -> dict[str, Any]:
    """Read config.yaml as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def repo_path(p: str) -> Path:
    """Resolve a config path relative to the repo root."""
    return REPO_ROOT / p
