"""Load config.yaml into a dict."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read a YAML config file and return it as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
