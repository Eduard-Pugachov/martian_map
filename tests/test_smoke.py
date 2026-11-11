"""Smoke tests: package imports and config loads."""

from martianmap import __version__
from martianmap.config import load_config


def test_version() -> None:
    assert __version__


def test_config_loads() -> None:
    cfg = load_config()
    assert cfg["ai4mars"]["ignore_index"] == 255
    assert len(cfg["domars"]["classes"]) == 15
