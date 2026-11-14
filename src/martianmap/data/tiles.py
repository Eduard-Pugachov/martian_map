"""Tiling and spatial block splits (never random tiles: neighbouring tiles leak)."""

import numpy as np

TRAIN, VAL, TEST, DROPPED = 0, 1, 2, -1
SPLIT_NAMES = {TRAIN: "train", VAL: "val", TEST: "test"}


def block_split_map(n_rows: int, n_cols: int, val_blocks: list[int], test_blocks: list[int]) -> np.ndarray:
    """Split id per block; ranges are [r0, r1, c0, c1) with exclusive ends. Default = TRAIN."""
    m = np.full((n_rows, n_cols), TRAIN, np.int8)
    for split, (r0, r1, c0, c1) in ((VAL, val_blocks), (TEST, test_blocks)):
        m[r0:r1, c0:c1] = split
    return m


def assign_split(r: int, c: int, tile: int, buffer: int, block: int, split_map: np.ndarray) -> int:
    """Split of the tile at pixel (r, c), or DROPPED if the tile grown by `buffer` px touches >1 split.

    Guarantees tiles of different splits are >= buffer px apart.
    """
    r0, c0 = max(r - buffer, 0) // block, max(c - buffer, 0) // block
    r1 = min((r + tile + buffer - 1) // block, split_map.shape[0] - 1)
    c1 = min((c + tile + buffer - 1) // block, split_map.shape[1] - 1)
    splits = np.unique(split_map[r0:r1 + 1, c0:c1 + 1])
    return int(splits[0]) if len(splits) == 1 else DROPPED


def tile_origins(height: int, width: int, tile: int) -> list[tuple[int, int]]:
    """Top-left pixel of every full, non-overlapping tile."""
    return [(r, c) for r in range(0, height - tile + 1, tile) for c in range(0, width - tile + 1, tile)]
