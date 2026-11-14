import itertools

import numpy as np

from martianmap.data.tiles import DROPPED, TEST, TRAIN, VAL, assign_split, block_split_map, tile_origins


def gap(a: tuple[int, int], b: tuple[int, int], tile: int) -> int:
    """Chebyshev gap in px between two tile squares (0 if they touch or overlap)."""
    return max(abs(a[0] - b[0]) - tile, abs(a[1] - b[1]) - tile, 0)


def test_no_tile_within_buffer_of_another_split() -> None:
    block, tile, buffer = 100, 40, 30
    split_map = block_split_map(6, 6, val_blocks=[0, 2, 4, 6], test_blocks=[3, 5, 1, 3])
    tiles = [(r, c, assign_split(r, c, tile, buffer, block, split_map)) for r, c in tile_origins(600, 600, tile)]
    kept = [t for t in tiles if t[2] != DROPPED]
    assert {t[2] for t in kept} == {TRAIN, VAL, TEST}
    for (ra, ca, sa), (rb, cb, sb) in itertools.combinations(kept, 2):
        if sa != sb:
            assert gap((ra, ca), (rb, cb), tile) >= buffer


def test_tile_lies_inside_its_split() -> None:
    block, tile = 100, 40
    split_map = block_split_map(4, 4, val_blocks=[0, 1, 0, 1], test_blocks=[2, 4, 2, 4])
    for r, c in tile_origins(400, 400, tile):
        s = assign_split(r, c, tile, 0, block, split_map)
        if s != DROPPED:
            assert (split_map[r // block:(r + tile - 1) // block + 1, c // block:(c + tile - 1) // block + 1] == s).all()


def test_block_split_map_ranges_are_end_exclusive() -> None:
    m = block_split_map(3, 3, val_blocks=[0, 1, 0, 1], test_blocks=[2, 3, 1, 3])
    assert m.tolist() == [[VAL, TRAIN, TRAIN], [TRAIN, TRAIN, TRAIN], [TRAIN, TEST, TEST]]


def test_tile_origins_full_tiles_only() -> None:
    assert tile_origins(100, 130, 50) == [(0, 0), (0, 50), (50, 0), (50, 50)]
