"""Delta + zig-zag + varint coordinate codec, with exact byte accounting.

The compressed store models a realistic progressively-decodable geometry format
(the TWKB / delta-varint family): coordinates are quantized to a fixed grid, then
each ring is stored as (first point) + zig-zag varint deltas between successive
points.  We never *decompress lossily* for correctness: the quantized coordinates
ARE the data of record — the exact ground-truth join runs on the very same
quantized geometry (see data.py:quantize_geom).  Douglas-Peucker LOD levels keep a
*subset* of these quantized vertices, so the finest level reproduces the stored
geometry bit-for-bit (eta = 0).  This makes certificate pruning provably sound:
coarse levels can only be *approximations* of the exact stored polygon, never a
different polygon.

`bytes_for_coords(n)` gives the decoded-byte cost of touching n coordinate pairs
in this encoding; the join instruments decode work in BOTH vertices and bytes.
"""
from __future__ import annotations

# Fixed coordinate grid: 1e-7 degrees ~ 1.1 cm at the equator (TIGER-grade).
GRID = 1e-7


def quantize(x: float) -> int:
    """Map a float coordinate to an integer grid cell (round-half-to-even)."""
    return int(round(x / GRID))


def _zigzag(n: int) -> int:
    return (n << 1) ^ (n >> 63)


def _varint_len(u: int) -> int:
    """Number of bytes an unsigned LEB128 varint occupies."""
    if u == 0:
        return 1
    n = 0
    while u:
        u >>= 7
        n += 1
    return n


def delta_varint_bytes(qcoords: list[tuple[int, int]]) -> int:
    """Exact byte length of delta+zigzag+varint encoding a quantized ring.

    First point is stored as two absolute zig-zag varints; each subsequent point
    as two zig-zag varint deltas.  This is the byte model used for decode cost.
    """
    if not qcoords:
        return 0
    total = 0
    px, py = 0, 0
    for (x, y) in qcoords:
        total += _varint_len(_zigzag(x - px)) + _varint_len(_zigzag(y - py))
        px, py = x, y
    return total


def bytes_for_coords(qcoords: list[tuple[int, int]]) -> int:
    """Decoded byte cost of materialising exactly these quantized vertices."""
    return delta_varint_bytes(qcoords)
