"""Brinkhoff'94 single-approximation multi-step baseline + per-pair metrics.

Brinkhoff-Kriegel-Schneider-Seeger (SIGMOD'94) use ONE intermediate approximation
(conservative + progressive) then exact refinement.  We reproduce it as the
2-level special case of the progressive certificate join: {coarsest, exact}.  This
is the honest "prior mechanism" baseline the decode-work law is measured against.
"""
from __future__ import annotations

from lod import LodLadder, Level, n_vertices
from codec import bytes_for_coords
from lod import _ring_qcoords
from joinq import JoinResult, PairResult, mbr_candidates, _exact_intersects


def _two_level_ladder(full: LodLadder) -> LodLadder:
    """View a full ladder as Brinkhoff's [coarsest, exact] two-level ladder."""
    lad = object.__new__(LodLadder)
    lad.exact = full.exact
    lad.levels = [full.levels[0], full.levels[-1]]
    return lad


def brinkhoff_join(geoms_r, geoms_s, candidates=None,
                   ladders_r=None, ladders_s=None) -> JoinResult:
    from joinq import progressive_join
    if ladders_r is None:
        ladders_r = [LodLadder(g) for g in geoms_r]
    if ladders_s is None:
        ladders_s = [LodLadder(g) for g in geoms_s]
    br = [_two_level_ladder(l) for l in ladders_r]
    bs = [_two_level_ladder(l) for l in ladders_s]
    return progressive_join(geoms_r, geoms_s, candidates, br, bs)
