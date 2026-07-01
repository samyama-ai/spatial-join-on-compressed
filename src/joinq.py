"""Band-limited progressive intersects join + baselines, with decode accounting.

Every method starts from the SAME MBR candidate set (shared STRtree filter) so the
measured decode delta is purely refinement depth (fairness pitfall #3, BRIEF).

Decode accounting is per candidate pair, WITHOUT cross-pair caching (apples-to-apples
with naive-refine, which decodes all vertices of every MBR survivor).  We report
vertices AND bytes.  A cached variant (decode each geometry to its deepest level once)
is available via `progressive_join(..., cache=True)` as a secondary metric.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import shapely
from shapely import STRtree
from shapely.geometry.base import BaseGeometry

from lod import LodLadder


@dataclass
class PairResult:
    i: int
    j: int
    intersects: bool
    dec_vert: int          # vertices decoded to certify this pair (both geoms)
    dec_bytes: int         # bytes decoded to certify this pair
    cert_level_r: int      # level of r at certification
    cert_level_s: int
    eta_star: float        # max(eta_r, eta_s) at certification (0 if exact)
    total_vert: int        # |r| + |s| (naive-refine cost for this pair)


@dataclass
class JoinResult:
    pairs: set[tuple[int, int]]                  # intersecting (i,j) pairs
    per_pair: list[PairResult] = field(default_factory=list)
    dec_vert: int = 0
    dec_bytes: int = 0
    total_vert: int = 0                          # sum over candidate pairs (naive cost)
    n_candidates: int = 0

    def add(self, pr: PairResult):
        self.per_pair.append(pr)
        self.dec_vert += pr.dec_vert
        self.dec_bytes += pr.dec_bytes
        self.total_vert += pr.total_vert
        self.n_candidates += 1
        if pr.intersects:
            self.pairs.add((pr.i, pr.j))


def mbr_candidates(geoms_r, geoms_s) -> list[tuple[int, int]]:
    """Shared MBR/STRtree filter: candidate (i,j) whose bounding boxes overlap."""
    tree = STRtree(geoms_s)
    cands: list[tuple[int, int]] = []
    for i, g in enumerate(geoms_r):
        for j in tree.query(g, predicate="intersects"):  # bbox-level via envelope
            cands.append((i, int(j)))
    return cands


def _exact_intersects(a: BaseGeometry, b: BaseGeometry) -> bool:
    return bool(a.intersects(b))


def progressive_join(geoms_r, geoms_s, candidates=None,
                     ladders_r=None, ladders_s=None) -> JoinResult:
    """The certificate-margin progressive join over DP-LOD ladders.

    For each candidate pair, start both geometries at level 0 and:
      - certify DISJOINT  if outer(r) ∩ outer(s) == empty
      - certify INTERSECT if inner(r) ∩ inner(s) != empty (non-degenerate)
      - else descend the geometry with the larger eta by one level; repeat.
    Falls through to an exact GEOS test at the finest level (eta = 0), which is
    always reached in the worst case (Omega(v) read bound).
    """
    if candidates is None:
        candidates = mbr_candidates(geoms_r, geoms_s)
    if ladders_r is None:
        ladders_r = [LodLadder(g) for g in geoms_r]
    if ladders_s is None:
        ladders_s = [LodLadder(g) for g in geoms_s]

    res = JoinResult(pairs=set())
    for (i, j) in candidates:
        Lr, Ls = 0, 0
        lr, ls = ladders_r[i], ladders_s[j]
        nlr, nls = lr.n_levels, ls.n_levels
        decided = None
        while decided is None:
            er, es = lr.levels[Lr].eta, ls.levels[Ls].eta
            if er == 0.0 and es == 0.0:
                # Both exact: definitive GEOS test.
                decided = _exact_intersects(lr.exact, ls.exact)
                break
            outer_r, outer_s = lr.outer(Lr), ls.outer(Ls)
            if not outer_r.intersects(outer_s):
                decided = False          # certified DISJOINT
                break
            inner_r, inner_s = lr.inner(Lr), ls.inner(Ls)
            if (not inner_r.is_empty) and (not inner_s.is_empty) \
                    and inner_r.intersects(inner_s):
                decided = True           # certified INTERSECT
                break
            # Ambiguous: descend the coarser (larger-eta) side.
            if er >= es and Lr < nlr - 1:
                Lr += 1
            elif Ls < nls - 1:
                Ls += 1
            elif Lr < nlr - 1:
                Lr += 1
            else:
                decided = _exact_intersects(lr.exact, ls.exact)
                break
        dv = lr.levels[Lr].n_vert + ls.levels[Ls].n_vert
        db = lr.levels[Lr].n_bytes + ls.levels[Ls].n_bytes
        tv = lr.levels[-1].n_vert + ls.levels[-1].n_vert
        res.add(PairResult(i, j, bool(decided), dv, db, Lr, Ls,
                           max(lr.levels[Lr].eta, ls.levels[Ls].eta), tv))
    return res


def naive_refine_join(geoms_r, geoms_s, candidates=None,
                      ladders_r=None, ladders_s=None) -> JoinResult:
    """Decode ALL vertices of every MBR survivor, then exact test.

    The honest decode baseline the decode-work law must beat (H3).  Same MBR
    candidate set; decode cost = full vertex/byte count of both geometries.
    """
    if candidates is None:
        candidates = mbr_candidates(geoms_r, geoms_s)
    if ladders_r is None:
        ladders_r = [LodLadder(g) for g in geoms_r]
    if ladders_s is None:
        ladders_s = [LodLadder(g) for g in geoms_s]
    res = JoinResult(pairs=set())
    for (i, j) in candidates:
        lr, ls = ladders_r[i], ladders_s[j]
        tv = lr.levels[-1].n_vert + ls.levels[-1].n_vert
        tb = lr.levels[-1].n_bytes + ls.levels[-1].n_bytes
        inter = _exact_intersects(lr.exact, ls.exact)
        res.add(PairResult(i, j, inter, tv, tb, lr.n_levels - 1, ls.n_levels - 1,
                           0.0, tv))
    return res


def exact_join(geoms_r, geoms_s, candidates=None) -> set[tuple[int, int]]:
    """Full-precision GEOS ground truth (the correctness reference)."""
    if candidates is None:
        candidates = mbr_candidates(geoms_r, geoms_s)
    out: set[tuple[int, int]] = set()
    for (i, j) in candidates:
        if _exact_intersects(geoms_r[i], geoms_s[j]):
            out.add((i, j))
    return out
