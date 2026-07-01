"""eta-ambiguous boundary band b(r,s) -- the H1 predictor.

PRE-REGISTERED DEFINITION (frozen 2026-07-01, before any run; band.py is the
predictor the decode-work law is fit against, so it must be a pure function of
GEOMETRY, independent of the join's decision path -- otherwise H1 is circular and
NC3 is meaningless).

  b(r, s) = #{exact vertices of r within w of boundary(s), inside MBR-overlap}
          + #{exact vertices of s within w of boundary(r), inside MBR-overlap}

with the reference band width w set purely by the two ladders' coarsest error
scales (NOT by where the pair certified):

  w(r, s) = geometric_mean(eta0_r, eta0_s)

where eta0_g is the Hausdorff error of geometry g's COARSEST LOD level.  Pairs whose
boundaries are cleanly separated at coarse scale contribute ~0 band vertices;
pairs whose boundaries interleave contribute many.  The law asserts decode work
D(r,s) is governed by b(r,s), not by total object size |r|+|s|.
"""
from __future__ import annotations
from math import sqrt

import shapely
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


def _boundary(g: BaseGeometry) -> BaseGeometry:
    return g.boundary


def _mbr_overlap(a: BaseGeometry, b: BaseGeometry):
    ax0, ay0, ax1, ay1 = a.bounds
    bx0, by0, bx1, by1 = b.bounds
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x0 > x1 or y0 > y1:
        return None
    return box(x0, y0, x1, y1)


def _verts_near(g: BaseGeometry, other_boundary: BaseGeometry, w: float,
                region) -> int:
    """Count exact vertices of g within w of other's boundary, inside region."""
    if region is None or w <= 0:
        return 0
    n = 0
    coords = shapely.get_coordinates(g)
    for x, y in coords:
        p = shapely.Point(x, y)
        if region.covers(p) and other_boundary.distance(p) <= w:
            n += 1
    return n


def band_vertices(r: BaseGeometry, s: BaseGeometry,
                  eta0_r: float, eta0_s: float) -> tuple[int, float]:
    """Return (b, w): pre-registered band-vertex count and reference width."""
    w = sqrt(max(eta0_r, 1e-12) * max(eta0_s, 1e-12))
    region = _mbr_overlap(r, s)
    if region is None:
        return 0, w
    b = (_verts_near(r, _boundary(s), w, region)
         + _verts_near(s, _boundary(r), w, region))
    return b, w
