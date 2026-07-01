"""Signed clearance margin m(r,s) -- the (amended) H1' predictor of decode work.

PRE-REGISTERED AMENDMENT (frozen 2026-07-01, logged in HYPOTHESIS.md):
The original band-vertex predictor (band.py) was REJECTED on the synthetic pilot
(held-out R^2=0.36, NEGATIVE slope): it counts vertices near the other boundary,
which is a proxy for overlap ROBUSTNESS and therefore ANTI-correlates with decode
(deep robust overlaps have a big near-boundary band yet certify instantly).

The correct, geometry-only, non-circular predictor is the MARGIN -- the signed
clearance of the pair from the predicate-flip boundary:

  m(r,s) = + inradius(r ∩ s)     if they intersect   (overlap depth)
         = - distance(r, s)      if disjoint          (gap)

The certificate must descend the LOD ladder until its resolution eta_L drops below
|m|; hence the certifying DEPTH L* ~ min{L : eta_L < |m|} depends on |m| alone and
NOT on object size, while the decoded VERTEX count adds the ladder's size profile.
This is the instance-optimal law: cost is set by proximity to the decision boundary.
"""
from __future__ import annotations

import shapely
from shapely.geometry.base import BaseGeometry

from codec import GRID

_MIC_TOL = GRID * 50   # inscribed-circle search tolerance (~0.5m); << feature size


def signed_margin(r: BaseGeometry, s: BaseGeometry) -> tuple[float, int]:
    """Return (m, intersects): signed clearance and whether they intersect.

    m > 0 : overlap depth (inradius of intersection)
    m < 0 : gap (negative distance) for disjoint pairs
    m ~ 0 : tangent / boundary-grazing (hardest to certify)
    """
    inter = r.intersection(s)
    if inter.is_empty:
        return -float(r.distance(s)), 0
    # Non-areal intersection (touch along a line/point) => margin ~ 0.
    if inter.area <= 0.0:
        return 0.0, 1
    try:
        mic = shapely.maximum_inscribed_circle(inter, _MIC_TOL)
        return float(mic.length), 1
    except Exception:
        # Fallback: representative clearance via negative-buffer probe.
        lo, hi = 0.0, float(max(inter.bounds[2] - inter.bounds[0],
                                inter.bounds[3] - inter.bounds[1])) or 1.0
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if inter.buffer(-mid).is_empty:
                hi = mid
            else:
                lo = mid
        return lo, 1
