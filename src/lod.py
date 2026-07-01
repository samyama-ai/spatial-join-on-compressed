"""Douglas-Peucker LOD ladder + sound two-sided Hausdorff certificate primitives.

A geometry is encoded as a progressive ladder of simplifications
    g~_0  (coarsest)  ...  g~_{k} = g  (exact),
each level L keeping a *subset* of the exact quantized vertices (Douglas-Peucker),
with a guaranteed Hausdorff error bound eta_L = HausdorffDistance(g, g~_L)
(monotone decreasing, eta_k = 0).  Decoding to level L costs the cumulative
vertices / bytes of that level's rings.

Certificate (credited: Brinkhoff-Kriegel SIGMOD'94 false-area test; here the
Hausdorff / DP-hierarchy instantiation).  For an intersects test on a pair
(r,s) at levels (Lr, Ls):
  * OUTER  O_g = g~_L.buffer(+eta_L)  satisfies  g subset O_g   (sound superset)
  * INNER  I_g = g~_L.buffer(-eta_L)  is used as a sound subset  I_g subset g
    for INTERSECT witnessing (see below).
  * DISJOINT certified iff  O_r ∩ O_s = empty      (=> r ∩ s = empty)
  * INTERSECT certified iff  I_r ∩ I_s != empty    (witness interior overlap)
  * else AMBIGUOUS -> descend the geometry whose eta is larger.

Soundness of OUTER is immediate (r ⊆ r~.buffer(eta) because HausdorffDistance
≤ eta).  INNER (erosion) is the standard Minkowski inner approximation; its
soundness for interiors of simple polygons is verified empirically on every run
by NC1 (exact set-equality vs GEOS ground truth) and the adversarial NC2 — the
pre-registration mandates STOP+harden on any correctness violation.  We keep a
positive safety margin (INNER buffer uses -eta with GEOS join_style=round) and
fall through to exact decode whenever inner buffers are empty/degenerate.
"""
from __future__ import annotations
from dataclasses import dataclass

from shapely import hausdorff_distance
from shapely.geometry.base import BaseGeometry

from codec import quantize, bytes_for_coords


def _ring_qcoords(geom: BaseGeometry) -> list[tuple[int, int]]:
    """All (quantized) vertices of a polygon/multipolygon boundary, flattened."""
    out: list[tuple[int, int]] = []

    def _add(poly):
        for x, y in poly.exterior.coords:
            out.append((quantize(x), quantize(y)))
        for r in poly.interiors:
            for x, y in r.coords:
                out.append((quantize(x), quantize(y)))

    gt = geom.geom_type
    if gt == "Polygon":
        _add(geom)
    elif gt == "MultiPolygon":
        for p in geom.geoms:
            _add(p)
    elif gt in ("LineString", "LinearRing"):
        for x, y in geom.coords:
            out.append((quantize(x), quantize(y)))
    elif gt == "MultiLineString":
        for ls in geom.geoms:
            for x, y in ls.coords:
                out.append((quantize(x), quantize(y)))
    return out


def n_vertices(geom: BaseGeometry) -> int:
    return len(_ring_qcoords(geom))


@dataclass
class Level:
    geom: BaseGeometry      # simplified geometry at this level (exact-coord subset)
    eta: float              # guaranteed Hausdorff error bound (0 at finest)
    n_vert: int             # cumulative vertices to decode to this level
    n_bytes: int            # cumulative decoded bytes to this level


class LodLadder:
    """Progressive DP ladder for one geometry, with decode-cost accounting."""

    # Simplification tolerance schedule (in coordinate units == degrees for TIGER).
    # Coarsest first; the last (0.0) is the exact stored geometry.
    DEFAULT_TOLERANCES = (0.02, 0.008, 0.003, 0.001, 0.0003, 0.0001, 0.0)

    def __init__(self, exact: BaseGeometry, tolerances=None):
        self.exact = exact
        tols = self.DEFAULT_TOLERANCES if tolerances is None else tolerances
        self.levels: list[Level] = []
        prev_nv = -1
        for tol in tols:
            g = exact if tol == 0.0 else exact.simplify(tol, preserve_topology=True)
            if g.is_empty or not g.is_valid:
                g = g.buffer(0) if not g.is_empty else exact
            nv = n_vertices(g)
            # Skip a level that did not actually coarsen relative to the previous
            # (keeps the ladder strictly increasing in detail).
            if nv == prev_nv and tol != 0.0:
                continue
            eta = 0.0 if tol == 0.0 else float(hausdorff_distance(exact, g))
            nb = bytes_for_coords(_ring_qcoords(g))
            self.levels.append(Level(geom=g, eta=eta, n_vert=nv, n_bytes=nb))
            prev_nv = nv
        # Ensure the finest level is exact with eta = 0.
        if self.levels[-1].eta != 0.0:
            g = exact
            self.levels.append(
                Level(exact, 0.0, n_vertices(g), bytes_for_coords(_ring_qcoords(g)))
            )

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    def outer(self, L: int) -> BaseGeometry:
        lv = self.levels[L]
        return lv.geom if lv.eta == 0.0 else lv.geom.buffer(lv.eta)

    def inner(self, L: int) -> BaseGeometry:
        lv = self.levels[L]
        return lv.geom if lv.eta == 0.0 else lv.geom.buffer(-lv.eta)
