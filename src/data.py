"""Datasets: quantization, real TIGER loader, synthetic control, adversarial fixtures.

All geometries are quantized to the codec grid ON INGEST (data.py:quantize_geom);
the quantized geometry is the data of record for BOTH the compressed store and the
exact GEOS ground truth, so certificate pruning is provably sound (lod.py header).
"""
from __future__ import annotations
import math
import os
import zipfile
import io

import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.base import BaseGeometry

from codec import GRID


def quantize_geom(g: BaseGeometry) -> BaseGeometry:
    """Snap all coordinates to the codec grid; the result is the data of record."""
    q = shapely.set_precision(g, GRID)
    if not q.is_valid:
        q = q.buffer(0)
    return q


# --------------------------------------------------------------------------- #
# Synthetic control: "blob" polygons with a tunable number of vertices and a
# controllable separation, so we can sweep selectivity x separation for H2 and
# generate the no-separation control (NC4).
# --------------------------------------------------------------------------- #
def _blob(cx, cy, radius, n_vert, roughness, rng) -> Polygon:
    ang = np.sort(rng.uniform(0, 2 * math.pi, n_vert))
    rad = radius * (1.0 + roughness * rng.uniform(-1, 1, n_vert))
    xs = cx + rad * np.cos(ang)
    ys = cy + rad * np.sin(ang)
    p = Polygon(np.column_stack([xs, ys]))
    if not p.is_valid:
        p = p.buffer(0)
    return p


def synthetic_pair_sets(n=60, n_vert=400, roughness=0.25, separation=0.0,
                        seed=0, extent=None):
    """Two INDEPENDENTLY-placed collections of blobs -- a genuine spatial join
    with natural, sweepable selectivity (not the degenerate paired placement).

    `separation` is a density/sparsity knob: it sets the placement extent, so
    larger `separation` => sparser => lower selectivity and larger MBR gaps.
    Polygon vertex counts VARY (so the size-independence test can run here too):
    n_vert is the mean; actual counts are drawn uniformly in [0.4, 1.6]*n_vert.
    Returns (geoms_r, geoms_s)."""
    rng = np.random.default_rng(seed)
    ext = extent if extent is not None else (6.0 + 6.0 * separation)

    def _coll():
        out = []
        for _ in range(n):
            cx, cy = rng.uniform(0, ext, 2)
            nv = int(rng.uniform(0.4, 1.6) * n_vert)
            out.append(quantize_geom(_blob(cx, cy, rng.uniform(0.2, 0.9),
                                           nv, roughness, rng)))
        return out

    return _coll(), _coll()


# --------------------------------------------------------------------------- #
# Adversarial NC2: interlocking combs -- boundaries interleave at every scale so
# the certificate can never prune early; expect decoded fraction -> 1.
# --------------------------------------------------------------------------- #
def interlocking_combs(n_teeth=40, seed=0):
    """One pair of combs whose teeth alternate; a genuine intersection decided
    only at the finest features."""
    # Comb R: teeth pointing up from y=0; Comb S: teeth pointing down, offset so
    # teeth interleave and the shapes truly overlap in a thin contested strip.
    def comb(y0, up, xoff):
        pts = []
        w = 0.1
        for k in range(n_teeth):
            x = xoff + k * 0.2
            if up:
                pts += [(x, y0), (x, y0 + 1.0), (x + w, y0 + 1.0), (x + w, y0)]
            else:
                pts += [(x, y0), (x, y0 - 1.0), (x + w, y0 - 1.0), (x + w, y0)]
        # close along a baseline
        if up:
            pts += [(xoff + n_teeth * 0.2, y0 - 0.2), (xoff, y0 - 0.2)]
        else:
            pts += [(xoff + n_teeth * 0.2, y0 + 0.2), (xoff, y0 + 0.2)]
        p = Polygon(pts)
        return quantize_geom(p if p.is_valid else p.buffer(0))

    R = [comb(0.5, up=True, xoff=0.0)]
    S = [comb(1.0, up=False, xoff=0.1)]   # teeth reach down into R's gaps
    return R, S


def known_answer_fixture():
    """Tiny hand-checked fixture: 3 R-squares vs 3 S-squares with known overlaps."""
    def sq(x, y, s=1.0):
        return quantize_geom(Polygon([(x, y), (x + s, y), (x + s, y + s), (x, y + s)]))
    R = [sq(0, 0), sq(5, 5), sq(10, 0)]
    S = [sq(0.5, 0.5), sq(2.0, 2.0), sq(5.5, 5.5)]
    # Truth: R0∩S0 (overlap), R1(5,5)∩S2(5.5,5.5) overlap; others disjoint.
    truth = {(0, 0), (1, 2)}
    return R, S, truth


# --------------------------------------------------------------------------- #
# Real data: US Census TIGER/Line AREAWATER (public domain).  Polygons carry
# 10^3-10^5 vertices -- the regime where "decode as little as possible" matters.
# --------------------------------------------------------------------------- #
TIGER_BASE = "https://www2.census.gov/geo/tiger/TIGER2023"


def tiger_areawater_url(fips: str) -> str:
    return f"{TIGER_BASE}/AREAWATER/tl_2023_{fips}_areawater.zip"


def tiger_edges_url(fips: str) -> str:
    return f"{TIGER_BASE}/EDGES/tl_2023_{fips}_edges.zip"


def tiger_arealm_url(fips: str) -> str:
    return f"{TIGER_BASE}/AREALM/tl_2023_{fips}_arealm.zip"


def download(url: str, dest_dir: str) -> str:
    import requests
    os.makedirs(dest_dir, exist_ok=True)
    fname = os.path.join(dest_dir, url.rsplit("/", 1)[-1])
    if not os.path.exists(fname):
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(fname, "wb") as f:
            f.write(r.content)
    return fname


def load_tiger_polygons(zip_path: str, max_geoms=None, min_vertices=50):
    """Load polygon geometries from a TIGER zip via geopandas/pyogrio."""
    import geopandas as gpd
    gdf = gpd.read_file(f"zip://{zip_path}")
    geoms = []
    for g in gdf.geometry:
        if g is None or g.is_empty:
            continue
        if g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        q = quantize_geom(g)
        if q.is_empty or not q.is_valid:
            continue
        from lod import n_vertices
        if n_vertices(q) < min_vertices:
            continue
        geoms.append(q)
        if max_geoms and len(geoms) >= max_geoms:
            break
    return geoms
