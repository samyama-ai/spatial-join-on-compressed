"""Workload-level metrics: join selectivity and coarse MBR-gap distribution."""
from __future__ import annotations
from math import sqrt

import numpy as np


def selectivity(exact_pairs: set, n_candidates: int) -> float:
    return (len(exact_pairs) / n_candidates) if n_candidates else 0.0


def mbr_gap(a_bounds, b_bounds) -> float:
    """Normalised bounding-box separation for a candidate pair.

    0 if the MBRs overlap; else Euclidean gap between boxes divided by the mean
    box diagonal (scale-free).  The forecaster (H2) uses the distribution of this
    over candidate pairs together with selectivity to predict decoded fraction.
    """
    ax0, ay0, ax1, ay1 = a_bounds
    bx0, by0, bx1, by1 = b_bounds
    dx = max(0.0, max(ax0, bx0) - min(ax1, bx1))
    dy = max(0.0, max(ay0, by0) - min(ay1, by1))
    gap = sqrt(dx * dx + dy * dy)
    da = sqrt((ax1 - ax0) ** 2 + (ay1 - ay0) ** 2)
    db = sqrt((bx1 - bx0) ** 2 + (by1 - by0) ** 2)
    mean_diag = 0.5 * (da + db) or 1.0
    return gap / mean_diag


def gap_distribution(geoms_r, geoms_s, candidates):
    gaps = [mbr_gap(geoms_r[i].bounds, geoms_s[j].bounds) for (i, j) in candidates]
    a = np.asarray(gaps, dtype=float)
    if a.size == 0:
        return {"gap_mean": 0.0, "gap_frac_overlap": 0.0, "gap_p50": 0.0}
    return {
        "gap_mean": float(a.mean()),
        "gap_frac_overlap": float((a == 0.0).mean()),
        "gap_p50": float(np.median(a)),
    }
