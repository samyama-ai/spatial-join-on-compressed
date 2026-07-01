"""Stage-3 correctness tests. NC1 (set-equality vs GEOS) is the HARD GATE:
any false pos/neg means the certificate is unsound -> STOP + harden (per the
pre-registered HYPOTHESIS), it is not a result.

Run:  python -m pytest tests/ -q   (from repo root, with src on the path)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import (known_answer_fixture, synthetic_pair_sets, interlocking_combs)
from joinq import (progressive_join, naive_refine_join, exact_join, mbr_candidates)
from baselines import brinkhoff_join
from lod import LodLadder


def _check_equals_ground_truth(R, S):
    cands = mbr_candidates(R, S)
    truth = exact_join(R, S, cands)
    lr = [LodLadder(g) for g in R]
    ls = [LodLadder(g) for g in S]
    prog = progressive_join(R, S, cands, lr, ls)
    naive = naive_refine_join(R, S, cands, lr, ls)
    brink = brinkhoff_join(R, S, cands, lr, ls)
    assert prog.pairs == truth, f"progressive != truth: " \
        f"missing={truth - prog.pairs} extra={prog.pairs - truth}"
    assert naive.pairs == truth, "naive-refine != truth (baseline bug)"
    assert brink.pairs == truth, "brinkhoff != truth"
    return truth, prog, naive


def test_known_answer_fixture():
    R, S, truth = known_answer_fixture()
    cands = mbr_candidates(R, S)
    got = exact_join(R, S, cands)
    assert got == truth, f"ground-truth fixture mismatch: {got} vs {truth}"
    _check_equals_ground_truth(R, S)


def test_nc1_synthetic_setequality_overlapping():
    # Overlapping regime (separation 0): many intersections.
    R, S = synthetic_pair_sets(n=50, n_vert=300, separation=0.0, seed=1)
    truth, prog, naive = _check_equals_ground_truth(R, S)
    assert len(truth) > 0, "expected some intersections in the overlapping regime"


def test_nc1_synthetic_setequality_separated():
    # Separated regime: few/no intersections, big MBR gaps.
    R, S = synthetic_pair_sets(n=50, n_vert=300, separation=2.0, seed=2)
    _check_equals_ground_truth(R, S)


def test_nc2_adversarial_combs_no_free_lunch():
    # Interlocking combs: certificate must NOT prune cheaply; decoded fraction
    # should be high (no free lunch). Correctness still exact.
    R, S = interlocking_combs(n_teeth=30, seed=0)
    truth, prog, naive = _check_equals_ground_truth(R, S)
    frac = prog.dec_vert / max(1, prog.total_vert)
    # On adversarial interleaving we expect to decode a large fraction.
    assert frac > 0.5, f"NC2 leak: pruned too cheaply on combs (frac={frac:.3f})"
