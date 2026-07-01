"""Experiment driver: run a workload through all methods, emit per-pair records
and a workload summary.  Deterministic (seeds only in synthetic data-gen).

Per-pair record (the unit H1 is fit on):
  workload, i, j, intersects, dec_vert, dec_bytes, total_vert, cert_level_r,
  cert_level_s, eta_star, band_b, band_w, mbr_gap, size_v

Workload summary (the unit H2/H3 use):
  workload, n_geoms_r, n_geoms_s, n_candidates, n_intersect, selectivity,
  gap_mean, gap_frac_overlap, gap_p50, phi_prog, phi_brink, decvert_prog,
  decvert_naive, decvert_brink, ratio_naive_over_prog, correct
"""
from __future__ import annotations
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from lod import LodLadder
from joinq import (progressive_join, naive_refine_join, exact_join, mbr_candidates)
from baselines import brinkhoff_join
from band import band_vertices
from margin import signed_margin
from metrics import selectivity, mbr_gap, gap_distribution
import data as D
import numpy as np


def run_workload(name, R, S, family="synthetic"):
    cands = mbr_candidates(R, S)
    lr = [LodLadder(g) for g in R]
    ls = [LodLadder(g) for g in S]
    truth = exact_join(R, S, cands)
    prog = progressive_join(R, S, cands, lr, ls)
    naive = naive_refine_join(R, S, cands, lr, ls)
    brink = brinkhoff_join(R, S, cands, lr, ls)

    correct = (prog.pairs == truth) and (naive.pairs == truth) and (brink.pairs == truth)

    per_pair = []
    for pr in prog.per_pair:
        eta0_r = lr[pr.i].levels[0].eta
        eta0_s = ls[pr.j].levels[0].eta
        b, w = band_vertices(lr[pr.i].exact, ls[pr.j].exact, eta0_r, eta0_s)
        g = mbr_gap(R[pr.i].bounds, S[pr.j].bounds)
        m, _ = signed_margin(lr[pr.i].exact, ls[pr.j].exact)
        phi_pair = pr.dec_vert / max(1, pr.total_vert)
        per_pair.append(dict(
            workload=name, i=pr.i, j=pr.j, intersects=int(pr.intersects),
            dec_vert=pr.dec_vert, dec_bytes=pr.dec_bytes, total_vert=pr.total_vert,
            cert_level_r=pr.cert_level_r, cert_level_s=pr.cert_level_s,
            eta_star=pr.eta_star, band_b=b, band_w=w, mbr_gap=g,
            size_v=pr.total_vert, margin=m, abs_margin=abs(m),
            log_abs_margin=float(np.log10(abs(m) + 1e-12)), phi_pair=phi_pair,
            family=family,
        ))

    gd = gap_distribution(R, S, cands)
    summary = dict(
        workload=name, n_geoms_r=len(R), n_geoms_s=len(S),
        n_candidates=prog.n_candidates, n_intersect=len(truth),
        selectivity=selectivity(truth, prog.n_candidates),
        **gd,
        phi_prog=prog.dec_vert / max(1, prog.total_vert),
        phi_brink=brink.dec_vert / max(1, brink.total_vert),
        decvert_prog=prog.dec_vert, decvert_naive=naive.dec_vert,
        decvert_brink=brink.dec_vert,
        decbytes_prog=prog.dec_bytes, decbytes_naive=naive.dec_bytes,
        ratio_naive_over_prog=(naive.dec_vert / max(1, prog.dec_vert)),
        ratio_brink_over_prog=(brink.dec_vert / max(1, prog.dec_vert)),
        correct=int(correct),
    )
    return summary, per_pair


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def synthetic_suite():
    """Selectivity x separation sweep for H1/H2 + NC controls."""
    summaries, pairs = [], []
    # H2 regime grid: separation controls selectivity & MBR-gap.
    for sep in (0.0, 0.3, 0.6, 1.0, 1.5, 2.5):
        for seed in (1, 2, 3):
            R, S = D.synthetic_pair_sets(n=70, n_vert=400, separation=sep, seed=seed)
            nm = f"syn_sep{sep}_s{seed}"
            s, pp = run_workload(nm, R, S, family="synthetic")
            s["family"] = "synthetic"; s["separation"] = sep; s["seed"] = seed
            summaries.append(s); pairs.extend(pp)
    # NC2 adversarial
    R, S = D.interlocking_combs(n_teeth=40, seed=0)
    s, pp = run_workload("nc2_combs", R, S, family="adversarial")
    s["family"] = "adversarial"; s["separation"] = -1; s["seed"] = 0
    summaries.append(s); pairs.extend(pp)
    return summaries, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--tiger-fips", default=None,
                    help="comma-separated county FIPS for TIGER AREAWATER x AREAWATER")
    ap.add_argument("--tiger-max", type=int, default=300)
    ap.add_argument("--tiger-pair", default=None,
                    help="two county FIPS 'A,B' for cross-county AREAWATER join")
    args = ap.parse_args()

    all_sum, all_pairs = [], []
    if args.synthetic:
        s, p = synthetic_suite()
        all_sum += s; all_pairs += p
        write_csv(os.path.join(args.out, "synth_pairs.csv"), p)
        write_csv(os.path.join(args.out, "synth_summary.csv"), s)
        print(f"[synthetic] {len(s)} workloads, {len(p)} pairs")

    if args.tiger_fips:
        from shapely.affinity import translate as _shift
        for fips in args.tiger_fips.split(","):
            fips = fips.strip()
            zp = D.download(D.tiger_areawater_url(fips), os.path.join(args.out, "tiger"))
            geoms = D.load_tiger_polygons(zp, max_geoms=args.tiger_max)
            import numpy as _np
            from lod import n_vertices as _nv
            vs = [_nv(g) for g in geoms]
            print(f"[tiger {fips}] {len(geoms)} water polygons; "
                  f"verts min/med/max={min(vs)}/{int(_np.median(vs))}/{max(vs)}")
            if len(geoms) < 8:
                continue
            # median polygon extent -> translation scale for the margin/selectivity sweep
            diag = _np.median([((g.bounds[2]-g.bounds[0])**2 +
                                (g.bounds[3]-g.bounds[1])**2) ** 0.5 for g in geoms])
            R = geoms
            for frac in (0.1, 0.25, 0.5, 1.0):
                dx = float(diag * frac)
                S = [D.quantize_geom(_shift(g, xoff=dx)) for g in geoms]
                nm = f"tiger_{fips}_d{frac}"
                s, pp = run_workload(nm, R, S, family="tiger")
                s["family"] = "tiger"; s["separation"] = frac; s["seed"] = None
                all_sum.append(s); all_pairs.extend(pp)
                print(f"  [{nm}] cand={s['n_candidates']} sel={s['selectivity']:.3f} "
                      f"phi={s['phi_prog']:.3f} naive/prog={s['ratio_naive_over_prog']:.2f} "
                      f"brink/prog={s['ratio_brink_over_prog']:.2f} correct={s['correct']}")
            write_csv(os.path.join(args.out, f"tiger_{fips}_pairs.csv"),
                      [p for p in all_pairs if p["workload"].startswith(f"tiger_{fips}")])

    if args.tiger_pair:
        for pair in args.tiger_pair.split(";"):
            fa, fb = [x.strip() for x in pair.split(",")]
            za = D.download(D.tiger_areawater_url(fa), os.path.join(args.out, "tiger"))
            zb = D.download(D.tiger_areawater_url(fb), os.path.join(args.out, "tiger"))
            R = D.load_tiger_polygons(za, max_geoms=args.tiger_max)
            S = D.load_tiger_polygons(zb, max_geoms=args.tiger_max)
            nm = f"tiger_{fa}x{fb}"
            s, pp = run_workload(nm, R, S, family="tiger")
            s["family"] = "tiger"; s["separation"] = None; s["seed"] = None
            all_sum.append(s); all_pairs.extend(pp)
            print(f"[{nm}] |R|={len(R)} |S|={len(S)} cand={s['n_candidates']} "
                  f"sel={s['selectivity']:.3f} phi={s['phi_prog']:.3f} "
                  f"naive/prog={s['ratio_naive_over_prog']:.2f} correct={s['correct']}",
                  flush=True)

    write_csv(os.path.join(args.out, "all_summary.csv"), all_sum)
    if all_pairs:
        write_csv(os.path.join(args.out, "all_pairs.csv"), all_pairs)
    print(f"[done] {len(all_sum)} workloads -> {args.out}/all_summary.csv")


if __name__ == "__main__":
    main()
