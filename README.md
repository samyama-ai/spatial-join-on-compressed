# Spatial joins on compressed geometry: a decode-work law

**A reproducible baseline + characterization for provably-exact spatial joins over compressed,
progressively-decodable geometry — not yet a new SOTA index, but a clean, honest reference the
spatial-DB community can build on.**

Filter-and-refine spatial joins have always avoided touching exact geometry for *certified* pairs,
but the field never **modeled the decompression cost of the survivors**. When geometry is stored in a
compressed multiresolution codec, the join's true cost is **bytes/vertices decoded**. This repo
measures that cost, gives a mechanism that minimizes it while returning **provably-exact** results,
and characterizes when it wins.

Problem: [`20-spatial-databases/spatial-join-on-compressed`](https://github.com/samyama-ai/dbms_research)
in the DBMS Research catalog.

## The claim, honestly stated

On real US Census **TIGER water polygons**, a progressive certificate join over a Douglas–Peucker
LOD ladder returns the **exact** intersection-join result while decoding:

- **3.4–16.8× (median 5.9×) fewer vertices** than naive decompress-then-refine, and
- **~4.9× fewer** than the Brinkhoff et al. (1994) single-approximation multi-step baseline,
- with **zero correctness violations** (set-equality vs GEOS ground truth on every workload).

**Why it works — the decode-work law:** decode work is governed by each candidate pair's
**signed-clearance margin** (how close it is to the predicate-flip boundary), *independent of object
size*. The certificate descends the ladder only until resolution η beats the margin — so near-tangent
pairs cost Ω(vertices) and robustly-decided pairs cost almost nothing. The law is **clean on
controlled geometry** (held-out depth R²=0.87, size-independent) and **directional on real data**
(R²≈0.55; decode fraction size-independent, β_size=0.07).

**Honest limits (see the paper's §Limitations):**
- The margin law is only *partially* predictive on real multi-scale boundaries (R²≈0.55, not the
  ≥0.80 we pre-registered) — a directional characterization, not a tight predictor.
- Our pre-registered *band-vertex* predictor was **rejected** (it proxies overlap robustness, not
  margin); reframed to the margin before the confirmatory run (pre-registration + amendment in
  `PREREGISTRATION`).
- The selectivity-based regime forecaster did **not** materialize; the real regime axis is the margin.
- Worst case is the trivial Ω(v) read bound (adversarial interlocking combs → decode ~everything).
- Not a new index; single-thread; polygon `intersects` (contains/within-ε are sanity-only).

## Reproduce

```bash
pip install -r requirements.txt
./run.sh          # experiments + analysis + figures -> results/
python -m pytest tests/ -q   # correctness gate (NC1 set-equality, NC2 adversarial)
```

Every number is regenerated into `results/` (`all_summary.csv`, `all_pairs.csv`, `verdict.json`,
`figures/`). See `REPRODUCIBILITY.md` for data provenance and the fairness accounting.

## What's here

| Path | What |
|---|---|
| `src/lod.py` | Douglas–Peucker LOD ladder + two-sided Hausdorff certificate primitives |
| `src/joinq.py` | progressive certificate join + naive-refine + exact baselines, decode accounting |
| `src/baselines.py` | Brinkhoff'94 single-approximation multi-step baseline |
| `src/margin.py` | signed-clearance margin (the decode-work-law predictor) |
| `src/band.py` | the **rejected** band-vertex predictor (kept for the honest record) |
| `src/codec.py` | delta+zigzag+varint byte accounting |
| `src/data.py` | TIGER loader, synthetic control, adversarial + known-answer fixtures |
| `src/experiment.py`, `src/analyze.py`, `src/make_figures.py` | driver, pre-registered analysis, figures |
| `tests/` | correctness gate |

## Positioning / prior art

We **concede the certificate mechanism** to Brinkhoff, Kriegel, Schneider & Seeger (*Multi-Step
Processing of Spatial Joins*, SIGMOD 1994) — their false-area test is our η-margin certificate; our
delta is the **multi-level** ladder (which beats their single approximation ~4.9×) and the **decode-cost
measurement over the native codec**. We measure in the decompression-sensitive cost model of Abboud et
al. (FOCS 2017) / Navarro's compact-data-structures program, and frame the result as **instance-optimal**
refinement (Afshani–Barbay–Chan, FOCS 2009). We differ from APRIL / Raster Intervals (SIGMOD 2023) — a
*secondary* raster approximation — by operating on the native geometry codec, and from distance-bounded
approximations (CIDR 2021) by returning **provably-exact** results.

## License
Code Apache-2.0. TIGER data is US Census public domain (not vendored). Catalog content CC BY 4.0.
