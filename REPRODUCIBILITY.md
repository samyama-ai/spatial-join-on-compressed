# Reproducibility

## Environment
- Python 3.12, dependencies pinned in `requirements.txt` (shapely 2.1.2 / GEOS 3.13.1,
  geopandas 1.1.4, pyogrio 0.13, numpy 2.4.4, scipy 1.17.1, matplotlib 3.10.8).
- Hardware: single-thread CPU (macOS/Apple-Silicon and x86 both fine); no GPU, no network
  except the one-time TIGER download from `www2.census.gov`.
- Determinism: all randomness is confined to synthetic data generation with fixed seeds
  (`numpy.random.default_rng(seed)`); the join, certificate, margin, and analysis are
  deterministic. GEOS is the exact reference for correctness.

## One command
```bash
./run.sh
```
regenerates every number and figure into `results/` (`all_summary.csv`, `all_pairs.csv`,
`verdict.json`, `figures/`).

## Data provenance
- **US Census TIGER/Line 2023 `AREAWATER`** (public domain): downloaded per county FIPS from
  `https://www2.census.gov/geo/tiger/TIGER2023/AREAWATER/`. NOT vendored — fetched by
  `src/data.py` and cached under `results/tiger/` (git-ignored).
  Counties used: 22051 (Jefferson, LA), 22103 (St Tammany, LA), 27031 (Cook, MN),
  27137 (St Louis, MN), 53033 (King, WA), 06075 (San Francisco, CA).
- **Real workloads:** (a) `AREAWATER ⋈ translate(AREAWATER, δ)` — a controlled-selectivity
  self-join over real water polygons, δ swept as a fraction of the median polygon diagonal
  (margin/selectivity regime sweep); (b) cross-county `AREAWATER ⋈ AREAWATER` for neighboring
  parishes sharing water (Lake Pontchartrain 22051×22103; Lake Superior/BWCA 27031×27137).
- **Synthetic control:** independently-placed rough blobs with varied vertex counts and a
  sparsity knob, plus adversarial interlocking-comb fixtures (NC2) and a known-answer fixture.

## Correctness (the hard gate)
Every workload asserts **set-equality** of the progressive join's output against the
full-precision GEOS ground truth (`correct=1`). Coordinates are quantized to a fixed grid on
ingest; that quantized geometry is the data of record for BOTH the compressed store and the
ground truth, and Douglas-Peucker LOD levels keep a subset of those exact vertices — so
certificate pruning can only ever be a sound approximation of the exact stored polygon.
`tests/test_correctness.py` enforces NC1 (set-equality) and NC2 (no free lunch on adversarial
combs); run `python -m pytest tests/ -q`.

## What each number comes from
- `results/all_pairs.csv` — one row per candidate pair: decoded vertices/bytes, certifying
  levels, signed `margin`, `phi_pair`, `band_b` (the rejected predictor), family.
- `results/all_summary.csv` — one row per workload: selectivity, MBR-gap distribution,
  decoded fraction, decode ratios vs naive-refine and Brinkhoff'94, correctness.
- `results/verdict.json` — H1'/H2/H3 evaluated against the pre-registered decision rules.
