# HYPOTHESIS (PRE-REGISTERED, FROZEN 2026-07-01) — Problem #14

`20-spatial-databases/spatial-join-on-compressed` · **Mode: (b)-with-teeth** · lead with the LAW.

> Frozen before any experiment is run (the pre-registration discipline). Decision rules + negative
> controls fixed up front. Amendments, if any, are appended with timestamp + reason below the line.

## Object under test

Provably-exact spatial join $R \bowtie_{\text{intersects}} S$ (primary; `contains`, `within-`$\epsilon$
secondary) over a Douglas–Peucker **LOD ladder** with Hausdorff error bounds $\eta_\ell$, certified by
the two-sided ($\eta$-dilate / $\eta$-erode) test, descending only into the MBR-overlap **band**.
**Metric = vertices/bytes decoded** to certify, at **set-equality** correctness vs full-precision.

## Definitions (fixed)

- $D(r,s)$ = vertices decoded (summed over levels, both geometries, incl. LOD headers counted in bytes)
  until $(r,s)$ is certified DISJOINT/INTERSECT.
- $b(r,s)$ = **$\eta^\*$-band vertices** = exact-level vertices of $r,s$ within the $\eta^\*$-neighborhood
  of the other's boundary inside the MBR overlap, $\eta^\*$ = certifying error level.
- $v(r,s) = |r| + |s|$ = total vertices (the "decode-everything" cost of naive-refine).
- $\varphi$ = aggregate decoded fraction = $\sum D / \sum v$ over candidate pairs (post-MBR-filter);
  speedup $S = 1/\varphi$.
- $\sigma$ = join selectivity = (#intersecting pairs)/(#MBR-candidate pairs). "MBR-gap" = normalized L0
  bounding-box separation for a candidate pair.

## Hypotheses + decision rules (falsifiable, numeric)

**H1 — THE LAW (primary; per-pair decode ∝ band, not object size).**
$D(r,s) \approx c\,b(r,s) + d$. On **held-out** geometries (train/test split by county), the linear fit
of $D$ on $b$ has **$R^2 \ge 0.80$**, AND after controlling for $b$ the partial effect of object size
$v$ is negligible (**standardized partial coeff $|\beta_v| \le 0.15$**, i.e. band explains decode, size
does not).
→ **PASS** = both hold. **FAIL** = $R^2 < 0.80$ or $|\beta_v| > 0.15$ (decode is size-driven, not
band-driven → law rejected, ship negative).

**H2 — FORECASTER + REGIME MAP (secondary; aggregate $\varphi$ predictable from selectivity + MBR-gap).**
A forecaster $\hat\varphi(\sigma, \text{L0 MBR-gap distribution})$ predicts measured $\varphi$ on held-out
(selectivity × separation) cells with **median relative error $\le 20\%$**, AND the sweep exhibits a
**crossover regime** — at least one cell with $\varphi \le 0.10$ (big win) and at least one with
$\varphi \ge 0.70$ (fall-back).
→ **PASS** = both. **FAIL** = median rel-err $>20\%$ or no crossover exhibited.

**H3 — BEATS NAIVE-REFINE AT MATCHED CORRECTNESS (the "teeth").**
On the primary **TIGER `AREAWATER ⋈ EDGES`** workload, the band-limited progressive join decodes
**≥ 2× fewer vertices** (median over counties) than naive refine-past-MBR, at **identical output**
(set-equality, zero violations), and the realized reduction is **within the band predicted by H1's law**
(measured $S$ inside the law's forecast interval).
→ **PASS** = ≥2× median reduction AND zero correctness violations AND $S$ consistent with the law.
**FAIL** = <2× or any correctness violation (→ mechanism unsound or law over-optimistic).

**Pre-registered success ladder (honesty-over-reach):**
- **Full win:** H1 + H2 + H3 all PASS → the law + regime map + teeth.
- **Partial (still ships):** H1 PASS + (H2 or H3) PASS → law validated, one of forecaster/teeth.
- **Negative (still ships):** H1 FAIL → ship the negative ("decode is size-driven, not band-driven, on
  real data") + the decode-instrumented harness + baselines. A legitimate day (Gate C).

## Negative controls / leak detectors (a "win" here = harness bug)

- **NC1 — Correctness (hard gate, every run):** set-equality vs full-precision ground truth. ANY
  false-pos/neg ⇒ STOP; certificate is unsound (we pruned without a proof). Not a result.
- **NC2 — Adversarial worst case:** interlocking-comb fixtures ⇒ **expect $\varphi \to 1$** (no
  shortcut, $\Omega(v)$). If we "win" (small $\varphi$) here, the certificate is leaking → bug.
- **NC3 — Random-relabel null:** shuffle pair identities ⇒ H1's band predictor must **lose** power
  ($R^2$ collapses). If it still predicts, we're fitting IDs/size, not geometry.
- **NC4 — No-separation control:** dense, heavily-overlapping data (low MBR-gap everywhere) ⇒ expect
  **small** speedup ($\varphi$ near 1). Huge speedup here ⇒ suspect unsound certification.
- **NC5 — Fairness audit:** LOD-header/index bytes counted for the progressive method; shared MBR
  candidate set for both; no double-counting cached decodes. Logged per run.

## Apparatus / reproducibility (fixed)

- Python (`~/projects/venv`), shapely 2.x + bundled GEOS, topojson/DP simplification,
  `hilbertcurve`/`pymorton`, TWKB via PostGIS `ST_AsTWKB` or custom delta/varint. PostGIS in Docker on
  **`mini`** for the exact reference + TWKB emit; Shapely-STRtree standalone fallback if `mini`
  unreachable. Deterministic; seeds only in synthetic data-gen. One-command `./run.sh` regenerates every
  number. TIGER/SpiderWeb re-fetchable (no vendored data); hardware/version/dataset-hash logged.

## Frozen scope (cut, don't dilute — Scope-down rule)

IN: intersects join, TIGER primary + OSM secondary + synthetic control, vertices-decoded metric,
naive-refine + decompress-then-join (PostGIS/Shapely) + Brinkhoff'94 baselines, H1–H3 + NC1–NC5.
OUT (future work): APRIL as a 4th baseline (add only if H1–H3 land early); GPU/vectorized decode;
`contains`/`within-`$\epsilon$ beyond a sanity pass; a formal decompression-sensitive theorem (6-mo
theory); predicate-aware codec design.

---
*Amendments (append below with timestamp + reason; do not edit above this line):*

### Amendment A1 — 2026-07-01 (after the synthetic PILOT, before any confirmatory run)

**H1 (band-vertex predictor) is REJECTED on the pilot** (synthetic sweep, 4,871 pairs):
held-out $R^2 = 0.36$ (< 0.80) with a **negative** slope ($\beta_{\text{band}} = -0.62$).
Diagnosed cause (not noise): `band_b` counts exact vertices near the *other* boundary, which is a
proxy for overlap **robustness**, so it **anti-correlates** with decode — deep, robustly-overlapping
pairs have a large near-boundary band yet certify INTERSECT instantly at level 0 (mean |margin| 0.11,
decode 416 v), while near-tangent pairs have a small band yet must descend to the finest level (mean
|margin| 0.0001, decode 802 v). The pilot was also degenerate for the size test (all blobs equal size).

**Reframed predictor (frozen now, before the confirmatory TIGER + fixed-synthetic run):** the
geometry-only **signed clearance margin** (see `margin.py`)
$m(r,s) = +\text{inradius}(r\cap s)$ if intersecting, $-\,\mathrm{dist}(r,s)$ if disjoint. The
certificate must descend until resolution $\eta_L < |m|$, so:

- **H1′ (margin law, PRIMARY, replaces H1):** the certifying **depth** and the per-pair **decoded
  fraction** $\varphi_{\text{pair}} = \text{dec\_vert}/\text{total\_vert}$ are governed by $\log|m|$.
  Decision: held-out $R^2 \ge 0.80$ predicting $\varphi_{\text{pair}}$ (or cert-level) from a monotone
  function of $\log|m|$, **AND** on real data with genuine size variation (TIGER) the standardized
  partial effect of object size $|\beta_{\text{size}}| \le 0.15$ after controlling for $\log|m|$
  (decode DEPTH is margin-driven, size-independent — the instance-optimal invariant).
- Pilot evidence motivating (not confirming) the reframe: corr(dec_vert, $\log_{10}|m|$) = **−0.81**,
  corr(cert_level, $\log_{10}|m|$) = **−0.83**, |margin|-by-level decays geometrically
  (0.11→0.012→0.0055→0.0016→0.0007→0.0001), i.e. each level resolves ~1 order of magnitude of margin.

H2, H3, NC1–NC5 unchanged. The confirmatory test is **real TIGER data + a fixed synthetic generator
with genuine selectivity variation** (the pilot generator produced selectivity ≡ 1.0, no disjoint
candidates); H1′ is judged there, and the original H1/band failure is reported honestly alongside.
