# LinkedIn draft — paper19 / spatial-join-on-compressed

> Stage-8 draft. Framing = invitation, never conquest. Every claim below is backed by a number in
> the repo. Post from Sandeep's account; resolve the tag handles manually (rationale below).

---

**When you join two maps, how much of the geometry do you actually need to decode?**

Spatial joins store polygons compressed — delta/varint coordinates, columnar geometry, TWKB. The
join's real cost isn't comparisons or wall-clock; it's **bytes decoded**. Filter-and-refine has
always avoided exact geometry for pairs it can certify early (Brinkhoff, Kriegel, Schneider & Seeger
nailed this in SIGMOD 1994) — but nobody modeled the *decode cost of the survivors*.

We built an open, reproducible harness that does, and found a simple law.

On real US Census TIGER water polygons, a progressive certificate join over a Douglas–Peucker
level-of-detail ladder returns the **provably-exact** intersection join while decoding **3.4–16.8×
(median 5.9×) fewer vertices** than naive decompress-then-refine, and ~4.9× fewer than a single
coarse approximation — zero correctness violations across 31 workloads (set-equality vs GEOS).

The law: **decode work is governed by each pair's signed-clearance margin** — how close it sits to the
predicate-flip boundary — *independent of object size*. The certificate descends the ladder only until
its resolution beats the margin. Near-tangent pairs cost Ω(vertices); robust overlaps cost almost
nothing.

We're honest about the edges: the law is clean on controlled geometry (R²=0.87) but only directional
on real, multi-scale boundaries (R²≈0.55). A predictor we pre-registered (near-boundary vertex count)
was **wrong** and we report it. Worst case is the trivial read bound. It's a baseline + characterization,
not a new index.

Code + preprint + one-command reproduction inside. **Did we get the decode accounting right? Where would
you push the law — predicate-aware codecs, a decode-vs-margin bound, the non-native-codec case?**

Code: github.com/samyama-ai/spatial-join-on-compressed
Preprint: arxiv.org/abs/2607.01182

#SpatialDatabases #ComputationalGeometry #GIS #DatabaseSystems #OpenScience

---

## Tag list (rationale — resolve handles manually; lead with their contribution)

- **Nikos Mamoulis** & **Thanasis Georgiadis** (APRIL / Raster Intervals, SIGMOD'23) — the closest
  modern compressed-domain intersection-join work; we benchmark the native-codec angle against their
  raster approximation. Tag as the people actively pushing this exact problem.
- **Ahmed Eldawy** (SpatialParquet, Sedona/Beast ecosystem) — columnar spatial storage + filter
  pushdown; our join-refinement-in-the-codec is the complementary step.
- **Eleni Tzirita Zacharatou** (Distance-Bounded Spatial Approximations, CIDR'21) — the exact-vs-
  bounded-error distinction we draw; a natural person to critique the framing.
- (Optional, historical, do NOT frame as a gotcha) **Hans-Peter Kriegel / Thomas Brinkhoff** — we
  build directly on their 1994 multi-step certificate; credit, don't challenge.

Guardrails: ≤ a handful of tags; never imply endorsement; lead with their work; the result is a
*characterization + honest partial law*, so say so — that honesty is why experts reply.
