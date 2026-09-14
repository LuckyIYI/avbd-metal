# Co-authored jar collision benchmark

`src/assetgen/design/implicit/jar.py` produces a visible mesh, a metric SDF and
convex cells from the same radial profile. The default open jar is 170 mm across
and 160 mm high. Its fillets are explicit polygonal arcs, not exact circular arcs.
The SDF is exact for their surface of revolution. The 64-sector mesh and convex
representation differ from that surface by at most 0.1024 mm radially.
There are 512 convex cells versus one field. No lid/thread or SDF/SDF claim is made.

The generator also exports two proven accelerators:

- Planar regions where the field is exactly a plane, allowing direct evaluation.
- Convex empty regions defined by normalized radial halfspaces. If all vertices
  of a triangle lie inside an eroded empty region, convexity and the 1-Lipschitz
  plane constraints certify the entire triangle's clearance. This skips search
  without replacing a failed query by an assumed separation.

These regions are mathematical author contracts, not hints inferred from samples.
The jar constructor derives them from its profile; independent tests compare
boundary/interior points against the unaccelerated field. Arbitrary field authors
must establish the same guarantees. Field transformations conservatively drop
these accelerators rather than retain invalid bounds.

## Standalone reproduction

Requires macOS 14+, Swift and native Metal access. No asset generator checkout is needed:

```sh
cd Tools/JarSDF
swift build -c release
JAR_HZ=240 JAR_ITERATIONS=4 .build/release/jar-sdf-pilot Fixtures/jar.json sdf /tmp/jar-sdf.json 0 1440
JAR_HZ=240 JAR_ITERATIONS=4 .build/release/jar-sdf-pilot Fixtures/jar.json convex /tmp/jar-convex.json 0 1440
.build/release/jar-sdf-pilot Fixtures/bolt-nut-overlap.json witness /tmp/jar-witness.json
```

Repeat seeds 0, 1, 2 with alternating backend order for the six-trial cohort.
Use a clean build after schema changes to avoid stale cross-module Swift layouts.
The executable exits nonzero on a failed result. Set JAR_PROGRESS=1 for progress.
The frozen fixture contains six original bolt/nut assets, each instantiated twice,
with the same mass, inertia and initial state across backends. Its source path is
provenance only; the executable never reads it. The jar and all cooked assets are embedded.

The six-second sequence settles, lifts, tips and discharges using three physical
world-anchor constraints, without per-step pose resets. Lift height is derived
from the rotated jar envelope, largest fastener diagonal and 20 mm exit clearance.
The original 180 mm lift wedged 50 mm bolts under a lip only 17 mm above the table;
the corrected lift is 236.6 mm. The original implementation at 240 Hz was insufficient for these impacts; the repaired
qualified comparison uses 240 Hz and four iterations for both representations.

Gates: no pre-pour escapes, less than 1 mm sampled jar penetration, all twelve
pieces discharged, less than 1 mm final table penetration, and final actuator
tracking within 5 mm / 0.05 rad. Samples use actual simulated poses and an independent polygon oracle at every step by default. Vertex sampling
is not continuous collision certification and does not measure every hardware pair.

`Validation/attempts.json` preserves every final-cohort trial and its trace;
`summary.json`, `contact-regressions.json`, `witness-regression.json` and
`run-manifest.json` retain measurements and original source hashes. Those results
were collected in the integration checkout before this standalone packaging.
The manifest's generator paths and base commit describe that original run.
Earlier failed experiments remain in the source workspace, not mislabeled as passes.

Current qualification is in **Validation/low-budget/README.md**: SDF passes 3/3 at 240 Hz / four iterations, with 0.960 mm worst every-step penetration and median 2.64 s including checks for six simulated seconds. Convex controls pass 0/3 at 240 Hz and 1/3 at 480 Hz. Original Validation results remain historical, using sparser checks.

## Engine repairs

- Generated surface geometry lives in module-scope Metal constant tables, avoiding
  per-thread copies/scratch allocations that caused the original GPU memory failure.
- The convex SAT fallback reconstructs actual touching features at its minimum
  translation instead of demanding agreement with a different clipped face normal.
- SDF queries use analytic feature normals, with finite differences only for a
  degenerate derivative at the zero set; creases retain a valid nearest feature.
- Triangle searches consider vertices as well as their centroid, and rotational
  motion contributes to speculative contact detection.
- Reused friction anchors must preserve the current certified normal gap. Curved
  surfaces cannot retain a stale tangent plane that admits millimetre penetration.

Existing runtime work budgets and witness checks remain enabled. Multiple deep
normal patches, fully contained fields, or unresolved search still fail explicitly.
No existing production family is silently switched to SDF by this benchmark.
