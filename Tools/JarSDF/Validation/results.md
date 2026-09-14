# Completed jar collision comparison

All **6/6 trials passed**: three seeds per backend, twelve saved bolt/nut instances,
identical geometry, masses, inertias, initial states and constraint-actuated motion.
Settings: 1/480 s, four iterations, six simulated seconds per trial.

| Measure | SDF | Convex decomposition |
|---|---:|---:|
| Passing trials | 3/3 | 3/3 |
| Jar colliders | 1 | 512 |
| Total scene colliders | 86 | 597 |
| Median physics steps/s | 353.9 | 236.3 |
| Throughput range | 313.7–470.7 | 89.4–265.1 |
| Worst sampled jar penetration | 0.872 mm | 0.603 mm |
| Contents discharged, each seed | 12/12 | 12/12 |
| Serialized jar collision data | 15,410 bytes | 856,469 bytes |

SDF median throughput was **1.50×** the convex median.
These are physics steps, not rendered frames; at 480 steps per simulated second,
median SDF throughput is 0.74× real time.
Rates include submission and fences, exclude validation reads and shader setup,
and vary with this shared machine's load. Serialized size is not allocated GPU memory.

The jar field and convex mesh share the same polygonal radial profile; convex
angular chord error is bounded by 0.1024 mm. The common quality gate is 1 mm.
Checks use actual simulated poses, include final table penetration and actuator
tracking, and retain every trial. Vertex checks are not continuous certification,
and do not quantify all hardware-to-hardware penetration.

The early GPU-memory error was resolved by moving generated geometry tables to
Metal constant storage. Further repairs cover convex SAT witnesses, analytic
crease normals, rotational detection, deepest triangle vertices and cached SDF
anchor revalidation. Certified planar and convex-cavity regions accelerate queries.

Earlier failed runs remain in the other `out/jar-sdf-*` directories. In particular,
1/240 s was insufficient for these impacts, and the original low pour height wedged
50 mm bolts below a lip only 17 mm above the table. This cohort uses a geometry-derived
236.6 mm lift and real constraint actuation, with no per-step pose overwrites.

Reproduce using `Tools/JarSDF/README.md` in this repository.
See `summary.json`, `attempts.json` and `run-manifest.json` in this directory. No production family was switched
to SDF automatically. SDF/SDF and competing deep normal patches remain unsupported.

Final regression verification: **31 Python tests passed**, **8 supported native contact cases passed**, the unsupported containment case was correctly rejected, and the captured bolt/nut convex-witness regression passed.
