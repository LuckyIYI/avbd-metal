# Qualified lower-budget result

The final SDF cohort passes **3/3** at **240 Hz and four iterations**, with no retained solver stiffness/penalty changes. Every physics step is checked against an independent radial polygon distance oracle. The same geometry, actuator motion, seeds and 1 mm threshold are used for the convex controls.

| Backend | Hz / iterations | Passes | Worst penetration | Median step rate | Median six-second run including checks |
|---|---:|---:|---:|---:|---:|
| SDF | 240 / 4 | 3/3 | 0.960 mm | 568.3 steps/s | 2.64 s |
| Convex | 240 / 4 | 0/3 | 3.776 mm | 724.4 steps/s | 2.10 s |
| Convex | 480 / 4 | 1/3 | 1.016 mm | 742.7 steps/s | 4.09 s |

SDF's median full run is **2.27× realtime**, including geometry checks but excluding initialization; the slowest measured run is 4.11 seconds for six simulated seconds (1.46× realtime). All SDF trials discharge 12/12 pieces. Convex at 480 Hz retains one piece in seed 0 and slightly exceeds the penetration gate in seed 1. The faster convex 240 Hz runs do not meet the same quality gate, so they are not qualifying alternatives. No universal speedup ratio against a fully qualified convex configuration is claimed.

## Retained fixes

- Field contacts use reference lever arms consistently in Taylor values and Jacobians, using the existing fixed-contact-frame solver path.
- Contact merging scales with the smaller object, preserving witnesses on small fasteners.
- Projected stationary minima avoid redundant field queries.
- Mixed SDF scenes retain the optimized convex kernel for ordinary hardware pairs. Field/raw-surface pairs remain owned by the field-aware kernel.
- The independent polygon oracle makes every-step geometry checks inexpensive, replacing sparse DAG-based sampling.

No new solver iterations, penalty multipliers, active-set solver changes, or relaxed quality thresholds remain. Exploratory stabilization, penalty and manifold changes that failed were reverted. Their outcomes are retained separately.

## Validation and limits

Eight supported native contact cases pass; full containment still intentionally rejects with reason 5. The captured convex witness regression passes. The earlier 480 Hz / sparse-sampling result is historical and superseded for qualification by these stricter checks.

120 Hz / four iterations is **not qualified**; exploratory runs fail, including incompatible deep normal patches. SDF/SDF, general CCD and robust multiple-normal contact patches remain missing. Vertex checks at every step still do not certify continuous between-step trajectories or all edge/interior intersections. The three-seed cohort is small; shared-machine load affects timings. Neither a threaded lid nor thread engagement was tested. Compound hierarchy acceleration remains disabled for field scenes.

Reproduce from Tools/JarSDF using the frozen fixture, seed 0/1/2, 1440 steps, JAR_HZ=240 and JAR_ITERATIONS=4. Keep the default JAR_SAMPLE_EVERY=1. The original six-trial results and all nine current final controls are retained; nothing failed has been relabeled a pass.
