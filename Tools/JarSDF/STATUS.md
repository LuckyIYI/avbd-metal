## Problem and resulting behavior

Procedurally authored distance fields can now be imported as opt-in rigid colliders, preserving cavities without many convex pieces. This draft adds a portable scalar-expression DAG, validation and gradients, and GPU sphere/field and triangle-backed surface/field contacts using AVBD manifolds and friction. Native boxes, cooked convex hulls, planes and two-sided triangle surfaces are supported within the limits below.

The latest increment qualifies the matched jar-and-fastener pour at **240 Hz / four iterations**, half the earlier step frequency, with **3/3 SDF trials passing** and every-step independent geometry checks. Median full run time including validation is **2.64 seconds for six simulated seconds** (2.27× realtime). No stiffness or penalty changes are retained. No production asset is switched automatically.

## Implemented

- JSON field import, explicit/automatic/finite-difference gradients, including `sign` and a zero-set fallback for degenerate analytic derivatives.
- Specialized flat-support queries using actual hull halfspaces and revalidated previous-face hints; bounded triangle search otherwise.
- Generated geometry tables in Metal constant storage, fixing the original per-thread allocation/GPU-memory failure.
- Author-certified exact planar regions and convex radial empty regions to bypass unnecessary field search. The importer validates structure; authors must prove semantic equivalence to their field.
- Triangle minimization includes vertices and stays inside its search subtriangle; field witnesses and normals remain checked.
- Speculative detection accounts for angular as well as linear motion.
- SDF contact-cache reuse revalidates the normal gap, preventing stale friction anchors from retaining an incorrect curved-surface tangent plane.
- Corrected convex SAT fallback witness reconstruction in both narrowphase paths, with a captured bolt/nut regression.
- Consistent reference lever arms for field-contact Taylor values and Jacobians, using the existing fixed-contact-frame path.
- Contact merging scaled to the smaller object; stationary projected queries avoid redundant evaluation.
- Optimized convex dispatch restored for ordinary hardware pairs in field scenes; field pairs remain on their own path.
- Independent polygon-distance validation at every simulation step, with optional GPU stage profiling.
- Standalone `Tools/ImplicitContacts` and new `Tools/JarSDF`, including the complete frozen jar/fastener fixture, numeric results and reproduction instructions.

## Measured validation — current qualification

Same frozen assets, masses/inertias, seeded initial poses, constraint-actuated motion and 1 mm gate. All runs use four iterations; there are no per-step pose overwrites. Vertex penetration is checked **every simulation step** with an independent radial polygon oracle cross-checked against the imported field.

| Backend | Physics Hz | Passes | Worst penetration | Median physics steps/s | Median six-second run including checks |
|---|---:|---:|---:|---:|---:|
| SDF | 240 | 3/3 | 0.960 mm | 568.3 | 2.64 s |
| Convex | 240 | 0/3 | 3.776 mm | 724.4 | 2.10 s |
| Convex | 480 | 1/3 | 1.016 mm | 742.7 | 4.09 s |

SDF median end-to-end throughput is **2.27× realtime**, including checks but excluding initialization. Its slowest trial is still faster than realtime: 4.11 seconds for six simulated seconds. Every SDF trial discharges 12/12 pieces. The convex 480 Hz control retains one piece in seed 0 and slightly exceeds penetration tolerance in seed 1. Faster failing runs do not count as qualified alternatives; this is **not a universal SDF speedup claim**.

The jar uses one SDF collider versus 512 convex cells (86 vs 597 total scene colliders). Serialized jar data is 15,410 vs 856,469 bytes, not measured GPU allocation. Convex angular chord error is at most 0.1024 mm. Shared-machine load and trajectories affect timings. Physics step rates are not rendering FPS.

The **earlier 480 Hz, 6/6 result used sparse checks and is historical**. It remains in `Tools/JarSDF/Validation/`; current all-nine-trial evidence, source hashes and limits are in **`Tools/JarSDF/Validation/low-budget/`**. Rejected stabilization/penalty/manifold experiments were reverted and their outcomes retained separately. No solver stiffness increase, extra iterations or relaxed tolerance remains in the final change.

- Eight supported native cases pass: plane, box, octagonal hull, inclined hull, tilted ring, triangle, open hole and edge miss.
- Unsupported containment intentionally rejects with reason 5; the captured convex bolt/nut regression passes.
- The generator-side 31 Python contract/export/evaluation tests passed in the source checkout.
- Historical strict ring impact improved from 4.48 mm to 0.0796 mm at unchanged dt 1/240 and four iterations; its original evidence remains preserved.

Checks include pre-pour containment, full discharge, final table penetration and actuator tracking. Every-step vertex sampling still is **not continuous collision certification or exhaustive hardware-to-hardware/edge-intersection validation**. The same geometry-derived 236.6 mm lift is used in all final controls; the original low height wedged long bolts against the table. Threads/lid engagement are not tested.

## Remaining shortcomings and TODO, in priority order

- [ ] **Multiple-normal manifolds at edges/corners and concave contacts.** One manifold still has one normal basis. Incompatible deep normal patches reject explicitly; support separate stable patches and persistent feature identities before claiming general concave contact coverage.
- [ ] **Reliable fast motion and small features.** **120 Hz / four iterations remains unqualified** in the strict jar tests. Speculative detection is bounded, not CCD. Qualify coarser timesteps with swept/conservative queries and continuous or denser independent penetration checks. The jar result does not validate threads, lid engagement or helical screw motion.
- [ ] **Performance qualification and dispatch.** Profile GPU narrowphase, solver, transfer, initialization and peak allocated memory separately; benchmark more shapes, pile densities and piece budgets on a quiet machine. Field scenes still disable compound hierarchy; optimized convex contact dispatch is now restored for ordinary hardware pairs. Restore remaining compatible acceleration and choose convex versus field by measured cost and accuracy; already-convex objects should keep native convex colliders.
- [ ] **Broader contract validation.** Metric distance, enclosing bounds, planar-region exactness and empty-region clearance remain author guarantees. Structural JSON checks cannot prove an arbitrary field or acceleration certificate. Add invalid-certificate/adversarial fixtures and stronger independent geometric checks; document transforms and scaling explicitly.
- [ ] **SDF↔SDF contacts**, including stable multi-patch contact generation, remain unsupported.
- [ ] **Full containment and unresolved searches.** They fail explicitly; robust penetration recovery remains missing. Keep runtime work limits and witness checks rather than suppressing failures.
- [ ] **CPU field contacts / parity** remain unsupported.
- [ ] **Import/initialization scalability.** Raw surfaces are limited to 508 triangles per collider and currently interact only with fields. Field-specific Metal compilation and bounded witness construction occur at initialization; caching and larger-surface strategies need qualification.
- [ ] **Release gates.** Extend beyond this small cohort to stacking, grasping, friction/rolling, repeated contact transitions and long-duration scenes. The current evidence is an experimental admission test, not robotics release certification.

The jar field is exact for a revolved polygonal profile with polygonal fillets, not an exact circular fillet or threaded closure. This remains a draft until the essential contact and validation gaps above are addressed.
