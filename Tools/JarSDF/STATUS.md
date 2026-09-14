## Problem and resulting behavior

Procedurally authored distance fields can now be imported as opt-in rigid colliders, preserving cavities without many convex pieces. This draft adds a portable scalar-expression DAG, validation and gradients, and GPU sphere/field and triangle-backed surface/field contacts using AVBD manifolds and friction. Native boxes, cooked convex hulls, planes and two-sided triangle surfaces are supported within the limits below.

The latest increment makes a matched jar-and-fastener pour complete successfully: **six of six final trials pass** using the same frozen assets, mass/inertia, seeded initial poses and constraint-actuated motion on both backends. No production asset is switched automatically.

## Implemented

- JSON field import, explicit/automatic/finite-difference gradients, including `sign` and a zero-set fallback for degenerate analytic derivatives.
- Specialized flat-support queries using actual hull halfspaces and revalidated previous-face hints; bounded triangle search otherwise.
- Generated geometry tables in Metal constant storage, fixing the original per-thread allocation/GPU-memory failure.
- Author-certified exact planar regions and convex radial empty regions to bypass unnecessary field search. The importer validates structure; authors must prove semantic equivalence to their field.
- Triangle minimization includes vertices and stays inside its search subtriangle; field witnesses and normals remain checked.
- Speculative detection accounts for angular as well as linear motion.
- SDF contact-cache reuse revalidates the normal gap, preventing stale friction anchors from retaining an incorrect curved-surface tangent plane.
- Corrected convex SAT fallback witness reconstruction in both narrowphase paths, with a captured bolt/nut regression.
- Standalone `Tools/ImplicitContacts` and new `Tools/JarSDF`, including the complete frozen jar/fastener fixture, numeric results and reproduction instructions.

## Measured validation

Matched jar cohort: three seeds per backend, twelve fasteners, six simulated seconds, **dt 1/480 s and four iterations on both paths**. Physical world-anchor constraints drive the jar; there are no per-step pose overwrites.

| Measure | SDF | Convex decomposition |
|---|---:|---:|
| Passing trials | 3/3 | 3/3 |
| Jar colliders | 1 | 512 |
| Total scene colliders | 86 | 597 |
| Median physics steps/s | 353.9 | 236.3 |
| Throughput range | 313.7–470.7 | 89.4–265.1 |
| Worst sampled jar penetration | 0.872 mm | 0.603 mm |
| Discharged pieces per trial | 12/12 | 12/12 |
| Serialized jar collision data | 15,410 bytes | 856,469 bytes |

**1.50× median throughput in this workload**, not a universal speedup. Physics steps/s is not rendering FPS; at 480 steps per simulated second the SDF median is **0.74× realtime**. Timings include submissions/fences, exclude setup and validation reads, and vary with shared-machine load and trajectories. Serialized size is not GPU allocation measurement. The convex approximation has at most 0.1024 mm angular chord error against the shared radial profile.

The unchanged 1 mm gate also checks pre-pour containment, full discharge, final tabletop penetration and actuator tracking. Penetration uses sampled mesh vertices at actual simulated poses, **not continuous collision certification or exhaustive hardware-to-hardware checks**. The earlier 240 Hz trial was insufficient; the original low pour height also wedged long bolts against the table. The final protocol uses a geometry-derived 236.6 mm lift. Earlier failures have not been relabeled as successes.

- 31 generator-side Python contract/export/evaluation tests pass (external source checkout).
- Eight supported native contact regression cases pass: plane, box, octagonal hull, inclined hull, tilted ring, triangle, open hole and edge miss.
- Unsupported full containment is intentionally rejected; it is not a supported success case.
- Captured convex bolt/nut witness regression passes.
- Historical strict ring impact improved from 4.48 mm to 0.0796 mm at unchanged dt 1/240 and four iterations. Existing `validation-main.json` and `validation-convex.json` retain that evidence.

See `Tools/JarSDF/README.md` and `Tools/JarSDF/Validation/` for the complete final cohort, traces and provenance. Results were collected in the integration checkout before standalone packaging; the recorded manifest identifies those original sources.

## Remaining shortcomings and TODO, in priority order

- [ ] **Multiple-normal manifolds at edges/corners and concave contacts.** One manifold still has one normal basis. Incompatible deep normal patches reject explicitly; support separate stable patches and persistent feature identities before claiming general concave contact coverage.
- [ ] **Reliable fast motion and small features.** Speculative detection is bounded, not CCD. Qualify coarser timesteps with swept/conservative queries and continuous or denser independent penetration checks. The jar result does not validate threads, lid engagement or helical screw motion.
- [ ] **Performance qualification and dispatch.** Profile GPU narrowphase, solver, transfer, initialization and peak allocated memory separately; benchmark more shapes, pile densities and piece budgets on a quiet machine. Field scenes currently disable compound hierarchy/optimized convex overlay. Restore compatible acceleration and choose convex versus field by measured cost and accuracy; already-convex objects should keep native convex colliders.
- [ ] **Broader contract validation.** Metric distance, enclosing bounds, planar-region exactness and empty-region clearance remain author guarantees. Structural JSON checks cannot prove an arbitrary field or acceleration certificate. Add invalid-certificate/adversarial fixtures and stronger independent geometric checks; document transforms and scaling explicitly.
- [ ] **SDF↔SDF contacts**, including stable multi-patch contact generation, remain unsupported.
- [ ] **Full containment and unresolved searches.** They fail explicitly; robust penetration recovery remains missing. Keep runtime work limits and witness checks rather than suppressing failures.
- [ ] **CPU field contacts / parity** remain unsupported.
- [ ] **Import/initialization scalability.** Raw surfaces are limited to 508 triangles per collider and currently interact only with fields. Field-specific Metal compilation and bounded witness construction occur at initialization; caching and larger-surface strategies need qualification.
- [ ] **Release gates.** Extend beyond this small cohort to stacking, grasping, friction/rolling, repeated contact transitions and long-duration scenes. The current evidence is an experimental admission test, not robotics release certification.

The jar field is exact for a revolved polygonal profile with polygonal fillets, not an exact circular fillet or threaded closure. This remains a draft until the essential contact and validation gaps above are addressed.
