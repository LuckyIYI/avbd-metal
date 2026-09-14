# Experimental implicit contacts

Portable scalar DAG import (`ImplicitField`, Codable) supplies distance and explicit,
automatic or finite-difference gradients. A metric field attached to
`SceneCollider.implicitField` enables GPU sphere, native box, cooked convex and
triangle-backed surface contacts. `ImplicitContactSurface` supplies finite or
infinite planes or zero-thickness two-sided triangles.

```sh
swift build --package-path Tools/ImplicitContacts -c release
Tools/ImplicitContacts/.build/release/implicit-contact-check Tools/ImplicitContacts/Fixtures/ring.json surface-convex 8 /tmp/sdf-box.json
```

Modes: `implicit` (sphere probes), `surface-plane`, `surface-infinite`,
`surface-infinite-far`, `surface-convex`, `surface-hull`, `surface-triangle`,
`surface-hole`, `surface-edge-miss`, `surface-tilted`. `surface-contain` deliberately
fails with reason 5. `surface-convex-generic` disables flat support acceleration;
`surface-boxes-baseline` substitutes 32 approximate ring boxes with identical body
mass/inertia. Run three alternating eight-replica trials for comparison.

The support shortcut requires a flat, spatially spread bounding-face patch and
real field boundary witnesses. Finite edges and uncertified/tilted cases retain
the bounded general search. No sampled convex hull replaces the SDF itself.

## Validation and limits

Local integration checkout: 16 Python export/evaluation tests passed. The native
scenarios above pass final-position checks except the intentional containment
rejection. An earlier tilted transition regression is fixed by strict flat-patch
eligibility. These are not release certificates.

Eight-ring medians, 240 steps, dt 1/240, four iterations: accelerated 1383 steps/s,
generic 214, native 32-box approximation 1438. Includes submission, synchronization
and per-step body reads; excludes shader initialization. These are historical local
checkout numbers; rerun on the PR checkout/hardware before relying on them.

**Known accuracy failure:** a ring dropped 80 mm reaches 4.48 mm transient floor
penetration; final flat-path penetration is 0.50 mm. The regular strict checker
fails this impact test. Do not interpret final-position acceptance as impact
accuracy. This PR improves query support/performance, not contact stiffness or CCD.

Other limits: exact metric field/bounds are an author contract; CPU field contacts
and SDF/SDF reject; triangle surfaces are limited to 508 triangles/collider and
currently interact only with fields. General search exhaustion, undefined normals,
full containment and competing deep normals reject explicitly. One manifold has
one normal basis. Custom-field scenes disable the compound hierarchy/convex overlay.
