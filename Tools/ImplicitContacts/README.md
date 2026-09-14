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

**Impact regression fixed:** SDF queries now use bounded speculative detection
for approaching pairs, while retaining the same solver collision margin. The
formerly failing ring-drop check passes at 0.0796 mm maximum penetration (previously
4.48 mm), with dt 1/240 and four iterations unchanged. This is not general CCD.

Convex hull triangles are grouped into unique outward face planes. A support face
can reuse the flat-patch path only when all other hull halfspaces establish that
no edge or other face intersects the field bounds. A cached previous face is tried
first and revalidated; generic search remains the fallback. No concavity is removed
from the field. `surface-hull-octagon` and `surface-hull-octagon-ramp` exercise a
non-box hull and inclined support. Append `-generic` for matched comparisons.

`validation-convex.json` retains all trials and the original ramp check that used
world Z incorrectly. The corrected check measures support distance along the ramp
normal, with the same 1 mm limit; maximum penetration is 0.0843 mm. Native box,
cooked hull, plane, triangle and tilted-ring impacts pass. Full containment still
rejects. Historical benchmark data above predates this impact fix. Current octagon
trial timings vary substantially under shared-machine load; inspect all samples.

Other limits: exact metric field/bounds are an author contract; CPU field contacts
and SDF/SDF reject; triangle surfaces are limited to 508 triangles/collider and
currently interact only with fields. General search exhaustion, undefined normals,
full containment and competing deep normals reject explicitly. One manifold has
one normal basis. Custom-field scenes disable the compound hierarchy/convex overlay.
