# Passive joint responses and detailed rigid scenes

`SceneJoint.response` adds a generalized effort law to an existing hinge or
prismatic constraint. It does not animate a body or replace its constraints.
`JointResponse(knots:damping:maxEffort:)` specifies a continuous piecewise-linear
coordinate/effort curve, nonnegative viscous damping, and a cap on total effort
including damping. Endpoints hold their effort beyond the knot range. Two to
sixteen finite, strictly ordered knots are accepted. Hinge coordinates use the
principal twist interval (-π, π); unsupported responses throw during upload.

Coordinates and efforts are radians and N·m for hinges, metres and N for slides.
Positive effort increases the coordinate. Spring closers, finite-range magnetic
catches, and multiple detent wells are different curves using the same API.
The local implicit solve uses the force derivative and projects negative
curvature to zero in its Hessian approximation. Both bodies receive the
corresponding force/torque; an external pose controller is unnecessary.

```swift
var joint = SceneJoint(bodyA: frame, bodyB: door, rA: anchorA, rB: anchorB,
    stiffnessLin: .infinity, stiffnessAng: .infinity,
    hingeAxis: F3(0, 0, 1), limitLo: 0, limitHi: 1.5)
joint.response = JointResponse(
    knots: [[0, 0], [0.025, -1], [0.1, 0], [1.5, 0]],
    damping: 0.2, maxEffort: 2)
```

`SceneJoint.breakLoad = JointBreakLoad(force:torque:)` specifies optional reaction
thresholds in N and N·m. Thresholds apply to constraint reactions, excluding
passive/motor effort and lever-arm torque from linear reaction. A broken joint
stays broken until repaired/reset. Existing `fracture` behavior remains available;
when a joint authors both, either criterion breaks it. A joint that breaks
mid-step stops applying its passive response for the rest of that step.
Fixed joints may have break loads but cannot have a scalar response curve.

Stateful push-push latches and audio are not represented by a passive effort
curve. Multi-turn hinge responses are not supported by this API yet.

## Island sleeping

The existing opt-in, synchronized rigid sleeping path now caches connectivity
while contact/joint edges remain unchanged. New contacts merge components;
removed contacts and broken joints allow rebuilding/splitting. Static supports
do not merge otherwise independent dynamic islands. Sleeping components retain
their connectivity until woken, since sleep-sleep pairs need not be emitted.
Optional author-supplied groups bind bodies explicitly; groups are not inferred
from asset names. Geometry remains available to collision detection and picking.

```swift
var sleep = RigidSleepSettings()
sleep.energyThreshold = 0.00005 // optional kinetic energy per unit mass, m²/s²
try solver.configureRigidSleeping(sleep)
```

Nil energy threshold preserves the previous linear/angular threshold policy.
Energy mode accounts for body inertia and pose drift as well as velocity.
Swept world AABBs, angular inflation, and collision domains limit unnecessary
wakeups. Picking is read-only; dragging or impulses wake the affected retained
island. Conservative global edits still wake all bodies. `bodyMass` reports
physical mass while asleep. `rigidIslandStatistics` exposes component/rebuild
counts. This remains host-managed at a synchronization boundary, not an
asynchronous GPU island scheduler, and does not support deformables.

## Sparse contact work and scan safety

Large rigid scenes gather active manifold IDs once per step. Repeated dual
updates address that list, including torsional contacts. Original manifold IDs,
primal adjacency order, and warm-start identity remain unchanged. The gather's
atomic order cannot affect independent dual updates. Small scenes and deformable
paths retain their existing schedule. Checkpoint restoration invalidates the derived
active list; contact queries enumerate restored manifolds until the next gather.

Compound BVH nodes also contain conservative box extents. Source hull vertices
and analytic boxes determine the extents. A bound on relative linear and angular
velocity limits speculative padding to motion possible in the current step,
under the existing maximum cap. The original sphere traversal envelope remains
unchanged; the additional box rejection uses the tighter bound. Stationary shapes use the ordinary margin;
fast linear and angular motion have dedicated coverage tests. Tests check required world-space source-geometry bounds independently
of the uploaded hierarchy and compare sparse/full contact solver results.

Shared prefix-sum scratch now covers **every scan consumer**, including compound
pair expansion and soft-contact ordering. The old grid/body-only allocation
could be overrun by pair scans, yielding duplicate contacts and unwritten pair
slots. Each scan checks input, output, and scratch capacity before dispatch.

A captured detailed-floor/hardware query also required more than 16K edge axes
for complete SAT recovery. Its bounded fallback now permits 64K axes; it still
fails closed beyond the budget and never truncates axes. Optional supporting-edge
enrichment returns to its already-valid manifold if its scratch list fills.


## Validation and measured limits

The focused release suite passes 72 tests covering response curves, physical
break loads, hinge stops, sleep/wake connectivity, checkpoint restoration,
scan capacities, sparse/full schedule parity, and convex GPU collision queries.

A local generated fixture with four homes, 657 native bodies, 32,392 authored
colliders and 133 joints passed 600 settling steps and a separate 240-step
spring-drag run. Mean physics step time was 4.73 ms. At 2560 × 1298 with all four
homes visible and simulating, the Fast renderer averaged 40.1 fps over 180
frames after 600 warmup frames: 11.82 ms GPU rendering and 7.58 ms physics per
displayed frame. The latter may contain multiple fixed simulation steps.
This is not a locked 60 fps result or an HQ-rendering benchmark. Generated
fixture assets are application-local; the repository tests contain standalone
regression fixtures for the engine changes.
