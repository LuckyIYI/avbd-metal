# Elastic cables

`PhysicsScene.addCable` creates a native elastic rod from a polyline. Each
segment has six degrees of freedom, cylinder mass/inertia, a capsule collider,
and a material frame whose local +Z follows the segment. `scene.cables` retains
body IDs, joint IDs, rest lengths and endpoint anchors. These IDs work with
ordinary body state updates, picking, rendering and rigid attachments.

```swift
import GPUSim

var scene = PhysicsScene(name: "Cable")
scene.settings.dt = 1 / 240
scene.settings.iterations = 24
let radius: Float = 0.015
let cable = try scene.addCable(
    points: (0...32).map { F3(Float($0) * 0.05, 0, 1.5) },
    radius: radius,
    density: 1200,
    material: .circular(radius: radius, youngModulus: 2e7,
                        poissonRatio: 0.3, dampingTime: 0.01),
    fixedSegments: [0])
let solver = try GPUSolver(scene: scene)
try solver.submitStep()
try solver.synchronize()
print(solver.bodyPosition(cable.bodyIDs.last!))
```

Use `Demos.make("cables")` for suspended cables of different bend stiffnesses
with a rigid payload. The existing analytic capsule renderer requires no new
mesh upload or CPU skinning pass. `scene.replicated` remaps cable metadata and
constraints along with the other scene objects.

## Material and attachment model

`CableMaterial` accepts stretch rigidity EA and shear rigidity GA in newtons,
bend rigidity EI and twist rigidity GJ in N m², and a Kelvin–Voigt relaxation
time in seconds. `circular` computes these from Young's modulus, Poisson's
ratio and the circular cross section; its shear correction factor is one.
Alternatively, author the four rigidities independently. Bend and twist may
be zero; stretch and shear must be positive and finite.

Each connection divides these rigidities by the mean adjacent segment length.
Changing resolution therefore preserves the underlying material coefficients
instead of retaining a resolution-dependent per-joint spring constant. Use
SI units consistently for geometry, density, gravity and material parameters.
The capsule's overlapping end caps contribute collision geometry, not extra
mass: total cable mass is density × cross-sectional area × polyline length.

The initial polyline is stress-free, including curved shapes. Frames are
parallel-transported along the polyline. Optional `rotations` can author the
material phase; each must be a unit quaternion with +Z along its edge. Invalid
geometry, fixed-segment indices or frames throw before any scene mutation.

`fixedSegments` clamps complete segments. To pin just an endpoint, author a
normal ball joint using `startAnchor` or `endAnchor`:

```swift
// Author before constructing the solver. The world point should coincide
// with the endpoint at rest (the body's rotation maps its local anchor).
let first = cable.bodyIDs[0]
let point = scene.bodies[first].position
    + scene.bodies[first].rotation.act(cable.startAnchor)
scene.addJoint(SceneJoint(bodyA: -1, bodyB: first,
                         rA: point, rB: cable.startAnchor))
```

For a moving rigid attachment, use that body's ID for `bodyA` and its local
anchor for `rA`. Set `stiffnessAng: .infinity` to clamp orientation as well.
Use an unfixed endpoint segment when pinning it. Branches can share a rigid
attachment; custom elastic connections can set `SceneJoint.cable` to a
`CableJointMaterial` while leaving the ordinary joint stiffnesses zero.

Adjacent segments are excluded from collision through their joint topology.
Nonadjacent segments and other bodies use the existing capsule narrow phase,
friction, collision groups and exclusion APIs. Cable friction anchors rotate
with the segment material. Contact is discrete: choose a timestep and segment
length appropriate to the radius, speed and curvature. There is no new CCD
guarantee. Very short, thick segments may need additional nearby-pair
exclusions because their capsule caps overlap beyond immediate neighbors.

## Energy and solver integration

For parent and child poses `(xA, RA)` and `(xB, RB)` with anchors `rA`, `rB`:

```
c = RAᵀ (xB + RB rB - xA) - rA
e = log((RA Rrest)ᵀ RB)
E = ½ cᵀ diag(GA, GA, EA)/L c + ½ eᵀ diag(EI, EI, GJ)/L e
```

`Rrest` is the authored relative orientation. The principal SO(3) logarithm
expresses bend/twist in the rest-child material frame. The angular law has
exact linear torque versus angle for pure bend or twist. This is a discrete
Cosserat-style material, distinct from Newton's curvature-binormal/Bishop
bend-twist split. At large simultaneous bend and twist the models differ.
Neighboring rest-relative rotations must stay below pi; refine the cable for
tight bends or concentrated twist. Plasticity, hysteresis, fracture, dedicated
cable friction, and a continuous surface mesh are not implemented.

The damping potential is `½ (dampingTime/dt) Δcᵀ K Δc` plus the analogous
angular term. Initial strains are cached before prediction, and analytic
Jacobians stamp the force and positive-semidefinite Gauss–Newton blocks into
the existing AVBD 6×6 solve. The moving parent frame is differentiated, which
preserves the coupled translation/angular response. A small-angle series
avoids cancellation in the logarithm Jacobian. No finite-difference gradients,
per-iteration allocations or extra CPU/GPU synchronization are used.

These finite material energies share the AVBD iterations with hard joints and
contacts, like existing elastic elements; they do not use adaptive penalties
or multiplier clamping. The cable flag selects a dedicated stamp within the
existing joint dispatch, including scalar, SIMD and fused solver paths. It
uses the inactive motor fields as explicitly tagged material storage and
preserves the 256-byte joint ABI. No cable-only buffers or dispatches are
added. Undamped joints skip history evaluation; zero angular rigidity skips
the angular logarithm. Reset preserves the material coefficients.

## Validation and performance

Run the standalone CPU/Metal validation without XCTest or external packages:

```sh
swift run -c release cable-validation
swift run -c release cable-validation --benchmark
```

`--cpu-only` runs without a Metal device. The cable GitHub Actions workflow
uses a full-Xcode macOS runner for the CPU XCTest cases and physical checks;
Metal validation and timings run on a local Apple GPU.

The validation covers atomic authoring, mass accounting, material anchors,
replication, analytic Jacobians, quaternion sign invariance, curved rest
shapes, axial Hooke equilibrium, torsion, free material spin, nonlinear
cantilever moment balance, floor contact, guided interior cable/cable contact,
CPU/Metal transient parity, and
reset/fresh equivalence. XCTest adds full force-versus-energy derivatives on
both bodies, positive-semidefinite blocks, contact-normal order reversal,
authoring edge cases and demo/ABI checks:

```sh
swift test --filter Cable
python3 -S Tools/verify_architecture.py
python3 -m unittest discover -s Tools/tests -p test_verify_architecture.py
```

The floor regression also fixes the existing CPU capsule/box normal sign;
the prior normal pushed a falling capsule into the box instead of supporting
it. The two collider orderings now follow the same B-to-A normal convention.

Benchmark timings are end-to-end wall time including submission and final
synchronization, excluding solver construction and shader compilation. The
benchmark uses a 1/240-second step, 12 iterations, 60 warmup frames and five
120-frame samples, reporting their median. It includes both suspended cables
with contact disabled and cables resting on box floors with contact enabled.
These measure this Metal backend; they are not a speed comparison with
Newton/CUDA on different hardware.

Measured on 2026-09-11: Apple M5, macOS 26.5.2, Swift 6.3.3, release build:

| Segments | No contact, ms/frame | Floor contact, ms/frame | Floor pairs |
|---:|---:|---:|---:|
| 16 | 0.2871 | 0.4967 | 16 |
| 256 | 0.2921 | 0.5657 | 256 |
| 1,024 | 0.3086 | 0.8464 | 1,024 |

All benchmark states remained finite. Maximum connector gaps at 1,024
segments were 0.005835 m suspended and 0.000044 m on the floor, for 0.1 m
segments with EA = 2,000 N, GA = 1,000 N, EI = 0.5 N m² and GJ = 0.2 N m².
The benchmark checks gaps below 0.01 m; the reported gaps include physical
elastic deformation and finite-iteration error, not just numerical error.

An ordinary finite-joint control (no cables or contact, 12 iterations, seven
240-frame samples after 60 warmup frames) compared this implementation with
base commit `4f5be05`: 256 bodies measured 0.2960 → 0.2997 ms/frame and 1,024
bodies 0.3260 → 0.3195 ms/frame. Sample ranges overlapped, so this control did
not show a meaningful regression or establish a speedup.

The standalone CPU and Metal checks, full release build, architecture gate,
and 23 architecture unit tests passed on that machine. `swift test --filter
Cable` could not compile because XCTest is absent from Command Line Tools;
the XCTest cases require a full Xcode installation. No Newton runtime
benchmark was run on this Mac.

Reference: [Newton's rod kernels at 811b7b1](https://github.com/newton-physics/newton/blob/811b7b1ac803e193f063819d6e5d085cebdcddf6/newton/_src/solvers/vbd/rigid_vbd_kernels.py)
and [rod authoring API](https://github.com/newton-physics/newton/blob/811b7b1ac803e193f063819d6e5d085cebdcddf6/newton/_src/sim/builder.py).
The Swift/Metal constitutive implementation here is independently derived.
