# Robotic Ethernet insertion

`cable-playground cableethernet` runs an automatic approach, alignment, insertion,
and hold. Replay resets the complete physical state. The plug housing is a
deformable tetrahedral solid with a thin shell latch; the jack casing is rigid. The moving tool
holds the rear boot and is coupled to a commanded wrist by three finite springs.
Only the wrist pose is prescribed. Neither the plug nor its contact deformation
is animated. The other cable end is attached to an already-mated connector on
the bench. An approximately 30 cm service loop supplies slack through the full
43 mm wrist stroke. The viewer starts close to the connector so the insertion fills the viewport;
**Whole cable** shows the complete connected apparatus and **Connector close-up** returns to the task. **Reset view**
(or F) restores the current view; scroll and trackpad pinch zoom smoothly. Zoom limits
and rendering tolerances follow the scene's presentation scale; physical units
remain metres.

## Geometry and units

All authored positions are metres, masses are kilograms, time is seconds, forces
are newtons, and elastic moduli are pascals. The renderer uses a smaller near
plane and shadow range for this scene; rendering scale does not change physics.

The plug envelope is 22.48 × 11.68 × 6.60 mm, with eight contacts on 1.02 mm pitch,
a 3.15 mm latch, and a 5.46 mm cable. These dimensions follow the
[Multicomp connector drawing](https://www.farnell.com/datasheets/1860766.pdf).
The rear crimp cavity, lead-in chamfer, boot, rounded latch entry lip, fixture and jack
clearances are explicit geometry. The jack is a designed test fixture, not a
manufacturer-certified CAD reproduction. Gold plating is a visual skin embedded
in the actual housing tetrahedra; it does not introduce rigid plug proxies or
numerically ill-conditioned coating elements.

## Constitutive and tool models

| Part | Model | Reference or assumption |
| --- | --- | --- |
| Plug housing | Stable Neo-Hookean FEM, E = 2.4 GPa, ν = 0.37, density = 1200 kg/m³ | E and density use [Covestro Makrolon 2405](https://solutions.covestro.com/en/products/makrolon/makrolon-2405_000000000000945088) as a generic polycarbonate reference; the connector's resin grade and Poisson ratio are unmeasured. |
| Latch | 0.47 mm plane-stress membrane and plate bending, same PC material, two shared root rows | Thickness and free angle are authored assumptions. The shell avoids the bending stiffness error of a single layer of linear tetrahedra. Its rounded contact boundary is included in the 3.15 mm envelope. Internal velocity smoothing damps unresolved shell vibration (0.3 per 480 Hz step); this is a numerical viscosity approximation, not measured PC loss data. |
| Socket contacts | Eight 8 mm × 0.30 mm spring wires; one rod and finite root spring each, tip stiffness 3EI/L³ | E = 110 GPa, density approximately 8800 kg/m³ from [Copper Development Association C52100](https://alloys.copper.org/alloy/C52100). Wire dimensions, preload and one-mode approximation are authored assumptions. |
| Cable | Native shearable, extensible rod; EA = 3000 N, GA = 1200 N, EI = 0.002 N·m², GJ = 0.001 N·m² | Effective properties of the entire jacketed cable; require measurement. Individual copper strands are not resolved. |
| Wrist/tool compliance | Three 8000 N/m springs at noncollinear anchors, plus a 15 N·m/rad orientation spring | Represents a compliant tool mount, not a calibrated robot servo. |
| Tool inertia | 65.61 g with a box inertia approximation | An authored effective inertial body; its density proxy does not represent the material of the visible gripper parts. |
| Boot attachments | Finite translation bonds and a finite cable weld | An established, no-slip boot grasp; grasp acquisition is outside the task. |
| Friction | Coulomb friction | Assumed, not measured for a particular connector pair. |

The motion uses smooth quintic approach and insertion trajectories. Lateral and
yaw offsets are task inputs, so an incorrectly aligned command can fail through
contact. Seating is assessed from the simulated nose position.
The shared `EthernetInsertionRun` controller stops its command clock at a 20 N
mount reaction and keeps simulating the held tool. This is a task force budget,
informed by the 20 N specification of a [Molex RJ45 connector family](https://www.content.molex.com/dxdam/literature/987650-8240.pdf),
not a measured limit of this particular assembly. An offset approach must stop
before seating. `gamma = 1` keeps the finite springs at their authored stiffness;
decaying their solver penalties would make a displacement-based force reading
incorrect. The support-load regression checks that reading against gravity.

## Stiff-solid convergence

`PhysicsScene.rigidMotionGroups` opts the Metal solver into an additional common
translation/rotation solve for a stiff deformable assembly and its handle. It preserves
the assembly's internal tet deformation while accelerating low-frequency motion.
Every vertex also receives its ordinary VBD update. Material moduli and nodal
masses are unchanged. The coarse pass combines complete contact stencils,
including barycentric cross terms, and uses a shared displacement fraction so
its update cannot invert an internal tet. Default scenes allocate no group
buffers and dispatch no coarse passes. Small assemblies use one persistent
workgroup for the complete colored solve, reusing the existing eight-lane vertex
kernel and a 64-lane coarse reduction. There are no cross-workgroup spin waits.
`AVBD_MOTION_DISPATCHED=1` selects the original dispatch schedule for comparison.

V-T/E-E surface queries now honor each connected component's authored
self-contact flag. A single mixed solid/shell component with self-contact off
skips queries that cannot produce contacts. Rigid/cable-to-surface queries and
contact between separate soft components remain active. This also removes
unintended latch-to-housing contacts from the insertion task.

Current scope is disjoint assemblies containing complete elastic elements,
with ball attachments and rigid handles. Shell features are supported in mixed
solid/shell assemblies. Volumetric self-contact, torsional contact and Planar-DAT
are rejected for this opt-in path. This FEM task and the coarse pass use Metal;
the CPU rigid/cable reference backend is not an equivalent soft-body task solver.

The native cable/soft-surface path finds a closest witness over the entire rod
axis, including triangle-interior and edge crossings. Closed tet boundary
normals eject contact wires outward even at shared face edges. This is spatial
coverage at each collision detection step, not continuous collision detection
over time. Ordinary, unflagged rigid capsules retain their compatibility path.

## Reproduction

The task contains 236 dynamic particles: 684 housing tetrahedra and 20 latch
triangles, with volume/area-lumped mass of 1.159 g. The qualified default uses
480 Hz and 16 iterations. On Apple M5 it seats at 14.780 mm with a 2.303 N peak
mount load. All eight contact wires deflect approximately 0.811 mm and the latch
bends 0.444 mm during insertion. Sampled contact-skin overlap stays below 29 µm
(the rigid contact margin is 25 µm), and minimum sampled tet volume ratio is
0.99973. Sampled contact-wire axes remain outside the housing at seating.

The pre-contact latch-tip excursion is below 53 µm, versus millimetres without
internal damping; the final half-second's tip variation is about 1.1 µm. The
remote cable anchor remains within 0.6 µm. These are checked over the trajectory,
not inferred from a final screenshot. A 1.2 mm lateral error stops the nose
approximately 0.40 mm before the socket mouth at the 20 N force budget.

The 24-iteration comparison seats at 14.762 mm with a 2.072 N peak load, and the
48-iteration reference seats at 14.762 mm with a 2.007 N peak load. This is a solver sensitivity check,
not a calibrated hardware force curve. The 240 Hz alternatives produced
inconsistent force spikes under iteration refinement and are not the default.
The headless 8-second cycle has measured roughly 7–13 seconds on this M5
(9.45 seconds in the final regression run). Background GPU load affects these
timings. The viewer limits physics work per draw to 12 ms to keep camera and
controls responsive, preserving the fixed physical timestep when it falls
behind wall time. This is not a guarantee of real-time performance.

```sh
swift build -c release --product cable-validation
.build/release/cable-validation --ethernet-authoring
.build/release/cable-validation --motion-groups
.build/release/cable-validation --surface-policy
.build/release/cable-validation --shader-regressions
.build/release/cable-validation --ethernet
ETHERNET_OFFSET_MM=1.2 .build/release/cable-validation --ethernet
```

`ETHERNET_ITERS`, `ETHERNET_DT`, and `ETHERNET_VISCOSITY` override the headless task's solver schedule
for convergence checks. Runs write SI-unit force/depth traces under `/tmp`.

## Sim-to-real status

This is a physical task model and a starting point for calibration, not evidence
of successful sim-to-real transfer. Measure the actual plug/jack geometry, tool
compliance, insertion force versus depth, latch loading/unloading, friction, and
cable bending/torsion before training a transferable controller. Repeat those
measurements across tolerances and verify that mesh and timestep refinement do
not materially change the force trace or success boundary. The latch bends over
the entry lip; a locking-hook retention test is not included. Electrical continuity,
contact wear, crimp damage and polymer fracture are not simulated.
