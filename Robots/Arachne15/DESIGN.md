# Arachne-15 phone-first body derivation

## Fixed requirements

- The bare iPhone 15 Pro is 146.60 × 70.60 × 8.25 mm and 187 g.
- It operates upright in landscape, at the robot centre, with rear cameras
  facing robot-forward (`+X`).
- The phone must insert and leave from above without removing legs or the body.
- The camera plateau, screen centre, and broad heat-rejection surfaces remain
  unobstructed.
- Eight two-axis legs target slow flat-floor wave-gait walking first.
- Purchased ROBOTIS frames carry servo-interface loads; printed parts do not
  imitate a bearing or precision servo frame.

## Derived architecture

The phone's 147.80 mm clearance width sets the minimum transverse structure.
Adding end-guide mounts and continuous load rails produces a 176 mm chassis
ring. The chassis is 154 mm fore-aft: enough for four hip stations without
making the unsupported centre panel large.

The chassis is a closed 14 mm perimeter torsion ring, not a solid tray. A 28 mm
transverse spine crosses its centre. That spine performs three jobs:

1. its 2 mm residual web directly supports the phone bottom;
2. it closes the torsional load path between left and right leg rows;
3. it locates the battery directly under the phone.

Two identical U-channel guides bolt to the spine/ring intersection. They only
capture the phone's short end edges and thickness, leaving the camera and both
broad faces open. One guide rotates 180 degrees, reducing unique parts and
assembly error. TPU tape provides compliant glass contact; a silicone strap is
the secondary retention path.

## Leg placement and clearance

Hip axes are at `X = -60, -24, +24, +60 mm` and `Y = ±92 mm`. Each hip gets a
local 28 × 28 × 7 mm load pod merged into the ring. The nearest XC330 envelope
begins 14 mm from the centreline while the phone guide ends at 11 mm, preserving
3 mm nominal clearance even where their Y/Z projections overlap.

The nominal feet span approximately 265 × 362 mm. With any single leg lifted,
the computed support polygon retains 95.0 mm minimum margin around the nominal
centre of mass projection.

## Mass and load closure

Generated closed-mesh volume, multiplied by conservative PETG/TPU densities,
is 278.6 g for all 30 printed pieces at 100% solid. The system mass budget rounds
this to 280 g and totals 1.220 kg.

Using 20% of the XC330-M288-T 5 V stall torque as the design limit gives:

- 0.186 N·m available design torque per knee;
- 0.159 N·m required in the five-leg, 1.5× dynamic case;
- 1.17 minimum five-leg safety factor;
- 1.63 safety factor in the intended seven-leg wave gait.

The estimated centre-of-mass height is 87.3 mm. The 95.0 mm worst support
margin corresponds to a first-order 1.09 g lateral tip threshold.

## Deliberate limitations

This architecture prioritizes a buildable flat-floor prototype and a safe,
observable gait. Two degrees of freedom per leg cannot actively place a foot
laterally on irregular terrain. A later rough-terrain version should use three
axes per leg and re-run the mass, power, thermal, and structural closure rather
than attaching another joint to this body without analysis.

Analytical closure is not physical validation. The mandatory progression is:
phone fit, static chassis load, single-joint current/thermal test, suspended
gait, tethered stand, tethered crawl, and only then untethered walking.
