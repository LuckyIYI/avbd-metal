# Arachne-15 research notes

## Topic and date window

An iPhone 15 Pro-centred, eight-legged robot that can be generated and checked
programmatically on Apple silicon. Sources checked on 2026-07-15; hardware
specifications use current manufacturer documentation rather than reseller
claims where possible.

## Key findings

- Apple specifies the iPhone 15 Pro at **146.6 × 70.6 × 8.25 mm and 187 g**.
  Apple's accessory drawing also defines camera/sensor keep-outs. The central
  landscape dock therefore points the rear cameras robot-forward, leaves both
  broad faces open, and models another 5.46 mm of camera-side clearance beyond
  the nominal 8.25 mm body.
- The ROBOTIS XC330-M288-T is a plausible small leg actuator: **20 × 34 × 26
  mm, 23 g, full metal gears, 0.93 N·m stall torque at 5 V, and 1.80 A stall
  current**. ROBOTIS explicitly recommends designing near **20% of stall
  torque** for general use; the load check follows that recommendation.
- Eight legs with a one-leg-at-a-time wave gait keep seven contacts nominally
  available. The design is allowed to use no fewer than five stance legs in
  its conservative dynamic load case. A four-leg fast gait is not accepted by
  the present torque model.
- OpenSCAD is an appropriate source-controlled CAD tool because its CLI can
  select and export individual parts and render assemblies without manual GUI
  steps. The installed build is the native universal 2026.06.12 snapshot;
  host verification reports that exact version.
- A fit audit against the meshes exposed by ROBOTIS's configurator found a
  conservative 20 × 29 × 34 mm complete actuator envelope. The FPX330-S102
  drawing confirms a 16 mm mounting pair, 8 mm centre opening, and PCD12 horn
  pattern; those dimensions are encoded in the printable interfaces.
- The iPhone should perform perception, planning, and policy inference, but a
  small wireless MCU bridge should own deterministic servo-bus deadlines and
  emergency stops. Actuator power must never come from iPhone USB-C.

## Emerging signals

- ROBOTIS's current X330 ecosystem exposes current-based control, feedback,
  official frames, CAD drawings, and a multidrop TTL bus. That is materially
  safer for an experimental legged platform than anonymous PWM hobby servos.
- A 3-DOF-per-leg revision would handle uneven ground better, but 24 XC330s add
  184 g, power, cost, and body load. The 16-DOF design deliberately targets
  flat-floor walking first.

## Gaps and unverified areas

- No fabricated prototype has walked yet. The PASS report is a conservative
  analytical screen, not empirical proof.
- Generated closed-mesh volume gives a 273.1 g full-solid PETG/TPU estimate;
  the load budget rounds that up to 275 g. A slicer and physical scale still
  need to replace the density-only estimate.
- Servo thermal behaviour, wiring voltage drop, joint backlash, foot friction,
  fatigue, shock loading, and phone retention require bench tests.
- The official ROBOTIS frame/actuator CAD should be imported into a fabrication
  revision before ordering custom metal parts. The OpenSCAD assembly uses the
  official actuator envelope and PCD12 horn pattern but intentionally relies on
  purchased FPX330 interface frames.
- A production iOS app and its wireless safety protocol are outside this CAD
  branch.

## Source quality notes

- High: Apple technical specifications and dimensional drawing.
- High: ROBOTIS model reference, selection guide, and official frame listing.
- High: OpenSCAD manual and Homebrew cask metadata for the installed tool.
- Low/no weight: reseller torque summaries were not used where manufacturer
  values were available.

## Confidence

**High** for phone envelope/mass, servo envelope/electrical limits, and CAD
reproducibility. **Medium** for the pre-prototype walking feasibility claim.

Sources:

- Apple iPhone 15 Pro tech specs: https://support.apple.com/en-asia/111829
- Apple accessory dimensional drawing: https://developer.apple.com/download/files/accessories/dimensional-drawings/iphone-15-pro.pdf
- ROBOTIS XC330-M288-T: https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/
- ROBOTIS selection guide: https://emanual.robotis.com/docs/en/reference/dxl-selection-guide/
- ROBOTIS FPX330-S102 frame: https://www.robotis.us/fpx330-s102-4pcs-set/
- ROBOTIS configurator meshes: https://robotis.us/viewer-external-hosting/
- FPX330-S102 mechanical drawing mirror: https://www.besttechnology.co.jp/modules/knowledge/gate.php/fpx330-s102.pdf?_noumb=&openfile=fpx330-s102.pdf&refer=BTX134+FPX330-S102+4pcs+Set&way=attach
- OpenSCAD manual: https://files.openscad.org/documentation/manual/OpenSCAD_User_Manual.pdf
- OpenSCAD Homebrew metadata: https://formulae.brew.sh/cask/openscad@snapshot
