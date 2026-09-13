# Finite grasp and helical constraints

Finite spring stiffness is a physical energy coefficient and now remains constant across warm starts. Hard constraints retain adaptive penalties. `setGrasp` activates an existing joint slot without resetting the payload's pose or velocity; passing a nil payload releases it. This is an authored compliant attachment, not a claim of emergent finger/contact grasping.

`setHelicalJoint` couples local +Z translation and twist reciprocally using positive pitch in metres per revolution. An optional lower travel stop represents a seated end. `helicalAngle` returns accumulated radians. This ideal constraint does not model thread friction, preload, cross-threading, automatic disengagement, or geometric thread contact. Samples between steps must be sufficiently close to unwrap rotation unambiguously.

The feature is covered by spring sag, carry/release, axial backdrive, multi-turn torque transmission and seated-stop regressions. Broader interaction validation is required before using it as a physical thread replacement.
