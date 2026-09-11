# Configurable GPU hinge stops

`SceneJoint.limitStiffness` sets the angular stop penalty in N m/rad. Its default is 40,000, preserving existing GPU behavior. It must be finite and positive. Set it after constructing the joint; the angular interval remains `limitLo...limitHi`.

Small, light mechanisms can require lower stop stiffness than furniture-scale doors at the same timestep. For example, a 6 g flip-top can use 2 N m/rad while retaining its original 0...1.85 rad interval and hard hinge-anchor constraints. This is a compliant stop, so some load-dependent angular deflection is intentional. It does not disable contacts or convert a lid to a fixed body. Choose a value for the mechanism's mass, dimensions, timestep and acceptable deflection.

The existing stop force clamp is unchanged. The GPU joint layout uses the previously unused limits.w slot, without changing buffer stride. This setting applies to the GPU solver; it does not add CPU hinge-limit parity.

Validation: `swift test -c release --filter HingeStopComplianceTests` passes two tests: historical default preservation and a loaded small-lid comparison checking finite motion, hinge-anchor error, bounded deflection, and greater deflection with a softer stop. A separate generated four-home scene retains its complete articulation and collision checks; that scene cohort is not certification of arbitrary parameter domains.
