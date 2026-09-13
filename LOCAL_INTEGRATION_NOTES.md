# Experimental small-feature contact work

This draft rejects extrapolated MPR witnesses and scales GJK simplex degeneracy calculations to the input geometry. Its captured disjoint thread/ring regression runs at three scales.

The optional convex-compound boundary filter removes contacts whose outward probe lies inside another hull owned by the same body. It is disabled by default. Only cooked convex colliders are accepted, adjacency is built once with a bounded quadratic scan, and geometry changes require rebuilding that adjacency. The probe distance is in scene units. The current support/open-gap test is not certification for arbitrary compounds, helical locking or high-speed contact. Moving and rotated compounds, performance and adverse overlap configurations need further validation before promotion from draft.

`applyAngularVelocityImpulses` accepts velocity increments, not torque impulses; the caller performs inverse-inertia conversion. `debugRigidContactWitnesses` is a synchronous diagnostic query.
