# Opt-in sleeping for interactive rigid scenes

```swift
var sleep = RigidSleepSettings()
sleep.quietTime = 0.8
try solver.configureRigidSleeping(sleep, groups: [[sofaBody, pillowBody]])
// Interaction APIs wake affected bodies/islands automatically.
solver.wakeRigidBodies() // explicit global wake
try solver.configureRigidSleeping(nil) // disable and restore original inverse masses
```

Sleeping is disabled by default. The implementation tracks quiet rigid contact/joint islands on the host. Optional groups bind bodies into a shared wake/sleep unit; they do not weld their geometry or joints. Sleeping geometry remains available for collisions and picking. Nearby moving bodies and modified supports wake sleepers; pose edits, settings changes, motors and drag/impulse operations also wake the appropriate state.

This path synchronizes at a retirement boundary each step. It is intended for mostly settled interactive rigid scenes, not fully pipelined training, and does not provide a GPU-native island scheduler. Deformables are rejected when enabling it. Do not infer reduced render cost from sleeping counts.

`RigidSleepingTests` covers quiet-body sleep, explicit groups, impulse wakes, changed gravity, moving supports, approaching bodies, invalid configuration and preventing premature sleep before the first integrated step.
