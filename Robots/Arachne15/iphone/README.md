# Arachne-15 iPhone deployment runtime

The repository now contains the complete deterministic policy path that can
run on iOS 17 with MLX. It does **not** claim that an unmeasured robot is safe
to walk. The code deliberately refuses to arm until an immutable qualified
policy and a commissioned per-robot calibration agree on the same checkpoint
fingerprint.

## Qualified field candidate

- Bundle: `../../../checkpoints/arachne15-goal-v1`
- Task: `arachne15-goal-v0`, revision 2000006
- Checkpoint fingerprint:
  `923e07c286f4fdb186b30a6fd95469e6848f4fec4ca1e3811320424b94c9dc02`
- Policy SHA-256:
  `9521c03cab6fc9e829cd2664fa0e086f69720d4aa46b1e5b893776a4df072c14`
- Deterministic control period: 20 ms / 50 Hz
- Tensor contract: 60 float observations to 16 normalized joint offsets

Goal v1 is an exact zero-update epoch-2 requalification of the historical v0
bundle, so the policy SHA-256 is unchanged while its metadata-bound checkpoint
fingerprint is deliberately different. Its schema-v2 manifest seals separate
four-seed nominal and full-collision validation suites plus the predeclared
cross-suite degradation gate.

`VectorPolicyDeploymentRuntime` verifies the manifest schema, task/revision,
policy digest, complete checkpoint fingerprint, tensor dimensions, timing,
normalizer, action distribution, task configuration, and training state before
MLX loads the actor. `GuardedPolicyController` rejects stale, future,
out-of-order, wrong-sized, or non-finite sensor frames; inference errors,
deadline misses, and invalid actions latch a safe stop. Re-arming is explicit.

`Arachne15DeploymentController` then maps the policy output through the
commissioned servo zeros and direction signs. Active commands include the
policy fingerprint and a one-period deadline. Safe-stop commands contain no
servo positions, so the bridge cannot mistake sixteen zeros for absolute
shaft angles.

## Policy input

All coordinates are right-handed: robot forward `+X`, left `+Y`, up `+Z`.
The first three vector groups are expressed in the robot body frame.

| Channels | Meaning | Hardware source |
|---:|---|---|
| 0–2 | body linear velocity, m/s | fused camera VIO + IMU |
| 3–5 | body angular velocity, rad/s | bias-corrected IMU gyro |
| 6–8 | world-up projected into body frame | fused attitude, normalized |
| 9–11 | desired body `vx`, `vy`, yaw rate | exact point-goal command function |
| 12–27 | relative joint position, rad | calibrated XC330 feedback |
| 28–43 | joint velocity, rad/s | timestamped differentiated feedback |
| 44–59 | previous applied normalized action | acknowledged bridge command |

The shared `Arachne15PolicyContract` generates this tensor in both the
simulator and iPhone path. A unit test compares the complete no-noise simulator
observation against the hardware encoder. Do not reconstruct the array in UI
code.

The policy does not consume an image or world goal directly. Perception first
estimates a world-space goal, then
`Arachne15PolicyContract.pointGoalCommand(...)` produces the exact body-twist
interface used in training. This keeps future hand-gesture perception separate
from locomotion qualification.

## Minimal app integration

Embed `AVBDCore` and `AVBDLearn` from this Swift package in an iOS 17 app and
copy the qualified policy bundle plus a commissioned calibration into app
resources:

```swift
let runtime = try Arachne15DeploymentController(
    bundleDirectory: policyBundleURL.path,
    calibration: try JSONDecoder().decode(
        Arachne15HardwareCalibration.self,
        from: Data(contentsOf: calibrationURL)))

runtime.arm() // only after bridge heartbeat, battery, temperature and tether checks
let command = runtime.command(
    for: measuredPolicyInput,
    sequence: sensorSequence,
    sensorTimestampSeconds: sensorTimestamp)
sendToSafetyBridge(command)
```

The generic physical-device Release build is continuously reproducible with
`make ios-ml`. It has passed for arm64 iPhoneOS with MLX's Metal library
packaged. App signing, sensor entitlements, transport selection, and measured
on-device latency remain commissioning work because they depend on the target
iPhone, bridge firmware, and chosen radio link.

The template at `hardware-calibration.template.json` is intentionally invalid:
`commissioned=false`, no robot serial, no timestamp, zero current limits, and
zero measured latency. Copy it into a dated run under the ignored
`artifacts/calibration/` directory; never edit it into a plausible-looking
default.

## Bridge requirements

The ESP32-S3 is a safety controller, not a transparent radio adapter. It must:

1. reject an unexpected policy fingerprint, duplicate/out-of-order sequence,
   expired deadline, malformed array, or non-finite value;
2. interpolate 50 Hz targets at the servo-bus rate without exceeding measured
   joint speed/acceleration;
3. enforce calibrated joint, current, voltage, and temperature limits locally;
4. publish timestamped position/current/temperature/fault feedback;
5. on heartbeat loss or `safe-stop`, lower the body using a verified local
   routine, torque-limit, then disable torque;
6. keep a physical kill switch and rail fuse independent of phone software.

BLE/Wi-Fi serialization and bridge firmware are not implemented in this
branch, because selecting that transport before measuring round-trip jitter and
servo-bus load would bake an assumption into the safety boundary. The policy
was trained with 0–40 ms action latency; the calibration validator refuses a
measured worst case above 40 ms, while the phone inference deadline is 18 ms.

## First hardware campaign

The next authorized test is not untethered navigation. Follow this order and
retain all raw telemetry with the policy and calibration fingerprints:

1. assign/verify unique servo IDs and directions with the rail current-limited;
2. measure every mechanical zero and powered joint limit;
3. identify one-leg step/chirp response, current and latency;
4. run 10,000 policy inferences on the iPhone and report p50/p95/p99/max time;
5. suspended 50 Hz command replay with watchdog interruption tests;
6. tethered neutral stand, then low-body straight crawl;
7. tethered random goals, then held-out floor/friction/payload trials.

The current policy passes its epoch-2 simulation qualification but inherits
weights from one final randomized training seed. It is a strong tethered-test
candidate, not yet the five-training-seed publication artifact required by
`../sim/SIM_TO_REAL.md`.
