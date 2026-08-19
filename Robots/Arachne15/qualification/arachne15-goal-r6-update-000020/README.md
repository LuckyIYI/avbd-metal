# Qualification summary

This directory is evidence for the immutable checkpoint identified by
`30c125b7f01b73bdd1524bc96cf8deb5e8a09897593a49e87aa6ce96f16d3027`.
The JSON reports are authoritative; this page is only a readable index.

## Exact training collision profile

Four held-out seeds, 512 episodes each:

| Seed | Success | Survival | Accepted |
|---:|---:|---:|:---:|
| 27001 | 94.73% | 100% | yes |
| 27002 | 95.31% | 100% | yes |
| 27003 | 94.14% | 100% | yes |
| 27004 | 95.70% | 100% | yes |

Pooled: **1,945 / 2,048 = 94.97%**. Worst seed: **94.14%**.
Every front/left/rear/right and near/far cohort passed its 85% floor.

## Full validation collision profile

The same frozen checkpoint was evaluated with the servo pod, phone guide,
camera plateau, BEC and bridge colliders enabled. This is an explicit task
transfer; no weights changed.

| Seed | Success | Survival | Accepted |
|---:|---:|---:|:---:|
| 28001 | 96.09% | 100% | yes |
| 28002 | 95.51% | 100% | yes |
| 28003 | 95.51% | 100% | yes |
| 28004 | 94.73% | 100% | yes |

Pooled: **1,955 / 2,048 = 95.46%**. Worst seed: **94.73%**.
The validation geometry changed success by +0.49 percentage points, safely
inside the maximum five-point regression gate.

Across these evaluations, full-horizon survival was 100%, penetration RMS was
approximately 0.36 mm, and only about 1.3–1.5% of foot-time samples exceeded
1 mm penetration. The acceptance contract also retains a -3 mm absolute
anti-tunnelling limit.

## Limitation

The selected revision-6 controller is one final randomized training seed. The
four evaluation seeds measure repeatability of that fixed controller; they do
not replace the required independent training-seed campaign. Hardware transfer,
disturbance recovery, slope, battery depletion, thermal and perception tests
remain open gates.

## Deployment inference smoke test

The release Xcode/MLX runtime loaded this exact bundle and compared deployed
actions with direct checkpoint replay. Across 500 Apple-Silicon macOS
inferences, maximum parity error was exactly zero; p50/p95/p99/max latency was
0.389/0.507/0.931/3.960 ms against the 20 ms control period. See
`apple-silicon-macos-inference-smoke.json`. This verifies artifact identity and
the packaged Metal path; it is not a substitute for the required on-iPhone
10,000-inference timing campaign.

The same package also passed a generic physical-device Release compile for
arm64 iPhoneOS with an iOS 17 deployment target. Xcode produced AVBDCore,
AVBDLearn, MLX/MLXNN modules and the iPhoneOS `default.metallib`; see
`ios-device-build-smoke.json`. This is compile evidence, not a signed app launch
or on-device latency measurement.
