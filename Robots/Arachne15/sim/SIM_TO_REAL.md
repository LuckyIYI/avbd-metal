# Arachne-15 sim-to-real experiment contract

Simulation randomization is not evidence of transfer by itself. This document
defines the measurements required to replace assumptions, the order in which
to add uncertainty, and the gates that keep a visually convincing policy from
being mistaken for a reliable robot controller.

## What is authoritative today

| Quantity | Current source | Required replacement |
|---|---|---|
| Link geometry and joint axes | Parametric CAD/MJCF generator | Printed-part and assembled-jig measurements |
| Whole-robot mass, COM, inertia | Component mass/envelope calculation | Per-link scale, balance jig, pendulum identification |
| Joint range | Mechanical CAD envelope | Powered low-current end-stop sweep |
| Torque limit | 20% of 5 V published stall torque | Dynamometer torque-speed/current curve at each battery voltage |
| Contact friction | Material estimate | TPU-foot incline and drag tests on target floors |
| Servo PD, damping, armature | Conservative controller estimate | One-leg chirp/step-response system identification |
| Control latency | Seeded 0--40 ms task range | Timestamped iPhone→bridge→servo→encoder round trip |

Raw measurements, scripts, firmware revision, temperature, battery voltage,
and fixture photos belong together under a dated `calibration/` run. Never
overwrite the nominal file without retaining the measurement provenance.

## Fidelity ladder

1. **Nominal stand:** no randomization, zero command, 60 s simulated and then
   tethered hardware. Reject any thermal/current violation or uncommanded
   drift.
2. **One-leg identification:** fit motor gain, damping, armature, latency, and
   backlash against held-out chirps. Report trajectory RMSE and peak-current
   error, not only the fitted parameters.
3. **Nominal velocity:** train on the explicit primitive collision profile.
   Evaluate fixed seeds on both training and validation collision profiles.
4. **Point-goal navigation:** initialize from a passing velocity controller,
   then evaluate random bearing/distance targets. Require stable arrival, not
   only minimum distance, and publish final/minimum distance distributions.
5. **Measured randomization:** expand each range only to cover repeated
   hardware measurements plus sensor uncertainty. Keep a nominal cohort so
   excessive randomization cannot hide model regressions.
6. **Disturbance validation:** randomized payload placement, foot friction,
   voltage, latency, slope, and external pushes. Hold out combinations and
   random seeds from training.
7. **Hardware crawl:** suspended gait, tethered stand, low-body straight crawl,
   turns, then untethered trials. The ESP32 watchdog and physical kill switch
   remain independent of the learned policy at every stage.

## Minimum publish gates

- At least five training seeds; publish median and worst seed.
- At least 512 fixed evaluation episodes per seed, plus a separate untouched
  final seed set.
- ≥95% full-horizon survival on nominal flat ground.
- ≥90% command success across the measured randomization envelope.
- Linear-velocity RMSE ≤0.15 m/s and yaw-rate RMSE ≤0.40 rad/s in simulation;
  tighten these after hardware range/speed measurements.
- No validation-profile regression larger than five percentage points.
- Hardware: zero watchdog misses, zero over-current/temperature stops, and all
  falls caught by the tether during the staged test campaign.

The task exports `episode/survived`, linear/yaw RMSE, root height, projected
gravity, contact count, and individual reward/penalty terms. Experiment reports
must retain the checkpoint fingerprint, complete serialized task configuration,
trainer configuration, commit, seed list, raw JSONL metrics, and evaluation
JSON. A screenshot or one successful replay is never an acceptance result.

`eval-rl` rejects any checkpoint/task configuration drift by default. An
intentional out-of-distribution check (for example, replaying a training-
collision checkpoint against `validationCollisionProfile=1`) must opt in with
`--allow-task-transfer`. Provenance-v3 evaluation JSON records both the
checkpoint contract and the evaluated contract plus an explicit transfer bit;
the aggregators refuse to mix reports from different contracts.

```sh
.xcbuild/Build/Products/Release/avbd eval-rl arachne15-goal-v0 \
  --checkpoint runs/arachne15-goal-v0/accepted/checkpoints/update-NNNNNN \
  --envs 512 --episodes 512 --seed 31001 --allow-task-transfer \
  --task-option validationCollisionProfile=1 --output validation-31001.json
```

Supply every non-default task option used by the accepted checkpoint as well;
the output makes any remaining difference inspectable rather than implicit.

After checkpoint selection and untouched-seed verification, export the exact
evaluated policy rather than copying a mutable run directory:

```sh
.xcbuild/Build/Products/Release/avbd export-policy-rl arachne15-goal-v0 \
  --checkpoint runs/arachne15-goal-v0/accepted/checkpoints/update-NNNNNN \
  --output field-bundles/arachne15-goal-v0-update-NNNNNN
```

The exporter refuses to overwrite a bundle. It copies only deterministic
inference files (no optimizer), verifies that their checkpoint fingerprint is
unchanged, and writes `deployment-manifest.json` with the policy SHA-256,
task/revision, exact task configuration, tensor dimensions, normalization and
action-distribution contract, 50 Hz control rate, and training provenance.
Hardware logs must include that manifest fingerprint on every trial.

### Current immutable candidate (2026-07-16)

`Robots/Arachne15/policies/arachne15-goal-r6-update-000020` is frozen at
checkpoint fingerprint
`30c125b7f01b73bdd1524bc96cf8deb5e8a09897593a49e87aa6ce96f16d3027`.
Four held-out seeds on the exact training collision profile achieved
1,945/2,048 successes (94.97%, worst seed 94.14%) with 100% full-horizon
survival. Four separate full-geometry validation seeds achieved 1,955/2,048
(95.46%, worst seed 94.73%) with 100% survival. Every directional and distance
cohort passed. The validation profile improved pooled success by 0.49 points.

The contact gate is time-resolved: foot penetration RMS must be no more than
0.5 mm, no more than 2.5% of foot-time samples may exceed 1 mm penetration,
and the absolute minimum clearance must remain above -3 mm. The candidate
measured about 0.36 mm RMS and 1.3--1.5% over 1 mm across the reported seeds.
This distinguishes normal short contact impacts from sustained floor
exploitation while retaining a hard anti-tunnelling bound.

These are fixed-policy evaluation seeds, not independent training seeds. The
candidate is eligible for suspended and tethered hardware experiments; it is
not the five-training-seed publication artifact required by the gate above.
See the bundle's `qualification/` directory for raw reports and
`Robots/Arachne15/iphone/README.md` for the on-device and safety contract.

The point-goal task additionally exports `episode/goal_reached`, goal-entry,
arrival-speed and dwell metrics,
`episode/final_goal_distance_m`, `episode/minimum_goal_distance_m`, live goal
distance, progress reward, and stable-arrival reward. Before camera input is
enabled, these measure navigation/controller performance with an exact task
goal. Camera and hand-gesture experiments must separately report goal-estimator
angular/range error and then the end-to-end navigation metrics; do not hide
perception failures inside locomotion success.

## Initial PPO run

The existing task-agnostic MLX PPO path can train this task directly:

```sh
make ml-tool
.xcbuild/Build/Products/Release/avbd train-rl arachne15-velocity-v0 \
  --run arachne-nominal-seed-1 --seed 1 --envs 2048 --updates 500 \
  --horizon 32 --epochs 5 --batch 16384 --lr 0.0003 \
  --gamma 0.99 --gae-lambda 0.95 --entropy 0.01 --target-kl 0.01 \
  --action-std 0.6 --hidden-layers 256,256,128 \
  --task-option domainRandomization=0 \
  --task-option maximumActionLatencySteps=0 \
  --task-option observationNoise=0
```

That command is a nominal discovery baseline, not a claimed tuned recipe.
After it passes the nominal gate, transfer the checkpoint into the default
measured-randomization task and compare against training from scratch. Use
Policy Replay with `AVBD_REPLAY_TASK=arachne15-velocity-v0` and a live run
directory to inspect each complete checkpoint without changing the task.
