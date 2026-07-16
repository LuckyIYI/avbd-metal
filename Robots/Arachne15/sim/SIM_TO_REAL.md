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
4. **Measured randomization:** expand each range only to cover repeated
   hardware measurements plus sensor uncertainty. Keep a nominal cohort so
   excessive randomization cannot hide model regressions.
5. **Disturbance validation:** randomized payload placement, foot friction,
   voltage, latency, slope, and external pushes. Hold out combinations and
   random seeds from training.
6. **Hardware crawl:** suspended gait, tethered stand, low-body straight crawl,
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
