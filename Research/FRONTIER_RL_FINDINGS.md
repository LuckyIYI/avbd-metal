# Frontier robot-learning experiments

Updated: 2026-07-18

## Objective

Beat a strong unified PPO baseline on honest contact-rich tasks, then test an
`imagine -> project through physics -> amortize` controller.  Complexity is
added only after the preceding experiment shows compute-matched headroom.

## Non-negotiable guardrails

- A reported success must come from physical state and contact manifolds, not
  phase labels, reference state, reward thresholds, or teleported objects.
- Training, validation, and test reset seeds remain disjoint.
- Every algorithm comparison uses the same task revision, simulator steps,
  evaluation seeds, and reports wall-clock cost.
- A planner must replay its selected trajectory in the same solver slot and
  reproduce its predicted terminal state before its score is trusted.
- Because contact transitions amplify micrometre-scale floating-point
  differences, the selected flow is also replayed in every resident replica;
  at least 80% must independently satisfy the full dynamic endpoint gate.
- Target-generator clone spread is reported separately. An unstable generating
  action sequence cannot invalidate a desired physical endpoint if a different
  reconstructed flow reaches it robustly, but the instability must stay visible.
- State cloning is tested through active contact, not only free flight.
- Humanoid box success eventually requires: unsupported lift, bilateral
  retention, at least two physical steps while carrying, destination support,
  release, and post-release survival.
- Failed hypotheses and simulator bugs stay in this ledger.

## Experiment ladder

1. Lock metrics and deterministic state-fork behavior.
2. Strong one-actor PPO baseline.
3. Gaussian cubic B-spline PPO under matched compute.
4. Frozen-policy spline MPPI to measure exact-physics planning headroom.
5. Planner distillation / DAgger.
6. Learned reachability, closest-reachable-state, and inverse-spline models.
7. Multi-timescale future-state generation; flow/diffusion only if simpler
   distributions fail on demonstrated multimodality.
8. Scale the winning system to humanoid box transport and other tasks.

## Findings log

### 2026-07-17 — E0 selection

- **Decision:** use batched Push-T as the first physical bridge benchmark.
- **Why:** it contains force-limited actuation, friction, rigid contact, and an
  object that must translate and rotate, while remaining cheap enough to run
  hundreds of reconstruction branches.
- **First falsifiable hypothesis:** for a source and target state both produced
  by AVBD, CEM/MPPI over a compact spline action trajectory reduces terminal
  reconstruction error substantially versus linear interpolation and random
  shooting, and the selected trajectory replays to the predicted state.
- **Go gate:** deterministic translated forks agree through contact, optimized
  reconstruction beats both baselines, and held-out replay agrees with the
  winning speculative branch.
- **No-go interpretation:** first distinguish incomplete solver state capture,
  insufficient action/horizon parameterization, and optimizer failure before
  adding a learned generator.

### 2026-07-18 — E0 exact-physics state-flow result

The benchmark now captures the pusher and welded T rigid poses, rotations,
linear/angular velocities, and rate-limited actuator command. A portable state
can be cold-forked into arbitrary collision-isolated replicas. Candidate flows
are endpoint-clamped cubic B-splines executed by AVBD, not kinematic paths.

The winning search is a state-only contact proposal followed by full-covariance
CEM over five internal 2D spline knots. It uses 256 candidates, 18 generations,
64 control steps, 16 target-settling steps, and 16 terminal-hold steps. Target
creation and reconstruction share the same 0.06 m/control-step actuator limit.

Matched three-seed results:

| seed | target translation | target yaw | optimized position | optimized yaw | linear loss | random loss | optimized loss | robust replay |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 101 | 1.558 m | 1.907 rad | 0.76 mm | 0.00344 rad | 22.054 | 2.791 | 0.000285 | 32/32 |
| 202 | 1.100 m | 1.454 rad | 3.76 mm | 0.01192 rad | 21.401 | 0.426 | 0.005591 | 32/32 |
| 303 | 0.940 m | 1.474 rad | 0.91 mm | 0.00033 rad | 11.881 | 1.047 | 0.002497 | 32/32 |

Across all 96 robust replays, the worst endpoint error was 3.77 mm and
0.01193 rad. Same-slot selected replay error was exactly zero in all three
runs. Wall time was 22.7--23.6 s per seed on the local Apple GPU.

The most interesting result is seed 202. The oracle action sequence that
created the target becomes contact-branch unstable at step 24 and its clones
end as much as 13.6 cm apart. CEM nevertheless discovers a different spline
that reaches the desired state in all 32 replicas. Physical reconstruction can
therefore act as a robustness projection, not merely recover hidden actions.

Exact reproduction command (replace the seed):

```sh
.build/debug/avbd experiment-pusht-flow \
  --envs 256 --iterations 18 --horizon 64 --seed 202 \
  --algorithm endpoint-contact-full-cem \
  --task-option simulationBatchSize=32 \
  --task-option controlPointCount=7 \
  --task-option eliteFraction=0.05 \
  --task-option targetSettlingSteps=16 \
  --task-option terminalHoldSteps=16 \
  --task-option targetGenerationMaximumStep=0.06
```

### Rejected results and dead ends

- A first apparent 53x CEM gain was rejected: the 16x16 batch exceeded the old
  fixed ground and selected an invalid falling branch. The ground is now
  centered and sized from batch layout.
- Slowing target generation while reconstructing with the old actuator limit
  made the reference replay fail. The infrastructure gate caught it; target
  and candidate action semantics are now identical.
- Recomputing `tip + retraction` each settling frame marched the pusher toward
  the wall and continued injecting energy. Retraction is now one fixed target.
- A diagonal CEM covariance recovered translation but often missed rotation.
  Full covariance materially helps coordinated contact-side and timing changes.
- Linear endpoint splines and uninformed random shooting are weak on contact
  discovery. The state-only `behind -> through -> retract` proposal supplies
  geometry, while exact simulation and CEM still choose the physical flow.
- Strict cross-replica equality is not a valid long-contact criterion. Step
  traces show exact forks remain within roughly 10--30 micrometres until a
  contact transition, then branch chaotically. Initial-color and disjoint hash
  hypotheses produced no change and were fully reverted. Same-slot determinism
  plus cross-replica task success is the accepted guardrail.

## Next experiment: amortized physical flow

Use the passing CEM planner as a teacher over many source/target pairs. Train
one structured MLX network to output the internal spline-knot distribution
(mean plus a low-rank/full Cholesky head) from relative body/actuator pose,
velocity, and geometry features. Then compare on held-out seeds:

1. network proposal with zero physics refinement;
2. network proposal plus one or two CEM generations;
3. current state-only proposal plus 18 CEM generations;
4. linear and random-shooting baselines.

Primary measures are robust endpoint success, CVaR/worst replay loss, simulator
steps, and wall time. Only after the learned proposal preserves success while
removing most of the 18-generation search cost does the experiment scale to the
humanoid box-transfer task.

## Evidence

- Base checkpoint: `ab83b95` on `codex/batched-rl-platform`; the robust-flow
  guardrails and matched sweep are the current worktree after that checkpoint.
- Generated runs and media remain ignored; source, tests, experiment commands,
  and this ledger are the reproducible record.
