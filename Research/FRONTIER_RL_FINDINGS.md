# Frontier robot-learning experiments

Updated: 2026-07-17

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
- A planner must replay its selected trajectory from the same snapshot and
  reproduce its predicted terminal state before its score is trusted.
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

## Evidence

- Current checkpoint: `08eb912` on `codex/batched-rl-platform`.
- Generated runs and media remain ignored; source, tests, experiment commands,
  and this ledger are the reproducible record.
