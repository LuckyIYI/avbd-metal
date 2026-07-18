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

### 2026-07-18 — E1 amortization and exact-physics arbitration

The first amortization dataset attempted 40 disjoint seeds with a cheaper
64-candidate, 12-generation teacher. Twenty-three rows passed every physical
and robust-replay gate; 17 were retained as explicit rejections rather than
silently entering training. Every accepted teacher flow replayed successfully
in all 32 resident replicas.

A single three-hidden-layer MLX MLP consumes a 22-value invariant encoding of
the two endpoint states and predicts a residual correction to the geometric
spline. The 18/5 train/validation split improved normalized validation MSE from
1.8338 to 1.3622, but neither the absolute-output model nor the residual model
produced a useful zero-shot controller. One-generation refinement passed 0/6
held-out seeds. This is a rejected scaling result: 23 multimodal contact flows
are not enough for a point-estimate network, even when the representation is
structured and invariant.

Nearest-neighbor residual transfer was a stronger low-data proposal. A fixed
50/50 mixture of retrieved and geometric candidates was therefore compared to
geometric-only CEM on 18 held-out seeds. It reduced mean/median terminal loss,
but tied task reliability and sometimes displaced a successful geometric
sample. The replacement performs one exact-physics probe batch (16 retrieved,
16 geometric), selects the better simulated neighborhood, and spends the
remaining 224 candidates around its best physical branch. Total candidate
count, resident replicas, horizon, and simulated replica-steps remain exactly
matched.

Matched 256-candidate x 2-generation result:

| proposal allocation | mean loss | median loss | robust endpoints | strict go | mean robust fraction | mean simulated replica-steps |
|---|---:|---:|---:|---:|---:|---:|
| geometric only | 0.4774 | 0.3750 | 6/18 | 4/18 | 33.68% | 39,283.6 |
| fixed 50/50 retrieval | 0.3643 | 0.2794 | 6/18 | 4/18 | 33.33% | 39,283.6 |
| exact-physics arbitration | **0.2674** | **0.2385** | 6/18 | 4/18 | **36.46%** | 39,283.6 |

Arbitration selected retrieval on 13 seeds and geometry on five. Relative to
the fixed mixture it reduced mean loss by 26.6% and median loss by 14.7%; versus
geometric only the reductions were 44.0% and 36.4%. Wall time was also within
noise (3.05 s/seed versus 3.09 s and 3.15 s). This validates the combination as
an optimizer: learned/retrieved imagination proposes modes, exact physics
reconstructs them, and evidence reallocates finite search compute. It does
**not** yet improve the 6/18 robust-success count, so it is not claimed as a
task-solving advance.

Paired failures expose the next issue. The legacy scalar loss weights position
far more heavily than endpoint velocity and yaw. Several low-loss candidates
miss the robust gate by only velocity (for example 0.109 m/s against a 0.100
m/s threshold). The next matched experiment must optimize a threshold-scaled
bottleneck objective while retaining the legacy loss as a reported diagnostic.

### 2026-07-18 — E2 balanced endpoint objective (validated)

Candidate ranking now has an optional `balancedEndpointBottleneck` objective.
It uses one shared set of frozen endpoint thresholds for both optimization and
robust replay classification. The score is the squared worst normalized error
plus 0.25 times the normalized mean-square error. The historical weighted loss
is still emitted for every candidate, so objective changes cannot erase the
longitudinal comparison.

On the initial 18 seeds, balanced geometry alone improved robust endpoints
from 6 to 7. Combining the balanced objective with exact-physics-arbitrated
retrieval reached 8/18 robust endpoints and 6/18 strict go passes. Its success
set strictly contained the balanced-geometric success set.

A fresh 32-seed confirmation was then frozen and run without tuning:

| method | robust endpoints | strict go | mean robust fraction | median worst normalized error | mean worst normalized error | p90 worst normalized error |
|---|---:|---:|---:|---:|---:|---:|
| balanced geometry | 13/32 | 7/32 | 40.33% | 1.240 | 1.604 | 2.335 |
| balanced + physics-arbitrated retrieval | **18/32** | **11/32** | **56.25%** | **0.928** | **1.374** | **2.319** |

All 13 geometric successes were retained; retrieval added five and lost zero.
Across the combined 50 seeds, the result is 26/50 versus 20/50 robust endpoints
and 17/50 versus 13/50 strict go passes. There are six one-way robust-success
discordances and zero in the opposite direction (exact two-sided paired sign /
McNemar probability 0.03125). Mean simulated replica-steps are identical
(39,216.64), and mean wall time is 3.08 s versus 3.14 s per seed.

The error distribution also improves: over 50 seeds, median/mean/p90 worst
normalized endpoint error move from 1.349/1.626/3.496 to
0.957/1.300/2.319. Mean historical weighted loss falls from 0.870 to 0.634.
One important negative remains: absolute worst error rises from 6.28 to 7.88
because a probe winner can receive all remaining samples and then refine badly.
A risk-aware version should reserve part of the exploitation batch for the
losing mode or retain multiple final hypotheses. This tail regression remains
visible and prevents a blanket dominance claim.

The validated combination is therefore:

1. invariant state-to-state proposal transfer as cheap "imagination";
2. a balanced, task-threshold-normalized physical reconstruction objective;
3. one exact-physics mode probe;
4. adaptive allocation of the remaining fixed compute;
5. same-slot determinism and all-replica endpoint replay as acceptance guards.

This is more useful than the point-estimate MLP found in E1: the memory proposes
contact modes, but the simulator remains the verifier and final decision-maker.

### 2026-07-18 — E3 first humanoid-box physical-state bridge

The balanced endpoint machinery is now task-general and the first H1 bridge is
implemented. A compatible revision-1000040 box policy establishes a real
bilateral grasp. A trajectory from the older v108 lift audit is then used only
to generate a hidden target; search is explicitly rejected if its proposal is
identical to that generating trajectory. The initial proposal instead comes
from the unrelated v98 left-transport audit.

The target selector rejects tilted/falling transients and found a useful state
16 control steps after the source grasp:

- box clearance 4.11 cm, with no source-table support;
- robot upright alignment 0.971 and box upright alignment 0.998;
- bilateral physical hand contact and the measured lifted milestone;
- 32/32 target replicas within every frozen endpoint constraint;
- exact fresh-task replay of the withheld generating action.

The endpoint vector covers root pose/twist, 19 joint angles and velocities,
both feet, box pose/twist, both terminal hand collision spheres, contact,
support, lift, upright, and failure state. It is not an observation-vector soup:
each component has an explicit physical unit and frozen success threshold.

Search progression:

| stage | candidates x generations | worst normalized endpoint | physical tier | robust replay |
|---|---:|---:|---|---:|
| zero arm delta | exact | 14.482 | supported / not lifted | 0% |
| unrelated v98 proposal | exact | 2.220 | supported / not lifted | 0% |
| broad reconstruction | 128 x 8 | 1.174 | lifted, bilateral, unsupported, upright | 0% |
| narrow continuation | 128 x 12 | 1.000172 | lifted, bilateral, unsupported, upright | 0% |
| final continuation | 128 x 6 | **0.917** | lifted, bilateral, unsupported, upright | **32/32** |

No tolerance was moved to turn the 1.000172 near miss into a pass. The last
search crossed the original thresholds physically. Its final errors include
2.27 cm box position, 0.0697 rad box rotation, 0.0916 m/s box velocity,
0.154 rad/s box angular velocity, 2.70 cm maximum hand position, and
0.229 m/s maximum hand velocity. Same-slot replay error is exactly zero. A
disjoint seed-1009 run using the resulting flow as a proposal also passes
32/32 without refinement (`0.917`), while the hidden target remains withheld.

This proves a short grasp-state -> lifted-state physical flow for the full
humanoid/object system. It does **not** solve box transport: the bridge is only
16 control steps and contains no walking, destination support, release, or
post-release survival. The next bridge must start from this robust lifted state
and target a later physically carried state before placement is attempted.

### 2026-07-18 — E4 simulator-discovered multistage carry frontier

Replaying a stored transport spline after the verified lift initially appeared
to move the box 0.35--0.38 m. That number was rejected: the box had already
left the stable grasp and the distance was post-failure flight. The experiment
now reports both unconstrained maximum carry and maximum carry while every
physical condition is simultaneously true. Across the stored trajectories,
the best stable carry was only 4.22 cm despite up to 38.4 cm unconstrained
motion. This distinction prevents a visually dramatic failure from entering
the success set.

The controller now supports explicit ordered source stages. A derived artifact
stores its complete physical lineage, and every evaluation replays each stage
through the simulator from the real bilateral-grasp source. A continuation is
therefore never treated as a standalone action. The first lineage was:

1. 16-frame reconstructed unsupported lift;
2. 12-frame reconstructed stable carry to 6.81 cm;
3. 8-frame reconstructed stable carry to 13.65 cm.

The 6 cm and 10 cm bridges use a two-part experiment matching the proposed
imagination/reconstruction split:

1. full-covariance CEM samples arm-spline futures from the verified source;
2. exact physics scores a goal set: carry progress, clearance, velocities,
   uprightness, bilateral contact, unsupported lift, and failure state;
3. the best valid simulator state becomes an exact hidden endpoint;
4. a separate CEM reconstructs that full state without receiving its generating
   trajectory;
5. same-slot determinism and 32-resident-replica physical success gate the
   resulting stage before it can extend the lineage.

Results:

| bridge | discovered physical target | reconstruction search | worst normalized endpoint | robust replay | same-slot replay |
|---|---|---:|---:|---:|---:|
| lift -> 6 cm frontier | 6.81 cm carry, 2.23 cm clearance, upright 0.974/0.937 | 128 x 8 + 128 x 8 refinement | **0.994** | **32/32** | 0 |
| 6 cm -> 10 cm frontier | 13.65 cm carry, 3.22 cm clearance, upright 0.984/0.906 | 128 x 6 | **0.351** | **32/32** | 0.0058 |

For the first frontier, an 8-candidate smoke reconstruction failed at worst
normalized error 4.0. A broad 128 x 8 search reached 1.070; the fixed gates
showed only box, hand, and foot velocity tolerances narrowly over threshold.
A narrower continuation crossed the original gate at 0.994. For the second
frontier, the zero-delta continuation was already near the hidden state at
1.287 (root angular velocity); 128 x 6 refinement reached 0.351. No endpoint
tolerance was changed during either progression.

This is the first genuinely transported state in the experiment: the box is
not merely lifted, it is carried 13.65 cm through collision/friction while the
humanoid remains upright and the grasp stays bilateral. It is still not the
complete table-to-table task. Destination approach, lowering, support transfer,
release, and post-release survival remain unverified.

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
- Saving the MLX validation-best parameters as an array list was a false early
  stopping result: MLX arrays remained aliased and continued changing. Best
  weights are now serialized to an immutable in-memory safetensor snapshot and
  restored before final reporting.
- Predicting absolute spline points made held-out proposals worse than the
  geometric prior. The network and retrieval index now predict only invariant
  residual corrections, and checkpoint schema v3 rejects old absolute models.

## Next experiment: carry to destination and transfer support

Extend the now-verified multistage lineage toward the receiving table. Target
discovery should include destination progress and a clearance corridor, then
branch into separate lowering/support-transfer and release/survival goal sets.
Do not collapse these into one sparse terminal reward: accept each stage only
after exact reconstruction and robust replay, then preserve the complete
simulator-executed ancestry in the next artifact.

The imagination layer should next retain multiple frontier hypotheses rather
than only the lowest scalar-loss mode. After enough accepted stages and failed
alternatives exist, amortize their distribution with a mixture or spline-policy
head in MLX; keep exact-physics arbitration as the final selector.

For amortization, compare on disjoint held-out seeds:

1. network proposal with zero physics refinement;
2. network proposal plus one or two CEM generations;
3. current state-only proposal plus 18 CEM generations;
4. linear and random-shooting baselines.

Primary measures are robust endpoint success, CVaR/worst replay loss, simulator
steps, and wall time. Only after the learned proposal preserves success while
removing most of the 18-generation search cost does the experiment scale to the
humanoid box-transfer task.

## Evidence

- Base checkpoints: `ab83b95` (robust physical-flow search) and `9691cef`
  (MLX amortization scaffold) on `codex/batched-rl-platform`; exact-physics
  arbitration and its matched sweep are the current worktree after the latter.
- Generated runs and media remain ignored; source, tests, experiment commands,
  and this ledger are the reproducible record.
