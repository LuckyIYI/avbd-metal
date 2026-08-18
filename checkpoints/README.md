# Policy Replay checkpoint catalog

Policy Replay ships only maintained examples with an executable pretrained
policy. Every native MLX checkpoint in this directory must reconstruct its
serialized task exactly; `VectorPolicyCompatibilityTests` rejects missing,
stale, extra, or deprecated task directories.

Serialized `initializationCheckpoint` and `checkpointDirectory` values are
immutable training provenance. They may name ignored `runs/` output or a
retired source snapshot; Policy Replay does not follow them. Runtime loading
uses the policy, metadata, and training state stored in each tracked checkpoint
directory (plus the deployment manifest where present).

Single-report qualifications keep their sealed held-out report beside the
weights as `evaluation.json`. Multi-seed native qualifications keep an
aggregate and raw fixed-seed reports under `qualification/`; Arachne Goal
retains its prior-contract report set under the canonical robot directory
linked below.

## Selectable replay evidence

- `humanoid-isaac-flat-v2` is the accepted native H1 Flat deployment on the
  epoch-2 deterministic-color solver (`taskRevision=2000011`). The task ID
  remains `humanoid-isaac-flat-v0`; v2 is its Policy Replay selection and
  sealed bundle ID. It is an exact zero-update requalification of v1: policy
  bytes are unchanged and the target records zero training and optimizer
  updates. Four manifest-locked, fixed-seed 512-episode evaluations passed
  2028/2048 episodes (99.02% pooled; 98.63% worst run), with worst-run linear
  and yaw-rate RMSE of 0.089 m/s and 0.132 rad/s. The deployment and
  requalification manifests bind the parent, candidate, producing commit,
  frozen criteria, raw reports, and reconstructed aggregate. Fingerprint:
  `00bc782d1845ddde94282b46f0d7fa2732feeb4a8e52215a5abe62128bccc756`.
- `external/unitree-h1` is the unchanged public Unitree RL Gym recurrent H1
  policy imported to MLX. Its source identity and recurrent parity contract are
  recorded in `manifest.json`.

The imported GEAR-SONIC entry remains a visible development reference, and
`Arachne Classical` remains a visible non-neural CPG/IK baseline with no
checkpoint.

## Historical learned-policy evidence

The remaining native checkpoints predate the epoch-2 contract and are retained
as immutable, nonselectable provenance. External imported policies retain their
own parity gates.

- `humanoid-isaac-flat-v1` is the former accepted H1 Flat deployment for the
  BSD-source collision hulls (`taskRevision=1000011`). It preserves the v0
  policy bytes exactly and records zero target training updates. It is the
  immutable epoch-1 parent of the accepted v2 zero-update requalification and
  remains historical/nonselectable. Fingerprint:
  `85571805cc7b688970cf5497beb5916be8fb3b1fcb7855207af6f55b208c7fd2`.
- `humanoid-isaac-flat-v0` is immutable historical evidence for the H1 Flat
  actor on the former revision-1000010 collision geometry. Its sealed
  512-episode test at seed 41010 passed 507/512 episodes (99.02%), with 0.089
  m/s linear and 0.134 rad/s yaw-rate RMSE. Runtime compatibility rejects this
  historical bundle; it is not selectable and is not evidence for revision
  1000011. Fingerprint:
  `d6b5d416e7f7d75fa2b9b9dd33f78ae387e3f2a8139aa6d25a69e5dbcae777ab`.
- `humanoid-isaac-goal-v0` is the epoch-1 H1 point-goal/impact actor on the
  corrected actuator contract (`taskRevision=1000004`). Every episode includes
  a real 8 kg projectile impact. Its sealed seed-42010 test reached 400/512
  goals (78.12%) and survived 80.66%, but aggregate final/minimum goal distance
  still fails the strict acceptance gate. It remains historical/nonselectable
  and requires epoch-2 requalification. Fingerprint:
  `15710d3f81b9ff4b5d14ab1a53d89381efd8effed02b237422ee72e625c113f6`.
- `arachne15-velocity-v0` is the corrected revision-6 straight-walk benchmark:
  fixed 0.15 m/s command, deterministic plant, no steering. Its single
  512-episode report at seed 45010 passed 512/512, with 0.068 m/s linear and
  0.284 rad/s yaw-rate RMSE and zero control steps deeper than 1 mm below the
  floor. It remains historical/nonselectable unless a robust multi-seed
  aggregate, matching deployment manifest, and epoch-2 requalification are
  published.
  Fingerprint:
  `aed643b062df4e0e07e70998212720909bc1b25229455489ba28d4319d202524`.
- `arachne15-goal-v0` is the formerly qualified epoch-1 sim-to-real point-goal actor.
  Its four-seed training-collider and four-seed validation-collider reports are
  retained under
  `Robots/Arachne15/qualification/arachne15-goal-r6-update-000020`.
  Those reports do not qualify the epoch-2 solver. Fingerprint:
  `30c125b7f01b73bdd1524bc96cf8deb5e8a09897593a49e87aa6ce96f16d3027`.

Both native Arachne learned policies remain historical and cannot be selected
through Policy Replay.

## Removed replay examples

The earlier native `humanoid-walk-v0` and `humanoid-goal-v0` examples were
superseded by the imported H1 tasks and are no longer shipped in Policy Replay.
The historical two-link `arm-pusht-v0` checkpoint used an obsolete observation
contract and has also been removed. The full Panda `maniskill-pusht-v1` task
remains available for research and training, but its best current held-out run
does not satisfy the 80% success gate, so it is not presented as a solved replay
example.
