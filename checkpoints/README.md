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

Newly qualified checkpoints keep their sealed held-out report beside the
weights as `evaluation.json`; Arachne Goal retains its larger multi-seed report
set under the canonical robot qualification directory linked below.

## Shipped learned policies

- `external/unitree-h1` is the unchanged public Unitree RL Gym recurrent H1
  policy imported to MLX. Its source identity and recurrent parity contract are
  recorded in `manifest.json`.
- `humanoid-isaac-flat-v0` is immutable historical evidence for the H1 Flat
  actor on the former revision-1000010 collision geometry. Its sealed
  512-episode test at seed 41010 passed 507/512 episodes (99.02%), with 0.089
  m/s linear and 0.134 rad/s yaw-rate RMSE. The current BSD-source hull task is
  revision 1000011, so runtime compatibility intentionally rejects this bundle
  until the unchanged weights are requalified and republished. Fingerprint:
  `d6b5d416e7f7d75fa2b9b9dd33f78ae387e3f2a8139aa6d25a69e5dbcae777ab`.
- `humanoid-isaac-goal-v0` is the current H1 point-goal/impact actor on the
  corrected actuator contract (`taskRevision=1000004`). Every episode includes
  a real 8 kg projectile impact. Its sealed seed-42010 test reached 400/512
  goals (78.12%) and survived 80.66%, but it remains a visible development
  policy because aggregate final/minimum goal distance still fails the strict
  acceptance gate. Fingerprint:
  `15710d3f81b9ff4b5d14ab1a53d89381efd8effed02b237422ee72e625c113f6`.
- `arachne15-velocity-v0` is the corrected revision-6 straight-walk benchmark:
  fixed 0.15 m/s command, deterministic plant, no steering. Its sealed
  512-episode test at seed 45010 passed 512/512, with 0.068 m/s linear and
  0.284 rad/s yaw-rate RMSE and zero control steps deeper than 1 mm below the
  floor. Fingerprint:
  `aed643b062df4e0e07e70998212720909bc1b25229455489ba28d4319d202524`.
- `arachne15-goal-v0` is the separately qualified sim-to-real point-goal actor.
  Its four-seed training-collider and four-seed validation-collider reports are
  retained under
  `Robots/Arachne15/qualification/arachne15-goal-r6-update-000020`.
  Fingerprint:
  `30c125b7f01b73bdd1524bc96cf8deb5e8a09897593a49e87aa6ce96f16d3027`.

`Arachne Classical` is deliberately also visible in Policy Replay, but is
labeled as a non-neural CPG/IK baseline and has no checkpoint.

## Removed replay examples

The earlier native `humanoid-walk-v0` and `humanoid-goal-v0` examples were
superseded by the imported H1 tasks and are no longer shipped in Policy Replay.
The historical two-link `arm-pusht-v0` checkpoint used an obsolete observation
contract and has also been removed. The full Panda `maniskill-pusht-v1` task
remains available for research and training, but its best current held-out run
does not satisfy the 80% success gate, so it is not presented as a solved replay
example.
