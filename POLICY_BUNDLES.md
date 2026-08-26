# Policy bundle format

A policy bundle is a copyable directory rooted at `policy-bundle.json`. It is
the unit imported by the macOS Policy Replay page. The app discovers pages from
manifests and renders their cameras, controls, and metrics without a
policy-specific Swift view.

The directory must contain only regular files and directories: symlinks and
special files are rejected. Every runtime path is relative to the bundle root;
absolute paths, `.`/`..` components, duplicate file mappings, and paths that
escape the bundle fail closed.

## Manifest v1

```json
{
  "schemaVersion": 1,
  "identifier": "my-h1-policy-v1",
  "title": "My H1 Policy",
  "summary": "What this policy does and which contract it expects.",
  "runtime": {
    "kind": "avbd-vector-ppo-v1",
    "files": {
      "metadata": "metadata.json",
      "policy": "policy.safetensors",
      "trainingState": "training-state.json",
      "deploymentManifest": "deployment-manifest.json"
    }
  },
  "simulation": {
    "task": "humanoid-isaac-flat-v0",
    "taskRevision": 2000011,
    "seed": 21001,
    "maxEpisodeSteps": 1000,
    "simulationStepSeconds": 0.005,
    "controlDecimation": 4,
    "includeInteractiveRobustnessProbes": true,
    "options": { "solverIterations": 20 }
  },
  "presentation": {
    "cameraPresets": [{
      "id": "follow", "label": "Follow", "anchor": "robot",
      "target": [0, 0, 0], "offset": [0, 0, 0.9],
      "distance": 8, "azimuth": -1.5707964, "elevation": 0.12
    }],
    "controls": [],
    "metrics": [{
      "id": "height", "label": "Height",
      "source": "task/root-height", "format": "%.3f", "unit": "m"
    }]
  }
}
```

`simulation` is exact rather than advisory. The loader cross-checks it against
the policy metadata/runtime manifest, then the runtime reconstructs the task
through the central task registry and requires the resulting task spec to
match. Bundle float options use the same exact Float semantics as checkpoint
metadata.

## Runtime ABIs

- `avbd-vector-ppo-v1` uses the four canonical files shown above. It can replay
  any registered task that implements `RLReplayTask`.
- `unitree-h1-recurrent-v1` uses `manifest.json`, `policy.safetensors`, and
  `LICENSE` through roles `manifest`, `policy`, and `license`. This ABI
  currently requires `includeInteractiveRobustnessProbes: true`; manifests
  that claim a different scene are rejected instead of being silently
  reconstructed with a projectile.

A new policy for a supported runtime and replay task requires only a new
bundle. A new network execution format or simulator/task ABI requires a new
versioned runtime adapter, but does not require a new app page.

## Presentation data

Camera presets select a named task/runtime anchor such as `robot`, `course`,
`goal`, or `world`. For `world`, `target` is the absolute look-at point. For
other anchors, `target` is an anchor-relative vector. `offset` is then applied
in both cases before the orbit parameters. The app always provides shared
play, pause, step, reset, playback-rate, and camera interaction.

Controls are ordered and have unique IDs:

- `slider`: finite `defaultValue`, `minimum`, `maximum`, optional `step` and
  display `format`.
- `toggle`: `defaultValue` is `0` or `1`.
- `button`: names a runtime/task `command`; its argument map references a
  value-bearing slider or toggle control ID.

Metrics bind a label to a named scalar source. Shared sources use `replay/*`,
task capabilities use `task/*`, and raw step metrics use `metric/*`. Unknown
sources render as unavailable instead of becoming code paths. Display formats
are parsed as bounded decimal precision/sign/suffix data, never executed as
format strings.

## Import and release trust

Use **Import Bundle…** in Policy Replay, or set
`AVBD_POLICY_BUNDLE=/absolute/path/to/bundle` while developing. Imported,
repository-relative, and operator-selected bundles are always visibly
unverified.

Qualification is deliberately not stored in a bundle. Packaged releases are
authenticated by the separate `checkpoints/policy-release-index.json`, which
pins the exact bundle-manifest SHA-256 and the commissioned policy/runtime
identity. A copied or edited manifest therefore cannot certify itself.
Unverified imports still fail closed unless their runtime manifest, policy
digest, metadata, and serialized simulator contract agree internally.

The tracked manifests under `checkpoints/` are complete examples for both v1
runtime ABIs.
