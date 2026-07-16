#!/usr/bin/env python3
"""Import Unitree's H1 TorchScript policy into AVBD's portable MLX format.

The runtime deliberately does not depend on Python or libtorch.  This importer
validates the exact recurrent architecture shipped by unitree_rl_gym, writes
its tensors as safetensors, and records deterministic TorchScript outputs that
the Swift/MLX runner can check independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
from pathlib import Path
from typing import Dict

import torch


SOURCE_URL = (
    "https://github.com/unitreerobotics/unitree_rl_gym/"
    "blob/main/deploy/pre_train/h1/motion.pt"
)
EXPECTED_SHAPES = {
    "memory.weight_ih_l0": [256, 41],
    "memory.weight_hh_l0": [256, 64],
    "memory.bias_ih_l0": [256],
    "memory.bias_hh_l0": [256],
    "actor.0.weight": [32, 64],
    "actor.0.bias": [32],
    "actor.2.weight": [10, 32],
    "actor.2.bias": [10],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision(path: Path) -> str | None:
    """Return the containing checkout's commit without requiring it."""
    for parent in [path.parent, *path.parents]:
        if (parent / ".git").exists():
            try:
                return subprocess.check_output(
                    ["git", "-C", str(parent), "rev-parse", "HEAD"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                return None
    return None


def write_safetensors(path: Path, tensors: Dict[str, torch.Tensor]) -> None:
    """Write contiguous float32 tensors using the documented safe format."""
    header: dict[str, object] = {
        "__metadata__": {
            "format": "avbd-unitree-h1-lstm-v1",
            "framework": "pytorch-to-mlx",
        }
    }
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().to(torch.float32).contiguous()
        raw = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)

    encoded = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for chunk in chunks:
            stream.write(chunk)


def golden_sequence(policy: torch.jit.ScriptModule) -> dict[str, object]:
    inputs = []
    for step in range(4):
        inputs.append(
            [
                0.0
                if step == 0
                else 0.25 * math.sin((step + 1) * (index + 1) * 0.137)
                + 0.05 * math.cos((step + 2) * (index + 1) * 0.071)
                for index in range(41)
            ]
        )

    policy.reset_memory()
    outputs = []
    hidden = []
    cell = []
    with torch.inference_mode():
        for values in inputs:
            output = policy(torch.tensor([values], dtype=torch.float32))
            outputs.append(output.squeeze(0).tolist())
            state = policy.state_dict()
            hidden.append(state["hidden_state"].reshape(-1).tolist())
            cell.append(state["cell_state"].reshape(-1).tolist())
    policy.reset_memory()
    return {
        "inputs": inputs,
        "actions": outputs,
        "hiddenStates": hidden,
        "cellStates": cell,
        "absoluteTolerance": 2e-5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")

    policy = torch.jit.load(str(checkpoint), map_location="cpu")
    methods = set(policy._c._method_names())
    if not {"forward", "reset_memory"}.issubset(methods):
        raise SystemExit(
            "unsupported TorchScript policy: expected forward and reset_memory"
        )
    state = policy.state_dict()
    actual_shapes = {
        name: list(state[name].shape)
        for name in EXPECTED_SHAPES
        if name in state
    }
    if actual_shapes != EXPECTED_SHAPES:
        raise SystemExit(
            "unsupported Unitree H1 architecture:\n"
            + json.dumps(actual_shapes, indent=2, sort_keys=True)
        )

    tensors = {name: state[name] for name in EXPECTED_SHAPES}
    output.mkdir(parents=True, exist_ok=True)
    write_safetensors(output / "policy.safetensors", tensors)

    manifest = {
        "schemaVersion": 1,
        "format": "avbd-unitree-h1-lstm-v1",
        "robot": "unitree-h1",
        "source": {
            "project": "unitreerobotics/unitree_rl_gym",
            "url": SOURCE_URL,
            "revision": source_revision(checkpoint),
            "checkpointSHA256": sha256(checkpoint),
            "license": "BSD-3-Clause",
        },
        "network": {
            "kind": "lstm-actor",
            "observationDimension": 41,
            "hiddenDimension": 64,
            "actionDimension": 10,
            "actorHiddenDimension": 32,
            "gateOrder": "ifgo",
            "activation": "elu",
        },
        "control": {
            "physicsTimeStep": 0.002,
            "controlDecimation": 10,
            "periodSeconds": 0.8,
            "actionScale": 0.25,
            "angularVelocityScale": 0.25,
            "jointPositionScale": 1.0,
            "jointVelocityScale": 0.05,
            "commandScale": [2.0, 2.0, 0.25],
            "defaultCommand": [0.5, 0.0, 0.0],
            "defaultJointPositions": [
                0.0, 0.0, -0.1, 0.3, -0.2,
                0.0, 0.0, -0.1, 0.3, -0.2,
            ],
            "jointNames": [
                "left_hip_yaw", "left_hip_roll", "left_hip_pitch",
                "left_knee", "left_ankle", "right_hip_yaw",
                "right_hip_roll", "right_hip_pitch", "right_knee",
                "right_ankle",
            ],
            "stiffness": [150, 150, 150, 200, 40, 150, 150, 150, 200, 40],
            "damping": [2, 2, 2, 4, 2, 2, 2, 2, 4, 2],
        },
        "observationLayout": [
            "body_angular_velocity[3] * 0.25",
            "projected_gravity[3]",
            "command[3] * [2,2,0.25]",
            "joint_position_minus_default[10]",
            "joint_velocity[10] * 0.05",
            "previous_action[10]",
            "sin_phase,cos_phase(period=0.8s)",
        ],
        "goldenSequence": golden_sequence(policy),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"imported {checkpoint}")
    print(f"  sha256: {manifest['source']['checkpointSHA256']}")
    print(f"  output: {output}")


if __name__ == "__main__":
    main()
