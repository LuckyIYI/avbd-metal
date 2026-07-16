#!/usr/bin/env python3
"""Headless reference run for Unitree RL Gym's H1 MuJoCo deployment."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch


def gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quaternion
    return np.asarray(
        [
            2 * (-qz * qx + qw * qy),
            -2 * (qz * qy + qw * qx),
            1 - 2 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--stats-every", type=int, default=25)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dt = 0.002
    decimation = 10
    kps = np.asarray([150, 150, 150, 200, 40] * 2, dtype=np.float32)
    kds = np.asarray([2, 2, 2, 4, 2] * 2, dtype=np.float32)
    default = np.asarray(
        [0, 0, -0.1, 0.3, -0.2, 0, 0, -0.1, 0.3, -0.2],
        dtype=np.float32,
    )
    command = np.asarray([0.5, 0, 0], dtype=np.float32)
    command_scale = np.asarray([2, 2, 0.25], dtype=np.float32)

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    policy = torch.jit.load(str(args.checkpoint.resolve()), map_location="cpu")
    policy.reset_memory()
    action = np.zeros(10, dtype=np.float32)
    target = default.copy()
    observation = np.zeros(41, dtype=np.float32)
    control_steps = int(round(args.seconds / (dt * decimation)))
    minimum_height = float(data.qpos[2])
    minimum_upright = 1.0
    first_fall_step = None

    with torch.inference_mode():
        for physics_step in range(1, control_steps * decimation + 1):
            data.ctrl[:] = (
                (target - data.qpos[7:]) * kps - data.qvel[6:] * kds
            )
            mujoco.mj_step(model, data)
            if physics_step % decimation:
                continue
            step = physics_step // decimation
            elapsed = physics_step * dt
            observation[0:3] = data.qvel[3:6] * 0.25
            observation[3:6] = gravity_orientation(data.qpos[3:7])
            observation[6:9] = command * command_scale
            observation[9:19] = data.qpos[7:] - default
            observation[19:29] = data.qvel[6:] * 0.05
            observation[29:39] = action
            phase = elapsed % 0.8 / 0.8
            observation[39:41] = [
                math.sin(2 * math.pi * phase),
                math.cos(2 * math.pi * phase),
            ]
            action = (
                policy(torch.from_numpy(observation).unsqueeze(0))
                .detach()
                .numpy()
                .squeeze(0)
            )
            target = default + 0.25 * action
            upright = -float(observation[5])
            minimum_height = min(minimum_height, float(data.qpos[2]))
            minimum_upright = min(minimum_upright, upright)
            if first_fall_step is None and (
                data.qpos[2] < 0.55 or upright < 0.5
            ):
                first_fall_step = step
            if args.trace:
                print(
                    "trace "
                    + json.dumps(
                        {
                            "step": step,
                            "root": data.qpos[:3].tolist(),
                            "q": data.qpos[7:].tolist(),
                            "dq": data.qvel[6:].tolist(),
                            "obs": observation.tolist(),
                            "action": action.tolist(),
                        },
                        separators=(",", ":"),
                    )
                )
            if step == 1 or step % max(args.stats_every, 1) == 0:
                print(
                    f"step {step:4d} t {elapsed:6.2f} "
                    f"root ({data.qpos[0]:+.3f},{data.qpos[1]:+.3f},"
                    f"{data.qpos[2]:.3f}) upright {upright:.3f} "
                    f"|action| {np.mean(np.abs(action)):.3f}"
                )

    report = {
        "simulator": f"mujoco-{mujoco.__version__}",
        "controlSteps": control_steps,
        "simulatedSeconds": control_steps * dt * decimation,
        "forwardDistanceMeters": float(data.qpos[0]),
        "lateralDistanceMeters": float(data.qpos[1]),
        "meanForwardSpeedMetersPerSecond": float(data.qpos[0]) / args.seconds,
        "finalPelvisHeightMeters": float(data.qpos[2]),
        "minimumPelvisHeightMeters": minimum_height,
        "minimumUprightAlignment": minimum_upright,
        "firstFallStep": first_fall_step,
        "firstFallTimeSeconds": (
            None if first_fall_step is None else first_fall_step * dt * decimation
        ),
        "fell": first_fall_step is not None,
        "finalJointPositions": data.qpos[7:].tolist(),
        "finalAction": action.tolist(),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(encoded, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
