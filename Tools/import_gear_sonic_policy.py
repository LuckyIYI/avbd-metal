#!/usr/bin/env python3
"""Import NVIDIA GEAR-SONIC's G1 controller into AVBD's MLX bundle.

The ONNX files are used only at import/verification time.  The resulting
bundle contains optimizer-free float32 safetensors, a complete control
contract, and golden ONNX Runtime outputs.  Runtime inference is native MLX.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper
except ImportError as error:  # pragma: no cover - exercised by users
    raise SystemExit(
        "GEAR-SONIC import requires `pip install onnx onnxruntime`"
    ) from error


SOURCE_PROJECT = "nvidia/GEAR-SONIC"
SOURCE_URL = "https://huggingface.co/nvidia/GEAR-SONIC"
CODE_PROJECT = "NVLabs/GR00T-WholeBodyControl"
CODE_URL = "https://github.com/NVLabs/GR00T-WholeBodyControl"
CODE_REVISION = "60de0df7ffedeef415fe58d435e92cc5b01ba3d9"
FORMAT = "avbd-gear-sonic-g1-v1"

ENCODER_INPUT_SHAPE = [1, 1762]
ENCODER_OUTPUT_SHAPE = [1, 64]
DECODER_INPUT_SHAPE = [1, 994]
DECODER_OUTPUT_SHAPE = [1, 29]
COMPACT_REFERENCE_DIMENSION = 640
HISTORY_DIMENSION = 930

REFERENCE_FILES = {
    "joint_pos.csv": 29,
    "joint_vel.csv": 29,
    "body_pos.csv": 42,
    "body_quat.csv": 56,
    "body_lin_vel.csv": 42,
    "body_ang_vel.csv": 42,
}

ENCODER_NAMES = [
    "module.encoders.g1.module.0",
    "module.encoders.g1.module.2",
    "module.encoders.g1.module.4",
    "module.encoders.g1.module.6",
    "module.encoders.g1.module.8",
]
ENCODER_DIMENSIONS = [640, 2048, 1024, 512, 512, 64]
DECODER_BIASES = [
    f"module.decoders.g1_dyn.module.{index}.bias"
    for index in (0, 2, 4, 6, 8, 10, 12)
]
DECODER_WEIGHTS = [f"onnx::MatMul_{index}" for index in range(136, 143)]
DECODER_DIMENSIONS = [994, 2048, 2048, 1024, 1024, 512, 512, 29]

# Hardware/MuJoCo order.  Policy observations/actions use Isaac Lab order.
ACTUATOR_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# hardware index -> policy index, copied from the official deployment header.
ACTUATOR_TO_POLICY = [
    0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
    11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28,
]
# policy index -> hardware index.
POLICY_TO_ACTUATOR = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]
POLICY_JOINT_NAMES = [ACTUATOR_JOINT_NAMES[i] for i in POLICY_TO_ACTUATOR]

DEFAULT_ANGLES = [
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path | None) -> str | None:
    if path is None:
        return None
    for parent in [path, *path.parents]:
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


def hugging_face_revision(model: Path) -> str | None:
    metadata = (
        model.parent / ".cache" / "huggingface" / "download"
        / f"{model.name}.metadata"
    )
    if not metadata.is_file():
        return None
    lines = metadata.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines else None


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    result: list[int | str] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            result.append(dimension.dim_value)
        else:
            result.append(dimension.dim_param)
    return result


def require_graph_contract(
    model: onnx.ModelProto,
    input_name: str,
    input_shape: list[int],
    output_name: str,
    output_shape: list[int],
) -> None:
    graph = model.graph
    actual_inputs = [(value.name, tensor_shape(value)) for value in graph.input]
    actual_outputs = [(value.name, tensor_shape(value)) for value in graph.output]
    if actual_inputs != [(input_name, input_shape)]:
        raise SystemExit(f"unsupported ONNX input contract: {actual_inputs}")
    if actual_outputs != [(output_name, output_shape)]:
        raise SystemExit(f"unsupported ONNX output contract: {actual_outputs}")


def initializer_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        tensor.name: numpy_helper.to_array(tensor).astype(np.float32, copy=False)
        for tensor in model.graph.initializer
    }


def require_shape(name: str, array: np.ndarray, shape: Iterable[int]) -> None:
    expected = tuple(shape)
    if array.shape != expected:
        raise SystemExit(
            f"unsupported tensor {name}: got {array.shape}, expected {expected}"
        )
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise SystemExit(f"tensor {name} must be finite float32")


def constant_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                result[node.output[0]] = numpy_helper.to_array(attribute.t)
    return result


def uniform_scalar(constants: dict[str, np.ndarray], name: str) -> float:
    if name not in constants:
        raise SystemExit(f"missing encoder quantizer constant {name}")
    values = constants[name].astype(np.float32).reshape(-1)
    if values.size == 0 or not np.all(values == values[0]):
        raise SystemExit(f"non-uniform encoder quantizer constant {name}")
    return float(values[0])


def silu(value: np.ndarray) -> np.ndarray:
    sigmoid = np.empty_like(value)
    positive = value >= 0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    sigmoid[~positive] = exponential / (1.0 + exponential)
    return value * sigmoid


def dense_stack(
    value: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
) -> np.ndarray:
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        # `einsum` avoids spurious Accelerate matmul overflow diagnostics seen
        # in Apple's system Python while preserving a transparent affine op.
        value = np.einsum("bi,oi->bo", value, weight, optimize=False) + bias
        if not np.isfinite(value).all():
            raise SystemExit(f"non-finite output in dense layer {index}")
        if index + 1 < len(weights):
            value = silu(value)
    return value.astype(np.float32, copy=False)


def pack_reference_for_encoder(reference: np.ndarray) -> np.ndarray:
    """Apply the official G1 graph's reshape/concat ordering."""
    if reference.ndim != 2 or reference.shape[1] != COMPACT_REFERENCE_DIMENSION:
        raise ValueError("compact reference must be [batch, 640]")
    joints = reference[:, :580].reshape(reference.shape[0], 10, 58)
    anchors = reference[:, 580:640].reshape(reference.shape[0], 10, 6)
    return np.concatenate([joints, anchors], axis=2).reshape(
        reference.shape[0], COMPACT_REFERENCE_DIMENSION
    )


def compact_to_onnx_reference(compact: np.ndarray) -> np.ndarray:
    """Expand the exact mode-0 MLP input into SONIC's 1762D superset.

    The compact input keeps the official semantic field order.  The exported
    graph performs the frame packing internally before running its MLP.
    """
    if compact.ndim != 2 or compact.shape[1] != COMPACT_REFERENCE_DIMENSION:
        raise ValueError("compact reference must be [batch, 640]")
    full = np.zeros((compact.shape[0], ENCODER_INPUT_SHAPE[1]), dtype=np.float32)
    full[:, 0] = 0.0  # g1 encoder mode; the next three mode slots stay zero.
    full[:, 4:584] = compact[:, :580]
    full[:, 601:661] = compact[:, 580:640]
    return full


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header: dict[str, object] = {
        "__metadata__": {"format": FORMAT, "framework": "onnx-to-mlx"}
    }
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        array = np.ascontiguousarray(tensors[name], dtype="<f4")
        raw = array.tobytes(order="C")
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
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


def copy_reference_clips(
    code_checkout: Path | None, output: Path
) -> list[dict[str, object]]:
    """Copy and content-lock NVIDIA's released example motion clips.

    References are generated artifacts, not repository source. Keeping them
    beside the imported model lets the app discover every released clip while
    `.gitignore` continues to exclude the several-megabyte CSV payloads.
    """
    if code_checkout is None:
        return []
    revision = git_revision(code_checkout)
    if revision is not None and revision != CODE_REVISION:
        raise SystemExit(
            "GEAR-SONIC code checkout revision mismatch: "
            f"got {revision}, expected {CODE_REVISION}"
        )
    source_root = (
        code_checkout / "gear_sonic_deploy" / "reference" / "example"
    )
    if not source_root.is_dir():
        raise SystemExit(
            f"GEAR-SONIC example reference directory is missing: {source_root}"
        )
    destination_root = output / "references"
    destination_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for source in sorted(path for path in source_root.iterdir() if path.is_dir()):
        frame_count: int | None = None
        hashes: dict[str, str] = {}
        for name, width in REFERENCE_FILES.items():
            path = source / name
            if not path.is_file():
                raise SystemExit(f"reference {source.name} is missing {name}")
            values = np.loadtxt(
                path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2
            )
            if values.ndim != 2 or values.shape[1] != width:
                raise SystemExit(
                    f"reference {source.name}/{name} has shape {values.shape}; "
                    f"expected [frames, {width}]"
                )
            if values.shape[0] == 0 or not np.isfinite(values).all():
                raise SystemExit(
                    f"reference {source.name}/{name} must contain finite frames"
                )
            if frame_count is None:
                frame_count = int(values.shape[0])
            elif values.shape[0] != frame_count:
                raise SystemExit(
                    f"reference {source.name} frame counts are inconsistent"
                )
            hashes[name] = sha256(path)
        destination = destination_root / source.name
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir()
        for name in [*REFERENCE_FILES, "info.txt", "metadata.txt"]:
            path = source / name
            if path.is_file():
                shutil.copyfile(path, destination / name)
        records.append({
            "name": source.name,
            "frameCount": frame_count,
            "files": hashes,
        })
    if not records:
        raise SystemExit("GEAR-SONIC checkout contains no example references")
    return records


def control_contract() -> dict[str, object]:
    armature_by_type = {
        "5020": 0.003609725,
        "7520_14": 0.010177520,
        "7520_22": 0.025101925,
        "4010": 0.00425,
    }
    motor_types = [
        "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
        "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",
        "7520_14", "5020", "5020",
        "5020", "5020", "5020", "5020", "5020", "4010", "4010",
        "5020", "5020", "5020", "5020", "5020", "4010", "4010",
    ]
    omega = 10.0 * 2.0 * math.pi
    base_stiffness = [armature_by_type[kind] * omega * omega for kind in motor_types]
    base_damping = [
        2.0 * 2.0 * armature_by_type[kind] * omega for kind in motor_types
    ]
    stiffness = list(base_stiffness)
    damping = list(base_damping)
    # Isaac Lab doubles the four ankles and both non-yaw waist axes.
    for high_gain_5020 in (4, 5, 10, 11, 13, 14):
        stiffness[high_gain_5020] *= 2.0
        damping[high_gain_5020] *= 2.0
    training_armature = [armature_by_type[kind] for kind in motor_types]
    for high_gain_5020 in (4, 5, 10, 11, 13, 14):
        training_armature[high_gain_5020] *= 2.0
    training_effort_limit = [
        139.0, 139.0, 88.0, 139.0, 50.0, 50.0,
        139.0, 139.0, 88.0, 139.0, 50.0, 50.0,
        88.0, 50.0, 50.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    ]
    training_velocity_limit = [
        20.0, 20.0, 32.0, 20.0, 37.0, 37.0,
        20.0, 20.0, 32.0, 20.0, 37.0, 37.0,
        32.0, 37.0, 37.0,
        37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0,
        37.0, 37.0, 37.0, 37.0, 37.0, 22.0, 22.0,
    ]
    # Isaac Lab derives the policy scale from each configured actuator's
    # actual effort clamp and stiffness. For ankles and non-yaw waist axes both
    # values are doubled, leaving the released motor-family ratio unchanged.
    action_scale = [
        0.25 * training_effort_limit[index] / stiffness[index]
        for index in range(len(motor_types))
    ]
    # The release's MuJoCo/hardware plant uses smaller hip pitch/roll clamps.
    # Keep it as a named deployment profile instead of silently weakening the
    # Isaac training plant used for source-parity replay.
    deployment_effort_limit = [
        88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
        88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
        88.0, 50.0, 50.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
        25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    ]
    return {
        "physicsTimeStep": 0.005,
        "controlDecimation": 4,
        "periodSeconds": 0.02,
        "referenceFrames": 10,
        "referenceFrameStride": 5,
        "historyFrames": 10,
        "historyOldestFirst": True,
        "actuatorJointNames": ACTUATOR_JOINT_NAMES,
        "policyJointNames": POLICY_JOINT_NAMES,
        "actuatorToPolicy": ACTUATOR_TO_POLICY,
        "policyToActuator": POLICY_TO_ACTUATOR,
        "defaultJointPositions": DEFAULT_ANGLES,
        "actionScale": action_scale,
        "stiffness": stiffness,
        "damping": damping,
        "trainingArmature": training_armature,
        "trainingEffortLimit": training_effort_limit,
        "trainingVelocityLimit": training_velocity_limit,
        "deploymentEffortLimit": deployment_effort_limit,
    }


def deterministic_inputs() -> tuple[np.ndarray, np.ndarray]:
    reference = np.zeros((2, COMPACT_REFERENCE_DIMENSION), dtype=np.float32)
    history = np.zeros((2, HISTORY_DIMENSION), dtype=np.float32)
    # Both encoder probes are deliberately separated from FSQ half-step
    # boundaries. This makes the golden result stable across ONNX CPU and
    # Apple-GPU float32 affine kernels despite the quantizer's discontinuity.
    for sample, phase in enumerate((262, 260)):
        reference[sample] = np.asarray([
            0.17 * math.sin((index + 1) * 0.017 + phase * 0.137)
            + 0.04 * math.cos((index + 3) * 0.071 + phase * 0.291)
            for index in range(COMPACT_REFERENCE_DIMENSION)
        ], dtype=np.float32)
    history[1] = np.asarray([
        0.13 * math.sin((index + 2) * 0.023)
        - 0.03 * math.cos((index + 5) * 0.053)
        for index in range(HISTORY_DIMENSION)
    ], dtype=np.float32)
    return reference, history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", required=True, type=Path)
    parser.add_argument("--decoder", required=True, type=Path)
    parser.add_argument("--observation-config", required=True, type=Path)
    parser.add_argument("--license-file", required=True, type=Path)
    parser.add_argument("--code-checkout", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    encoder_path = args.encoder.expanduser().resolve()
    decoder_path = args.decoder.expanduser().resolve()
    config_path = args.observation_config.expanduser().resolve()
    license_path = args.license_file.expanduser().resolve()
    for path in (encoder_path, decoder_path, config_path, license_path):
        if not path.is_file():
            raise SystemExit(f"required source file does not exist: {path}")

    config_text = config_path.read_text(encoding="utf-8")
    for required in (
        "token_state",
        "his_base_angular_velocity_10frame_step1",
        "his_body_joint_positions_10frame_step1",
        "his_body_joint_velocities_10frame_step1",
        "his_last_actions_10frame_step1",
        "his_gravity_dir_10frame_step1",
        "motion_joint_positions_10frame_step5",
        "motion_joint_velocities_10frame_step5",
        "motion_anchor_orientation_10frame_step5",
    ):
        if required not in config_text:
            raise SystemExit(f"observation config is missing {required}")

    encoder_model = onnx.load(str(encoder_path))
    decoder_model = onnx.load(str(decoder_path))
    onnx.checker.check_model(encoder_model)
    onnx.checker.check_model(decoder_model)
    require_graph_contract(
        encoder_model, "obs_dict", ENCODER_INPUT_SHAPE,
        "encoded_tokens", ENCODER_OUTPUT_SHAPE,
    )
    require_graph_contract(
        decoder_model, "obs_dict", DECODER_INPUT_SHAPE,
        "action", DECODER_OUTPUT_SHAPE,
    )

    encoder_initializers = initializer_arrays(encoder_model)
    decoder_initializers = initializer_arrays(decoder_model)
    encoder_weights: list[np.ndarray] = []
    encoder_biases: list[np.ndarray] = []
    tensors: dict[str, np.ndarray] = {}
    for layer, base in enumerate(ENCODER_NAMES):
        weight = encoder_initializers.get(f"{base}.weight")
        bias = encoder_initializers.get(f"{base}.bias")
        if weight is None or bias is None:
            raise SystemExit(f"missing G1 encoder layer {base}")
        require_shape(
            f"{base}.weight", weight,
            [ENCODER_DIMENSIONS[layer + 1], ENCODER_DIMENSIONS[layer]],
        )
        require_shape(f"{base}.bias", bias, [ENCODER_DIMENSIONS[layer + 1]])
        encoder_weights.append(weight)
        encoder_biases.append(bias)
        tensors[f"encoder.{layer}.weight"] = weight
        tensors[f"encoder.{layer}.bias"] = bias

    decoder_weights: list[np.ndarray] = []
    decoder_biases: list[np.ndarray] = []
    for layer, (weight_name, bias_name) in enumerate(
        zip(DECODER_WEIGHTS, DECODER_BIASES)
    ):
        source_weight = decoder_initializers.get(weight_name)
        bias = decoder_initializers.get(bias_name)
        if source_weight is None or bias is None:
            raise SystemExit(f"missing G1 decoder layer {layer}")
        require_shape(
            weight_name, source_weight,
            [DECODER_DIMENSIONS[layer], DECODER_DIMENSIONS[layer + 1]],
        )
        require_shape(bias_name, bias, [DECODER_DIMENSIONS[layer + 1]])
        weight = np.ascontiguousarray(source_weight.T, dtype=np.float32)
        decoder_weights.append(weight)
        decoder_biases.append(bias)
        tensors[f"decoder.{layer}.weight"] = weight
        tensors[f"decoder.{layer}.bias"] = bias

    constants = constant_arrays(encoder_model)
    quantizer = {
        "offset": uniform_scalar(
            constants, "/quantizer/Constant_1_output_0"),
        "scale": uniform_scalar(
            constants, "/quantizer/Constant_2_output_0"),
        "halfStep": uniform_scalar(
            constants, "/quantizer/Constant_3_output_0"),
        "levels": uniform_scalar(
            constants, "/quantizer/Constant_4_output_0"),
    }

    reference, history = deterministic_inputs()
    encoder_session = ort.InferenceSession(
        str(encoder_path), providers=["CPUExecutionProvider"])
    decoder_session = ort.InferenceSession(
        str(decoder_path), providers=["CPUExecutionProvider"])
    # NVIDIA exports fixed batch-one graphs.  Run golden samples separately;
    # native MLX remains genuinely batched at runtime.
    expanded_reference = compact_to_onnx_reference(reference)
    onnx_tokens = np.concatenate([
        encoder_session.run(
            ["encoded_tokens"], {"obs_dict": expanded_reference[index:index + 1]}
        )[0].astype(np.float32)
        for index in range(reference.shape[0])
    ], axis=0)
    encoder_latent = dense_stack(
        pack_reference_for_encoder(reference), encoder_weights, encoder_biases)
    quantizer_pre_round = (
        np.tanh(encoder_latent + quantizer["offset"]) * quantizer["scale"]
        - quantizer["halfStep"]
    )
    quantizer_tie_margin = float(np.min(np.abs(
        quantizer_pre_round - (np.floor(quantizer_pre_round) + 0.5))))
    if quantizer_tie_margin < 0.02:
        raise SystemExit(
            "golden encoder probe is too close to an FSQ rounding boundary "
            f"({quantizer_tie_margin})")
    direct_tokens = np.round(quantizer_pre_round) / quantizer["levels"]
    direct_tokens = direct_tokens.astype(np.float32)
    token_error = float(np.max(np.abs(onnx_tokens - direct_tokens)))
    if token_error > 1e-6:
        raise SystemExit(
            f"simplified encoder disagrees with ONNX (max error {token_error})")

    decoder_input = np.concatenate([onnx_tokens, history], axis=1)
    onnx_actions = np.concatenate([
        decoder_session.run(
            ["action"], {"obs_dict": decoder_input[index:index + 1]}
        )[0].astype(np.float32)
        for index in range(decoder_input.shape[0])
    ], axis=0)
    direct_actions = dense_stack(
        decoder_input, decoder_weights, decoder_biases)
    action_error = float(np.max(np.abs(onnx_actions - direct_actions)))
    if action_error > 2e-5:
        raise SystemExit(
            f"simplified decoder disagrees with ONNX (max error {action_error})")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "policy.safetensors"
    write_safetensors(weights_path, tensors)
    copied_license = output / "LICENSE.nvidia-model.txt"
    shutil.copyfile(license_path, copied_license)
    code_checkout = (
        args.code_checkout.expanduser().resolve()
        if args.code_checkout else None
    )
    reference_clips = copy_reference_clips(code_checkout, output)

    manifest = {
        "schemaVersion": 1,
        "format": FORMAT,
        "robot": "unitree-g1-29dof",
        "source": {
            "project": SOURCE_PROJECT,
            "url": SOURCE_URL,
            "modelRevision": hugging_face_revision(encoder_path),
            "encoderSHA256": sha256(encoder_path),
            "decoderSHA256": sha256(decoder_path),
            "observationConfigSHA256": sha256(config_path),
            "codeProject": CODE_PROJECT,
            "codeURL": CODE_URL,
            "codeRevision": git_revision(code_checkout),
            "license": "NVIDIA-Open-Model-License",
            "licenseFile": copied_license.name,
            "licenseSHA256": sha256(copied_license),
        },
        "weightsFile": weights_path.name,
        "weightsSHA256": sha256(weights_path),
        "network": {
            "kind": "gear-sonic-g1-encoder-decoder",
            "compactReferenceDimension": COMPACT_REFERENCE_DIMENSION,
            "tokenDimension": 64,
            "historyDimension": HISTORY_DIMENSION,
            "decoderInputDimension": 994,
            "actionDimension": 29,
            "encoderLayerDimensions": ENCODER_DIMENSIONS,
            "decoderLayerDimensions": DECODER_DIMENSIONS,
            "activation": "silu",
            "encoderMode": "g1",
            "encoderModeID": 0,
            "quantizer": quantizer,
        },
        "control": control_contract(),
        "referenceObservationLayout": [
            "future_joint_position[10,29], frames current+0,5,...45",
            "future_joint_velocity[10,29], frames current+0,5,...45",
            "base_to_reference_anchor_rotation_6d[10,6], frames current+0,5,...45",
        ],
        "historyObservationLayout": [
            "base_angular_velocity[10,3], oldest_to_newest",
            "joint_position_minus_default[10,29], policy_order, oldest_to_newest",
            "joint_velocity[10,29], policy_order, oldest_to_newest",
            "previous_raw_action[10,29], policy_order, oldest_to_newest",
            "projected_gravity[10,3], oldest_to_newest",
        ],
        "references": reference_clips,
        "golden": {
            "referenceInputs": reference.tolist(),
            "historyInputs": history.tolist(),
            "tokens": onnx_tokens.tolist(),
            "actions": onnx_actions.tolist(),
            "tokenAbsoluteTolerance": 1e-6,
            # Seven large float32 affine layers accumulate slightly
            # differently in MLX Metal than ONNX Runtime CPU. The importer
            # still enforces 2e-5 direct graph parity above; this is the
            # separately measured cross-backend deployment envelope.
            "actionAbsoluteTolerance": 2e-3,
            "minimumQuantizerTieMargin": quantizer_tie_margin,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"imported GEAR-SONIC G1 model revision {manifest['source']['modelRevision']}")
    print(f"  encoder direct parity max error: {token_error:.3g}")
    print(f"  decoder direct parity max error: {action_error:.3g}")
    print(f"  golden FSQ tie margin: {quantizer_tie_margin:.3g}")
    print(f"  bundle: {output}")


if __name__ == "__main__":
    main()
