from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import verify_policy_evidence as verifier  # noqa: E402


class MultiSuiteEvidenceVerifierTests(unittest.TestCase):
    def context(self, bundle: Path) -> verifier.CheckpointContext:
        entry = verifier.CatalogEntry(
            selection_id="fixture", task_id="arachne15-velocity-v0",
            runtime="native", qualification="accepted",
            checkpoint_relative_directory="fixture",
            evidence_relative_path="external.json",
        )
        return verifier.CheckpointContext(entry, bundle, {}, {}, "0" * 64)

    def test_schema2_evidence_must_be_the_in_bundle_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "requalification-manifest.json").write_text("{}")
            external = root / "external.json"
            external.write_text("{}")
            with self.assertRaisesRegex(
                verifier.VerificationError, "in-bundle non-symlink"
            ):
                verifier.verify_schema2_requalification(
                    root, self.context(bundle), external
                )

    def test_schema2_rejects_a_symlinked_bundle_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            target = root / "manifest.json"
            target.write_text("{}")
            manifest = bundle / "requalification-manifest.json"
            manifest.symlink_to(target)
            with self.assertRaisesRegex(
                verifier.VerificationError, "in-bundle non-symlink"
            ):
                verifier.verify_schema2_requalification(
                    root, self.context(bundle), manifest
                )

    def test_configuration_delta_is_exact_and_float32_aware(self) -> None:
        nominal = {"plant": 1, "validationCollisionProfile": 0}
        validation = {"plant": 1.0, "validationCollisionProfile": 1}
        self.assertEqual(
            verifier.configuration_changed_fields(nominal, validation),
            ["validationCollisionProfile"],
        )
        validation["extra"] = 1
        self.assertEqual(
            verifier.configuration_changed_fields(nominal, validation),
            ["extra", "validationCollisionProfile"],
        )

    def test_velocity_profile_pins_collision_metrics(self) -> None:
        criteria = verifier.arachne_evaluation_criteria(
            "arachne15-velocity-v0", {}
        )
        self.assertEqual(criteria["minimumSuccessRate"], 0.90)
        self.assertEqual(
            criteria["minimumTaskMetrics"]["episode/survived"], 0.95
        )
        self.assertEqual(
            criteria["maximumTaskMetrics"][
                "episode/foot_collider_penetration_rmse_m"
            ],
            0.0005,
        )


if __name__ == "__main__":
    unittest.main()
