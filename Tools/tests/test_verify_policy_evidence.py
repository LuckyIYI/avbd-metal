from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))

import verify_policy_evidence as verifier  # noqa: E402


class MultiSuiteEvidenceVerifierTests(unittest.TestCase):
    @staticmethod
    def git(root: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True, text=True, check=True)
        return completed.stdout.strip()

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

    def test_published_arachne_profiles_pin_every_file_and_outcome(self) -> None:
        root = TOOLS.parent
        for task, profile in verifier.ARACHNE_QUALIFICATION_PROFILE.items():
            with self.subTest(task=task):
                candidate_files = verifier.arachne_candidate_files(profile)
                self.assertEqual(len(candidate_files), 15)
                candidate = root / "checkpoints" / profile[
                    "checkpointRelativeDirectory"
                ]
                parent = root / profile["parentDirectory"]
                verifier.verify_pinned_file_tree(
                    candidate, candidate_files, f"{task} candidate"
                )
                verifier.verify_pinned_file_tree(
                    parent, profile["parentFiles"], f"{task} parent"
                )
                self.assertEqual(
                    verifier.checkpoint_fingerprint(candidate),
                    profile["candidateFingerprint"],
                )
                self.assertEqual(
                    verifier.checkpoint_fingerprint(parent),
                    profile["parentFingerprint"],
                )
                for suite in profile["suites"].values():
                    self.assertEqual(
                        sum(suite["successes"]), suite["totalSuccesses"]
                    )
                    self.assertEqual(len(suite["seeds"]), 4)
                    self.assertEqual(len(suite["reportSHA256"]), 4)

    def test_pinned_tree_rejects_mutated_or_extra_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            bundle.mkdir()
            payload = bundle / "payload.json"
            payload.write_bytes(b"sealed")
            expected = {
                "payload.json": verifier.sha256_file(payload),
            }
            verifier.verify_pinned_file_tree(bundle, expected, "fixture")
            payload.write_bytes(b"mutated")
            with self.assertRaisesRegex(
                verifier.VerificationError, "SHA-256 changed"
            ):
                verifier.verify_pinned_file_tree(bundle, expected, "fixture")
            payload.write_bytes(b"sealed")
            (bundle / "extra.json").write_text("{}")
            with self.assertRaisesRegex(
                verifier.VerificationError, "file inventory changed"
            ):
                verifier.verify_pinned_file_tree(bundle, expected, "fixture")

    def test_release_app_inventory_is_exact(self) -> None:
        root = TOOLS.parent
        source = root / "checkpoints"
        with tempfile.TemporaryDirectory() as temporary:
            packaged = Path(temporary) / "checkpoints"
            packaged.mkdir()
            shutil.copy2(source / "README.md", packaged / "README.md")
            for relative in (
                "humanoid-isaac-flat-v2",
                "external/unitree-h1",
                "arachne15-velocity-v1",
                "arachne15-goal-v1",
            ):
                shutil.copytree(source / relative, packaged / relative)
            verifier.verify_app_checkpoint_package(root, packaged)
            (packaged / "arachne15-velocity-v0").mkdir()
            with self.assertRaisesRegex(
                verifier.VerificationError, "directory inventory differs"
            ):
                verifier.verify_app_checkpoint_package(root, packaged)

    def test_declared_source_commit_must_remain_in_head_ancestry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.git(root, "init", "--quiet")
            self.git(root, "config", "user.name", "Evidence Test")
            self.git(root, "config", "user.email", "evidence@example.invalid")
            (root / "first.txt").write_text("sealed\n", encoding="utf-8")
            self.git(root, "add", "first.txt")
            self.git(root, "commit", "--quiet", "-m", "sealed source")
            source_commit = self.git(root, "rev-parse", "HEAD")
            verifier.verify_declared_source_commit(
                root, source_commit, "fixture")

            self.git(root, "checkout", "--quiet", "--orphan", "rewritten")
            (root / "second.txt").write_text("replacement\n", encoding="utf-8")
            self.git(root, "add", "-A")
            self.git(root, "commit", "--quiet", "-m", "rewritten history")
            with self.assertRaisesRegex(
                verifier.VerificationError, "not an ancestor of HEAD"
            ):
                verifier.verify_declared_source_commit(
                    root, source_commit, "fixture")


if __name__ == "__main__":
    unittest.main()
