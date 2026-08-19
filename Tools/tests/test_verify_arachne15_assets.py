#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


TOOLS = Path(__file__).resolve().parents[1]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import verify_arachne15_assets as verifier


SOURCE = verifier.REPOSITORY_ROOT / verifier.ASSET_RELATIVE_DIRECTORY


class ArachneAssetVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.assets = Path(self.temporary.name) / "arachne15"
        shutil.copytree(SOURCE, self.assets)

    def repin(self, relative: str) -> dict[str, str]:
        result = dict(verifier.EXPECTED_FILE_SHA256)
        result[relative] = hashlib.sha256(
            (self.assets / relative).read_bytes()).hexdigest()
        return result

    def test_packaged_snapshot_passes(self) -> None:
        reports = verifier.verify_asset_directory(self.assets)
        self.assertEqual([report.collisions for report in reports], [39, 60])

    def test_missing_and_extra_files_are_rejected(self) -> None:
        (self.assets / "meshes/foot_pad.stl").unlink()
        with self.assertRaisesRegex(verifier.VerificationError, "file inventory"):
            verifier.verify_asset_directory(self.assets)
        shutil.copy2(SOURCE / "meshes/foot_pad.stl",
                     self.assets / "meshes/foot_pad.stl")
        (self.assets / "extra").write_text("unexpected")
        with self.assertRaisesRegex(verifier.VerificationError, "file inventory"):
            verifier.verify_asset_directory(self.assets)

    def test_symlink_is_rejected(self) -> None:
        readme = self.assets / "README.md"
        readme.unlink()
        readme.symlink_to("/not/an/asset")
        with self.assertRaisesRegex(verifier.VerificationError, "symlink"):
            verifier.verify_asset_directory(self.assets)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFO unavailable")
    def test_special_file_is_rejected(self) -> None:
        os.mkfifo(self.assets / "fifo")
        with self.assertRaisesRegex(verifier.VerificationError, "special file"):
            verifier.verify_asset_directory(self.assets)

    def test_changed_hash_is_rejected(self) -> None:
        readme = self.assets / "README.md"
        readme.write_text(readme.read_text() + "changed\n")
        with self.assertRaisesRegex(verifier.VerificationError, "SHA-256 changed"):
            verifier.verify_asset_directory(self.assets)

    def test_truncated_stl_is_rejected_after_rehash(self) -> None:
        relative = "meshes/foot_pad.stl"
        mesh = self.assets / relative
        mesh.write_bytes(mesh.read_bytes()[:-1])
        with self.assertRaisesRegex(verifier.VerificationError, "STL length"):
            verifier.verify_asset_directory(
                self.assets, expected_hashes=self.repin(relative))

    def test_xml_contract_is_rejected_after_rehash(self) -> None:
        relative = "arachne15_training.xml"
        model = self.assets / relative
        model.write_text(model.read_text().replace(
            'ctrlrange="-0.186 0.186"', 'ctrlrange="-0.2 0.2"', 1))
        with self.assertRaisesRegex(verifier.VerificationError, "torque contract"):
            verifier.verify_asset_directory(
                self.assets, expected_hashes=self.repin(relative))

    def test_mesh_escape_is_rejected_after_rehash(self) -> None:
        relative = "arachne15_validation.xml"
        model = self.assets / relative
        model.write_text(model.read_text().replace(
            'file="chassis.stl"', 'file="../chassis.stl"', 1))
        with self.assertRaisesRegex(verifier.VerificationError, "escapes"):
            verifier.verify_asset_directory(
                self.assets, expected_hashes=self.repin(relative))


if __name__ == "__main__":
    unittest.main()
