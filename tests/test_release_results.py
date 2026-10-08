"""CPU integrity checks for the exported evidence and important failure cases."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.verify_release import (ReleaseError, candidate_files, verify_condition,
                                    verify_evaluation, verify_manifest, verify_public_files,
                                    verify_release)


ROOT = Path(__file__).resolve().parents[1]


class HistoricalResultsTests(unittest.TestCase):
    def test_exported_results_are_internally_consistent(self):
        result = verify_release(ROOT)
        self.assertGreater(result["exported_artifacts"], 0)
        self.assertGreater(result["verified_final_answers"], 0)

    def test_factual_change_is_detected_from_output_not_saved_flag(self):
        data = json.loads((ROOT / "docs/results/memory-scaling-v1/evaluation.json").read_text())
        condition = copy.deepcopy(data["results"]["24"]["our_delta"])
        condition["trials"][0]["final"]["text"] = "A contradictory answer"
        with self.assertRaisesRegex(ReleaseError, "inconsistent trial metric"):
            verify_condition(condition, "altered", 24)

    def test_unmeasured_capacity_limit_cannot_be_reported_as_zero_accuracy(self):
        data = json.loads((ROOT / "docs/results/memory-scaling-v1/evaluation.json").read_text())
        conditions = [c for methods in data["results"].values() for c in methods.values()
                      if c.get("available") is False]
        self.assertTrue(conditions)
        condition = copy.deepcopy(conditions[0])
        condition["summary"] = {"n": 0, "returned_exact": {"count": 0, "rate": 0}}
        with self.assertRaisesRegex(ReleaseError, "no measured result"):
            verify_condition(condition, "altered", 0)

    def test_question_order_must_match_across_methods(self):
        folder = ROOT / "docs/results/memory-scaling-v1"
        data = json.loads((folder / "evaluation.json").read_text())
        completed = json.loads((folder / "completion.json").read_text())
        trials = data["results"]["24"]["rag_qa"]["trials"]
        trials[0], trials[1] = trials[1], trials[0]
        with self.assertRaisesRegex(ReleaseError, "question order differs"):
            verify_evaluation(data, completed, "altered")

    def test_source_hash_is_not_checked_against_normalized_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "docs/results"
            folder.mkdir(parents=True)
            artifact = folder / "sanitized.json"
            original, released = b'{"path":"private"}', b'{"path":"redacted"}'
            artifact.write_bytes(released)
            manifest = dict(schema_version=1, normalization="Paths redacted; original evidence digest retained.",
                            files=[dict(path="docs/results/sanitized.json", bytes=len(released), normalized=True,
                                        original_sha256=hashlib.sha256(original).hexdigest(),
                                        released_sha256=hashlib.sha256(released).hexdigest())])
            (folder / "export-manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(verify_manifest(root), 1)
            artifact.write_bytes(original)
            with self.assertRaises(ReleaseError):
                verify_manifest(root)

    def test_archive_fallback_scans_only_supplied_root_and_excludes_local_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "release"
            root.mkdir()
            (root / "README.md").write_text("Public release.")
            (root / "docs").mkdir()
            (root / "docs/results").mkdir()
            (root / "docs/results/result.json").write_text("{}")
            (root / "models").mkdir()
            (root / "models/local.pt").write_bytes(b"local")
            (root / "__pycache__").mkdir()
            (root / "__pycache__/local.pyc").write_bytes(b"local")
            (Path(temporary) / "outside.pt").write_bytes(b"outside")
            # Even if Git identifies an enclosing repository, use the archive's own tree.
            import subprocess
            probe = subprocess.CompletedProcess([], 0, str(Path(temporary)).encode(), b"")
            with patch("scripts.verify_release.subprocess.run", return_value=probe) as command:
                selected = candidate_files(root)
                self.assertEqual({p.relative_to(root).as_posix() for p in selected},
                                 {"README.md", "docs/results/result.json"})
                self.assertEqual(command.call_count, 1)
                self.assertEqual(verify_public_files(root), 2)
                (root / "accidental_weights.pt").write_bytes(b"weights")
                with self.assertRaisesRegex(ReleaseError, "artifact included"):
                    verify_public_files(root)


if __name__ == "__main__":
    unittest.main()
