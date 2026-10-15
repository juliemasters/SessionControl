"""CPU integrity checks for aggregate-only publication and omitted private data."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from scripts.verify_release import ReleaseError,verify_manifest,verify_release,verify_aggregate_tables,verify_aggregate_privacy

ROOT=Path(__file__).resolve().parents[1]


class FinalEvidenceTests(unittest.TestCase):
    def test_public_aggregate_release(self):
        result=verify_release(ROOT)
        self.assertEqual(result['verified_measured_condition_rows'],34)
        self.assertEqual(result['unique_original_questions'],8)

    def mutate_table(self,method,change):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            shutil.copytree(ROOT/'docs/results',root/'docs/results')
            path=root/'docs/results'/method/'comparison_table.json'
            table=json.loads(path.read_text());change(table['rows']);path.write_text(json.dumps(table))
            with self.assertRaises(ReleaseError):verify_aggregate_tables(root)

    def test_unavailable_is_not_zero_accuracy(self):
        self.mutate_table('scaling-real512-ttft-v1',lambda rows:next(r for r in rows if r['status']=='context_limit').update(exact=0))

    def test_timing_repeats_are_not_independent_accuracy_items(self):
        self.mutate_table('scaling-real512-ttft-v1',lambda rows:rows[0].update(accuracy_queries=24))

    def test_memory_tiers_are_checked_against_resource_summaries(self):
        self.mutate_table('scaling-real512-compact-v1',lambda rows:rows[0].update(retained_gib=30.15))

    def test_ttft_cannot_exceed_p95(self):
        self.mutate_table('scaling-real512-ttft-v1',lambda rows:rows[0].update(ttft_median_ms=999999))

    def test_per_item_records_stay_local(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'docs/results';folder.mkdir(parents=True)
            (folder/'accidental.json').write_text(json.dumps(dict(question='fixture',answer='fixture')))
            with self.assertRaises(ReleaseError):verify_aggregate_privacy(root)

    def test_original_hash_does_not_replace_released_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'docs/results';folder.mkdir(parents=True)
            original,released=b'private',b'aggregate'
            artifact=folder/'summary.txt';artifact.write_bytes(released)
            manifest=dict(schema_version=1,normalization='Aggregate-only export',files=[dict(
                path='docs/results/summary.txt',bytes=len(released),normalized=True,
                original_sha256=hashlib.sha256(original).hexdigest(),released_sha256=hashlib.sha256(released).hexdigest())])
            (folder/'export-manifest.json').write_text(json.dumps(manifest));self.assertEqual(verify_manifest(root),1)
            artifact.write_bytes(original)
            with self.assertRaises(ReleaseError):verify_manifest(root)
