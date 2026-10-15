"""CPU tests of original cohort and preservation-corpus reconstruction."""
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
from scripts import prepare_experiment as module

def record(qid, source=None):
    source = source or "source-" + qid
    return dict(question_id=qid, question_type="single-session-user", question="Original " + qid + "?",
                answer="literal answer " + qid, answer_session_ids=[source], haystack_session_ids=[source],
                haystack_sessions=[[dict(role="user", content="Full session with {literal} café."),
                                    dict(role="assistant", content="Assistant turn retained.")]])

def entry(row, history="earlier_pilot"):
    qid = row["question_id"]
    return dict(qid=qid, sid="qa-" + hashlib.sha256(qid.encode()).hexdigest()[:16],
                source_session_id=row["answer_session_ids"][0], history=history)

class PreparationTests(unittest.TestCase):
    def test_original_session_and_manifest_order_preserved(self):
        first, second = record("a"), record("b")
        entries = [entry(second), entry(first)]
        items = module.reconstruct_items([first, second], dict(sessions=entries))
        self.assertEqual([r["qid"] for r in items], ["b", "a"])
        self.assertEqual(items[0]["question"], second["question"])
        self.assertEqual(items[0]["answer"], second["answer"])
        self.assertEqual(items[0]["context"], second["haystack_sessions"][0])
        self.assertEqual(len(items[0]["context"]), 2)

    def test_rejects_ineligible_duplicate_and_mismatched_identifiers(self):
        row = record("a")
        manifest = entry(row)
        for field, value in [("sid", "wrong"), ("source_session_id", "wrong")]:
            wrong = dict(manifest, **{field: value})
            with self.assertRaises(ValueError):
                module.reconstruct_items([row], [wrong])
        with self.assertRaises(ValueError):
            module.reconstruct_items([row], [manifest, manifest])
        abstention = dict(row, question_id="a_abs")
        with self.assertRaises(ValueError):
            module.reconstruct_items([abstention], [entry(abstention)])
        with self.assertRaises(ValueError):
            module.reconstruct_items([dict(row, question_type="multi-session")], [manifest])

    def test_global_answer_exclusion_complete_user_turns_and_first_occurrence(self):
        text = "A café memory contains exactly ten separate words for testing."
        other = "A second memory contains ten complete words for this test."
        row = record("a")
        row["haystack_session_ids"] += ["distractor", "protected_elsewhere"]
        row["haystack_sessions"] += [[dict(role="assistant", content=other), dict(role="user", content=text),
                                     dict(role="user", content=text), dict(role="user", content="too short"),
                                     dict(role="user", content=other)], [dict(role="user", content=other)]]
        second = record("b", source="protected_elsewhere")
        corpus, pool, excluded = module.distractor_records([row, second], count=2)
        expected = [dict(text=text, source_session_id="distractor", turn=1),
                    dict(text=other, source_session_id="distractor", turn=4)]
        random.Random(0).shuffle(expected)
        self.assertEqual(corpus, expected)
        self.assertEqual(pool, 2)
        self.assertIn("protected_elsewhere", excluded)
        with self.assertRaises(ValueError):
            module.distractor_records([row, second], count=3)

    def synthetic_cohort(self):
        records = [record(str(i)) for i in range(64)]
        for i in range(200):
            records[0]["haystack_session_ids"].append("distractor-" + str(i))
            records[0]["haystack_sessions"].append([dict(role="user", content=f"Turn {i} includes café and enough separate words for this preservation test.")])
        manifest = [entry(row, "previous_test" if i < 24 else "writer_development" if i < 40 else "earlier_pilot")
                    for i, row in enumerate(records)]
        return records, manifest

    def test_cpu_preparation_has_no_completed_experiment_and_verifies_bytes(self):
        records, manifest = self.synthetic_cohort()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data, manifest_path, out = root / "data.json", root / "manifest.json", root / "prepared"
            module.write_json(data, records)
            module.write_json(manifest_path, dict(sessions=manifest))
            corpus, _, _ = module.distractor_records(records)
            encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in corpus).encode()
            with patch.object(module, "DATASET_SHA256", module.sha256(data)), patch.object(module, "CORPUS_SHA256", hashlib.sha256(encoded).hexdigest()):
                module.prepare(data, manifest_path, out)
                metadata = json.loads((out / "preparation.json").read_text())
                self.assertEqual(metadata["stage"], "prepared_only")
                self.assertFalse(metadata["experiment_completed"])
                self.assertEqual((out / "distractors.jsonl").read_bytes(), encoded)
                self.assertEqual(len(json.loads((out / "items.json").read_text())), 64)
                with self.assertRaises(FileExistsError):
                    module.prepare(data, manifest_path, out)
                data.write_text("[]")
                with self.assertRaises(ValueError):
                    module.prepare(data, manifest_path, root / "bad")
                self.assertFalse((root / "bad").exists())
