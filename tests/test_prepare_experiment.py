"""CPU tests of original cohort/corpus reconstruction and honest bootstrap metadata."""
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import prepare_experiment as module
from scripts import bootstrap_experiment as bootstrap


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

    def test_source_completion_requires_actual_unchanged_delta_files(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            module.write_json(source / "completion.json", dict(completed=False, base_weights_restored=False))
            with self.assertRaises(ValueError):
                bootstrap.verify_source(source)
            hashes = {}
            for i in range(40):
                file = source / "deltas" / str(i) / "v1.pt"
                file.parent.mkdir(parents=True)
                file.write_bytes(b"synthetic integrity fixture")
                hashes[str(i)] = module.sha256(file)
            module.write_json(source / "completion.json", dict(completed=True, base_weights_restored=True,
                              selected_writer="answer_plus_eos", physical_gpu=1, new_delta_sha256=hashes))
            self.assertEqual(bootstrap.verify_source(source)["new_delta_sha256"], hashes)
            (source / "deltas/0/v1.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                bootstrap.verify_source(source)

    def test_supplied_model_path_cannot_replace_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "downloaded"
            target.mkdir()
            (target / "config.json").write_text("{}")
            expected = root / "models/runtime"
            bootstrap.local_model_link(target, expected)
            self.assertEqual(expected.resolve(), target)
            bootstrap.local_model_link(target, expected)
            other = root / "other"
            other.mkdir()
            (other / "config.json").write_text("{}")
            with self.assertRaises(FileExistsError):
                bootstrap.local_model_link(other, expected)
            self.assertEqual(expected.resolve(), target)

    def test_frozen_scaling_design_has_no_fabricated_prior_evaluation(self):
        records, manifest = self.synthetic_cohort()
        items = module.reconstruct_items(records, manifest)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source-baseline"
            source.mkdir()
            hashes = {}
            for row in items[:40]:
                file = source / "deltas" / row["sid"] / "v1.pt"
                file.parent.mkdir(parents=True)
                file.write_bytes(b"synthetic delta fixture; not model weights")
                hashes[row["sid"]] = module.sha256(file)
            module.write_json(source / "completion.json", dict(completed=True, base_weights_restored=True,
                              selected_writer="answer_plus_eos", physical_gpu=1, new_delta_sha256=hashes))
            for name, value in [("items.json", items[:40]), ("design.json", {"synthetic_test_fixture": True}), ("writes.json", [])]:
                module.write_json(source / name, value)
            projection = root / "projection.pt"
            projection.write_bytes(b"synthetic projection fixture")
            with patch.object(bootstrap, "check_preparation", return_value=({"dataset_sha256": module.DATASET_SHA256}, items)), \
                    patch.object(bootstrap.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024 ** 3)):
                out = bootstrap.prepare_scaling(root, root / "prepared", projection, root)
            design = json.loads((out / "design.json").read_text())
            self.assertEqual(len(design["reused_delta_sids"]), 40)
            self.assertEqual(len(design["new_delta_sids"]), 24)
            self.assertIsNone(design["previous_comparison"])
            self.assertFalse(design["previous_comparison_evaluated"])
            self.assertFalse(design["historical_validation_reproduced"])
            self.assertEqual(design["comparison_hashes"], {})
            self.assertEqual(set(design["references"]), {"embedding", "kv", "rag"})
            self.assertFalse((out / "evaluation.json").exists())
            self.assertFalse(json.loads((out / "completion.json").read_text())["completed"])
            frozen = json.loads((out / "manifest_sha256.json").read_text())
            self.assertEqual(frozen, {name: module.sha256(out / name) for name in ["items.json", "design.json"]})

    def test_default_flow_replaces_cuda_process_before_scaling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prepared, projection, out = root / "runs/prepared", root / "runs/prepared/projector.pt", root / "runs/reproduction"
            argv = ["bootstrap_experiment.py", "--prepared", str(prepared), "--projection", str(projection), "--out", str(out)]
            with patch.object(bootstrap, "ROOT", root), patch.object(sys, "argv", argv), \
                    patch.object(bootstrap, "check_preparation"), \
                    patch.object(bootstrap, "check_projection", return_value=(projection, {})), \
                    patch.object(bootstrap, "local_model_link"), patch.object(bootstrap, "bootstrap") as writer, \
                    patch.object(bootstrap, "prepare_scaling") as scaling, \
                    patch.object(bootstrap.os, "chdir"), patch.object(bootstrap.os, "execv") as replace:
                bootstrap.main()
                writer.assert_called_once_with(root, prepared, projection, out)
                scaling.assert_not_called()
                replace.assert_called_once()
                executable, command = replace.call_args.args
                self.assertEqual(executable, sys.executable)
                self.assertEqual(command[:2], [sys.executable, str(Path(bootstrap.__file__).resolve())])
                self.assertEqual(command[-1], "--run-only")
                for flag, value in [("--prepared", prepared), ("--projection", projection), ("--out", out),
                                    ("--model", root / "models/Llama-3.1-8B-Instruct"),
                                    ("--embedding-model", root / "models/all-MiniLM-L6-v2")]:
                    self.assertEqual(command[command.index(flag) + 1], str(value))


if __name__ == "__main__":
    unittest.main()
