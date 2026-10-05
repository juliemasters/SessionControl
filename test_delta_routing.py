import unittest
import json
import os
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
import AlphaEditSessionStore as module
from demo_delta_routing import TinyDialogueFFN, WEIGHT, fake_alphaedit, populate, probe, score, generate
from DeltaProbeText import DeltaProbeText


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        writer = patch.object(module, "apply_AlphaEdit_to_model", fake_alphaedit)
        writer.start()
        self.addCleanup(writer.stop)
        self.model = TinyDialogueFFN()
        self.store = module.SessionFFNMemory(self.model, [WEIGHT], self.temp.name)
        with patch.object(module.time, "time_ns", side_effect=[100, 200]):
            populate(self.store)

    def route(self, question, scoring=score, probing=probe, answering=generate):
        return self.store.route_and_run(question, probing, scoring, answering)

    def test_all_deltas_and_final_binding(self):
        def scoring(q, candidate):
            self.assertEqual(self.model.answer(), "unknown")
            return score(q, candidate)
        result = self.route("Do I play piano?", scoring)
        self.assertEqual([p.candidate for p in result.probes], ["guitar", "piano"])
        self.assertEqual(result.selected.session_id, "B")
        self.assertEqual(result.answer, "You play piano.")
        self.assertFalse(result.tied)
        self.assertEqual(self.model.answer(), "unknown")
        self.assertTrue(self.model.training)

    def test_tie(self):
        result = self.route("What instrument do I play?")
        self.assertTrue(result.tied)
        self.assertEqual(result.selected.session_id, "B")
        self.assertEqual(result.answer, "You play piano.")

    def test_routing_only_does_not_generate_final_answer(self):
        with patch.object(self.store, "run", wraps=self.store.run) as calls:
            result = self.store.route_and_run("Do I play piano?", probe, score)
        self.assertEqual(result.selected.session_id, "B")
        self.assertEqual(result.answer, "")
        self.assertEqual(calls.call_count, 2)  # Only one probe per session.
        self.assertEqual(self.model.answer(), "unknown")

    def test_recency_survives_reopen_and_probes(self):
        self.route("Do I play guitar?")
        self.store = module.SessionFFNMemory(self.model, [WEIGHT], self.temp.name)
        self.assertEqual(self.route("What instrument do I play?").selected.session_id, "B")

    def test_score_beats_recency(self):
        self.assertEqual(self.route("Do I play guitar?").selected.session_id, "A")

    def test_fact_scorer_receives_candidate_provenance_under_base_weights(self):
        seen = []
        def fact_score(question, candidate, session_id, version):
            self.assertEqual(self.model.answer(), "unknown")
            seen.append((session_id, version, candidate))
            return 1.0 if session_id == "A" else 0.0
        def old_score(*args):
            raise AssertionError("Plausibility scorer must not run")
        self.store.activate("B", 1)
        result = self.store.route_and_run("question", lambda m, q: "identical", old_score,
                                          score_session=fact_score)
        self.assertEqual(result.selected.session_id, "A")
        self.assertEqual(seen, [("A", 1, "identical"), ("B", 1, "identical")])
        self.assertEqual(self.model.answer(), "unknown")

    def test_fact_scorer_failure_restores_base(self):
        def fail(*args):
            raise RuntimeError("verification failed")
        self.store.activate("B", 1)
        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            self.store.route_and_run("question", probe, score, score_session=fail)
        self.assertEqual(self.model.answer(), "unknown")

    def test_legacy_recency_and_equal_timestamps(self):
        for session, timestamp in [("A", 4000000000), ("B", 3000000000)]:
            metadata = self.store.root / session / "v1.json"
            data = json.loads(metadata.read_text())
            del data["saved_at_ns"]
            metadata.write_text(json.dumps(data))
            os.utime(self.store.root / session / "v1.pt", ns=(timestamp, timestamp))
        self.assertEqual(self.route("question").selected.session_id, "A")
        os.utime(self.store.root / "B" / "v1.pt", ns=(4000000000, 4000000000))
        self.assertEqual(self.route("question").selected.session_id, "A")

    def test_latest_version(self):
        self.store.write_session("A", None, [{"delta": torch.tensor([[0.], [-10.], [20.]])}],
                                 None, None, None, 2, parent_version=1)
        result = self.route("Do I play piano?")
        self.assertEqual([(p.session_id, p.version) for p in result.probes], [("A", 2), ("B", 1)])
        self.assertEqual(result.probes[0].candidate, "piano")
        self.assertEqual(result.selected.session_id, "A")

    def test_exceptions_restore_base(self):
        def fail(*args):
            raise RuntimeError("callback failed")
        for kwargs in [{"probing": fail}, {"scoring": fail}, {"answering": fail}]:
            with self.assertRaises(RuntimeError):
                self.route("question", **kwargs)
            self.assertEqual(self.model.answer(), "unknown")
            self.assertTrue(self.model.training)

    def test_invalid_scores_and_candidates(self):
        for value in [float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                self.route("question", lambda q, z: value)
        with self.assertRaises(ValueError):
            self.route("question", probing=lambda m, q: "")
        self.assertEqual(self.model.answer(), "unknown")

    def test_empty_store(self):
        with TemporaryDirectory() as directory:
            empty = module.SessionFFNMemory(self.model, [WEIGHT], directory)
            with self.assertRaisesRegex(ValueError, "No saved"):
                empty.route_and_run("question", probe, score, generate)

    def test_concurrent_routes(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(self.route, ["Do I play guitar?", "Do I play piano?"] * 10))
        self.assertEqual([r.selected.session_id for r in results], ["A", "B"] * 10)
        self.assertEqual(self.model.answer(), "unknown")

    def test_text_score_alignment_and_length_normalization(self):
        class FakeLM:
            def __call__(self, input_ids, attention_mask):
                logits = torch.zeros(1, input_ids.shape[1], 5)
                # Prefix is two tokens. Positions 1 and 2 predict the candidate.
                logits[0, 1, 3] = 2
                logits[0, 2, 4] = 1
                return SimpleNamespace(logits=logits)
        adapter = DeltaProbeText(FakeLM(), None)
        adapter._prompt = lambda question: "prefix"
        adapter._input = lambda text: {"input_ids": torch.tensor(
            [[1, 2]] if text == "prefix" else [[3, 4]])}
        actual = adapter.score("question", "candidate")
        expected = (torch.log_softmax(torch.tensor([0., 0., 0., 2., 0.]), 0)[3]
                    + torch.log_softmax(torch.tensor([0., 0., 0., 0., 1.]), 0)[4]) / 2
        self.assertAlmostEqual(actual, expected.item(), places=6)


if __name__ == "__main__":
    unittest.main()
