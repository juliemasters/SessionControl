"""Routing measurement integrity, independent of answer generation quality."""
import copy
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from comparison_core import read, save, digest
from experiments.zsre_generated.core import order
from experiments.zsre_generated.routing_report import audit_cell, aggregate_passes
from experiments.zsre_generated import routing_report


def cell(n=8, repeat=0, hits=6):
    return dict(completed=True, available=True, method='our_delta', n=n, repeat=repeat, physical_gpu=1,
        trials=[dict(case_id=i, selected=f"case_{i if i<hits else (i+1)%n:04d}",
                     source_hit=i<hits, correct_rank=1 if i<hits else 2,
                     final={'text':''}, token_f1=0., exact=False) for i in order(n, repeat)])


class RoutingMeasurementTests(unittest.TestCase):
    def test_empty_or_wrong_answers_still_count_in_routing_denominator(self):
        result, decisions = audit_cell(cell(), 8, 0)
        self.assertEqual((result['correct'], result['questions'], result['routing_accuracy']), (6, 8, .75))
        self.assertEqual(len(decisions), 8)

    def test_corrupted_flags_ranks_membership_and_cohort_are_rejected(self):
        base = cell()
        for changes in [{'source_hit':False}, {'correct_rank':2}, {'selected':'case_1000'}]:
            broken = copy.deepcopy(base)
            # The first question in this deterministic order is a routed hit.
            good = next(t for t in broken['trials'] if t['source_hit'])
            good.update(changes)
            with self.assertRaises(ValueError):audit_cell(broken, 8, 0)
        broken = copy.deepcopy(base); broken['trials'][1] = broken['trials'][0]
        with self.assertRaises(ValueError):audit_cell(broken, 8, 0)

    def test_mean_and_sample_sd_require_ten_distinct_complete_passes(self):
        rows = [audit_cell(cell(repeat=r, hits=8 if r==0 else 4), 8, r)[0] for r in range(10)]
        result = aggregate_passes(rows, 8)
        self.assertAlmostEqual(result['routing_accuracy_mean'], .55)
        self.assertAlmostEqual(result['routing_accuracy_sd'], statistics.stdev([1.]+[.5]*9))
        self.assertEqual(result['total_decisions'], 80); self.assertEqual(result['total_correct'], 44)
        with self.assertRaises(ValueError):aggregate_passes(rows[:9], 8)
        with self.assertRaises(ValueError):aggregate_passes(rows[:9]+[rows[0]], 8)

    def test_report_waits_for_success_preserves_measurements_and_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp); design = {'code_sha256':{'fixed-inference-source':'unchanged'}}
            save(out/'input_manifest.json', {})
            save(out/'routing_extension.json', dict(source_sha256=digest(routing_report.__file__),
                 input_manifest_sha256=digest(out/'input_manifest.json')))
            save(out/'completion.json', dict(completed=False))
            with patch.object(routing_report, 'check', return_value=design):
                with self.assertRaises(ValueError):routing_report.build(out)
            self.assertFalse((out/'routing').exists())
            save(out/'completion.json', dict(completed=True))
            save(out/'driver_status.json', dict(stage='completed'))
            requests = [dict(prompt=f'Question {i}?', target_new=f'Answer {i}') for i in range(2)]
            save(out/'requests.json', requests)
            original = {'report.md':'Original summary\n', 'report.html':'<!doctype html><h1>Original report</h1>'}
            for name, text in original.items():(out/name).write_text(text)
            save(out/'report_manifest.json', {name:digest(out/name) for name in original})
            raw = {}
            for repeat in range(10):
                source = out/'our_delta'/f'run-{repeat:02d}'
                save(source/'provenance.json', design)
                c = cell(n=2, repeat=repeat, hits=1)
                for t in c['trials']:
                    t.update(question=requests[t['case_id']]['prompt'], reference=requests[t['case_id']]['target_new'])
                save(source/'bank-2.json', c)
                raw[source/'bank-2.json'] = digest(source/'bank-2.json')
            with patch.object(routing_report, 'check', return_value=design), patch.object(routing_report, 'SIZES', [2]):
                routing_report.build(out)
                self.assertEqual(read(out/'routing/accuracy.json')['summary'][0]['routing_accuracy_mean'], .5)
                self.assertEqual(read(out/'routing/completion.json')['verified_decisions'], 20)
                before = (out/'report.html').read_bytes()
                routing_report.build(out)
                self.assertEqual((out/'report.html').read_bytes(), before)
            for file, want in raw.items():self.assertEqual(digest(file), want)
            for name, want in read(out/'report_manifest.json').items():self.assertEqual(digest(out/name), want)
            self.assertEqual((out/'routing/original_report.md').read_text(), original['report.md'])


if __name__ == '__main__':unittest.main()
