"""CPU fixture audits exercise the complete five-arm report input schema."""
import copy
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from comparison_core import read, save
from experiments.zsre_topk import report
from experiments.zsre_topk.core import METHODS, question_order


def fixture(root, dataset):
    """Keep the source bank size while measuring one eight-question checkpoint."""
    source, out = root / 'source', root / 'measurement'
    model = dict(model='fixture-model', model_revision='fixture-revision', answer_cap=4, layers=[1])
    if dataset == 'zsre':
        bank = [f'case_{index:04d}' for index in range(1000)]
        requests = [dict(case_id=i, prompt=f'Question {i}?', target_new=f'Answer {i}',
                         answer_sids=[bank[i]], abstention=False) for i in range(1000)]
        save(source / 'requests.json', [{key: row[key] for key in ('case_id', 'prompt', 'target_new')}
                                        for row in requests])
        ids = list(range(8))
    else:
        bank = [f'session_{index:03d}' for index in range(500)]
        ids = [f'question_{i}' for i in range(8)]
        requests = [dict(case_id=ids[i], prompt=f'Official question {i}?', target_new=f'Official answer {i}',
                         answer_sids=bank[:2] if i == 0 else [bank[i]], abstention=i == 7)
                    for i in range(8)]
        save(source / 'queries.json', [dict(qid=row['case_id'], question=row['prompt'],
                                             answer=row['target_new'], support_sids=row['answer_sids'],
                                             abstention=row['abstention']) for row in requests])
        save(source / 'sessions.json', [dict(sid=sid) for sid in bank])
    save(source / 'design.json', dict(sizes=[8], plan=[dict(n=8, qids=ids)], **model))
    save(source / 'completion.json', dict(completed=True))
    design = dict(dataset=dataset, source_run=str(source), sizes=[8], bank_sids=bank,
                  plan=[dict(n=8, case_ids=ids)], repeats=10, methods=METHODS,
                  physical_gpu=1, precision='float32', code_sha256={'fixture.py': 'frozen'}, **model)
    save(out / 'design.json', design)
    save(out / 'requests.json', requests)
    by_id = {row['case_id']: row for row in requests}
    for method_index, method in enumerate(METHODS):
        for repeat in range(10):
            folder = out / method / f'run-{repeat:02d}'
            save(folder / 'provenance.json', dict(physical_gpu=1, code_sha256=design['code_sha256'],
                                                  started_utc='2026-11-13T00:00:00+00:00'))
            trials = []
            for case_id in question_order(design, 8, repeat):
                request = by_id[case_id]
                # The second official support is a valid hit for the multi-session item.
                selected = request['answer_sids'][-1]
                coarse = [selected] + [sid for sid in bank[:8] if sid != selected]
                shortlist = None if method == 'full_scan' else coarse[:min(int(method.split('_')[1]), 8)]
                abstention = request['abstention']
                trials.append(dict(
                    case_id=case_id, question=request['prompt'], reference=request['target_new'],
                    final=dict(text=request['target_new'], tokens=1, token_ids=[1],
                               hit_cap=False, finish_reason='eos'),
                    token_f1=1., exact=True, output_tokens=1, hit_cap=False,
                    selected=selected, source_hit=None if abstention else True,
                    shortlist_hit=None if abstention or method == 'full_scan' else True,
                    coarse_rank=None if abstention or method == 'full_scan' else 1,
                    fine_rank=None if abstention else 1,
                    shortlisted_sids=shortlist, coarse_order=None if method == 'full_scan' else coarse,
                    fine_order=coarse if shortlist is None else shortlist,
                    routing_seconds=.001 * (repeat + 1), first_generated_token_seconds=.02 + .001 * repeat,
                    ttft_seconds=.03 + .002 * repeat, query_seconds=.1 + .01 * repeat + .001 * method_index,
                ))
            index_disk, index_cpu = (0, 0) if method == 'full_scan' else (512, 256)
            resources = dict(factor_bank_disk_bytes=1024, cpu_factor_payload_bytes=512,
                             key_index_disk_bytes=index_disk, cpu_key_index_payload_bytes=index_cpu,
                             extra_disk_bank_bytes=1024 + index_disk, host_peak_rss_bytes=4096,
                             peak_gpu_allocated_bytes=8192, peak_gpu_reserved_bytes=16384)
            save(folder / 'bank-8.json', dict(
                completed=True, available=True, method=method, n=8, repeat=repeat, physical_gpu=1,
                finished_utc='2026-11-13T01:00:00+00:00', trials=trials,
                groups={'all': report.summary(trials)}, routing=report.routing_summary(trials, method),
                resources=resources,
            ))
    return out, design


class TopKReportTests(unittest.TestCase):
    def test_both_datasets_audit_every_cell_and_preserve_answerable_denominators(self):
        for dataset, routing_per_pass in [('zsre', 8), ('longmemeval', 7)]:
            with self.subTest(dataset=dataset), TemporaryDirectory() as temporary:
                out, design = fixture(Path(temporary), dataset)
                original = {path: path.read_bytes() for path in out.glob('*/run-*/bank-8.json')}
                with patch.object(report, 'check', return_value=design):
                    actual_design, rows, perpass, outputs, verified = report.audit(out)
                self.assertEqual(actual_design, design)
                self.assertEqual((verified, len(outputs), len(perpass), len(rows)), (400, 400, 50, 5))
                self.assertEqual(len({(row['method'], row['repeat'], row['case_id']) for row in outputs}), 400)
                for method_index, row in enumerate(rows):
                    self.assertEqual(row['method'], METHODS[method_index])
                    self.assertEqual((row['questions_per_pass'], row['total_questions']), (8, 80))
                    self.assertEqual(row['routing_questions_per_pass'], routing_per_pass)
                    self.assertEqual(row['abstention_questions_per_pass'], 8 - routing_per_pass)
                    self.assertEqual(row['routing_questions'], routing_per_pass * 10)
                    self.assertEqual(row['routing_correct'], routing_per_pass * 10)
                    self.assertEqual(row['routing_accuracy_mean'], 1.)
                    self.assertEqual(row['mean_f1_mean'], 1.)
                    self.assertEqual(row['mean_f1_sd'], 0.)
                    self.assertEqual(row['ttft_mean_ms_valid_passes'], 10)
                    self.assertAlmostEqual(row['ttft_mean_ms_mean'], 39.)
                    self.assertAlmostEqual(row['ttft_mean_ms_sd'], statistics.stdev([30. + 2 * r for r in range(10)]))
                    self.assertAlmostEqual(row['query_latency_mean_ms_delta_vs_full_scan_mean'], method_index)
                    if row['method'] == 'full_scan':
                        self.assertIsNone(row['shortlist_recall_mean'])
                        self.assertIsNone(row['shortlist_correct'])
                    else:
                        self.assertEqual(row['shortlist_recall_mean'], 1.)
                        self.assertEqual(row['shortlist_correct'], routing_per_pass * 10)
                for measurement in perpass:
                    self.assertEqual(measurement['routing_questions'], routing_per_pass)
                for path, original_bytes in original.items():
                    self.assertEqual(path.read_bytes(), original_bytes)
                    self.assertNotIn('routing_accuracy', read(path)['groups']['all'])
                self.assertFalse((out / 'comparison.json').exists())
                self.assertFalse((out / 'completion.json').exists())

    def test_corrupt_measurements_are_rejected_before_a_report_can_be_emitted(self):
        with TemporaryDirectory() as temporary:
            out, design = fixture(Path(temporary), 'longmemeval')
            path = out / 'topk_1/run-00/bank-8.json'
            original = read(path)

            def answerable(cell):
                return next(trial for trial in cell['trials'] if trial['source_hit'] is not None)

            def wrong_order(cell):
                cell['trials'][0], cell['trials'][1] = cell['trials'][1], cell['trials'][0]

            changes = {
                'unavailable': lambda cell: cell.update(available=False),
                'wrong cohort order': wrong_order,
                'source flag': lambda cell: answerable(cell).update(source_hit=False),
                'fine rank': lambda cell: answerable(cell).update(fine_rank=2),
                'coarse rank': lambda cell: answerable(cell).update(coarse_rank=2),
                'selected outside shortlist': lambda cell: answerable(cell).update(shortlisted_sids=['session_499']),
                'abstention source label': lambda cell: next(t for t in cell['trials'] if t['source_hit'] is None).update(source_hit=True),
                'saved summary': lambda cell: cell['groups']['all'].update(mean_f1=.5),
                'retained memory': lambda cell: cell['resources'].update(extra_disk_bank_bytes=0),
                'finish before start': lambda cell: cell.update(finished_utc='2026-11-12T00:00:00+00:00'),
            }
            for name, change in changes.items():
                with self.subTest(corruption=name):
                    broken = copy.deepcopy(original)
                    change(broken)
                    save(path, broken)
                    with patch.object(report, 'check', return_value=design), self.assertRaises(ValueError):
                        report.audit(out)
                    self.assertFalse((out / 'comparison.json').exists())
            save(path, original)

    def test_changed_source_support_and_missing_measurement_cannot_pass_audit(self):
        with TemporaryDirectory() as temporary:
            out, design = fixture(Path(temporary), 'longmemeval')
            requests = read(out / 'requests.json')
            requests[0]['answer_sids'] = ['session_002']
            save(out / 'requests.json', requests)
            with patch.object(report, 'check', return_value=design), self.assertRaises(ValueError):
                report.audit(out)
            source_queries = read(Path(design['source_run']) / 'queries.json')
            requests[0]['answer_sids'] = source_queries[0]['support_sids']
            save(out / 'requests.json', requests)
            (out / 'topk_32/run-09/bank-8.json').unlink()
            with patch.object(report, 'check', return_value=design), self.assertRaises(FileNotFoundError):
                report.audit(out)


if __name__ == '__main__':
    unittest.main()
