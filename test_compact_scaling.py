"""Meaningful numerical/protocol checks; these perform no LLM inference."""
import unittest
from fixed_query_scaling_core import nested_bank, latency_summary, check_complete, SIZES


class CompactNumerics(unittest.TestCase):
    def test_factor_energy_matches_materialized_independent_updates(self):
        import torch
        from compact_session_memory import factor_scores
        g=torch.Generator().manual_seed(9)
        payload={str(i):(torch.randn(11,generator=g),torch.randn(17,generator=g)) for i in range(3)}
        keys={str(i):torch.randn(5,17,generator=g) for i in range(3)}
        numerator=torch.zeros(5,dtype=torch.float64);denominator=0.
        for name,(left,right) in payload.items():
            matrix=torch.outer(left,right)
            numerator+=(matrix@keys[name].T).double().square().sum(0)
            denominator+=matrix.double().square().sum()
        actual=torch.tensor(factor_scores(payload,keys),dtype=torch.float64)
        torch.testing.assert_close(actual,numerator/denominator,rtol=1e-6,atol=1e-7)

    def test_sherman_morrison_matches_dense_solve(self):
        import torch
        from compact_session_memory import one_key_factors
        generator=torch.Generator().manual_seed(7)
        key=torch.randn(13,dtype=torch.float64,generator=generator)
        residual=torch.randn(5,dtype=torch.float64,generator=generator)
        q,_=torch.linalg.qr(torch.randn(13,13,dtype=torch.float64,generator=generator))
        projector=q[:,:7]@q[:,:7].T
        left,right=one_key_factors(projector,key,residual,.1)
        a=projector@torch.outer(key,key)+.1*torch.eye(13,dtype=torch.float64)
        reference=torch.linalg.solve(a,torch.outer(projector@key,residual)).T
        torch.testing.assert_close(torch.outer(left,right),reference,rtol=1e-11,atol=1e-11)

    def test_rank_one_fit(self):
        import torch
        from compact_session_memory import fit_rank_one
        g=torch.Generator().manual_seed(4)
        matrix=torch.outer(torch.randn(17,generator=g),torch.randn(23,generator=g))
        left,right,error=fit_rank_one(matrix)
        self.assertLess(error,1e-6)
        torch.testing.assert_close(torch.outer(left,right),matrix,rtol=1e-5,atol=1e-6)


class FixedPanelProtocol(unittest.TestCase):
    def test_sampler_waits_for_status_and_excludes_completed_condition(self):
        import json
        import subprocess
        import sys
        import tempfile
        import time
        from pathlib import Path
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            # /proc can expose host PIDs while subprocess uses a sandbox PID namespace.
            proc_pid=Path('/proc/self').resolve().name
            with subprocess.Popen([sys.executable,str(Path(__file__).parent/'scripts/sample_scaling_rss.py'),
                                   '--pid',proc_pid,'--out',str(out)]) as sampler:
                try:
                    time.sleep(.6)
                    (out/'status.json').write_text(json.dumps(dict(stage='our_delta',n=8)))
                    (out/'measurements.json').write_text(json.dumps({'our_delta_8':{}}))
                    time.sleep(1.1)
                finally:
                    sampler.terminate()
                    sampler.wait(timeout=5)
            samples=[json.loads(line) for line in (out/'resource_usage_cpu.jsonl').read_text().splitlines()]
            self.assertTrue(samples)
            self.assertTrue(all(r['stage']=='between_conditions' and r['n'] is None for r in samples))
            self.assertTrue(all(r['host_rss_bytes']>0 for r in samples))

    def test_report_recomputes_accuracy_from_outputs(self):
        import copy
        from build_real512_report import verify_outcomes
        from experiment_utils import evaluated, metrics
        ids=['a','b']
        final=dict(text='Luna',emitted_text='Luna',tokens=2,hit_cap=False,finish_reason='eos')
        trials=[]
        for repeat in range(3):
            for qid in ids:
                trial=evaluated(dict(qid=qid,question='Name?',answer='Luna',sid=qid),
                                final,1.,selected_sid=qid)
                trials.append(dict(trial,repeat=repeat))
        cell=dict(available=True,trials=trials,summary=metrics(trials[:2]))
        results={str(n):{method:copy.deepcopy(cell) for method in ('our_delta','rag_qa')}
                 for n in SIZES}
        for cells in results.values():
            cells['kv_full_qa']=dict(available=False,reason='context_limit',trials=[],summary=None)
        verify_outcomes(results,ids,3)
        results['512']['our_delta']['trials'][0]['final']['text']='wrong'
        with self.assertRaises(ValueError):verify_outcomes(results,ids,3)

    def test_rss_uses_recall_samples_not_loading_hwm_or_untagged_bank(self):
        from three_tier_memory import recall_rss,historical_rss
        samples=[dict(stage='model_loading',n=8,host_rss_bytes=999),
            dict(stage='our_delta',n=8,host_rss_bytes=10,lifetime_high_water_bytes=999),
            dict(stage='our_delta',n=8,host_rss_bytes=20),
            dict(stage='our_delta',host_rss_bytes=100),
            dict(stage='kv_setup_8',host_rss_bytes=50),
            dict(stage='kv_full_qa',n=8,host_rss_bytes=40)]
        result=recall_rss(samples,SIZES)
        self.assertEqual(result['8/our_delta']['sampled_peak_rss_bytes'],20)
        self.assertEqual(result['8/kv_full_qa']['sampled_peak_rss_bytes'],50)
        pooled=historical_rss(samples)
        self.assertEqual(pooled['our_delta']['sampled_peak_rss_bytes'],100)
        self.assertFalse(pooled['our_delta']['bank_size_in_samples'])

    def test_report_refuses_incomplete_run_before_creating_graphs(self):
        import json
        import tempfile
        from pathlib import Path
        from build_real512_report import build
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/'completion.json').write_text(json.dumps(dict(completed=False)))
            with self.assertRaises(ValueError):build(out)
            self.assertEqual(list(out.glob('*.png')),[])
            self.assertEqual(list(out.glob('*.svg')),[])

    def test_nested_panel_is_constant_and_missing_pilot_rejected(self):
        rows=[dict(qid=str(i),source_session_id=str(i)) for i in range(64)]
        pilot=list(reversed(rows[8:16]))
        bank,panel=nested_bank(rows,pilot)
        self.assertEqual([r['qid'] for r in panel],[r['qid'] for r in pilot])
        for n in (8,16,32,64):self.assertTrue(set(r['qid'] for r in panel)<=set(r['qid'] for r in bank[:n]))
        with self.assertRaises(ValueError):nested_bank(rows,pilot[:-1]+[dict(qid='unknown')])

    def test_repeats_do_not_inflate_accuracy_and_unavailable_is_not_zero(self):
        ids=['a','b'];trials=[dict(qid=q,repeat=r,query_seconds=1+r) for r in range(3) for q in ids]
        summary=latency_summary(trials)
        self.assertEqual(summary['warm_n'],4)
        results={str(n):{m:dict(available=True,trials=trials) for m in ('our_delta','rag_qa')} for n in SIZES}
        for cells in results.values():cells['kv_full_qa']=dict(available=False,reason='context_limit',summary=None,trials=[])
        check_complete(results,ids,3)
        results['512']['kv_full_qa']['summary']={'rate':0}
        with self.assertRaises(ValueError):check_complete(results,ids,3)


if __name__=='__main__':unittest.main()
