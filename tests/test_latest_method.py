"""Focused CPU checks for the current activation-routing kernels."""
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments/activation_qa500'))


class LatestMethodTests(unittest.TestCase):
    def test_general_factor_energy_matches_dense_delta(self):
        import torch
        from full500_core import factor_energy
        torch.manual_seed(6)
        payload={};keys={};numerator=0.;denominator=0.
        for name in ['layer13','layer14','layer15']:
            u=torch.randn(5,2);v=torch.randn(9,2);h=torch.randn(3,9)
            payload[name]=(u,v);keys[name]=h
            delta=u@v.T
            numerator=numerator+(h@delta.T).double().square().sum(-1)
            denominator=denominator+delta.double().square().sum()
        self.assertTrue(torch.allclose(factor_energy(payload,keys),numerator/denominator,atol=1e-7,rtol=1e-6))

    def test_rank_one_score_matches_dense_normalized_response(self):
        import torch
        from compact_session_memory import factor_scores
        torch.manual_seed(2)
        u=torch.randn(5);v=torch.randn(9);h=torch.randn(2,9)
        actual=torch.tensor(factor_scores({'layer':(u,v)},{'layer':h}),dtype=torch.float64)
        delta=u[:,None]*v[None,:]
        expected=(h@delta.T).double().square().sum(-1)/delta.double().square().sum()
        self.assertTrue(torch.allclose(actual,expected,atol=1e-7,rtol=1e-6))

    def test_write_target_is_complete_answer_and_eos(self):
        from qa500_core import answer_requests
        class Tok:eos_token='<EOS>'
        answer="Every word of the answer, with 'quotes' and {braces}."
        record=dict(context=[dict(role='user',content='Full session context')],
            bindings=[dict(question='Exact question?',answer=answer)])
        request=answer_requests(record,Tok(),0)[0]
        self.assertEqual(request['target_new']['str'],answer+'<EOS>')
        self.assertEqual(request['subject'],'Q: Exact question?\nA:')
        self.assertIn('Full session context',request['value_context_templates'][0][1])

    def test_answer_boundary_preserves_unrelated_same_line_text(self):
        from answer_only_generation import answer_boundary
        full='20 playlists. An unrelated claim follows.'
        self.assertEqual(answer_boundary(full),(full,None))
        self.assertEqual(answer_boundary('20 playlists.\nQ: Next question'),('20 playlists.','next_turn'))

    def test_execution_snapshot_hashes(self):
        from verify_latest_source import verify
        self.assertTrue(verify(ROOT))


if __name__=='__main__':unittest.main()
