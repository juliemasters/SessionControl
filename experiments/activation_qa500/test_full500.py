"""Check low-rank scoring, isolated probe batches and multi-source denominators."""
import unittest

from full500_report import source_scores


class Full500Tests(unittest.TestCase):
    def test_partial_source_recall_is_not_complete_retrieval(self):
        self.assertEqual(source_scores(['a','c'],['a','b']),dict(support_recall=.5,complete_support=False))
        self.assertEqual(source_scores(['b','a','c'],['a','b']),dict(support_recall=1.,complete_support=True))

    def test_rank_two_energy_matches_dense_update(self):
        import torch
        from full500_core import factor_energy
        torch.manual_seed(9)
        U=torch.randn(5,2);V=torch.randn(7,2);x=torch.randn(3,7);delta=U@V.T
        actual=factor_energy({'p':(U,V)},{'p':x})
        expected=(x@delta.T).double().square().sum(-1)/delta.double().square().sum()
        torch.testing.assert_close(actual,expected,rtol=2e-6,atol=1e-8)

    def test_probe_batch_does_not_mix_session_updates(self):
        import torch
        from full500_experiment import delta_batch
        torch.manual_seed(7)
        model=torch.nn.Module();model.ffn=torch.nn.Linear(7,5,bias=False)
        payloads=[{'ffn.weight':(torch.randn(5,r),torch.randn(7,r))} for r in [1,2]]
        x=torch.randn(2,4,7);baseline=model.ffn(x)
        with delta_batch(model,payloads):actual=model.ffn(x)
        expected=torch.stack([baseline[i]+x[i]@(p['ffn.weight'][0]@p['ffn.weight'][1].T).T for i,p in enumerate(payloads)])
        torch.testing.assert_close(actual,expected,rtol=2e-6,atol=1e-6)
        torch.testing.assert_close(model.ffn(x),baseline,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
