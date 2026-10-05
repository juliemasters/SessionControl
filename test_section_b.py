import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
import AlphaEditSessionStore as module
from demo_delta_routing import TinyDialogueFFN, WEIGHT, fake_alphaedit, populate, probe, generate
from section_b import route_session


class SectionBTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.model = TinyDialogueFFN()
        self.store = module.SessionFFNMemory(self.model, [WEIGHT], self.temp.name)
        with patch.object(module, 'apply_AlphaEdit_to_model', fake_alphaedit):
            populate(self.store)

    def test_regenerate_every_delta_once_and_restore(self):
        seen=[]
        def retry(model,q):
            seen.append(model.answer());return 'retry-'+model.answer()
        def score(q,text):
            self.assertEqual(self.model.answer(),'unknown')
            return .9 if text=='retry-guitar' else .1
        result=route_session(self.store,'instrument?',probe,score,generate,regenerate=retry)
        self.assertEqual(seen,['guitar','piano'])
        self.assertEqual(result['selected_session'],'A')
        self.assertTrue(result['regenerated'])
        self.assertEqual(self.model.answer(),'unknown')

    def test_high_score_skips_retry_and_recency_breaks_near_tie(self):
        def retry(*args):raise AssertionError('Retry should not run')
        result=route_session(self.store,'instrument?',probe,lambda q,p:.9,generate,regenerate=retry)
        self.assertTrue(result['tied']);self.assertEqual(result['selected_session'],'B')
        self.assertFalse(result['regenerated'])

    def test_failure_restores_weights(self):
        def fail(*args):raise RuntimeError('judge failed')
        with self.assertRaises(RuntimeError):
            route_session(self.store,'instrument?',probe,fail,generate,regenerate=probe)
        self.assertTrue(torch.equal(self.model.down_proj.weight,torch.zeros_like(self.model.down_proj.weight)))

if __name__=='__main__':unittest.main()
