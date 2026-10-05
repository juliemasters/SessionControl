import unittest
from routing_ladder_core import load_sessions,trials,select,evaluate,summarize,wilson,low_confidence
from judge_training_data import build

class DesignTests(unittest.TestCase):
    def test_ambiguity_is_not_unique(self):
        rows=trials(load_sessions('L3'))
        self.assertTrue(all(r['ambiguous'] and len(r['compatible'])==2 for r in rows))
    def test_l0_and_controls(self):
        rows=trials(load_sessions('L0'))
        self.assertFalse(any(r['ambiguous'] for r in rows))
        self.assertEqual(sum(r['active']==r['target'] for r in rows),40)
    def test_tolerance_and_negative_target_margin(self):
        selected,band=select({'a':.80,'b':.81},{'a':2,'b':1},.02)
        self.assertEqual(selected,'a');self.assertEqual(len(band),2)
        t=dict(target='a',active='b',aliases=['dogs'],compatible=['a'],ambiguous=False)
        row=evaluate(t,{'a':.1,'b':.9},'b',['b'],'Seoul','dogs','Seoul')
        self.assertAlmostEqual(row['margin'],-.8)
        self.assertEqual(summarize([row],['a','b'])['unique_answerable']['switch_accuracy'],0)
    def test_absent_metrics_not_zero(self):
        self.assertIsNone(summarize([],['a','b'])['unique_answerable']['false_switch_rate'])
        self.assertLess(wilson(10,10)[0],1)
    def test_training_eval_separation(self):
        train,val=build('train'),build('validation')
        self.assertFalse({r['entity'] for r in train if r['entity']} & {r['entity'] for r in val if r['entity']})
        self.assertFalse({r['question'] for r in train}&{r['question'] for r in val})
        evalrows=load_sessions('L0')+load_sessions('L2')
        self.assertFalse({r['entity'] for r in train+val if r['entity']}&{r['entity'] for r in evalrows})
        self.assertTrue(all(r['label']==1 for r in train if r['kind']=='alternate_valid_value'))
    def test_no_silent_session_shortfall(self):
        with self.assertRaises(ValueError):load_sessions('L3',8)
    def test_regeneration_threshold(self):
        self.assertTrue(low_confidence({'a':.54,'b':.1},.55))
        self.assertFalse(low_confidence({'a':.55,'b':.1},.55))
        self.assertFalse(low_confidence({'a':.01,'b':.9},.55))
        with self.assertRaises(ValueError):low_confidence({'a':float('nan')},.55)

if __name__=='__main__':unittest.main()
