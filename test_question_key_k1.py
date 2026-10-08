import unittest
from question_key_k1_core import candidates, request, prompt, summarize, interval


def record(i):
    return dict(question_type='single-session-user', question_id=str(i), question=f'Fact {i}?',
                answer='Seoul', answer_session_ids=[str(i)], haystack_session_ids=[str(i)],
                haystack_sessions=[[dict(role='user', content='I live in Seoul.')]])


class QuestionKeyTests(unittest.TestCase):
    def test_identical_write_read_prompt(self):
        item=next(candidates([record(0)]))
        req=request(item,'V1',0)
        self.assertEqual(req['prompt'].format(req['subject']),prompt(item['question']))
        self.assertEqual(req['subject'],prompt(item['question']))
        self.assertEqual(req['target_new']['str'],'Seoul')

    def test_answer_in_question_rejected(self):
        r=record(0);r['question']='Do I live in Seoul?'
        self.assertEqual(list(candidates([r])),[])

    def test_assistant_only_rejected(self):
        r=record(0);r['haystack_sessions'][0][0]['role']='assistant'
        self.assertEqual(list(candidates([r])),[])

    def test_variants_keep_inference_question(self):
        item=next(candidates([record(0)]))
        self.assertEqual(request(item,'V4',0)['subject'],'My memory:')
        self.assertEqual(request(item,'V5',0)['target_new']['str'],'I live in Seoul.')
        self.assertEqual(request(item,'V5',0)['subject'],prompt(item['question']))

    def test_failed_writes_count_in_denominator(self):
        rows=[dict(write_success=True,fired_immediate=True,fired_delayed=True,exact_immediate=True,
                   base_fired=False,immediate=dict(tokens=2)),dict(write_success=False)]
        s=summarize(rows)
        self.assertEqual(s['write_success_rate'],.5)
        self.assertEqual(s['fire_rate_immediate'],.5)
        self.assertEqual(s['mean_probe_length'],2)
        self.assertTrue(s['fire_rate_immediate_ci95'][0]<.5<s['fire_rate_immediate_ci95'][1])
        self.assertIsNone(interval(0,0))


if __name__=='__main__':
    unittest.main()
