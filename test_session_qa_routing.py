import unittest

from session_qa_routing_core import select_items, write_request, section_a, section_b, metrics
from question_key_k1_core import prompt
from grounding_core import subject_last_index


def records():
    return [dict(question_type='single-session-user', question_id=str(i), question=f'Question {i}?',
                 answer=answer, answer_session_ids=[str(i)], haystack_session_ids=[str(i)],
                 haystack_sessions=[[dict(role='user', content='Grounded '+answer+' {literal}'),
                                     dict(role='assistant', content='Entire session retained.')]])
            for i, answer in enumerate(['guitar','piano'])]


class Adapter:
    def __init__(self, items):
        self.items = {i['sid']: i for i in items}
        self.stored = {}
        self.calls = []
    def generate(self, text):
        return dict(text='Unknown')
    def write(self, sid, req):
        self.stored[sid] = req
    def probe(self, sid, question):
        assert sid in self.stored
        self.calls.append((sid, question))
        return dict(text=self.items[sid]['answer']+'\nQ: unrelated scaffold')


class IntegrationTests(unittest.TestCase):
    def test_full_session_grounding_and_identical_recall_key(self):
        item = select_items(records(), 2)[0]
        req = write_request(item, 0)
        bare, full = [t.format(req['prompt']).format(req['subject']) for t in req['value_context_templates'][0]]
        self.assertEqual(bare, prompt(item['question']))
        self.assertIn('{literal}', full)
        self.assertIn('Entire session retained.', full)
        self.assertTrue(full.endswith(bare))
        self.assertEqual(req['target_new']['str'], item['answer'])
        class CharacterTokenizer:
            def encode(self, text):
                return list(text)
        template = req['value_context_templates'][0][1].format(req['prompt'])
        self.assertEqual(subject_last_index(CharacterTokenizer(), template, req['subject']), len(full)-1)

    def test_connected_bank_and_fresh_final_answer(self):
        items = select_items(records(), 2)
        adapter = Adapter(items)
        saved = {}
        save = lambda name, value: saved.update({name:value})
        writes = section_a(items, adapter, save)
        expected = {i['question']:i['answer'] for i in items}
        class Judge:
            def score(self, q, candidate):
                return int(candidate == expected[q])
        trials = section_b(items, writes, adapter, Judge(), save)
        self.assertEqual(len(adapter.calls), 2+4+2)
        self.assertTrue(all(t['routing_correct'] and t['answer_exact'] for t in trials))
        self.assertTrue(all(not t['raw_answer_exact'] for t in trials))
        self.assertTrue(all(len(t['candidates']) == 2 for t in trials))
        self.assertEqual(metrics(items,writes,trials)['routing_accuracy'], 1)

    def test_tie_rule_and_failed_write_denominator(self):
        items = select_items(records(), 2)
        adapter = Adapter(items)
        save = lambda *args: None
        writes = section_a(items, adapter, save)
        class Judge:
            def score(self, q, candidate):
                return 0.5
        trials = section_b(items, writes, adapter, Judge(), save)
        self.assertTrue(all(t['tied'] for t in trials))
        self.assertTrue(all(t['selected_sid'] == min(i['sid'] for i in items) for t in trials))
        writes[0]['write_success'] = False
        trials = section_b(items, writes, adapter, Judge(), save)
        m = metrics(items, writes, trials)
        self.assertEqual(m['write_success_rate'], 0.5)
        self.assertEqual(m['complete_probe_bank_rate'], 0)

    def test_no_base_or_short_answer_filter(self):
        rs = records()
        rs[0]['answer'] = 'A longer answer which is not literally present in the session'
        self.assertEqual(len(select_items(rs,2)),2)


if __name__ == '__main__':
    unittest.main()
