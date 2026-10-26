"""Guard the corrected write contract, complete support eligibility and ten passes."""
from pathlib import Path
import statistics
import sys
import unittest

from qa500_core import CODE,answer_requests,question_plan,mean_sd,document
sys.path.insert(0,str(CODE))


class CorrectedProtocolTests(unittest.TestCase):
    def test_reference_answer_not_capsule_is_full_write_target(self):
        class Tok:eos_token='<EOS>'
        answer='The full answer includes {braces}, punctuation, and every word. '*20
        record=dict(sid='s',context=[dict(role='user',content='Unabridged context {with data}')],
                    bindings=[dict(qid='q',question='Exact original question?',answer=answer,capsule='WRONG TARGET')])
        req=answer_requests(record,Tok(),0)[0]
        self.assertEqual(req['target_new']['str'],answer+'<EOS>')
        self.assertEqual(req['subject'],'Q: Exact original question?\nA:')
        self.assertIn('Unabridged context',req['value_context_templates'][0][1])
        self.assertIn(answer,document(record))
        self.assertNotIn('WRONG TARGET',document(record))

    def test_multi_question_conversation_preserves_both_complete_answers(self):
        class Tok:eos_token='<EOS>'
        record=dict(sid='s',context=[dict(role='user',content='All conversation turns')],bindings=[
            dict(qid='a',question='Question A?',answer='Complete A'),dict(qid='b',question='Question B?',answer='Complete B')])
        req=answer_requests(record,Tok(),7)
        self.assertEqual([r['target_new']['str'] for r in req],['Complete A<EOS>','Complete B<EOS>'])

    def test_query_waits_until_all_supporting_sessions_exist(self):
        sessions=[dict(sid=s) for s in ['a','b','c']]
        queries=[dict(qid='single',support_sids=['a']),dict(qid='multi',support_sids=['b','c'])]
        plan=question_plan(sessions,queries,sizes=[1,2,3])
        self.assertEqual([p['qids'] for p in plan],[['single'],['single'],['single','multi']])

    def test_ten_pass_mean_and_sample_sd(self):
        a=mean_sd(list(range(10)))
        self.assertEqual(a['mean'],4.5);self.assertEqual(a['sd'],statistics.stdev(range(10)))
        self.assertEqual(mean_sd([.9]*10)['sd'],0.)
        with self.assertRaises(ValueError):mean_sd([1]*9)

    def test_real_tokenizer_preserves_quoted_answer_alignment(self):
        from transformers import AutoTokenizer
        from grounding_core import subject_last_index
        import ast
        source=(CODE/'AlphaEdit/grounded_value.py').read_text()
        tree=ast.parse(source)
        decoders=[n for n in ast.walk(tree) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Attribute) and n.func.attr=='decode']
        self.assertEqual(len(decoders),1)
        self.assertTrue(any(k.arg=='clean_up_tokenization_spaces' and
                            isinstance(k.value,ast.Constant) and k.value.value is False
                            for k in decoders[0].keywords))
        tok=AutoTokenizer.from_pretrained('/home/julie/alphaeditSessionControl/models/Llama-3.1-8B-Instruct',local_files_only=True)
        record=dict(context=[dict(role='user',content='The complete conversation contains {braces}.')],
                    bindings=[dict(question='What alternative terms did you suggest?',
                    answer="I suggested 'sexual fixations', 'problematic sexual behaviors', 'sexual impulsivity', and 'compulsive sexuality'.")])
        req=answer_requests(record,tok,0)[0]
        ids=tok.encode(' '+req['target_new']['str'].lstrip())
        if ids[0] in [tok.bos_token_id,tok.unk_token_id]:ids=ids[1:]
        for context in req['value_context_templates'][0]:
            base=context.format(req['prompt'])
            prefix=tok.encode(base.format(req['subject']))
            expected=prefix+ids[:-1]
            old=tok.encode((base+tok.decode(ids[:-1],clean_up_tokenization_spaces=True)).format(req['subject']))
            corrected=tok.encode((base+tok.decode(ids[:-1],clean_up_tokenization_spaces=False)).format(req['subject']))
            self.assertNotEqual(old,expected)
            self.assertEqual(corrected,expected)
            idx=subject_last_index(tok,base,req['subject'])
            self.assertEqual(len(corrected)-idx,len(ids))


if __name__=='__main__':unittest.main()
