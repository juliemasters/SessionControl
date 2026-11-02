"""Meaningful routing/math/interface contracts before an expensive GPU run."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from experiments.session_zsre.core import capture_query_keys, select_candidate, nullspace, raw_write_request


class MatchedSessionTests(unittest.TestCase):
    def test_compact_routing_matches_dense_energy_including_normalization_and_ties(self):
        torch.manual_seed(4)
        names=['a.weight','b.weight'];keys={name:torch.randn(1,7) for name in names}
        bank={sid:{name:(torch.randn(5),torch.randn(7)) for name in names} for sid in ['c','b','a']}
        expected={}
        for sid,payload in bank.items():
            matrices={name:u[:,None]*v[None,:] for name,(u,v) in payload.items()}
            expected[sid]=sum(float((matrix.double()@keys[name][0].double()).square().sum()) for name,matrix in matrices.items())/sum(float(matrix.double().square().sum()) for matrix in matrices.values())
        chosen,value,margin,ranks=select_candidate(bank,keys)
        order=sorted(expected,key=lambda sid:(-expected[sid],sid))
        self.assertEqual(chosen,order[0]);self.assertEqual(ranks,{sid:i+1 for i,sid in enumerate(order)})
        self.assertAlmostEqual(value,expected[chosen],places=5)
        self.assertAlmostEqual(margin,expected[order[0]]-expected[order[1]],places=5)
        bank['a']=bank['b']=bank['c']
        self.assertEqual(select_candidate(bank,keys)[0],'a')

    def test_nullspace_agrees_with_dense_second_moment_threshold(self):
        torch.manual_seed(7);x=torch.randn(10,17)
        p,detail=nullspace(x,.02)
        eigen,v=torch.linalg.eigh(x.T@x/x.shape[0]);expected=v[:,eigen<.02]@v[:,eigen<.02].T
        self.assertTrue(torch.allclose(p,expected,atol=2e-6))
        self.assertTrue(torch.allclose(p@p,p,atol=2e-6));self.assertEqual(detail['tokens'],10)

    def test_writer_never_reads_held_out_fields_and_preserves_complete_target(self):
        class Allowed(dict):
            def __getitem__(self,key):
                if key not in ['case_id','prompt','target_new']:raise AssertionError('Held-out field exposed: '+key)
                return super().__getitem__(key)
        r=Allowed(case_id=5,prompt='Which city?',target_new='The full answer.')
        request=raw_write_request(r,SimpleNamespace(eos_token='</s>'))
        self.assertEqual(request['subject'],'Which city?')
        self.assertEqual(request['target_new']['str'],'The full answer.</s>')
        self.assertEqual(request['value_context_templates'],[['{}']])

    def test_raw_query_keys_match_alphaedit_lookup_on_tiny_mistral(self):
        from transformers import MistralConfig,MistralForCausalLM,AutoTokenizer
        from AlphaEdit.compute_ks import compute_ks
        torch.manual_seed(8);torch.set_num_threads(2)
        tok=AutoTokenizer.from_pretrained('/home/julie/alphaeditSessionControl/models/Mistral-7B-v0.1',local_files_only=True)
        tok.pad_token_id=tok.eos_token_id;tok.padding_side='right'
        cfg=MistralConfig(vocab_size=32000,hidden_size=16,intermediate_size=32,num_hidden_layers=3,
                          num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
        model=MistralForCausalLM(cfg).float().eval();name='model.layers.1.mlp.down_proj.weight'
        query='Which city is the capital of France?'
        captured=capture_query_keys(model,tok,[name],query)
        hp=SimpleNamespace(rewrite_module_tmp='model.layers.{}.mlp.down_proj',fact_token='subject_last')
        native=compute_ks(model,tok,[{'prompt':'{}','subject':query}],hp,1,[['{}']])
        self.assertTrue(torch.equal(captured[name],native))
        # Verify selected-delta metric calls restore base weights even if scoring
        # raises, and targets never enter the query-key or routing functions.
        from compact_session_memory import CompactSessionMemory
        from experiments.session_zsre.worker import routed_metric
        with tempfile.TemporaryDirectory() as tmp:
            memory=CompactSessionMemory(model,[name],Path(tmp));base=model.get_parameter(name).detach().clone()
            bank={'case_0000':{name:(torch.ones(16)*.01,torch.ones(32))}}
            def metric(*args,**kwargs):
                self.assertFalse(torch.equal(model.get_parameter(name),base))
                self.assertEqual(args[4],'secret held-out target')
                raise RuntimeError('Synthetic metric failure')
            with patch('experiments.session_zsre.core.capture_query_keys',wraps=capture_query_keys) as capture:
                with self.assertRaisesRegex(RuntimeError,'Synthetic metric failure'):
                    routed_metric(memory,bank,metric,model,tok,hp,query,'secret held-out target')
                self.assertEqual(capture.call_args.args[-1],query)
            self.assertTrue(torch.equal(model.get_parameter(name),base))


if __name__=='__main__':unittest.main()
