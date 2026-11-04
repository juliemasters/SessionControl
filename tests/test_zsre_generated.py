"""Numerical, native-state, raw-generation and reusable-Mistral-cache contracts."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
import torch
from experiments.zsre_generated.core import cumulative_right,wise_state,install_wise,finalized_merge_snapshot,generate_raw,VENDOR,modules


class GeneratedZsRETests(unittest.TestCase):
    def test_cumulative_woodbury_matches_dense_preservation_system(self):
        torch.manual_seed(13)
        for count in [0,3,20]:
            q=torch.linalg.qr(torch.randn(17,5)).Q;p=torch.eye(17)-q@q.T
            previous=torch.randn(17,count);key=torch.randn(17)
            got=cumulative_right(p,key,previous,.1)
            pd=p.double();kd=key.double();hd=previous.double()
            matrix=pd@(kd[:,None]@kd[None,:]+hd@hd.T)+.1*torch.eye(17,dtype=torch.float64)
            expected=torch.linalg.solve(matrix,pd@kd).float()
            self.assertTrue(torch.allclose(got,expected,atol=2e-6,rtol=2e-5))
            self.assertTrue(torch.allclose(matrix@got.double(),pd@kd,atol=2e-5,rtol=2e-5))
        # The accumulator must affect the solve: this is not the independent writer.
        self.assertFalse(torch.allclose(got,cumulative_right(p,key,previous[:,:0],.1),atol=1e-4))

    def tiny(self):
        from transformers import MistralConfig,MistralForCausalLM,AutoTokenizer
        torch.manual_seed(8);torch.set_num_threads(2)
        cfg=MistralConfig(vocab_size=32000,hidden_size=16,intermediate_size=32,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=512,sliding_window=8)
        model=MistralForCausalLM(cfg).float().eval()
        tok=AutoTokenizer.from_pretrained('/home/julie/alphaeditSessionControl/models/Mistral-7B-v0.1',local_files_only=True)
        tok.pad_token_id=tok.eos_token_id
        return cfg,model,tok

    def test_both_native_wise_deployment_states_preserve_exact_logits(self):
        _,model,_=self.tiny();native=modules()
        for retrieve in [False,True]:
            hp=native['wise_hparams'].WISEHyperParams.from_hparams(str(VENDOR/'hparams/WISE/mistral-7b.yaml'))
            hp.model_name='mistral-tiny';hp.inner_params=['model.layers[1].mlp.down_proj.weight'];hp.retrieve=retrieve
            editor=native['WISE'].WISE(hp,copy.deepcopy(model),'cpu');a=editor.get_adapter_layer()
            with torch.no_grad():a.new_weight.add_(.1)
            a.editing_mean_act.update(.01);a.editing_total_cnt=500;a.save_weight()
            state=wise_state(editor,500);query=torch.tensor([[3,4,5,6]])
            with torch.no_grad():before=editor.model(input_ids=query).logits.clone()
            if not retrieve:
                with torch.no_grad():editor(input_ids=query) # native finalization
                finalized=wise_state(editor,500)
                self.assertIsNotNone(finalized['main_weight'])
                install_wise(editor,state)
                with torch.no_grad():after=editor.model(input_ids=query).logits
                self.assertTrue(torch.equal(before,after))
            with tempfile.TemporaryDirectory() as tmp:
                file=Path(tmp)/'state.pt';torch.save(state,file);state=torch.load(file,weights_only=True)
            a.memory_weight=[];a.editing_mean_act.min_a=1e9
            with torch.no_grad():a.new_weight.zero_()
            install_wise(editor,state)
            with torch.no_grad():after=editor.model(input_ids=query).logits
            self.assertTrue(torch.equal(before,after));self.assertEqual(a.editing_total_cnt,500)

    def test_production_finalization_restores_training_outside_inference_mode(self):
        _,model,_=self.tiny();native=modules()
        hp=native['wise_hparams'].WISEHyperParams.from_hparams(str(VENDOR/'hparams/WISE/mistral-7b.yaml'))
        hp.model_name='mistral-tiny';hp.inner_params=['model.layers[1].mlp.down_proj.weight'];hp.retrieve=False
        editor=native['WISE'].WISE(hp,model,'cpu');a=editor.get_adapter_layer()
        with torch.no_grad():a.new_weight.add_(.1)
        a.editing_mean_act.update(.01);a.editing_total_cnt=500;a.save_weight()
        before=wise_state(editor,500);query=torch.tensor([[3,4,5,6]])
        # Exercise the exact production helper even when its caller is in
        # inference mode, which previously made the next copy_/edit fail.
        with torch.inference_mode():deployment=finalized_merge_snapshot(editor,500,dict(input_ids=query))
        self.assertIsNotNone(deployment['main_weight'])
        self.assertFalse(torch.is_inference(a.layer.weight));self.assertFalse(torch.is_inference(a.new_weight))
        self.assertTrue(torch.equal(a.layer.weight,a.original_layer.weight))
        self.assertEqual(a.editing_total_cnt,500);self.assertEqual(len(a.memory_weight),len(before['memory_weight']))
        for old,new in zip(before['memory_weight'],a.memory_weight):self.assertTrue(torch.equal(old,new))
        # Prove the next optimizer step is possible and has a finite gradient.
        a.editing=True;a.set_parameter_tunable();optimizer=torch.optim.SGD([a.new_weight],lr=.01)
        loss=editor.model(input_ids=query).logits.square().mean();loss.backward()
        self.assertIsNotNone(a.new_weight.grad);self.assertTrue(torch.isfinite(a.new_weight.grad).all())
        optimizer.step()

    def test_generation_consumes_raw_question_without_gold_prefix(self):
        _,model,tok=self.tiny();query='Which city is the capital of France?';seen=[]
        h=model.register_forward_pre_hook(lambda module,args,kwargs:seen.append(kwargs['input_ids'].detach().clone()),with_kwargs=True)
        try:result=generate_raw(model,tok,query,3)
        finally:h.remove()
        self.assertTrue(torch.equal(seen[0],tok(query,return_tensors='pt').input_ids))
        self.assertLessEqual(result['tokens'],3);self.assertEqual(len(result['token_ids']),result['tokens'])

    def test_sliding_window_dynamic_kv_prefix_survives_repeated_queries(self):
        cfg,model,tok=self.tiny()
        from memory_comparison_core import prefill,cache_payload_bytes,kv_bytes_per_token,prefix_fingerprint,assert_prefix_unchanged
        from ttft_timing import decode_cached_observed,FirstAnswerClock
        ids=tok.encode('A stored memory record. '*5)
        self.assertGreater(len(ids),cfg.sliding_window)
        cache=prefill(model,ids,chunk_size=7);fingerprint=prefix_fingerprint(cache)
        self.assertEqual(cache_payload_bytes(cache),len(ids)*kv_bytes_per_token(cfg))
        outputs=[]
        for _ in range(2):
            observer=FirstAnswerClock(tok,time.monotonic())
            result,cache=decode_cached_observed(model,tok,cache,tok.encode('Question?',add_special_tokens=False),observer,cap=2)
            outputs.append(result['token_ids']);self.assertEqual(cache.get_seq_length(),len(ids))
            assert_prefix_unchanged(cache,fingerprint)
        self.assertEqual(outputs[0],outputs[1])


if __name__=='__main__':unittest.main()
