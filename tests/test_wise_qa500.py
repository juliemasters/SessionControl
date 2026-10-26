"""Real tiny-Llama loss/gradient and WISE-state parity; no benchmark mocks."""
import copy
from pathlib import Path
import tempfile
import unittest

from experiments.wise_qa500.core import (HERE,editor_class,vendor_modules,tokens_for_binding,export_state,install_state,order)


class WISEIntegrationTests(unittest.TestCase):
    def editors(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.manual_seed(9);torch.set_num_threads(2)
        official,params=vendor_modules()
        hp=params.WISEHyperParams.from_hparams(str(HERE/'config/official-llama3.1-8b.yaml'))
        hp.model_name='tiny-llama';hp.inner_params=['model.layers[1].mlp.down_proj.weight'];hp.sequential_edit=True
        cfg=LlamaConfig(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=3,num_attention_heads=4,
                        num_key_value_heads=2,max_position_embeddings=128,attention_dropout=0.)
        model=LlamaForCausalLM(cfg).float().eval()
        a=official.WISE(copy.deepcopy(hp),copy.deepcopy(model),torch.device('cpu'))
        b=editor_class()(copy.deepcopy(hp),copy.deepcopy(model),torch.device('cpu'))
        return torch,a,b

    def test_masked_projection_and_frozen_prefix_match_official_gradients(self):
        torch,reference,fast=self.editors()
        torch.manual_seed(4)
        ids=torch.randint(3,60,(3,12));labels=torch.full_like(ids,-100)
        labels[:2,-4:]=ids[:2,-4:]
        tokens=dict(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels)
        for editor in [reference,fast]:
            editor.get_adapter_layer().editing=True;editor.get_adapter_layer().set_parameter_tunable()
        for step in range(3):
            for editor in [reference,fast]:editor.get_adapter_layer().new_weight.grad=None
            expected=reference._cal_ft_loss(tokens);actual=fast._cal_ft_loss(tokens)
            self.assertTrue(torch.allclose(actual,expected,atol=1e-6,rtol=1e-6))
            expected.backward();actual.backward()
            a=reference.get_adapter_layer().new_weight;b=fast.get_adapter_layer().new_weight
            self.assertTrue(torch.allclose(a.grad,b.grad,atol=1e-6,rtol=1e-5))
            with torch.no_grad():a.add_(a.grad,alpha=-.03);b.add_(b.grad,alpha=-.03)
        self.assertIsNotNone(fast._frozen_prefix)

    def test_deployment_checkpoint_preserves_native_routing_and_logits(self):
        torch,_,editor=self.editors();a=editor.get_adapter_layer()
        with torch.no_grad():a.new_weight.add_(.2)
        a.editing_mean_act.update(.05);a.memory_mean_act=[copy.deepcopy(a.editing_mean_act)]
        a.memory_weight=[a.new_weight.detach().clone()-.1];a.editing_total_cnt=508
        x=torch.tensor([[3,4,5,6]])
        with torch.inference_mode():expected=editor.model(input_ids=x).logits.clone()
        state=export_state(editor,500)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'memory.pt';torch.save(state,path)
            restored=torch.load(path,map_location='cpu',weights_only=True)
        with torch.no_grad():a.new_weight.zero_()
        a.memory_weight=[];a.editing_mean_act.min_a=1e9
        install_state(editor,restored)
        with torch.inference_mode():actual=editor.model(input_ids=x).logits
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(a.editing_total_cnt,508);self.assertEqual(len(a.memory_weight),1)

    def test_write_retains_full_context_literal_question_and_every_answer_token(self):
        class Tok:
            pad_token_id=0;eos_token='<EOS>'
            def encode(self,text,add_special_tokens=True):
                return ([1] if add_special_tokens else [])+[ord(c)+2 for c in text]
        tok=Tok();answer="A full answer with 'quoted text', {braces}, and all details."
        record=dict(context=[dict(role='user',content='Full conversation text '+('x'*3000))])
        binding=dict(question='Literal question?',answer=answer)
        tokens,act,deact=tokens_for_binding(tok,record,binding,'An unrelated locality turn.','cpu')
        for row in tokens['labels'][:2]:
            self.assertEqual(row[row!=-100].tolist(),tok.encode(' '+answer+tok.eos_token,add_special_tokens=False))
        grounded=''.join(chr(i-2) for i in tokens['input_ids'][1].tolist() if i>1)
        self.assertIn('x'*3000,grounded);self.assertIn('Q: Literal question?\nA:',grounded)
        self.assertEqual(act,[None]);self.assertEqual(deact,[None])

    def test_matched_per_pass_question_order(self):
        qids=['a','b','c','d']
        import hashlib
        expected=sorted(qids,key=lambda q:hashlib.sha256(f'3/128/{q}'.encode()).hexdigest())
        self.assertEqual(order(qids,3,128),expected)


if __name__=='__main__':unittest.main()
