"""Native source, teacher-forcing metric and trusted checkpoint restore contracts."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from experiments.wise_paper.native import VENDOR,modules,evaluator,restore


class NativePaperTests(unittest.TestCase):
    def test_unmodified_publication_era_code_and_paper_hyperparameters(self):
        native=modules()
        hp=native['wise_hparams'].WISEHyperParams.from_hparams(str(VENDOR/'hparams/WISE/mistral-7b.yaml'))
        self.assertEqual((hp.edit_lr,hp.n_iter,hp.mask_ratio),(1.,70,.2))
        self.assertEqual((hp.alpha,hp.beta,hp.gamma,hp.act_ratio),(5.,20.,10.,.88))
        self.assertEqual(hp.inner_params,['model.layers[27].mlp.down_proj.weight'])
        self.assertTrue(native['WISE'].WISE.__module__.startswith('_paper_easyeditor.'))
        self.assertEqual(native['wise_main'].apply_wise_to_model.__module__,'_paper_easyeditor.models.wise.wise_main')

    def test_official_metric_is_teacher_forced_and_locality_returns_tokens(self):
        import torch
        metric=evaluator()
        class Batch(dict):
            def to(self,device):return self # Synthetic CPU-only transport for the native CUDA helper.
        class Tok:
            pad_token_id=0;padding_side='left'
            def encode(self,text):return [1]+[{'who':3,'knows':4,'gold':6}[w] for w in text.split()]
            def __call__(self,texts,**kwargs):
                if isinstance(texts,str):texts=[texts]
                ids=[self.encode(s) for s in texts];maximum=max(map(len,ids))
                values=[[0]*(maximum-len(row))+row for row in ids]
                return Batch(input_ids=torch.tensor(values),attention_mask=torch.tensor([[int(v!=0) for v in row] for row in values]))
        class Model:
            seen=[]
            def __call__(self,input_ids,**kwargs):
                self.seen.append(input_ids.tolist());b,l=input_ids.shape;logits=torch.zeros(b,l,12)
                for i in range(l-1):logits[:,i].scatter_(1,input_ids[:,i+1,None],10.)
                return SimpleNamespace(logits=logits)
            def generate(self,**kwargs):raise AssertionError('Official primary metric must not be described as free generation')
        tok=Tok();model=Model();hp=SimpleNamespace(max_length=30,use_chat_template=False)
        self.assertEqual(metric(model,tok,hp,'who knows','gold',0),[1.])
        self.assertEqual(metric(model,tok,hp,'who knows','gold',0,locality=True),[[6]])
        self.assertIn(6,model.seen[0][0]);self.assertEqual(tok.padding_side,'left')

    def test_native_saved_checkpoint_restores_merge_and_retrieve_logits(self):
        import torch
        from transformers import MistralConfig,MistralForCausalLM
        native=modules();torch.manual_seed(9);torch.set_num_threads(2)
        cfg=MistralConfig(vocab_size=64,hidden_size=16,intermediate_size=32,num_hidden_layers=3,
             num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,sliding_window=64)
        for retrieve in [False,True]:
            hp=native['wise_hparams'].WISEHyperParams.from_hparams(str(VENDOR/'hparams/WISE/mistral-7b.yaml'))
            hp.model_name='mistral-tiny';hp.inner_params=['model.layers[1].mlp.down_proj.weight'];hp.retrieve=retrieve
            editor=native['WISE'].WISE(hp,MistralForCausalLM(cfg).float().eval(),'cpu');a=editor.get_adapter_layer()
            with torch.no_grad():a.new_weight.add_(.2)
            a.weight_mask=torch.ones_like(a.new_weight,dtype=torch.long);a.editing_total_cnt=10;a.editing_mean_act.update(.01)
            if retrieve:
                a.memory_weight=[a.new_weight.detach().clone()-.1];a.memory_mean_act=[copy.deepcopy(a.editing_mean_act)]
            query=torch.tensor([[3,4,5,6]])
            with torch.no_grad():expected=editor(input_ids=query).logits.clone()
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'native.pt';editor.save(str(path))
                with torch.no_grad():a.new_weight.zero_()
                a.editing_mean_act.min_a=1e9;a.memory_weight=[];a.memory_mean_act=[]
                self.assertEqual(restore(editor,path,torch),10)
            with torch.no_grad():actual=editor(input_ids=query).logits
            self.assertTrue(torch.equal(expected,actual))


if __name__=='__main__':unittest.main()
