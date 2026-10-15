import time
import unittest
from ttft_timing import FirstAnswerClock,FirstAnswerStreamer,observe_generate,decode_cached_observed,ttft_summary


class Tokenizer:
    def decode(self,ids,**kwargs):return ' '.join(str(t) for t in ids if t not in (0,2))


class TimingTest(unittest.TestCase):
    def test_report_refuses_unfinished_experiment(self):
        import json
        import tempfile
        from pathlib import Path
        from build_real512_ttft_report import build
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);(p/'completion.json').write_text(json.dumps(dict(completed=False)))
            with self.assertRaises(ValueError):build(p)
            self.assertEqual(list(p.glob('*.png')),[])

    def test_prompt_special_and_empty_tokens_are_not_visible_answers(self):
        import torch
        ticks=iter([11.,12.,13.]);observer=FirstAnswerClock(Tokenizer(),10.,clock=lambda:next(ticks))
        streamer=FirstAnswerStreamer(observer);streamer.put(torch.tensor([[1,9,9]]))
        self.assertIsNone(observer.first_generated_at)
        streamer.put(torch.tensor([0]));self.assertIsNone(observer.first_visible_at)
        streamer.put(torch.tensor([7]));self.assertEqual(observer.result()['ttft_seconds'],2.)
        self.assertEqual(observer.result()['tokens_before_visible'],2)

    def test_streaming_does_not_change_hf_generated_tokens(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.manual_seed(0)
        model=LlamaForCausalLM(LlamaConfig(vocab_size=24,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
            bos_token_id=1,eos_token_id=2,pad_token_id=0)).eval()
        x=torch.tensor([[1,3,4]])
        with torch.inference_mode():expected=model.generate(x,do_sample=False,max_new_tokens=4)
        observer=FirstAnswerClock(Tokenizer(),time.monotonic())
        original=model.generate
        with observe_generate(model,observer),torch.inference_mode():actual=model.generate(x,do_sample=False,max_new_tokens=4)
        self.assertTrue(torch.equal(expected,actual));self.assertEqual(model.generate,original)
        self.assertIsNotNone(observer.first_visible_at)

    def test_observed_cache_decode_preserves_tokens_and_reusable_prefix(self):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        from memory_comparison_core import prefill,decode_cached,cache_payload_bytes
        torch.manual_seed(0)
        model=LlamaForCausalLM(LlamaConfig(vocab_size=24,hidden_size=16,intermediate_size=32,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
            bos_token_id=1,eos_token_id=2,pad_token_id=0)).eval()
        prefix=[1,3,4,5,6,7];cache=prefill(model,prefix,chunk_size=2);payload=cache_payload_bytes(cache)
        expected,cache=decode_cached(model,Tokenizer(),cache,[8,9],cap=4)
        observer=FirstAnswerClock(Tokenizer(),time.monotonic())
        actual,cache=decode_cached_observed(model,Tokenizer(),cache,[8,9],observer,cap=4)
        self.assertEqual(expected['token_ids'],actual['token_ids'])
        self.assertEqual(cache.get_seq_length(),len(prefix));self.assertEqual(cache_payload_bytes(cache),payload)
        self.assertIsNotNone(observer.first_visible_at)

    def test_warm_percentiles_exclude_first_pass_and_invalid_bounds_fail(self):
        trials=[dict(repeat=0,ttft_seconds=50.,query_seconds=51.),
                dict(repeat=1,ttft_seconds=1.,query_seconds=2.),dict(repeat=2,ttft_seconds=3.,query_seconds=4.)]
        s=ttft_summary(trials);self.assertEqual(s['median_seconds'],2.);self.assertEqual(s['p95_seconds'],3.)
        self.assertEqual(s['warm_n'],2);trials[1]['query_seconds']=.5
        with self.assertRaises(ValueError):ttft_summary(trials)


if __name__=='__main__':unittest.main()
