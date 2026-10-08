import unittest
import torch
from types import SimpleNamespace
from memory_comparison_core import kv_bytes_per_token,cache_payload_bytes,prefill,decode_cached,token_f1,rank_vectors


class ComparisonTest(unittest.TestCase):
    def test_grouped_query_cache_formula(self):
        c=SimpleNamespace(hidden_size=4096,num_attention_heads=32,num_hidden_layers=32,num_key_value_heads=8)
        self.assertEqual(kv_bytes_per_token(c),262144)

    def test_retrieval_and_number_f1(self):
        vectors=torch.tensor([[1.,0.],[0.,1.]])
        self.assertEqual(rank_vectors(torch.tensor([0.,1.]),vectors,1)[0][0],1)
        self.assertEqual(token_f1('5000','500'),0.)
        self.assertEqual(token_f1('The Grand Ballroom','The Grand Ballroom'),1.)

    def test_chunked_cache_matches_full_prefix_and_reuses_safely(self):
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.manual_seed(0)
        config=LlamaConfig(vocab_size=24,hidden_size=16,intermediate_size=32,num_hidden_layers=2,
                           num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128,
                           bos_token_id=1,eos_token_id=2,pad_token_id=0)
        model=LlamaForCausalLM(config).eval();model.generation_config.eos_token_id=[2]
        ids=[1,3,4,5,6,7];suffix=[8,9]
        cache=prefill(model,ids,chunk_size=2)
        self.assertEqual(cache_payload_bytes(cache),len(ids)*kv_bytes_per_token(config))
        class Tok:
            def decode(self,ids,**kwargs):return ' '.join(str(i) for i in ids if i!=2)
        first,cache=decode_cached(model,Tok(),cache,suffix,cap=1)
        with torch.inference_mode():expected=model(input_ids=torch.tensor([ids+suffix])).logits[0,-1].argmax().item()
        self.assertEqual(first['token_ids'][0],expected)
        self.assertEqual(cache.get_seq_length(),len(ids))
        second,cache=decode_cached(model,Tok(),cache,suffix,cap=1)
        self.assertEqual(first['token_ids'],second['token_ids'])


if __name__=='__main__':unittest.main()
