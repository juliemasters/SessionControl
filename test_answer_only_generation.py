import unittest
from answer_only_generation import answer_boundary, reference_tokens_present


class AnswerBoundaryTest(unittest.TestCase):
    def test_stops_at_generated_turn(self):
        self.assertEqual(answer_boundary('20\nQ: How many songs?\nA: 30'), ('20', 'next_turn'))
        self.assertEqual(answer_boundary('Luna\n\nUser: Thanks'), ('Luna', 'next_turn'))

    def test_preserves_multiline_lists_and_decimals(self):
        text = '1. Blue\n2. Green\nThe amount is $12.50.'
        self.assertEqual(answer_boundary(text), (text, None))
        self.assertEqual(answer_boundary('A: is part of the answer'), ('A: is part of the answer', None))

    def test_does_not_hide_same_line_additions(self):
        text = 'The Grand Ballroom in New York City.'
        self.assertEqual(answer_boundary(text), (text, None))

    def test_containment_has_number_boundaries(self):
        self.assertFalse(reference_tokens_present('5000', '500'))
        self.assertFalse(reference_tokens_present('110%', '10%'))
        self.assertTrue(reference_tokens_present('10% off', '10%'))
        self.assertTrue(reference_tokens_present('6:30 PM.', '6:30 pm'))
        self.assertFalse(reference_tokens_present('16:30 PM', '6:30 pm'))

    def test_stopper_has_per_row_results_and_ignores_prompt(self):
        import torch
        from answer_only_generation import make_turn_stopper
        class Tokenizer:
            def batch_decode(self, tails, **kwargs):
                return ['20\nQ:' if row[0].item() == 1 else 'University of Melbourne' for row in tails]
        stop = make_turn_stopper(Tokenizer(), 3)
        value = stop(torch.tensor([[7, 8, 9, 1], [7, 8, 9, 2]]), None)
        self.assertEqual(value.tolist(), [True, False])

    def test_real_generate_finishes_rows_independently(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, BatchEncoding, LogitsProcessor
        from answer_only_generation import generate_answer_batch

        class Tokenizer:
            pad_token_id = 0
            def __call__(self, prompts, **kwargs):
                return BatchEncoding(dict(input_ids=torch.tensor([[1, 1]]*len(prompts)),
                                          attention_mask=torch.ones((len(prompts), 2), dtype=torch.long)))
            def decode(self, ids, **kwargs):
                pieces = {0:'', 1:'', 2:'', 3:'20', 4:'\nQ:', 5:'University ', 6:'of ', 7:'Melbourne'}
                return ''.join(pieces[int(t)] for t in ids)
            def batch_decode(self, ids, **kwargs):
                return [self.decode(row, **kwargs) for row in ids]
            def encode(self, text, **kwargs):
                return list(text)

        class ForcedTokens(LogitsProcessor):
            def __call__(self, ids, scores):
                step = ids.shape[1]-2
                scores.fill_(-float('inf'))
                sequences = [[3, 4, 2, 2], [5, 6, 7, 2]]
                for row, sequence in enumerate(sequences):
                    scores[row, sequence[min(step, len(sequence)-1)]] = 0
                return scores

        model = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=8,
                                         bos_token_id=1, eos_token_id=2, pad_token_id=0)).eval()
        model.config.max_position_embeddings = 1024
        model.generation_config.eos_token_id = [2]
        generate = model.generate
        model.generate = lambda **kwargs: generate(**kwargs, logits_processor=[ForcedTokens()])
        output = generate_answer_batch(model, Tokenizer(), ['count?', 'university?'], cap=8)
        self.assertEqual(output[0]['text'], '20')
        self.assertEqual(output[0]['finish_reason'], 'next_turn')
        self.assertEqual(output[0]['tokens'], 2)
        self.assertEqual(output[1]['text'], 'University of Melbourne')
        self.assertEqual(output[1]['finish_reason'], 'eos')
        self.assertEqual(output[1]['tokens'], 4)


if __name__ == '__main__':
    unittest.main()
