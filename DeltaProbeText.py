"""Causal-LM callbacks for delta probing.

Score is mean candidate-token log likelihood given the question under the base
model: a compatibility baseline, not calibrated session-intent confidence.
"""
import torch


class DeltaProbeText:
    def __init__(self, model, tokenizer, probe_tokens=8, answer_tokens=128):
        self.model, self.tokenizer = model, tokenizer
        self.probe_tokens, self.answer_tokens = probe_tokens, answer_tokens

    def _prompt(self, question, short=True):
        instruction = ("Answer with only the key answer word or short phrase."
                       if short else "Answer the question directly.")
        messages = [{"role": "user", "content": f"{instruction}\n{question}"}]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        return f"{instruction}\nQuestion: {question}\nAnswer:"

    def _input(self, prompt):
        device = self.model.get_input_embeddings().weight.device
        return self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

    def _generate(self, model, question, short):
        encoded = self._input(self._prompt(question, short))
        output = model.generate(
            **encoded, do_sample=False,
            max_new_tokens=self.probe_tokens if short else self.answer_tokens,
            pad_token_id=self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        return self.tokenizer.decode(
            output[0, encoded["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    def probe(self, model, question):
        return self._generate(model, question, True)

    def generate(self, model, question):
        return self._generate(model, question, False)

    def score(self, question, candidate):
        # The router restores base weights before calling the scorer.
        prefix = self._input(self._prompt(question))["input_ids"]
        continuation = self._input(candidate)["input_ids"]
        if not continuation.shape[1]:
            raise ValueError("Candidate contains no tokens")
        ids = torch.cat([prefix, continuation], dim=1)
        logits = self.model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        prediction = logits[:, prefix.shape[1] - 1:-1, :].float().log_softmax(dim=-1)
        return prediction.gather(-1, continuation.unsqueeze(-1)).mean().item()
