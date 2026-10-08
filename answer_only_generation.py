"""Answer-boundary stopping for bare question-key recall, independent of gold data.

The emitted token sequence is retained for auditing. Returned text removes only
a newly generated conversation-turn delimiter; ordinary sentences, lists,
decimal points, and multiline answers are preserved.
"""
import re

TURN = re.compile(r'\n[ \t]*(?:Q|A|Question|User|Assistant)[ \t]*:', re.IGNORECASE)


def answer_boundary(text):
    match = TURN.search(text)
    return (text[:match.start()].strip(), 'next_turn') if match else (text.strip(), None)


def make_turn_stopper(tokenizer, prompt_width):
    import torch
    from transformers import StoppingCriteria

    class AnswerTurnStopping(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            tails = tokenizer.batch_decode(input_ids[:, prompt_width:].detach().cpu(),
                                           skip_special_tokens=True)
            return torch.tensor([TURN.search(text) is not None for text in tails],
                                device=input_ids.device, dtype=torch.bool)

    return AnswerTurnStopping()


def generate_answer_batch(model, tokenizer, questions, *, stop_at_turn=True, cap=64):
    """Only literal question strings reach generation; no references or sessions."""
    import torch
    from transformers import StoppingCriteriaList
    from question_key_k1_core import prompt

    if any(not isinstance(q, str) or not q.strip() for q in questions):
        raise ValueError('Nonempty literal question strings required')
    tokenizer.padding_side = 'left'
    encoded = tokenizer([prompt(q) for q in questions], return_tensors='pt',
                        padding=True, truncation=False).to(model.device)
    width = encoded.input_ids.shape[1]
    if width + cap > model.config.max_position_embeddings:
        raise ValueError('Question exceeds model capacity; no truncation')
    kwargs = dict(do_sample=False, max_new_tokens=cap, pad_token_id=tokenizer.pad_token_id)
    if stop_at_turn:
        kwargs['stopping_criteria'] = StoppingCriteriaList([make_turn_stopper(tokenizer, width)])
    with torch.inference_mode():
        generated = model.generate(**encoded, **kwargs)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    output = []
    for sequence in generated[:, width:].detach().cpu().tolist():
        length, reason = len(sequence), 'cap'
        # Reconstruct the first per-row finish; padded companions are excluded.
        for i, token in enumerate(sequence):
            if token in eos:
                length, reason = i + 1, 'eos'
                break
            if stop_at_turn:
                text = tokenizer.decode(sequence[:i+1], skip_special_tokens=True)
                if TURN.search(text):
                    length, reason = i + 1, 'next_turn'
                    break
        tokens = sequence[:length]
        emitted = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        text, boundary = answer_boundary(emitted) if stop_at_turn else (emitted, None)
        output.append(dict(text=text, emitted_text=emitted, token_ids=tokens,
                           tokens=length, answer_tokens=len(tokenizer.encode(text, add_special_tokens=False)),
                           finish_reason=reason, boundary_removed=boundary is not None,
                           hit_cap=reason == 'cap', stop_at_turn=stop_at_turn))
    return output


def reference_tokens_present(text, answer):
    """Evaluation-only containment with token boundaries: 500 does not match 5000."""
    from question_key_k1_core import normalize
    value, target = normalize(text).split(), normalize(answer).split()
    return bool(target) and any(value[i:i+len(target)] == target
                               for i in range(len(value)-len(target)+1))
