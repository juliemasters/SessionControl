"""First generated/visible answer timestamps, independent of reference labels."""
import math
import statistics
import time
from contextlib import contextmanager


class FirstAnswerClock:
    def __init__(self, tokenizer, started, *, clock=time.monotonic):
        self.tokenizer=tokenizer;self.started=started;self.clock=clock
        self.ids=[];self.first_generated_at=None;self.first_visible_at=None
        self.tokens_before_visible=None

    def observe(self, ids):
        if self.first_generated_at is None:self.first_generated_at=self.clock()
        if self.first_visible_at is not None:return
        self.ids.extend(int(t) for t in ids)
        text=self.tokenizer.decode(self.ids,skip_special_tokens=True)
        if text.strip():
            self.first_visible_at=self.clock();self.tokens_before_visible=len(self.ids)

    def result(self):
        return dict(ttft_seconds=None if self.first_visible_at is None else self.first_visible_at-self.started,
                    first_generated_token_seconds=None if self.first_generated_at is None else self.first_generated_at-self.started,
                    tokens_before_visible=self.tokens_before_visible)


class FirstAnswerStreamer:
    """HF greedy generation sends the prompt first; it is never an answer token."""
    def __init__(self, observer):self.observer=observer;self.prompt_seen=False
    def put(self, value):
        if not self.prompt_seen:self.prompt_seen=True;return
        self.observer.observe(value.reshape(-1).tolist())
    def end(self):pass


@contextmanager
def observe_generate(model, observer):
    original=model.generate
    def generate(*args,**kwargs):
        if 'streamer' in kwargs:raise ValueError('Unexpected existing streamer')
        return original(*args,**kwargs,streamer=FirstAnswerStreamer(observer))
    model.generate=generate
    try:yield
    finally:model.generate=original


def decode_cached_observed(model, tokenizer, cache, suffix_ids, observer, *, cap=64):
    """Same greedy/cache/crop algorithm as decode_cached, with a token observer."""
    import torch
    from memory_comparison_core import cache_payload_bytes,prefix_fingerprint,assert_prefix_unchanged
    from answer_only_generation import answer_boundary
    start_length=cache.get_seq_length();fingerprint=prefix_fingerprint(cache)
    eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
    generated=[];reason='cap';maximum=cache_payload_bytes(cache);next_ids=suffix_ids
    begin=time.monotonic()
    with torch.inference_mode():
        for step in range(cap):
            position=cache.get_seq_length();x=torch.tensor([next_ids],device=model.device)
            result=model.model(input_ids=x,past_key_values=cache,use_cache=True,
                               cache_position=torch.arange(position,position+len(next_ids),device=model.device))
            cache=result.past_key_values
            logits=model.lm_head(result.last_hidden_state[:,-1,:]).float()
            token=int(logits.argmax(-1).item());generated.append(token)
            observer.observe([token])
            maximum=max(maximum,cache_payload_bytes(cache));del result,logits,x
            if token in eos:reason='eos';break
            emitted=tokenizer.decode(generated,skip_special_tokens=True)
            if answer_boundary(emitted)[1]:reason='next_turn';break
            next_ids=[token]
    if model.device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.monotonic()-begin
    emitted=tokenizer.decode(generated,skip_special_tokens=True).strip()
    returned,boundary=answer_boundary(emitted);cache.crop(start_length)
    assert_prefix_unchanged(cache,fingerprint)
    return dict(text=returned,emitted_text=emitted,tokens=len(generated),token_ids=generated,
                finish_reason=reason,hit_cap=reason=='cap',boundary_removed=boundary is not None,
                max_query_cache_payload_bytes=maximum,decode_seconds=elapsed),cache


def ttft_summary(trials):
    warm=[t for t in trials if t['repeat']>0]
    values=sorted(t['ttft_seconds'] for t in warm if t['ttft_seconds'] is not None)
    if not values or len(values)!=len(warm):raise ValueError('Missing visible first token')
    if any(t['ttft_seconds']<0 or t['ttft_seconds']>t['query_seconds'] for t in trials):
        raise ValueError('TTFT outside measured query duration')
    return dict(warm_n=len(values),median_seconds=statistics.median(values),
                p95_seconds=values[math.ceil(.95*len(values))-1],
                first_pass_mean_seconds=statistics.mean(t['ttft_seconds'] for t in trials if t['repeat']==0),
                definition='Local query arrival to first nonempty decoded final-answer text; no network/queue. OS file cache not flushed.')
