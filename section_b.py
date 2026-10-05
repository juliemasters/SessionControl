"""BERT-scored Section B routing over already-written AlphaEdit deltas."""
from routing_ladder_core import low_confidence, select


class BertProbeJudge:
    """Binary relevance judge. Scores are not factual correctness probabilities."""

    def __init__(self, checkpoint, device='cuda:0'):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device).eval()
        if self.model.config.id2label.get(1) != 'relevant':
            raise ValueError('Checkpoint must identify class 1 as relevant')

    def score(self, question, probe):
        inputs = self.tokenizer(question, probe, truncation=False, return_tensors='pt').to(self.device)
        if inputs.input_ids.shape[1] > self.model.config.max_position_embeddings:
            raise ValueError('Judge context exceeded; refusing silent truncation')
        with self.torch.inference_mode():
            return self.model(**inputs).logits.float().softmax(-1)[0, 1].item()


def route_session(store, question, probe, judge, generate, *, regenerate,
                  threshold=0.55, tie_epsilon=0.02):
    """Probe all sessions, optionally regenerate all once, score, select, answer.

    probe/regenerate/generate: synchronous callback(model, question) -> str.
    judge: callback(question, candidate) -> relevance probability in [0,1].
    One store owns the model; external model/checkpoint mutation is unsupported.
    """
    import torch
    if not isinstance(question, str) or not question.strip():
        raise ValueError('A nonempty question is required')
    if not 0 <= threshold <= 1 or not 0 <= tie_epsilon <= 1:
        raise ValueError('Probability thresholds must be in [0,1]')
    with store._lock:
        training = store.model.training
        store.model.eval()
        try:
            store.deactivate()
            sessions = store.latest_sessions()
            if len(sessions) < 2:
                raise ValueError('At least two saved session deltas are required')
            versions = dict(sessions)
            recency = {sid: store._saved_at_ns(sid, ver) for sid, ver in sessions}

            def run_probes(callback):
                values = {}
                for sid, ver in sessions:
                    candidate = store.run(sid, ver, callback, question)
                    if not isinstance(candidate, str) or not candidate.strip():
                        raise ValueError(f'Empty or invalid probe for {sid}')
                    values[sid] = candidate
                return values

            with torch.inference_mode():
                initial = run_probes(probe)
                initial_scores = {sid: float(judge(question, text)) for sid, text in initial.items()}
                regenerated = low_confidence(initial_scores, threshold)
                candidates = run_probes(regenerate) if regenerated else initial
                scores = ({sid: float(judge(question, text)) for sid, text in candidates.items()}
                          if regenerated else initial_scores)
                still_low = low_confidence(scores, threshold)
                selected, band = select(scores, recency, tie_epsilon)
                answer = store.run(selected, versions[selected], generate, question)
            return dict(question=question, initial_probes=initial, initial_scores=initial_scores,
                        probes=candidates, scores=scores, regenerated=regenerated,
                        still_low=still_low, selected_session=selected,
                        selected_version=versions[selected], tied=len(band)>1,
                        tie_candidates=band, answer=answer)
        finally:
            store.deactivate()
            store.model.train(training)


def require_gpu1():
    """Call BEFORE importing torch/transformers; logical cuda:0 becomes physical GPU 1."""
    import os
    import subprocess
    output = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    devices = dict(line.strip().split(', ', 1) for line in output.splitlines() if line.strip())
    if '1' not in devices:
        raise RuntimeError('Physical GPU 1 unavailable; no fallback permitted')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = devices['1']
    return devices['1']
