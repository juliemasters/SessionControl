"""Synthetic routing demo. Toy scores validate flow, not semantic accuracy."""
from tempfile import TemporaryDirectory
from unittest.mock import patch
import torch
from torch import nn
import AlphaEditSessionStore as module

LABELS = ["unknown", "guitar", "piano"]
WEIGHT = "down_proj.weight"


class TinyDialogueFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(1, 3, bias=False)
        nn.init.zeros_(self.down_proj.weight)

    def answer(self):
        return LABELS[self.down_proj(torch.ones(1, 1)).argmax().item()]


def fake_alphaedit(model, tok, requests, hparams, cache_c=None, P=None):
    with torch.no_grad():
        model.down_proj.weight.add_(requests[0]["delta"])
    return model, cache_c


def populate(store):
    for session, label in [("A", "guitar"), ("B", "piano")]:
        delta = torch.zeros(3, 1)
        delta[LABELS.index(label), 0] = 10
        store.write_session(session, None, [{"delta": delta}], None, None, None, 1)


def probe(model, question):
    return model.answer()


def score(question, candidate):
    # Toy scorer has no access to session IDs or expected answers.
    return 1.0 if candidate in question.lower() else 0.0


def generate(model, question):
    return f"You play {model.answer()}."


def main():
    with patch.object(module, "apply_AlphaEdit_to_model", fake_alphaedit), TemporaryDirectory() as directory:
        model = TinyDialogueFFN()
        store = module.SessionFFNMemory(model, [WEIGHT], directory)
        populate(store)
        for question in ["Do I play guitar?", "Do I play piano?", "What instrument do I play?"]:
            result = store.route_and_run(question, probe, score, generate)
            print(f"Q: {question}")
            print(f"Candidates: {[(p.candidate, p.score) for p in result.probes]}")
            print(f"Selected: {result.selected.session_id}; tied={result.tied}")
            print(result.answer)
            assert model.answer() == "unknown"


if __name__ == "__main__":
    main()
