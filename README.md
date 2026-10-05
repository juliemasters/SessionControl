# SessionControl

Research prototype for **Section B: session routing by probing independently stored AlphaEdit FFN deltas**.

Given a new question, load each session delta separately, generate a probe, score the question–probe pair, select a session, and generate the final answer under the selected delta. The user does not supply a session ID or answer a routing follow-up.

**Status:** routing mechanics are tested; reliable semantic routing is not established. A relevant hallucination can receive a high BERT score. This is research code, not a validated memory system.

## Architecture

```text
Section A (external): write each session -> save independent delta + version metadata

Section B (this repository):
question -> restore base -> load delta A -> probe A --+
         -> restore base -> load delta B -> probe B --+-> BERT relevance scores
         -> ...                                     |
                                                    v
                     all scores < 0.55? regenerate EVERY probe once
                                                    |
                     select highest score (near-tie band 0.02)
                                                    |
                     load selected delta -> final answer -> restore base
```

Near ties prefer the most recently **saved** delta. Probing does not change write recency. If scores remain low after the single retry, the prototype still chooses a session and exposes `still_low=True`; this is not a confidence guarantee.

## Files

| File | Purpose |
|---|---|
| `AlphaEditSessionStore.py` | Save/load versioned deltas, restore base weights, serialize inference; historical exact-tie router |
| `section_b.py` | BERT judge, near-tie routing, regenerate-all policy, GPU-1 guard |
| `DeltaProbeText.py` | Hugging Face generation callbacks; historical likelihood-scoring baseline |
| `routing_ladder_core.py` | Trial cohorts, ambiguity flags, tolerance selection, metrics |
| `h2_routing.py` | Synthetic benchmark and evaluation helpers |
| `routing_ladder_bank.json` | Authored synthetic session facts |
| `judge_training_data.py` | Synthetic relevance-training/validation examples; no trained weights |
| `demo_delta_routing.py` | CPU toy demonstration with mocked edits |
| `test_*.py` | Storage, restoration, routing, retry, and experimental-design tests |

The new `section_b.py` packages the existing experiment's routing policy into a reusable callback interface. It is unit-tested integration code; it has not itself been rerun with the large model. Context-augmented oracle-session experiments and Section A dataset extraction are outside this upload.

## Quick start: CPU mechanics test

Use Python 3.10+ (tested locally with Python 3.12). Install an appropriate PyTorch build for your system; the commands below use pip's default build.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell alternative: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -p 'test_*.py' -v
python demo_delta_routing.py
```

The toy demo mocks AlphaEdit writes. Passing it verifies software flow, not real-model recall or routing accuracy.

## Use existing session deltas with BERT

You must provide the exact unedited base model/tokenizer used for writing, a trained binary BERT relevance checkpoint (`id2label[1] == "relevant"`), and matching FFN deltas. No model weights, dataset conversations, BERT checkpoint, credentials, server paths, or experimental delta files are included.

```python
from section_b import require_gpu1
require_gpu1()  # BEFORE torch/transformers imports; physical GPU 1 only

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from AlphaEditSessionStore import SessionFFNMemory
from DeltaProbeText import DeltaProbeText
from section_b import BertProbeJudge, route_session

model_path = "models/base-model"       # supply the matching model
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float32
).to("cuda:0").eval()                  # logical 0 is physical GPU 1

store = SessionFFNMemory(
    model, ["model.layers.7.mlp.down_proj.weight"], "session_deltas"
)
text = DeltaProbeText(model, tokenizer, probe_tokens=256, answer_tokens=256)
judge = BertProbeJudge("checkpoints/bert-relevance", device="cuda:0")

def retry(model, question):
    return text.probe(model, "Recall a specific earlier detail before answering: " + question)

result = route_session(
    store, "Where do I live?", text.probe, judge.score, text.generate,
    regenerate=retry, threshold=0.55, tie_epsilon=0.02,
)
print(json.dumps(result, indent=2))
```

This example uses chat-style callbacks. **Match probe prompts to the prompts used in your memory-write experiment.** The earlier QA-key pilot used `Q: {question}\nA:`; its shared-memory readback used `My memory:`. Prompt changes can change recall dramatically. Retry must change the prompt or decoding policy; repeating identical deterministic generation will not help. Generation caps are configurable, and inputs must fit the model/judge context windows.

Expected checkpoint layout:

```text
session_deltas/
  A/v1.pt       # dict: exact FFN parameter name -> tensor delta
  A/v1.json     # session_id, version, parent_version, weight_names, saved_at_ns
  B/v1.pt
  B/v1.json
```

The highest numeric version is probed for each session. Deltas are relative to the same base weights, never added cumulatively across sessions. Use one store per shared model; do not mutate its model or checkpoints concurrently outside the store. A lock spans probing, scoring, selection and final generation. The base is restored even on failure. The implementation targets a single-process research worker.

Section A's optional `write_session` imports the upstream `AlphaEdit` package lazily. Loading and probing existing deltas does not require that package. AlphaEdit is the editing algorithm, not the base language model. Install the upstream research implementation separately if you need to write new memories; its license and model-provider terms apply independently.

## Evaluation and interpretation

Track routing accuracy, inactive-session switch accuracy, active-session false-switch rate, final-answer accuracy, tie rate, and target score margin. Report random chance `1/N` and oracle-session answer recall separately. `routing_ladder_core.py` provides these helpers and marks semantically ambiguous questions. Wilson intervals are descriptive because paraphrases are correlated.

Historical two-session question-key tests reached 3/6 final routing across the tested K conditions, equal to the 50% chance baseline. Written-key recall improved with extra question keys, but held-out recall remained incomplete and BERT often gave both correct and unsupported probes scores near 1. These are small pilot observations, not general performance claims. Historical result artifacts are not bundled here.

Relevance scoring cannot establish that an answer came from a particular stored memory. The 0.55 retry threshold and 0.02 tie band are provisional. Recency can dominate when scores saturate. Context-assisted answering is a separate experiment and must not be presented as successful question-only weight recall.

## Scope of this upload

This repository contains project-authored routing code, synthetic data, and documentation only. Large assets and local artifacts are excluded through `.gitignore`. No new distribution license is assigned by this upload.
