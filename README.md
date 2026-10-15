# SessionControl

Independent AlphaEdit session memories with question-only activation routing, compact delta storage, and measured comparisons against matched RAG and full-bank KV caching.

The final experiment measures **8, 16, 32, 64, 128, 256, and 512 real LongMemEval sessions** on physical GPU 1. Every bank size uses the same **eight original questions**, exactly as supplied at write time. No paraphrases or BERT scorer are used.

| At 512 stored sessions | Compact AlphaEdit | Matched RAG |
| --- | --- | --- |
| Correct source selected | 8/8 | 8/8 |
| Whole-answer exact match | 7/8 | 7/8 |
| Mean word-token F1 | 93.27% | 98.86% |
| Warm median pipeline TTFT | 810 ms | 3,458 ms |
| Retained bank size | 109.29 MiB, disk | 6.84 MiB, CPU/disk |
| Peak allocated GPU memory | 30.15 GiB | 31.15 GiB |

Memory peaks include the common float32 Llama weights; retained bank bytes exclude them. These are separate storage tiers, not quantities to add. Full-bank KV ran through 32 sessions and exceeded the model's native context capacity at 64 and above. RAG retains a smaller bank than compact deltas.

The first 64 sessions have official Q/A annotations. The additional 448 are real conversations with verified, newly constructed extractive write annotations. Accuracy remains an **exploratory eight-question mechanism test on previously inspected questions**, not a 512-question or held-out generalization result. Repeated passes supply timing samples, not additional accuracy items.

[Architecture](docs/ARCHITECTURE.md) · [Results](docs/RESULTS.md) · [Reproduction](docs/REPRODUCING.md) · [Scaling report](docs/results/scaling-real512-compact-v1/report.html) · [TTFT/F1 report](docs/results/scaling-real512-ttft-v1/report.html)

![SessionControl architecture](docs/figures/sessioncontrol-architecture.svg)

## Write, route, generate

1. Write one independent memory using the **full answer-bearing conversation, original question, and supplied answer**. The key is `Q: {question}\nA:` and the target is the answer plus EOS. AlphaEdit updates FFN layers 13–15 and restores base weights after each write.
2. Capture the base LLM's final-prompt-token FFN inputs. Score every delta independently using

   `score(q,s) = sum_l ||delta_W[s,l] @ k[l](q)||² / sum_l ||delta_W[s,l]||_F²`.

3. Select the maximum, with lexical session-ID tie breaking. Apply only that delta, generate the answer greedily with EOS/next-turn stopping, and restore base weights.

Compact storage saves two factor vectors per layer. The 64 legacy dense memories passed numerical reconstruction, all 64 own-question token-sequence checks, and 32 routing checks before expansion. This is an empirically validated approximation, not globally lossless compression. The one-key compact writer is a separate storage implementation.

## Final code

| File | Role |
| --- | --- |
| `experiment_real512_compact.py` | Connected write, routing, generation, RAG/KV comparison and resource measurement |
| `compact_session_memory.py` | Factor storage, compact writer and activation energy scoring |
| `experiment_real512_ttft.py` | Read-only TTFT/F1 replay of a completed bank |
| `ttft_timing.py` | First visible answer timing and reusable cached decoding |
| `experiment_utils.py` | Shared IO, physical GPU 1 selection and evaluation |
| `session_write.py`, `question_key.py` | Grounded write request and exact question key |
| `build_real512_report.py`, `build_real512_ttft_report.py` | Completion-gated graphs and standalone HTML |
| `scripts/verify_release.py` | CPU verification of aggregate tables, hashes and data exclusions |

Earlier BERT prototypes, pilot runners, duplicate historical reports and temporary workers have been removed from this working tree. Third-party AlphaEdit/ROME utilities and the dense store remain because compact validation and writing depend on them.

## Install and check

Use Python 3.12. Install the appropriate PyTorch 2.6.0 CUDA or CPU wheel, then:

```bash
python -m pip install -r requirements.txt
CUDA_VISIBLE_DEVICES="" python -m unittest discover -v
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
python scripts/verify_release.py
```

Running the model experiment requires local checkpoints, the pinned dataset, a projector, and the completed 64-session source bank; these are excluded from GitHub. See [reproduction requirements](docs/REPRODUCING.md). Aggregate reports can be viewed without those artifacts. Per-item questions, reference answers, generated outputs and raw session records remain local.

The cleanup extracts shared helpers without changing the scoring or generation algorithms. Published measurements come from the completed pre-cleanup GPU runs; packaging is not another GPU experiment. See [validation](docs/VALIDATION.md) and [artifact provenance](docs/EXPORT.md).

Third-party attribution and license scope: [NOTICE.md](NOTICE.md).
