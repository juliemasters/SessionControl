# WISE-Retrieve on the matched 500-conversation experiment

This experiment adds **official WISE-Retrieve** to the completed SessionControl, plain AlphaEdit, matched Q/A RAG and full-bank KV comparison. It trains a new WISE trajectory; AlphaEdit deltas cannot be reused as WISE weights. Existing baseline aggregates are reused explicitly.

Implementation source: [zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit/tree/4c109870955a4522ac3d7cf10ad00f34de8e4f0d/easyeditor/models/wise), pinned to commit `4c109870955a4522ac3d7cf10ad00f34de8e4f0d`. The unmodified text WISE dependencies are vendored under [`third_party/wise`](../../third_party/wise), with the original MIT license, source URL and per-file SHA-256 manifest. This comparison does not substitute a newly implemented router for WISE.

## Protocol

| Stored conversations | Official questions per pass | Q/A bindings written |
|---:|---:|---:|
| 8 | 8 | 8 |
| 16 | 16 | 16 |
| 32 | 32 | 32 |
| 64 | 64 | 64 |
| 128 | 98 | 130 |
| 256 | 162 | 262 |
| 500 | 283 | 508 |

- Exact same conversation order, official questions, complete reference answers and base Llama-3.1-8B-Instruct as the preceding experiment. No paraphrases or invented questions.
- Each write includes the **full conversation + literal question + complete answer**, with question-only and grounded teaching templates. A deterministic locality row comes from a 200-turn LongMemEval distractor pool that excludes every official supporting conversation. Only these auxiliary locality turns are capped at 128 tokens; stored conversations and answers are never truncated.
- WISE's recommended Llama-3.1 settings: layer 29 FFN down projection, SGD learning rate 0.9, 30 iterations, mask ratio 0.2, margins 5/10/10, activation ratio 0.88, norm constraint 1.0, `retrieve=True`, `replay=False`, `save_freq=500`, `merge_freq=1000`. Sequential editing is enabled.
- Checkpoints count **conversations**; WISE save/merge frequencies count individual Q/A bindings. The first completed shard appears after binding 500. This is the official configuration, not a tuned one-shard-per-session variant.
- Native WISE inference chooses its main/side/sharded memory through its own activation mechanism. SessionControl's external delta router is not added to WISE.
- Ten **actual inference passes** on one trained trajectory, at every checkpoint. The N=8 functional pilot is the first of those ten passes. All eligible official questions are evaluated, in the same per-pass order as the previous experiment.
- Whole final-answer word-token F1 and normalized exact match only. No substring-containment credit, answer extraction, BERT scoring or source-selection accuracy.
- Same greedy 128-token generation cap, EOS/new-turn stopping, physical GPU 1, FP32 and TF32-disabled setting. Identical shared generation and timing helper files are used.
- TTFT is query arrival through WISE routing and prefill to the first nonempty final-answer text. Missing visible timings are counted, not imputed; empty responses still count in answer accuracy. Mean, median, p95 and complete query latency are exported.
- Separate deployed side-memory disk bytes, host RSS and GPU allocated/reserved peaks. Common base weights count in GPU peaks. Original unchanged weights are reconstructed from the common model; training-only restart state is excluded from deployed bank storage.

All methods see the reference answers during writing. For multi-session items the full global answer is taught to each associated conversation. The bank is selected from official support metadata. This is an **exact-question supervised memory test**, not an official blind LongMemEval score or an unseen-question reasoning evaluation. Changing eligible cohorts can change difficulty. The original64, new-item, newly eligible, question-type and abstention groups are exported separately.

WISE retains its own recommended layer/optimizer/router rather than AlphaEdit's layer settings. This compares complete systems, not only the effect of session routing. Baseline latency measurements come from the previous completed run and are not interleaved with the new WISE measurements. Ten inference repetitions do not estimate variation across ten independently trained models.

## Integration adaptations

The official algorithm files are unchanged. The integration uses the same bare `Q: ...\nA:` key as the existing comparison instead of WISE's optional chat wrapping. Token assembly preserves the complete answer, punctuation and EOS.

`EfficientWISE` computes the **same per-sequence masked cross entropy** while projecting only supervised hidden states through the vocabulary head. During a fixed write batch it also caches only the unchanged frozen decoder prefix before layer 29. The edited layer and all following layers are recomputed at every optimizer step. No post-edit KV state is cached during optimization. Tests compare losses and gradients against the official full calculation across successive updates.

A tensor-only checkpoint stores active side weights, completed shard weights and activation thresholds. Unchanged original matrices come from the same base model. Resume checkpoints separately include the exact gradient mask, RNG and edit-history state. Tests verify native routing/logit parity after a checkpoint round trip.

## Files

| File | Role |
|---|---|
| `core.py` | Frozen dataset reconstruction, official WISE imports, complete-answer tensors, exact execution optimizations and checkpoint codec |
| `worker.py` | Sequential writes and matched GPU 1 inference measurements |
| `run.py` | Detached driver, resume, TTFT-first progress page and aggregate-result export |
| `report.py` | Completion-gated audit, ten-pass mean/sample SD, CSVs, HTML and plots |
| `bootstrap_reference.py` | Reconstruct matching local input files from the official data and published hashes; reuse measured baseline aggregates |
| `config/` | Unmodified official model hyperparameters and the initial 64 source-session IDs |
| `reference/` | Previous measured baseline aggregates and input/model checksums; no raw conversations or model weights |
| `../../tests/test_wise_qa500.py` | Real tiny-Llama loss/gradient parity, checkpoint routing/logit parity and full-write contract tests |

## Reproduce

Use the repository's pinned requirements and a matching PyTorch CUDA installation. The original dataset and base-model weights are supplied separately. `reference/metadata.json` records their required checksums.

```bash
python -m unittest tests.test_wise_qa500 -v

# Reconstruct inputs that match the previously measured baseline run.
# This copies existing aggregate scores; it does not claim to rerun the baselines.
python -m experiments.wise_qa500.bootstrap_reference \
  --data data/longmemeval_s.json \
  --model models/Llama-3.1-8B-Instruct \
  --out runs/wise-reference-inputs

python -m experiments.wise_qa500.run prepare \
  --data data/longmemeval_s.json \
  --model models/Llama-3.1-8B-Instruct \
  --reference-run runs/wise-reference-inputs \
  --out runs/wise-qa500-v1

python -m experiments.wise_qa500.run launch --out runs/wise-qa500-v1
```

An existing completed reference-run directory may be used directly instead of the bootstrap command. GPU workers resolve physical index 1 to its UUID before importing CUDA and refuse GPU 0 or CPU fallback. A busy GPU is waited for, never terminated. The detached job survives terminal disconnection while the GPU host/container remains running.

Monitor `progress.html`, `status.json`, `ledger.json` and `driver.log` in the run directory. TTFT values appear as soon as complete cells exist, with explicit pass counts; partial values are provisional. On success, `report.html`, `comparison.csv`, per-pass and raw local answer CSVs, and accuracy/TTFT/memory figures are written. Plots and a final mean are created only after every WISE pass/checkpoint is audited. KV infeasibility from the reference study remains unavailable.

The driver stages only aggregate results and figures under `docs/results/wise-qa500-v1`. Raw conversations, answer transcripts, model weights and restart files are never staged for GitHub. A separate publication step is required to push the staged results; local completion does not claim GitHub publication.

## Status

Implementation and CPU integration tests are available. The real GPU 1 pilot and full experiment are tracked separately; no WISE accuracy result is claimed until measured. The previous baseline results under `reference/` are not WISE results.

## Citation

Peng Wang et al. *WISE: Rethinking the Knowledge Memory for Lifelong Model Editing of Large Language Models.* NeurIPS 2024. [Paper](https://arxiv.org/abs/2405.14768).
