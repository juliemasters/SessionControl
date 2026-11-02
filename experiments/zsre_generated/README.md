# ZsRE with our generated-answer, TTFT and memory metrics

Six conditions: SessionControl, plain cumulative AlphaEdit, matched top-one Q/A
RAG, full-bank KV cache, native WISE-Merge and native WISE-Retrieve. All use the
same first **1,000 official ZsRE edit records**, Mistral-7B-v0.1 revision and FP32
on physical GPU 1. Checkpoints: **8,16,32,64,128,256,500,1000**.

At each checkpoint, ask **all first-N original src questions**, ten fresh passes
per method. Score the **complete final generated answer** with the same custom
word-token F1 and normalized exact match as LongMemEval. Also measure visible and
first-generated TTFT, total query latency, output/cap rates, deployed bank disk
bytes, sampled process RAM and peak allocated/reserved GPU memory. These are not
teacher-forced token metrics or ten independently trained model replicas.

Reuse all compatible saved independent factors and the Mistral-only preservation
projector. Build the missing cumulative AlphaEdit and native WISE prefixes.
Ordinary AlphaEdit keeps its continuous post-update key preservation accumulator;
an algebraically equivalent Woodbury solve uses the key factors rather than a
14336-square factorization. FP64 core solves followed by FP32 updates are
explicit; tests establish tractable dense-system parity, not full-size bitwise
FP32 trajectory identity. The value optimizer and fixed hyperparameters stay the
same as SessionControl, with complete alt targets plus EOS.

WISE's publication-era algorithm, optimizer, masking, augmentation and native
router remain unmodified. Its own target policy is retained. The N=500 Merge
evaluation copy uses native finalization; its training state is restored before
continuing to 1,000. No threshold tuning, oracle selection or forced-memory scores.

RAG and KV retain every exact question and complete counterfactual answer, using
the same fixed memory-record instruction and query suffix. Mistral base has no
chat template. Parametric methods use their raw ZsRE question keys. No question
paraphrases or evaluation answer tokens enter generation. RAG/KV are explicitly
supervised Q/A memory comparators. Native Mistral's 4,096-token sliding window
remains enabled; full-bank capacity, allocated cache payload and unavailable
checkpoints are recorded without truncation or retrieval fallback.

The previous LongMemEval run used Llama-3.1, so this dataset/model transfer is not
an isolated causal test of dataset alone. This is an exact-question supervised
memory experiment, not an official blind benchmark. No measured write-time
comparison is claimed. All scores/timings in this report are fresh.

```bash
python -m unittest tests.test_zsre_generated -v
python -m experiments.zsre_generated.prepare --out runs/zsre-generated-sixmethod-1000-v2
python -m experiments.zsre_generated.run launch --out runs/zsre-generated-sixmethod-1000-v2
```

The detached server driver verifies model/source/data hashes, runs functional
eight-item pilots without accuracy thresholds, builds the full trajectories,
executes all ten measurement passes with rotating method order, audits every
answer/timestamp and writes final HTML/CSV/plots only after complete success.
It waits for GPU 1 and never kills another job or falls back to GPU 0. It survives
chat disconnection while the server remains powered on; failures are explicit.
Raw datasets, generated answer transcripts, model weights and checkpoints stay
local. Only source and appropriately labeled aggregate reports belong on GitHub.
