# Method evaluated by the top-k experiment

The treatment is an activation-key prefilter followed by the original
SessionControl delta-response scorer. It preserves the saved editing results
and each dataset's original answer-generation procedure.

1. Reuse independent compact session deltas at FFN layers 13, 14 and 15.
2. Build normalized FP32 keys from the unedited model's write-question
   activations, concatenated across those layers.
3. Capture an incoming question's activations once and compare its concatenated
   key with all stored keys by cosine similarity.
4. Retain the top-k distinct session IDs. LongMemEval can have multiple written
   questions per session, so its coarse session score is the maximum similarity
   over that session's question keys.
5. Run the unchanged original delta-response scorer on only those candidates,
   apply one selected delta, generate the answer, and restore base weights.

At k=1 the coarse retrieval determines the selected session. At larger k the
original delta-response scorer can override the coarse winner. This is an
exact-write-question memory experiment, not a paraphrase/generalization claim.

## Two separate timing regimes

- **Preloaded factors:** `experiments/zsre_topk/` freshly compares full scan and
  k=1/8/16/32 at every original checkpoint, ten inference passes each. Both arms
  preload the same complete factor bank on CPU. This isolates the reduced
  candidate-scoring/transfer work; it does not claim fewer candidate file loads.
- **Streaming factors:** `experiments/topk_streaming.py` queues a separate
  largest-bank comparison of full scan versus k=8, ten passes per dataset.
  The factor bank retains only metadata. Each candidate lookup loads a factor
  file; application reloads the selected factor once. The report audits N+1
  versus min(k,N)+1 deserialize calls per request. Warm OS cache is allowed;
  these are filesystem calls rather than cold physical-device read counts.

Both regimes count retained factor storage plus the activation-key index, and
keep disk, process RAM and GPU memory separate. The index adds about **77%**
to compact-factor storage in both current banks. Streaming changes factor
residency and query I/O, not the number of factor files retained on disk.

## Scope of the current datasets

- **LongMemEval:** 500 saved conversation deltas, Llama-3.1-8B-Instruct FP32,
  layers 13–15, 508 written question bindings. Largest-bank answer metrics use
  all 283 eligible questions; routing excludes 15 abstentions, leaving 268.
- **ZsRE transfer:** 1,000 saved Q/A deltas, Mistral-7B-v0.1 FP32, the same
  activation-prefilter algorithm and layers. It is labelled a model/dataset
  transfer rather than a literal Llama reproduction. These saved deltas cannot
  be applied to a different base model.

The supplied 512-session, BF16, two-GPU Llama prototype's 536/51 ms routing,
1,332/337 ms completion and 505/117 ms TTFT values remain historical references.
They are neither substituted results nor promised outcomes for the current
FP32 experiments. Its eight reconstructed questions do not establish broader
accuracy. Current results use their actual eligible cohorts and fresh timings.

The original frozen preloaded run is not modified by adding the streaming
supplement. The supplement waits for its successful completion and free GPU 1.
