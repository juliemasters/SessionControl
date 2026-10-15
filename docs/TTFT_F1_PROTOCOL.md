# TTFT and token F1 for the completed 512-session bank

This replay reuses the verified compact factors and matched RAG question-key
index from `scaling-real512-compact-v1`. It writes no memories. All model
inference runs on physical GPU 1 with Llama-3.1-8B-Instruct in float32, greedy
generation, a 64-token cap, and the existing EOS/next-turn stopping policy.

The same eight literal original questions are asked at N = 8, 16, 32, 64,
128, 256, and 512. The bank contains 64 original annotated sessions and 448
additional real sessions with new extractive write annotations. It is not a
512-question accuracy test or a held-out generalization estimate.

TTFT starts at local query arrival and ends when the first nonempty decoded
final-answer text is available. It includes:

- Deltas: fresh base activation capture, scoring every factor, selection,
  reading/applying the selected delta, question tokenization and generation.
- RAG: question embedding, CPU vector search, record/prompt assembly,
  tokenization, retrieved-context prefill and first-token generation.
- Warm KV: fresh question tokenization, suffix prefill and generation against
  an already constructed full-bank cache.

Network, request queues, model startup and memory writing/index creation are
excluded. Existing RAG vectors are on CPU and the encoder runs on GPU 1.
OS file caches are not flushed. The methods run sequentially with batch size
one; this is a local replay, not a production serving benchmark.

For each feasible KV bank, one first request additionally measures the actual
contiguous span from full-bank prompt construction/tokenization, capacity
checking and prefix prefill through the first visible answer. The model is
already resident. This single cold-prefix sample is not a model cold start or
a p95 estimate. Unavailable KV conditions have no measured TTFT or F1.

HF streamer prompt events are ignored. Cached decoding preserves the original
greedy/cache-crop algorithm and records the sampled token's availability.
Each emitted token sequence and selected source must match the prior
corresponding trial. First-token decoding adds small observer overhead.
Complete query work also includes final decoding and cleanup/integrity checks.

The first pass supplies eight unique accuracy items; the two later passes
supply 16 warm timing samples per condition. Report median and nearest-rank
p95 TTFT. Mean token F1 averages per-question casefolded word-overlap scores
with articles retained, matching the original custom metric. It is not
standard SQuAD normalization or a factual-correctness certificate. Report it
alongside source selection, whole-answer exact match and output inspection.

```bash
python experiment_real512_ttft.py \
  --source runs/scaling-real512-compact-v1 \
  --out runs/scaling-real512-ttft-v1

python build_real512_ttft_report.py --out runs/scaling-real512-ttft-v1
```

Use a new output directory and an idle physical GPU 1. The report builder
requires completed replay and integrity checks, verifies source/factor/code
hashes, recomputes metrics and creates a self-contained HTML report, CSV/JSON
tables and PNG/SVG/PDF figures. Model inputs and the completed source bank
remain read-only.
