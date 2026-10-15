# Real-session scaling with compact storage

This experiment evaluates **N = 8, 16, 32, 64, 128, 256, 512** on physical
GPU 1. Every condition asks the same eight literal original LongMemEval pilot
questions. These questions were previously inspected. This is an exploratory
interference and resource experiment, not unseen-query generalization.

The first 64 memories reuse the completed official `single-session-user` bank.
The additional 448 memories use distinct, untouched conversation sessions from
LongMemEval's distractor haystacks. Their write annotations are deterministic
extractive quote-completion questions: a unique six-word prefix identifies a
complete sentence copied from a user turn. The full conversation is retained
at write time. Each added annotation is checked against its source user turn.
These are **new write annotations**, not 448 additional official benchmark
questions. The distractor question distribution changes above N = 64; report
that limitation when interpreting the curve.

## Compact implementation

Existing dense float32 deltas are approximated with one outer product per FFN
layer. Before expansion, all 64 existing own-question emitted token sequences
must match dense inference, and the eight-question top-1 routing results must
match at N = 8, 16, 32, and 64. The maximum relative Frobenius reconstruction
error must not exceed 0.002. These are empirical gates, not a global guarantee
of losslessness or preservation of all session facts.

New writes use the same grounded value optimization, answer-plus-EOS target,
50-step maximum, L2 = 0.1, and layers 13–15. Each write starts from the base.
For a single key with a zero preservation accumulator, the dense AlphaEdit
linear system admits the Sherman-Morrison form:

```
A = lambda I + (P k) kᵀ
A⁻¹ (P k) = (P k) / (lambda + kᵀ P k)
Delta W = residual [(P k)/(lambda + kᵀ P k)]ᵀ
```

This saves two float32 vectors per layer rather than a full matrix. It changes
the numerical solver, so this is reported as a **separate compact implementation**,
not an invisible replacement of the published dense condition. A numerical
unit test compares the factor formula with a dense solve in float64.

The factor vectors contain 216 KiB of raw parameters per memory for this
architecture; actual file bytes include serialization headers and are measured.
Common model weights and base-weight backups are reported separately.

## Matched methods and measurement

- Compact AlphaEdit: capture base-model FFN last-token keys; read and score
  every stored factor independently; apply the selected delta; generate;
  restore base weights. Routing never receives gold source labels or answers.
- Matched RAG: embed stored question keys with local MiniLM, search all keys,
  retrieve one full conversation plus stored Q/A, and generate with the base
  Llama model. This has the same supplied-Q/A memory supervision.
- Full-bank KV: prefill every full conversation plus stored Q/A into a
  reusable uncompressed float32 cache; append each original question and decode.
  No oracle retrieval, truncation, or sliding-window substitute is allowed.

Accuracy counts the **eight unique first-pass queries**, not 24 independent
items. Two repeated passes measure latency and reproducibility. Report source
selection, normalized whole-answer exact match, reference containment, token
F1 and mean rank/MRR. Full-bank KV has no explicit source-selection output.
These lexical answer metrics do not establish that every output statement is
factually correct. Immediate own-key write retention for the 448 added
annotations is a separate diagnostic, not 448 routing test questions.

Latency is serial batch-one wall time with CUDA synchronization. Compact
timings include routing and memory loading; RAG includes embedding/search,
tokenization, prefill and decode. KV prefix setup is measured separately from
reusable-query latency. First-pass and later-pass timings are distinguished.
OS cache is not flushed; method execution is sequential, not counterbalanced.
This is not a production serving-throughput benchmark.

Memory measures separate actual retained disk/CPU representations, actual
retained GPU KV tensor payload, and peak CUDA allocated/reserved bytes. A
sidecar samples process RSS every 0.5 seconds; sampled RSS includes shared
allocations, the full metadata manifest, allocator retention and base backups,
and is not an exact per-method incremental CPU peak. RAG storage grows with N.
These three tiers are never summed. Lifetime host high-water marks include
model loading and are excluded from recall RSS. RAG's serialized text/vector
bank is reported separately from the live CPU vector copy and GPU encoder.
Host RSS includes the vectors; CUDA peaks include the encoder. The harness retains the full 512-vector array at
smaller bank sizes; this is not an isolated deployment-memory benchmark.
The actual Llama-3.1-8B GQA architecture uses **256 KiB/token** for float32 KV.

When the full prefix exceeds the model's 131,072-token context limit, KV is
unavailable at that N. Report the actual encoded token requirement and the
calculated KV payload; do not report measured allocation, latency, or zero
accuracy for an unavailable condition. Dense delta payload above 64 is a
calculated counterfactual; only compact deltas are actually retained for all512.

## Execution

Use an environment with the existing local checkpoint, data and completed
64-session delta bank. The runner masks CUDA by physical GPU-1 UUID before
importing torch and refuses another occupied GPU or a CPU/GPU-0 fallback.

```bash
set -e
python experiment_real512_compact.py \
  --data-root /path/to/original/workspace \
  --out runs/scaling-real512-compact-v1 --prepare-only

python experiment_real512_compact.py \
  --data-root /path/to/original/workspace \
  --out runs/scaling-real512-compact-v1 &
sessioncontrol_gpu_pid=$!

python scripts/sample_scaling_rss.py \
  --pid "$sessioncontrol_gpu_pid" --out runs/scaling-real512-compact-v1 &
sessioncontrol_sampler_pid=$!

wait "$sessioncontrol_gpu_pid"
wait "$sessioncontrol_sampler_pid"

python build_real512_report.py --out runs/scaling-real512-compact-v1
```

The sampler waits for the runner's first status record and excludes status
records for conditions already marked complete. The recorded run used an
independent final-recall completion marker (`cpu_recall_end.json`) to exclude
later capacity-planning samples while the controller retained its last recall
status. No host RSS values were inferred from GPU measurements or lifetime HWM.

The report builder refuses to create graphs until the complete 512-session
experiment, compact validation and integrity checks succeed. It verifies
the frozen protocol and all saved factor hashes. Original dense artifacts
are read-only; new files are written only into the new run directory.

The 100% accuracy line is an aspirational target, not a forecast. Historical
K1 results are labeled as a different protocol, including its different
question set, writer and 16-token cap. They are not paired estimates of
current routing accuracy.
