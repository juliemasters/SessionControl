# Queued streaming-factor supplement

This separate experiment compares full scan with top-k 8 at the largest frozen
bank only: ZsRE has 1,000 edits and 1,000 queries; LongMemEval has 500 conversation
deltas and 283 eligible queries, including 15 abstentions. Two arms × ten actual
passes × (1,000 + 283) queries = **25,660 fresh inference requests**. It changes
factor loading to test whether fewer candidate factors are read. No existing
primary runtime source, input, factor, index or result is modified.

```bash
python -m experiments.topk_streaming prepare \
  --out runs/activation-topk-streaming-v1 \
  --primary runs/activation-topk-two-dataset-v1
python -m experiments.topk_streaming launch --out runs/activation-topk-streaming-v1
```

Use the existing experiment Python environment. `launch` detaches a queue
watcher. It requires both the primary root's `completion.json` to say
`completed: true` and `driver_status.json` to say `stage: completed`, and waits
for physical GPU 1 to be free. A recorded primary failure blocks the supplement
without starting GPU work. It never resumes or repairs the primary. Each
dataset/method/pass runs in a fresh subprocess using the launch interpreter;
methods rotate by pass and query order matches the primary's deterministic
`question_order`. A failure ends the driver with an explicit failure record.

`prepare` freezes copies of the designs, normalized requests and original input
manifests, plus index metadata and hashes. It reuses the source factor and index
paths without creating duplicate index tensors. The supplement adds its own
module to the frozen execution-source hashes. Original completed runs remain
the factor `source_run`, so the original integrity checker still applies.
Every worker verifies all frozen sources and artifacts. Setup validates every
factor hash and measures logical factor tensor bytes by loading one at a time.

`LazyFactorBank` implements `Mapping`: iteration yields only SIDs, and one
lookup deserializes one factor payload. There is no tensor cache or preload.
The unchanged `ActivationRouter` therefore loads N payloads for full-scan fine
scoring and min(8, N) for top-k scoring. Both arms reload the selected payload
once for application, making total counts N+1 and min(8,N)+1. Every load checks
FP32, layer names, finite values, shapes and ranks. LongMemEval payloads use the
original canonicalization before the original rank-one/rank-two scorer.

Models/backbones, precision (FP32), saved factors, prompts, greedy 128-token
generation, stopping, restore placement, two one-token warmups, visible-answer
TTFT observer, source-routing flags, final-answer F1/exact match and process/GPU
peak measurement match the frozen primary. The only changed treatment is factor
residency and consequent candidate filesystem/deserialization work. Support
labels enter after timed routing and generation. LongMemEval abstentions count
in answer and latency metrics and are excluded from source-routing denominators.

Warm operating-system file cache is allowed; no caches are flushed. Recorded
load counts and bytes describe application filesystem/deserialization calls and
serialized file sizes, **not physical cold-disk I/O**. Setup and original offline
index building are outside query timing. Deployment resource accounting includes
the shared factor files and, for top-k, the reused CPU key index. Logical total
factor bytes are separate from zero resident factor-cache bytes and from measured
process RSS. The allocator can retain freed memory; no exact RSS reduction is
inferred from the cache policy.

Completed runs produce root-level `comparison.csv/json`, `per_pass.csv`,
`question_outputs.csv`, `summary.md`, `report.html`, and PNG/SVG/PDF plots for
F1/exact/source routing, TTFT/query/routing latency, disk/RAM/GPU, and factor file
read counts/bytes. Error bars are sample SD across ten pass-level metrics on a
fixed trained bank. The report verifies all 40 cells, every expected query,
raw answer/routing/timing fields and streaming read-count invariants before
writing `completion.json` with `completed: true`.

Manual entry points are `drive --out ROOT`, `evaluate --out ROOT/DATASET
--method full_scan|topk_8 --repeat 0..9`, and `report --out ROOT`. They write only
under the supplement root. Do not point `--out` at the primary run.
