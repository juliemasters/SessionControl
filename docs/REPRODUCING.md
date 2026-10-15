# Running the final implementation

Use Python 3.12 and the pinned dependencies in `requirements.txt`. Model inference explicitly selects physical GPU 1 by UUID before importing PyTorch and refuses fallback or an occupied GPU. CPU checks require no model weights.

## Required local artifacts

The final scaling runner extends a **completed 64-session dense source bank**. It does not fabricate that bank from published summaries. Provide a local workspace with:

- `models/Llama-3.1-8B-Instruct/` and `models/all-MiniLM-L6-v2/` checkpoints.
- `data/longmemeval_s.json`, SHA256 `08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894`.
- `results/memory-scaling-v1/` containing original `items.json`, `design.json`, `completion.json`, `evaluation.json`, and all 64 `deltas/<sid>/v1.pt` files. Completion must record successful writes, restored base weights and delta digests.
- `results/session-qa-routing-v2/items.json`, the original eight-question pilot panel.
- `results/k1-question-keys-v2/summary.json`, the historical K1 comparison reference.
- The layers 13–15 projector at the path and digest recorded in the source design.

These datasets, weights, dense deltas and projectors are not redistributed. The public result files are normalized evidence exports and cannot substitute for the original source bank's integrity files. A fresh checkout alone cannot reproduce the neural experiment. Retired pilot/BERT and source-bank bootstrap runners are intentionally absent from this final-only code tree.

For data auditing, `scripts/prepare_experiment.py` reconstructs the original 64-session cohort and 200-turn LongMemEval distractor corpus from the pinned dataset and `config/session_manifest.json`. `prepare_deep_projection_gpu1.py` builds a projector from that corpus; supply `--corpus`, `--layers 13,14,15`, and `--out` explicitly. Corpus reconstruction is not a completed source-bank write.

## Execute

Run from the repository root so upstream `globals.yml` is available. Use a new output directory for every run. Follow [REAL512_PROTOCOL.md](REAL512_PROTOCOL.md) for preparation, execution and host-RSS sampling:

```bash
python experiment_real512_compact.py \
  --data-root /path/to/local/workspace \
  --out runs/scaling-real512-compact-v1 --prepare-only

python experiment_real512_compact.py \
  --data-root /path/to/local/workspace \
  --out runs/scaling-real512-compact-v1

python build_real512_report.py --out runs/scaling-real512-compact-v1

python experiment_real512_ttft.py \
  --source runs/scaling-real512-compact-v1 \
  --out runs/scaling-real512-ttft-v1

python build_real512_ttft_report.py --out runs/scaling-real512-ttft-v1
```

The full protocol includes the optional parallel RSS sidecar; without sampling, host RSS is missing rather than inferred. Graph generation requires completed runs and matching original integrity artifacts. TTFT replay must reproduce every prior emitted token sequence and selected source. Frozen runtime code must still match when regenerating a TTFT report from that run.

The released runners have shared helpers extracted from earlier scripts. This packaging refactor was CPU checked, not rerun on the full neural benchmark. Historical embedded code digests refer to the original pre-cleanup source, not these refactored files. Published HTML and figures preserve the completed measured results; do not rewrite their provenance to suggest a new run.
