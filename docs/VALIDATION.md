# Validation scope

The published compact512 run records successful completion, restored base weights, unchanged original dense deltas, unchanged factors and a frozen protocol/code. Its compact validation checks 64 own-question token sequences, 32 panel routing selections and the numerical reconstruction gate. The later TTFT replay records identical emitted token IDs and selected sources for all 408 corresponding trials.

Cleanup extracted common IO/GPU/evaluation helpers and grounded question-key construction from retired scripts. It retained the scoring, compact solver, cache decoding and answer-stopping algorithms. No new neural benchmark was run during packaging; recorded measurements and original source hashes remain historical.

CPU validation checks numerical factor scores against materialized deltas, the compact solver against a dense linear solve, answer boundaries, reusable KV-cache correctness, timing observers against unchanged generated tokens, frozen panel/repeat rules and cohort reconstruction. Public release checks validate hashes, aggregate table consistency, memory tiers, timing bounds and the exclusion of per-item data. Detailed question/answer records remain local.


```bash
CUDA_VISIBLE_DEVICES="" python -m unittest discover -v
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
python scripts/verify_release.py
```

GPU reruns require the private source bank and checkpoints described in [REPRODUCING.md](REPRODUCING.md). CPU tests do not certify end-to-end GPU execution of the refactored packaging version.

The final aggregate-only release passed 23 core tests and 11 preparation/release tests. All 19 final application/helper modules imported with CUDA disabled. Public checks cover 33 exported artifacts and 34 measured condition rows. Local detailed evidence checks covered 816 recorded answers from the same eight unique original questions; those records are not published.
