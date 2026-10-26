# Latest completed activation-routing experiment

This directory contains the source for the completed `qa500-checkpoints-tenpass-v2` experiment: 500 stored LongMemEval conversations, checkpoints 8/16/32/64/128/256/500, and ten newly measured inference passes per checkpoint. The final checkpoint evaluates 283 completely supported official questions. All writes use the full conversation, exact question and complete reference answer. No BERT scorer is used.

The experiment's 17 execution-tracked source files match the final pass's saved SHA-256 hashes exactly. See [`SOURCE_MANIFEST.json`](../../SOURCE_MANIFEST.json). Shared method code lives at the repository root and under `AlphaEdit/`; the measured experiment workers live here. Reporting/launch helpers are the latest workspace versions and are identified separately from execution-verified files.

## Entry points and roles

| File | Role |
|---|---|
| `qa500_core.py` | Frozen write protocol, bank/question plan, full-answer requests and rank-one/rank-two activation scoring dispatch |
| `qa500_build.py` | Write independent compact deltas or cumulative plain AlphaEdit checkpoints |
| `qa500_eval.py` | Actual checkpoint inference, final-answer F1/exact match, TTFT and memory measurements |
| `full500_core.py` | The exact compact AlphaEdit solver, base restoration and general low-rank response-energy function used by the current writer/router; older capsule-preparation entry points in this dependency are not used by the current experiment |
| `comparison_core.py` | Atomic JSON/tensor saves and checksums |
| `official64_core.py` | Source snapshots and host-memory sampling used by current workers |
| `qa500_remeasure.py` | Ten additional inference passes on a saved frozen bank |
| `qa500_run.py` / `qa500_launch.py` | Full write/pilot/evaluation driver and detached launch |
| `qa500_report.py` | Audit and ten-pass final-answer accuracy, timing and memory report |
| `qa500_ttft_report.py` | TTFT-first progress table |
| `add_focused_memory_report.py` | RAG-versus-our-method disk/RAM/GPU figure added to the completed report |
| `qa500_progress.py` | Progress page |
| `qa500_resume_alignment.py` | Documented one-time punctuation-alignment fix migration from the actual run; not a normal launch step |

## Current method

1. Write one independent session delta from full context, exact official question and complete answer, restoring the base weights between writes. Layers 13–15 are edited. Store compact factors rather than a full dense tensor per session.
2. At recall, process `Q: {question}\nA:` through the base model once and capture the last-position FFN down-projection inputs.
3. Compute each delta's normalized local response energy from those captured activations. This is algebraic activation probing, without generating one candidate answer per delta.
4. Select the maximum-scoring delta, apply only it, generate the final answer and restore the base model.
5. Evaluate the entire returned final answer using word-token F1 and normalized exact match. Measure query-to-first-visible-answer TTFT and separate disk, host RAM and GPU tiers.

With `Delta W_s,l = U_s,l V_s,l^T` and last-question-position FFN input `h_l(q)`, the score is:

```text
score(s,q) = sum_l || Delta W_s,l h_l(q) ||_2^2
             / sum_l || Delta W_s,l ||_F^2
```

The implementation uses factors directly and sums over layers 13,14,15. `s` indexes a stored conversation, `q` is the question, `l` is the edited layer, `U,V` are compact update factors, and `Delta W` is the FFN weight update. The numerator measures the delta's local response to the question; dividing by total squared update norm reduces the advantage of a globally larger update. These activations come from the base model; this is not a full forward under every edited model.

## Using the captured experiment source

The method kernels and original experiment files are preserved rather than rewritten for packaging. Historical experiment launchers contain the original workspace/model/dataset/bank locations. Model weights, the official dataset, the null-space projector and stored deltas are **external assets**, not part of this source release. Copying the source does not copy a trained memory bank.

The canonical runnable workspace for the captured run is `/home/julie/Desktop/session_method_comparison`; shared method dependencies are `/home/julie/Desktop/draft_1`. For the existing workspace:

```bash
cd /home/julie/Desktop/session_method_comparison

# A new output directory is mandatory: completed cells in an existing run are
# intentionally skipped rather than relabeled as fresh measurements.
/home/julie/alphaeditSessionControl/.venv-server/bin/python qa500_remeasure.py prepare \
  --out /home/julie/Desktop/session_method_comparison/runs/qa500-new-readout
/home/julie/alphaeditSessionControl/.venv-server/bin/python qa500_remeasure.py launch \
  --out /home/julie/Desktop/session_method_comparison/runs/qa500-new-readout
```

Those commands intentionally start another experiment and are provided for reference; packaging this release does not execute them or alter the active WISE experiment. On a different machine, configure the launcher paths and supply the matching external assets before running. Preserve the measured mathematical and generation settings if comparing against the published results.

The shared code uses PyTorch 2.6.0, Transformers 4.51.3 and the pinned requirements at the repository root. Runtime CUDA is restricted to physical GPU 1. The dataset/reference answers are open at write time; this is supervised exact-question memory recall, not a blind LongMemEval or unseen-question generalization score.
