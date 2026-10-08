# Reproduce the connected experiment

Run commands from the repository root. The measured environment was Python 3.12.3, PyTorch 2.6.0+cu118, Transformers 4.51.3, and an A100 with approximately 80 GiB on physical GPU 1. Other package versions are pinned in `requirements-experiments.txt`.

The GPU entry points resolve `nvidia-smi` index **1** to its UUID before importing GPU libraries. They expose that device as logical `cuda:0` and refuse a CPU/GPU 0 fallback. A machine must have physical index 1 available. Use an idle GPU for comparable allocation measurements. The largest measured live recall allocation was 57.49 GiB for the 32-session KV condition; AlphaEdit writing and projection construction have separate costs.

A fresh bank retains roughly 42 GiB of dense delta files plus a roughly 2.3 GiB projector. Allow at least 50 GiB of free output space, in addition to the local model checkpoints and dataset. The runner checks per-stage free-space requirements. No published tensor files are needed: all 64 memories are written locally.

## 1. Install the pinned environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-experiments.txt
```

The separate CPU installation in the README is for unit checks. Full experiments use the CUDA wheel. Installing pinned packages is not a guarantee of bit-identical outputs across different hardware or model revisions.

## 2. Obtain the original dataset and local models

The benchmark used the original `longmemeval_s` release, not its cleaned successor. Preparation verifies the complete downloaded file against SHA256:

```text
08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894
```

```bash
mkdir -p data models
curl -L --fail https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s -o data/longmemeval_s.json
```

If the provider changes or removes those bytes, use an original copy matching the digest; do not silently substitute another edition.

Obtain [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) with the required model access, and [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2). Place complete local checkpoint directories at:

```text
models/Llama-3.1-8B-Instruct/
models/all-MiniLM-L6-v2/
```

Alternatively, create local directory symlinks to checkpoints you already have. The projector builder requires the first model at that exact location **before** bootstrap. The bootstrap CLI also accepts `--model` and `--embedding-model` paths and links them to the expected locations, refusing to replace a different existing checkpoint. No access tokens belong in the repository.

The historical reports record model identities and runtime versions but do not record a complete checkpoint revision/file-hash manifest. They therefore cannot certify bit-for-bit checkpoint identity for a new download. Record the revisions you use alongside fresh results.

## 3. Reconstruct the cohort and preservation corpus on CPU

```bash
python scripts/prepare_experiment.py \
  --data data/longmemeval_s.json \
  --manifest config/session_manifest.json \
  --out runs/prepared
```

The identifier-only manifest preserves the original order of all 64 question/session pairs, including the first 24 anchor questions and their prior-use labels. Preparation retrieves literal original questions, supplied answers, and **complete answer-bearing sessions**. It reconstructs 200 unique distractor user turns, excluding all answer-session IDs, using the original word-count rule and seed. Corpus SHA256 must equal:

```text
e23dff571defe17a41c5a1fbe936efefea65029db2bcdd8a78cd063e70134525
```

Preparation does not write deltas or create an experiment-success marker. This CPU step was checked against the real source dataset during packaging: all 64 question/answer/session fields, order, and corpus bytes matched the completed experiment.

## 4. Build the layer-specific null-space projector on GPU 1

```bash
python prepare_deep_projection_gpu1.py \
  --corpus runs/prepared/distractors.jsonl \
  --layers 13,14,15 \
  --threshold 0.02 \
  --out runs/prepared/projector.pt
```

The projection uses empirical uncentered token second moments and thin SVD. It is derived from this small benchmark distractor corpus, rather than a Wikipedia covariance download. The original corpus produced 5,916 tokens with protected ranks 177, 239, and 345. This is a pilot preservation construction, not a full locality/preservation evaluation. Projector metadata remains beside the tensor file.

## 5. Write all memories and run the nested comparisons

```bash
python scripts/bootstrap_experiment.py \
  --prepared runs/prepared \
  --projection runs/prepared/projector.pt \
  --model models/Llama-3.1-8B-Instruct \
  --embedding-model models/all-MiniLM-L6-v2 \
  --out runs/reproduction-v1
```

This first writes **40 real source-bank deltas** with the frozen answer-plus-EOS objective. It records actual write-time generation checks, verifies saved hashes, and confirms base-weight restoration. It then writes the remaining **24 deltas**, scores the 64×64 activation matrix, and generates answers at bank sizes 24, 32, 48, and 64.

The primary comparison gives all three methods the full sessions and supplied original Q/A at memory creation:

- Deltas: normalized activation energy selects one stored memory; the edited LLM answers the bare original question.
- Matched dense RAG: normalized MiniLM question-key vectors select one full session/Q/A record; the unedited LLM generates an answer.
- Full-bank KV: the unedited LLM reuses a float32 cache of all records, with query tails cropped between questions.

Context-only RAG retrieves three conversation chunks without the supplied Q/A annotations. It is a secondary condition with different write-time information. KV conditions that exceed the complete context window or GPU capacity are reported as unavailable; no truncation or zero-accuracy substitution is applied.

Output directories:

```text
runs/reproduction-v1/source-baseline/  # 40 newly written source deltas/checks
runs/reproduction-v1/scaling/          # remaining writes, nested comparison and measurements
```

To separate the two stages, use `--bootstrap-only`, then rerun the same command with `--run-only`. `--run-only` requires the successful local source bootstrap and unchanged delta hashes. It begins a new scaling stage; it does not resume a partially failed scaling run. Completed and failed stage directories are immutable; use a fresh `--out` to retry a failed experiment. Nothing deletes previous deltas automatically.

The new bootstrap does not reproduce the earlier calibration, writer-selection evaluation, or the separate historical 24-session comparison. It marks only operations actually executed. The scaling runtime excludes the preceding 40 bootstrap writes and projection construction; write-time diagnostics and recall peaks are reported separately.

## 6. Generate graphs only after successful completion

```bash
python build_memory_scaling_report.py --out runs/reproduction-v1/scaling
```

The report builder requires completed GPU 1 runs, frozen protocol hashes, restored base weights, unchanged delta/source artifacts, and paired complete trials. It independently recomputes aggregate measures before plotting. It writes `report.html` and PNG/SVG figures beside fresh results. A new bootstrap has no historical-comparison control, so that check is recorded as null. Historical semantic-review notes are not assigned to fresh answers.

Fresh bootstrap model execution has **not** been validated end to end during this packaging task. CPU preparation, metadata contracts, cache reuse, answer stopping, exported evidence, and import closure were checked. The existing published reports came from the completed historical GPU runs.

Original experimental scripts are included for provenance and reusable implementation. Their direct preparation paths depend on earlier local result directories; the supported clean-checkout path is the sequence above. To inspect published evidence without models, run `python scripts/verify_release.py` and open the HTML files under `docs/results/` in a browser.
