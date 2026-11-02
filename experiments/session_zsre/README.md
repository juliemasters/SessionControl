# Our session controller on the matched WISE ZsRE benchmark

This is a new experiment, not a reinterpretation of the LongMemEval results.
Use the same ordered 1,000 official ZsRE edits, Mistral-7B-v0.1 revision, FP32,
GPU 1, seed 42, checkpoints 1/10/100/1000, unmodified official teacher-forced
metrics and unedited locality baseline as `experiments/wise_paper`.

ZsRE supplies no conversations: one memory represents one official Q&A edit.
Reuse the existing compact independent AlphaEdit writer (layers 13–15) and
normalized activation update-energy router. Write the exact raw question and
complete counterfactual answer plus EOS. Omit the older Q:/A: wrapper in both
phases to match native WISE's raw question interface. No context is invented.
The value context retains an explicit space before decoded target tokens because
SentencePiece drops that leading space. Before writing, every one of the 1,000
value-optimizer token sequences is checked against direct question-ID plus
target-prefix-ID concatenation. This fixes tokenizer alignment without changing
the existing optimizer, objective or key position.

Mistral needs its own nullspace projectors: first 200 **training-split** locality
prompts, all token keys, uncentered second moment, thin SVD, eigenvalue threshold
0.02, exactly the earlier small-corpus projector procedure. Never reuse Llama's
projector or deltas. Never use held-out paraphrases, evaluation-locality queries
or edit targets to construct preservation projectors. This is not full-corpus
AlphaEdit preservation statistics.

At each checkpoint, independently route every original question, held-out
paraphrase and locality query against all first-T deltas. Routing sees only the
query and base FFN input activations; target tokens enter **after selection** in
the unchanged official metric. Always select argmax; do not add a locality gate,
oracle identity lookup, BERT scorer or lexical fallback. Restore base weights
between independent writes, routing queries and selected-delta evaluation.

Report the same six token-average/all-token metrics as WISE, plus query-only
source-selection hit/rank and retained compact storage. These are not generated
answer F1, exact match or TTFT. Compare the full systems, not the isolated causal
effect of session control: their writers and model-edit layers differ.

## Run

The matched WISE run must already be complete and its assets present.

```bash
python -m unittest tests.test_session_zsre -v
python -m experiments.session_zsre.prepare --out runs/session-zsre-mistral-v2
python -m experiments.session_zsre.run launch --out runs/session-zsre-mistral-v2
```

The detached driver verifies frozen model/source/data hashes, checks ten baseline
items for exact native parity, runs ten functional edits with no accuracy
threshold, resumes the same committed independent bank to 1,000, audits all
checkpoint rows and writes the final HTML/CSV/figures only on full success.
It waits for GPU 1 and never touches GPU 0 or stops another process. It survives
chat disconnection while the server remains powered on. Failures are recorded
explicitly rather than passed off as completed results.

Source question/answer files, model weights and deltas stay outside GitHub.
Keep the prior WISE and LongMemEval results unchanged.
