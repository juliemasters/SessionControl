# Native WISE on the paper's ZsRE benchmark

This is a separate reproduction using the authors' released ZsRE edit/training
data, Mistral-7B-v0.1, publication-era native WISE code, generated prefix
augmentation, and released reliability/generalization/locality metrics.
It does not change or replace the earlier LongMemEval experiment.

The native WISE implementation, optimizer, activation loss, masks, thresholds,
tokenizer, augmentation generator, merger and inference router are unmodified.
`third_party/wise_paper/UPSTREAM.json` pins source revision
`119c004395aab6f9101637ccc9e8d5520c084a09` (2024-12-19) with per-file hashes.
The driver imports only the necessary native modules; it does not substitute the
previous EfficientWISE subclass. Official metric function ASTs are loaded without
altering their bodies, avoiding unrelated EasyEdit trainer dependencies.

## Protocol

- First 1,000 records in the authors' designated `zsre_mend_edit.json` (1,037
  records), as in their released runner. Unrelated training prompts come from
  the separate `zsre_mend_train.json` (163,196 records).
- Edit input is `src`, target is `alt`. Original `rephrase` tests generalization;
  evaluation `loc` tests locality against the unedited model's predictions.
- Original paper model family: `mistralai/Mistral-7B-v0.1`, pinned Hub revision
  `27d67f1b5f57dc0953326b2601d68371d40ea8da`, FP32, physical GPU 1, seed 42.
- Run WISE-Merge first (`retrieve=False`), then the documented WISE-Retrieve
  variant (`retrieve=True`). No other native YAML hyperparameter changes.
- One sequential 1,000-edit trajectory per variant. At T=1,10,100,1000 evaluate
  all first-T prompts after T edits, preserving the paper's retention question.
- Preserve native checkpoints, augmentation templates, RNG state and all metrics;
  the ten-edit functional pilot is resumed, not restarted or cherry-picked.
- The released primary metrics use teacher forcing and target-token accuracy.
  They are **not** free-generated-answer F1 or whole-answer exact match. Report
  both token averages and all-target-token indicators under explicit names.
- Locality compares post-edit predictions with pre-edit predictions; it is not
  simply correctness against NQ's reference answer.
- Results/figures are generated only after both full runs pass the completion
  audit. No substitution of forced-memory results or tuned thresholds.

Exact paper subset IDs/seed, original runtime and original experiment commit are
not disclosed. The selected revision is contemporaneous with the paper's v3;
this is not a claim that it is the undisclosed experiment commit. The released
prefix generator creates five length-5 and five length-10 prefixes, which differs
from the prose description of ten length-10 sequences. Follow released code
unchanged and disclose this difference. Lower-T cells here contain T questions,
not an undisclosed aggregation over many independent edit sequences.

## Execute

The authors' data archive and model are stored outside the code tree.

```bash
python -m unittest tests.test_wise_paper -v
python -m experiments.wise_paper.prepare --out runs/wise-paper-zsre-mistral-v1
python -m experiments.wise_paper.run launch --out runs/wise-paper-zsre-mistral-v1
```

`driver_status.json`, per-variant `status.json`, `driver.log`, `ledger.json` and
`checkpoint-T.json` show live progress. The detached driver downloads/verifies
the model, waits for GPU 1 rather than stopping other jobs, runs both variants,
and writes `report.html`, `report.md`, `summary.csv`, `results.json` and figures
on success. A failure is recorded explicitly in `completion.json`; it does not
produce a completed report. The job survives chat/terminal disconnection while
the server/container remains powered on. It does not send messages or email.

## Sources

- Paper: https://arxiv.org/abs/2405.14768
- Official data: https://drive.google.com/file/d/1YtQvv4WvTa4rJyDYQR2J-uK8rnrt0kTA/view
- Official runner/config/source: https://github.com/zjunlp/EasyEdit/tree/119c004395aab6f9101637ccc9e8d5520c084a09

Hallucination and Temporal files are retained from the same official archive;
they are not silently represented as completed experiments in this ZsRE run.
