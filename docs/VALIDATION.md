# Packaging validation

Checks performed locally on the prepared tree:

- 42 existing and experiment CPU unit tests passed, including weight restoration, delta routing, independent answer stopping, and reusable KV prefix/tail integrity.
- 14 preparation and release CPU tests passed, including literal cohort reconstruction, real-file bootstrap metadata contracts, process replacement before the scaling stage, immutable artifacts, and independent exported metric verification.
- Real source-dataset preprocessing reproduced all 64 original question/answer/full-session fields and ordering, with an identical 200-turn preservation-corpus digest.
- The released modules import successfully without loading checkpoints. Documentation links and local dependency closure were checked.
- Export verification checked all 37 artifact digests and independently reproduced measures from 656 historical final answers: 96 in the original comparison and 560 across nested scaling conditions.
- A historical report-builder smoke check regenerated four figure pairs and HTML from copies of actual completed-run evidence, reproduced all three earlier 24-question controls, and left original results unchanged.
- A standalone public-tree copy verified without `.git`. Public files were scanned for accidental credentials, machine-specific paths, model tensors, logs, and files exceeding GitHub's size limit.

No new GPU experiment, model write, remote upload, or GitHub Actions run was performed during packaging. The clean-checkout bootstrap is covered by CPU metadata/process tests; its complete new GPU execution remains unverified. The published accuracy and memory numbers belong to the completed historical runs.

To repeat the checks, use the CPU commands in the root README. Run `git diff --check` in a Git checkout. Full reproduction and its hardware requirements are documented separately in `REPRODUCING.md`.
