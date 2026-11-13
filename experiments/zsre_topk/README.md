# Matched top-k experiments: ZsRE and LongMemEval

This package tests whether an offline activation-key shortlist can improve source routing and reduce visible-answer time to first token (TTFT) on **both datasets**. The package keeps its `experiments.zsre_topk` name, while each frozen design records `dataset: zsre` or `dataset: longmemeval` and is reported separately. Each dataset gets a **fresh full-scan baseline** alongside all top-k arms. Historical quality and latency scores do not supply these comparisons.

Original factor banks, dataset artifacts and frozen execution code stay unchanged. Each experiment writes a separate run directory and records checksums for the protected original sources and new experiment sources. No GPU run or publication is implied by preparing this package.

## Original protocols retained

| Dataset | Reused source | Stored units | Model and prompt | Eligible questions per pass |
|---|---|---|---|---|
| ZsRE | `runs/zsre-generated-sixmethod-1000-v3` | 1,000 Q/A deltas | Original Mistral FP32, raw question | N questions at every prefix |
| LongMemEval | `runs/qa500-checkpoints-tenpass-v2` | 500 conversation deltas | Original Llama FP32, `Q: {question}\nA:` wrapper | 8, 16, 32, 64, 98, 162, 283 |

ZsRE checkpoints are **8, 16, 32, 64, 128, 256, 500 and 1,000** saved deltas. LongMemEval checkpoints are **8, 16, 32, 64, 128, 256 and 500** conversations. The LongMemEval bank has 508 write bindings across 500 conversations; multiple activation keys may map to one conversation. Its shortlist always contains distinct conversation IDs. Only questions whose complete official supporting-source set is present in the prefix are evaluated.

For each dataset, the original base model, FP32 arithmetic, edited layers, factors, raw questions/reference answers, generation cap, decoder and stopping rules are preserved. The LongMemEval fine scorer retains its original rank-one/rank-two dispatch. Runtime execution uses physical GPU 1. The experiments do not compress factors or change model precision, prompts or decoding.

## Compared methods

| Method | Coarse candidates | Fine selection |
|---|---|---|
| `full_scan` | All N sources | Original scorer over all N |
| `topk_1` | Best activation-key source | Original scorer over one candidate |
| `topk_8` | Best min(8, N) distinct sources | Original scorer over shortlist |
| `topk_16` | Best min(16, N) distinct sources | Original scorer over shortlist |
| `topk_32` | Best min(32, N) distinct sources | Original scorer over shortlist |

Every arm preloads the same FP32 factor bank on the CPU. Only the top-k arms additionally load one normalized FP32 activation index on the CPU. The index is built offline from original write questions using the unedited base model and the dataset’s original question wrapper. At query time, the index supplies the coarse order and the unchanged original FP32 fine score chooses a source from the shortlist. Stable ties retain the original smallest source ID. No reference answer or support label is given to the router.

Matching CPU factor residency across arms isolates the shortlist effect from a different factor storage policy. Every arm also retains the original per-query selected-factor disk read before applying that delta. The new full-scan latency need not equal the historical run’s latency. Index disk storage and CPU tensor payload are counted explicitly in top-k resource measurements.

`topk_1` is **pure coarse selection** because its fine scorer receives one candidate. Both datasets query exact original write questions. Successful retrieval therefore measures source self-recognition; it does not establish paraphrase, unseen-question or factual generalization. LongMemEval reference answers were explicitly taught at write time, including the complete answer in each supporting conversation’s bindings for a multi-session item. This is a supervised memory mechanism experiment, without a blind LongMemEval or unseen-reasoning claim.

## Measurement sequence

The driver automatically proceeds through these phases for each dataset:

1. Freeze the design, original Q/A, ordered bank/source identities and old/new source hashes in a fresh output directory.
2. Build and validate the offline CPU activation index.
3. Run a routing-only preflight on the direct questions and save its decisions.
4. Run the first N=8 pilot cells for all five arms and verify the measurement path.
5. Complete all five arms, all dataset checkpoints and ten actual inference passes, resuming completed cells as needed.
6. Audit the complete matrix and generate the final local report.

The ZsRE preflight consists of **1,000 direct routing queries**, not ten inference passes. Preflight decisions are readiness checks, not completed benchmark quality or timing results. N=8 pilots are the first actual scheduled cells and are reused when the driver resumes the full matrix; they are not duplicated to inflate the repeat count. No final comparison report or successful completion marker is produced from a partial experiment.

Ten passes reuse each fixed trained bank. They evaluate every eligible original question in a deterministic, repeat-specific order matched across arms. ZsRE has **400 cells and 100,200 actual generation requests**. LongMemEval has **350 cells and 33,150 actual generation requests**. Means and sample standard deviations summarize ten pass-level measurements, not ten independently trained banks or training-seed uncertainty. The two datasets retain separate models, units and eligible cohorts and should be interpreted separately.

Prepare both datasets and start their detached GPU-1 driver with:

```sh
python -m experiments.zsre_topk.prepare --out runs/activation-topk-two-dataset-v1
python -m experiments.zsre_topk.run launch --out runs/activation-topk-two-dataset-v1
```

Use a fresh run directory. The driver serializes both datasets on physical GPU 1, manages the ordered phases, and produces their final reports automatically. A completed matrix for either dataset can also be audited and reported directly:

```sh
python -m experiments.zsre_topk.report --out runs/NAME-OF-NEW-TOPK-RUN/zsre
python -m experiments.zsre_topk.report --out runs/NAME-OF-NEW-TOPK-RUN/longmemeval
```

## Metrics and accounting

- **Routing accuracy:** selected source belongs to the question’s original answer-source set. ZsRE has one source per question; LongMemEval can have several supporting conversations. The denominator includes answerable eligible questions and excludes abstentions.
- **Shortlist recall:** at least one original answer source occurs in the min(k, N) coarse shortlist, using the same answerable denominator. Full scan has no coarse shortlist, so its value is null. Coarse and fine source ranks are the minimum rank over supporting sources.
- **Abstentions:** source hit, shortlist hit and source ranks are null. They remain in final-answer F1, exact-match and latency denominators. At 500 LongMemEval conversations there are 283 evaluated questions, 15 abstentions and 268 answerable source-routing questions per pass.
- **Answer quality:** word-token F1 and normalized exact match on the complete final generated answer, using the unchanged original scoring. No reference containment, gold-assisted extraction or teacher forcing is added.
- **Latency:** query arrival to first visible answer content, first generated token and complete query completion. Query timing includes activation capture, CPU coarse selection, original fine routing, the selected-factor disk read, delta application, prefill and generation. Offline key construction and bank setup are excluded. Routing time is also retained explicitly.
- **Resources:** retained factor bytes plus index bytes on disk; common preloaded CPU factor payload; additional CPU index payload; sampled peak process RSS; peak allocated and reserved GPU bytes. Disk, RAM and GPU remain separate tiers. GPU peaks include the common model. No compression claim is made.

Missing visible TTFT is counted explicitly. Such queries remain in answer and source-routing denominators, according to their original answerability, and are excluded only from visible-TTFT statistics. A pass with no visible answer has a null TTFT aggregate; the comparison records valid pass counts. Timing order and generation metadata must pass the audit.

Each cell is stored at `{method}/run-{repeat:02d}/bank-{n}.json`. It contains raw questions, reference answers, complete generated answers, source decisions, coarse/fine source ranks, shortlist flags, routing and generation timings, token counts and resource measurements. Every process folder has source-hash and physical-GPU provenance.

## Completion-gated artifacts

The report checks every planned available, completed cell before writing comparison artifacts. It verifies original ordered banks, raw Q/A, official support metadata and eligible question cohorts; matched query order; complete final-answer F1/exact match; timestamps; selected source IDs, ranks and routing flags; recomputed generation/routing summaries; matched factor residency; and activation-index accounting across arms. Frozen execution and input hashes must still match.

Each final run directory contains:

- `comparison.csv` and `comparison.json`: per-method/per-size means, sample SD, denominators, resource fields and paired per-repeat differences from the new full-scan baseline.
- `per_pass.csv`: every audited pass-level measurement.
- `question_outputs.csv`: all audited answers, source decisions, official source labels and timings.
- `accuracy.{png,svg,pdf}`: final F1, exact match and source routing accuracy.
- `latency.{png,svg,pdf}`: visible TTFT, complete query latency and routing latency.
- `memory.{png,svg,pdf}`: separate retained disk, process RAM and GPU plots.
- `summary.md` and `report.html`: dataset-specific results with interpretation and limitations.
- `report_manifest.json`: checksums of every report artifact.
- `completion.json`: successful completion only after the full audit and artifact generation.

Figures show the fresh full scan in gray and top-k variants in blue, with distinct markers. No plot or text assumes that top-k improves either outcome before full measurements exist. Higher routing accuracy need not yield higher answer quality, and a small shortlist need not improve latency at every bank size. LongMemEval’s eligible cohort also changes with bank size, so cross-size differences reflect both memory capacity and cohort difficulty.
