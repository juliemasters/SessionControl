# Measured results and limits

The completed experiments support activation routing under exact-question supervision. They do not establish an overall accuracy or memory advantage over RAG. At the largest bank, activation selected the correct delta for all 64 questions, but 58 generated answers matched the reference exactly. Matched-information RAG returned 63 exact answers with a much smaller retained text/index bank.

The [24-session comparison](results/memory-comparison-v1/report.html) and [24–64-session scaling report](results/memory-scaling-v1/report.html) include graphs, per-question outputs, and execution checks. The tables below were verified against the original completed `evaluation.json`, `measurements.json`, and `resource_summary.json` files. Rounded memory values use GiB = 2^30 bytes and MiB = 2^20 bytes.

## Conditions compared

All primary methods use Llama-3.1-8B-Instruct in float32 and greedy batch-one generation with a 64-token cap and EOS/next-turn stopping. All receive the full answer-bearing session plus the supplied original question and answer at memory creation. Recall asks that same original question. This is an open-book write followed by an exact-question recall test.

| Method | Retained representation and recall |
| --- | --- |
| Independent dense deltas | AlphaEdit answer-plus-EOS writes at layers 13–15; base weights restored between writes. Normalized delta activation energy selects one delta, then the edited model generates from the bare question key. |
| Matched-information dense RAG | MiniLM embeddings index the supplied question keys. Top-one retrieval supplies the full session and its stored question/answer to the unedited model for generation. |
| Full-bank KV | An uncompressed float32 reusable cache contains every session and supplied question/answer. The unedited model receives the query suffix. No session selector or truncation is used. Query additions are removed before reuse. |

The final router does not use BERT or probability calibration. A 64 × 64 activation matrix contains 4,096 question–delta energy scores, not 4,096 independently generated answer probes.

## Accuracy

“Exact” below means the complete delivered answer matches after case, punctuation, and whitespace normalization. It is stricter than conveying the requested fact, but less strict than verbatim matching. Source accuracy measures whether the selected/retrieved session is the reference answer session. It is not defined for full-bank KV because that method reads the entire bank.

| Sessions | Delta exact answers | Delta source correct | Matched RAG exact answers | Matched RAG source correct | KV exact answers |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 24/24 (100%) | 24/24 | 24/24 (100%) | 24/24 | 6/24 (25%) |
| 32 | 31/32 (96.9%) | 32/32 | 32/32 (100%) | 32/32 | 15/32 (46.9%) |
| 48 | 45/48 (93.8%) | 48/48 | 47/48 (97.9%) | 48/48 | Unavailable: context limit |
| 64 | 58/64 (90.6%) | 64/64 | 63/64 (98.4%) | 64/64 | Unavailable: context limit |

These are the fresh nested-bank scaling measurements. The earlier 24-session comparison produced the same exact-answer outcomes. The same 24 anchor questions remained 24/24 exact for deltas and matched RAG at every bank size. Therefore, the fall in the all-question delta rate cannot by itself be attributed to routing interference: the larger banks also introduce additional questions. All selected delta ranks were one.

At 64 sessions, mean token F1 was 0.932 for deltas and 0.999 for matched RAG. Token F1 and bounded reference containment are lexical measures; neither verifies the truth of every generated statement. The graphs provide Wilson intervals for binary proportions. Small, previously exposed question sets limit the interpretation of these intervals.

The six nonexact delta outputs at 64 sessions all used the correct delta. Post hoc inspection found three clearly wrong/missing answers: a year instead of a three-month duration, a generic Spotify refusal instead of the playlist name, and “classic gin fizz” instead of “lavender gin fizz.” The other three involved omitted detail, wording, or uncertainty. They should not all be described as wholly factually wrong. The corresponding independent write-time outputs also failed exact matching, so these cases motivate better memory writing and answer fidelity rather than a replacement router.

Likewise, KV's 6/24 exact answers do not mean only six answers conveyed the requested fact. An exploratory content review of the original 24-session comparison judged 23/24 KV outputs to convey the reference fact without an explicit contradiction; one contradicted itself. This review was performed after inspection, by one unblinded Codex reviewer. It is not independent semantic grading and does not validate every additional claim. No equivalent independent factual-accuracy estimate exists for the larger run.

## Memory

The following values come from the scaling experiment. “Bank payload” excludes common model weights and describes the actual retained representation. “Peak live GPU” is PyTorch peak allocated memory during recall and required routing/index/prefill setup, including common model weights. Allocator-reserved memory is separately shown in the reports. AlphaEdit writing is excluded from these recall peaks.

| Sessions | Delta bank, disk | RAG text/index, CPU/disk | KV bank, GPU | Delta peak live GPU | RAG peak live GPU | KV peak live GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 15.75 GiB | 0.370 MiB | 17.92 GiB | 31.04 GiB | 31.31 GiB | 50.20 GiB |
| 32 | 21.00 GiB | 0.502 MiB | 24.31 GiB | 31.04 GiB | 31.31 GiB | 57.49 GiB |
| 48 | 31.50 GiB | 0.732 MiB | Not constructed | 31.04 GiB | 31.31 GiB | Unavailable |
| 64 | 42.00 GiB | 0.988 MiB | Not constructed | 31.04 GiB | 31.31 GiB | Unavailable |

Disk storage, CPU memory, and GPU memory are different resources. Deltas are loaded individually from disk; their runtime also holds a 672 MiB CPU backup of the edited base matrices. RAG retains full session text plus its vector index. Common LLM parameters occupy 29.92 GiB; RAG additionally uses 86.64 MiB of encoder parameters, included in its GPU peaks. The compact RAG payload figures do not mean the entire RAG system occupies only about 1 MiB.

The 64-delta activation matrix was computed once. Its measured setup peak is conservatively included in the reported delta peaks for every bank size, rather than claiming four separately measured router setup peaks. The earlier comparison measured 31.02 GiB for deltas and 31.31 GiB for RAG; the scaling run used a fresh 24-session control and the same expandable-segment allocator setting for every condition. Reserved-memory differences between runs must not be mistaken for changes in retained representation size.

At 64 sessions, dense delta storage is much larger than RAG storage. At the feasible 32-session KV condition, delta peak live GPU use is approximately 46% lower than KV's. These findings support a GPU-memory advantage over this uncompressed full-bank KV baseline, while providing no general memory-efficiency advantage over RAG.

Every condition uses a transient decoding KV cache. The full-bank KV baseline additionally retains the entire memory bank as KV between queries. The comparison does not evaluate quantized, compressed, offloaded, or retrieval-assisted KV variants.

Writing 24 new deltas took 9.94 minutes and peaked at 34.74 GiB allocated GPU memory. These costs are separate from the table above. Forty verified deltas were reused; the complete scaling run took 39.38 minutes on physical GPU 1. Its latency measurements separate bulk activation from generation and are not a serving-throughput benchmark.

## Calculated values and capacity limits

Measured 24/32-session KV tensor payloads match the configuration formula: 32 layers × 2 (key/value) × 8 KV heads × 128 dimensions × 4 bytes = 262,144 bytes per prefix token. Dashed expected-memory curves use this formula and the actual encoded prefix lengths. They are representation-size calculations, not measurements of allocated/reserved GPU memory.

| Sessions | Encoded full-bank prefix | Calculated KV payload | Execution |
| ---: | ---: | ---: | --- |
| 24 | 73,384 tokens | 17.92 GiB | Completed; actual payload verified |
| 32 | 99,582 tokens | 24.31 GiB | Completed; actual payload verified |
| 48 | 145,892 tokens | 35.62 GiB | Exceeds 131,072-token context limit |
| 64 | 196,104 tokens | 47.88 GiB | Exceeds 131,072-token context limit |

The capacity check includes the longest question suffix and 64 generated tokens. Unavailable conditions have no assigned accuracy. Calculating their payload does not show that they can run or fit GPU memory. Delta payload grows by approximately 672 MiB per session in the present dense format; compression was not tested. Decoding-cache sizes for deltas are configuration-derived estimates, while RAG/KV cache sizes are counted from actual tensors. The 100% accuracy line is an aspirational target, not a forecast or an unperformed measurement.

## Scope and reproducibility

The source is original LongMemEval-S, with SHA-256 `08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894`. The [official original dataset card](https://huggingface.co/datasets/xiaowu0162/longmemeval) now recommends a cleaned successor. Reproducing these measurements requires the recorded original edition; replacing it changes the experiment.

The 64 sessions include earlier pilot, writer-development, training, calibration, and test items. The writer and router were frozen before this capacity run; the run did not tune new parameters. Nevertheless, these are previously inspected items, so the results are exploratory capacity evidence rather than an independent held-out accuracy estimate. No paraphrases were used. Exact question keys make matched RAG retrieval especially easy, consistent with the same supervision provided to the delta writer.

A secondary context-only RAG condition used no supplied question/answer annotations and retrieved three conversation chunks. It returned 8/24 and 30/64 exact answers. Its information at memory creation differs from the primary conditions, so it is reported separately and is not a fair basis for claiming the supervised delta method outperforms matched RAG.

Only one supplied question per session was evaluated. Receiving the full session during writing does not establish that a delta retained every session fact. Successful completion means the protocol ran, base weights were restored, stored-delta hashes were unchanged, and reusable KV prefixes passed integrity checks. It does not mean every generated answer was correct.

Public artifact copies may replace workstation paths and GPU UUIDs. Historical source hashes identify the original completed artifacts; export manifests identify the released copies. The published package omits model weights, dense deltas, full conversation banks, and projection tensors. Reproducing neural execution requires preparing those inputs as described in the repository's reproduction instructions.
