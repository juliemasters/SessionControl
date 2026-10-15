# Final measured results

The compact bank and read-only TTFT replay completed on physical GPU 1 using Llama-3.1-8B-Instruct in float32. The nested banks contain 8, 16, 32, 64, 128, 256, and 512 distinct real conversations. The same eight previously inspected original questions are evaluated at every size.

Compact deltas and matched RAG select the correct source for **8/8 questions** and achieve **7/8 whole-answer exact matches** at every size. At N=512, mean custom word-token F1 is **93.27%** for compact deltas and **98.86%** for RAG. Articles are retained in this metric; it is not standard SQuAD F1 or factual grading.

One compact output omitted a location qualifier; the nonexact RAG output differed by an article. The other seven answers match exactly. Correct routing and answer fidelity are separate measures. Per-item text remains local.

At N=512, compact factors retain **109.29 MiB** and matched RAG retains **6.84 MiB**. The 64-session dense source bank retained 42 GiB before compact conversion; the compact 64-session bank retains about 13.66 MiB. Dense storage at 512 is a calculated counterfactual, not an allocated bank. Actual peak allocated GPU memory at N=512 is approximately **30.15 GiB compact** and **31.15 GiB RAG**, including about 29.92 GiB of common LLM parameters. Host RSS, disk retention and GPU allocation remain separate measurements.

The warm median time to first visible answer at N=512 is **810 ms compact** and **3,458 ms RAG**; p95 is **871 ms** and **4,242 ms**, respectively. Each condition has 16 warm timing samples from two repeated passes. TTFT includes routing/retrieval, memory loading, tokenization and prefill, and excludes model startup, writing, network and queues. OS file caches were not flushed.

Full-bank KV is feasible at N=8,16,32, with exact match 6/8,3/8,5/8 and reference containment 8/8 at each size. Sentence framing reduces lexical scores, so these are not counts of factual errors. At N=32, warm median TTFT is 744 ms; one measured first request including prefix construction takes 316 seconds. From N=64, the complete prefix exceeds the 131,072-token context limit. Accuracy, live memory and latency are unmeasured for those unavailable conditions.

Legacy compact validation preserved all **64/64** own-question token sequences and **32/32** panel routing selections, with maximum relative Frobenius reconstruction error below 0.002. Added-session own-question write checks matched exactly for **442/448**, with six failures retained. This write diagnostic is separate from eight-question routing accuracy.

The larger bank includes 448 real distractor conversations with new extractive write annotations, alongside 64 official annotated sources. It does not contain 512 independent official evaluation questions. Exact question keys were supplied at write time; a direct exact-key lookup is a relevant additional baseline, not measured here. The results do not establish unseen-query generalization, retention of every session fact, or an overall memory advantage over RAG.

[Scaling report and memory graphs](results/scaling-real512-compact-v1/report.html) · [TTFT/F1 report](results/scaling-real512-ttft-v1/report.html)

[Scaling protocol](REAL512_PROTOCOL.md) · [TTFT protocol](TTFT_F1_PROTOCOL.md) · [Export provenance](EXPORT.md)
