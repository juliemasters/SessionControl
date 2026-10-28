# WISE-Retrieve: full-context, exact-question, complete-answer memory comparison

Completed 500 sequential conversation writes and ten actual inference passes at every checkpoint on physical GPU 1. WISE uses its own native activation router. Accuracy is measured only on the complete final generated answer: token F1 and normalized exact match.

The four baseline conditions are the previous audited ten-pass run. Their scores and timings are explicitly reused; only WISE was newly trained and evaluated. These are ten inference repetitions on one trained trajectory, not ten independently trained models.

| Sessions | Questions/pass | Method | F1 (%) | Exact (%) | Mean visible TTFT (ms) | Bank (MiB) | Peak GPU (GiB) |
|---:|---:|---|---:|---:|---:|---:|---:|
| 8 | 8 | SessionControl | 93.27 ± 0.00 | 87.50 ± 0.00 | 198.46 ± 47.27 | 1.71 ± 0.00 | 30.36 ± 0.00 |
| 8 | 8 | AlphaEdit without session control | 87.50 ± 0.00 | 87.50 ± 0.00 | 72.84 ± 23.63 | 672.00 ± 0.00 | 30.14 ± 0.00 |
| 8 | 8 | WISE-Retrieve | 9.60 ± 0.00 | 0.00 ± 0.00 | 48.83 ± 18.61 | 224.00 ± 0.00 | 30.80 ± 0.00 |
| 8 | 8 | RAG | 100.00 ± 0.00 | 100.00 ± 0.00 | 3296.44 ± 58.61 | 0.13 ± 0.00 | 31.26 ± 0.00 |
| 8 | 8 | Full-bank KV cache | 100.00 ± 0.00 | 100.00 ± 0.00 | 268.32 ± 18.09 | 0.00 ± 0.00 | 37.53 ± 0.00 |
| 16 | 16 | SessionControl | 96.63 ± 0.00 | 93.75 ± 0.00 | 162.68 ± 10.48 | 3.42 ± 0.00 | 30.36 ± 0.00 |
| 16 | 16 | AlphaEdit without session control | 85.10 ± 0.00 | 81.25 ± 0.00 | 61.53 ± 9.32 | 672.00 ± 0.00 | 30.14 ± 0.00 |
| 16 | 16 | WISE-Retrieve | 11.63 ± 0.00 | 0.00 ± 0.00 | 42.54 ± 0.14 | 224.00 ± 0.00 | 30.80 ± 0.00 |
| 16 | 16 | RAG | 98.44 ± 0.00 | 93.75 ± 0.00 | 3382.72 ± 34.71 | 0.27 ± 0.00 | 31.48 ± 0.00 |
| 16 | 16 | Full-bank KV cache | 86.74 ± 0.00 | 75.00 ± 0.00 | 458.14 ± 7.25 | 0.00 ± 0.00 | 45.63 ± 0.00 |
| 32 | 32 | SessionControl | 98.32 ± 0.00 | 96.88 ± 0.00 | 163.64 ± 7.34 | 6.83 ± 0.00 | 30.36 ± 0.00 |
| 32 | 32 | AlphaEdit without session control | 85.85 ± 0.00 | 81.25 ± 0.00 | 59.19 ± 6.17 | 672.00 ± 0.00 | 30.14 ± 0.00 |
| 32 | 32 | WISE-Retrieve | 23.26 ± 0.00 | 15.62 ± 0.00 | 44.76 ± 0.11 | 224.00 ± 0.00 | 30.80 ± 0.00 |
| 32 | 32 | RAG | 99.22 ± 0.00 | 96.88 ± 0.00 | 3258.62 ± 6.62 | 0.53 ± 0.00 | 31.48 ± 0.00 |
| 32 | 32 | Full-bank KV cache | 85.52 ± 0.00 | 71.88 ± 0.00 | 814.63 ± 4.49 | 0.00 ± 0.00 | 60.37 ± 0.00 |
| 64 | 64 | SessionControl | 93.16 ± 0.00 | 90.62 ± 0.00 | 174.43 ± 4.10 | 13.66 ± 0.00 | 30.36 ± 0.00 |
| 64 | 64 | AlphaEdit without session control | 88.76 ± 0.00 | 84.38 ± 0.00 | 51.41 ± 0.89 | 672.00 ± 0.00 | 30.14 ± 0.00 |
| 64 | 64 | WISE-Retrieve | 10.17 ± 0.00 | 1.56 ± 0.00 | 43.12 ± 0.08 | 224.00 ± 0.00 | 30.80 ± 0.00 |
| 64 | 64 | RAG | 99.61 ± 0.00 | 98.44 ± 0.00 | 3249.77 ± 6.01 | 1.06 ± 0.00 | 31.48 ± 0.00 |
| 64 | — | Full-bank KV cache | Unavailable: context_limit | — | — | — | — |
| 128 | 98 | SessionControl | 89.96 ± 0.00 | 87.76 ± 0.00 | 209.98 ± 3.89 | 27.72 ± 0.00 | 30.36 ± 0.00 |
| 128 | 98 | AlphaEdit without session control | 86.10 ± 0.00 | 81.63 ± 0.00 | 55.07 ± 2.29 | 672.00 ± 0.00 | 30.14 ± 0.00 |
| 128 | 98 | WISE-Retrieve | 24.62 ± 0.00 | 10.20 ± 0.00 | 45.32 ± 0.11 | 224.00 ± 0.00 | 30.80 ± 0.00 |
| 128 | 98 | RAG | 87.09 ± 0.00 | 81.63 ± 0.00 | 3104.58 ± 4.40 | 2.05 ± 0.00 | 31.51 ± 0.00 |
| 128 | — | Full-bank KV cache | Unavailable: context_limit | — | — | — | — |
| 256 | 162 | SessionControl | 85.38 ± 0.00 | 81.48 ± 0.00 | 281.78 ± 3.52 | 55.84 ± 0.00 | 30.36 ± 0.00 |
| 256 | 162 | AlphaEdit without session control | 51.16 ± 0.00 | 32.72 ± 0.00 | 55.52 ± 0.91 | 672.00 ± 0.00 | 30.14 ± 0.00 |
| 256 | 162 | WISE-Retrieve | 14.84 ± 0.00 | 1.85 ± 0.00 | 48.26 ± 0.08 | 224.00 ± 0.00 | 30.80 ± 0.00 |
| 256 | 162 | RAG | 72.89 ± 0.00 | 67.90 ± 0.00 | 3144.55 ± 2.43 | 4.17 ± 0.00 | 31.55 ± 0.00 |
| 256 | — | Full-bank KV cache | Unavailable: context_limit | — | — | — | — |
| 500 | 283 | SessionControl | 80.58 ± 0.00 | 75.97 ± 0.00 | 415.91 ± 16.65 | 108.25 ± 0.00 | 30.36 ± 0.00 |
| 500 | 283 | AlphaEdit without session control | 9.65 ± 0.00 | 1.77 ± 0.00 | 62.41 ± 0.28 | 672.00 ± 0.00 | 30.14 ± 0.00 |
| 500 | 283 | WISE-Retrieve | 11.40 ± 0.00 | 1.41 ± 0.00 | 50.42 ± 0.07 | 448.00 ± 0.00 | 30.85 ± 0.00 |
| 500 | 283 | RAG | 62.70 ± 0.00 | 56.54 ± 0.00 | 3126.75 ± 2.50 | 8.18 ± 0.00 | 31.55 ± 0.00 |
| 500 | — | Full-bank KV cache | Unavailable: context_limit | — | — | — | — |

## Protocol and limits

Official WISE-Retrieve sequential edits; layer 29, learning rate 0.9, 30 iterations, save_freq=500, merge_freq=1000, replay=False.
Each binding receives the complete conversation, literal question and complete official answer plus EOS. One question-only and one fully grounded template; one deterministic auxiliary distractor locality row. No session or answer truncation.
Disable upstream chat wrapping to keep the exact Q: question / A: recall key used by the comparison. Custom token assembly retains every answer token. An integration subclass projects only supervised hidden states through the LM head; masked CE reduction and its gradients are tested against the original implementation. Within one write batch, cache only the unchanged frozen decoder prefix before layer 29; recompute the edited layer and all downstream layers at every optimizer step. Official SGD, activation loss, masks and router unchanged.
Checkpoints count conversations, not individual bindings. 500 conversations contain 508 bindings; official save/merge frequencies count bindings. No frequency tuning against evaluation answers. Native WISE routing is used, without SessionControl routing.
Ten measured inference passes on one sequentially trained WISE trajectory. Same eligible questions and per-pass order as all baselines. Whole final-answer token F1 and normalized exact match only. No answer extraction or source-routing accuracy.
Same FirstAnswerClock and greedy decoding as the completed experiment; query arrival to first nonempty final output. Includes WISE routing in every relevant forward. Excludes model loading, checkpoint installation and writing. Baselines are the prior completed ten-pass run, not simultaneous interleaved measurements.
Separate deployable side-memory checkpoint bytes, process host RSS, and GPU allocated/reserved peaks including common model weights. Unchanged original weights are reconstructed from the common base model. Training-only mask/RNG/history state excluded from deployed bank bytes.
Supervised exact-question mechanism check, not a blind LongMemEval benchmark. Full global reference answer is supplied for each supporting session. Eligible cohorts change with N. Precision and data match, but each method retains its own layer/optimizer/router configuration.

At checkpoints 8,16,32,64,128,256,500 the query counts are 8,16,32,64,98,162,283. The final 500 conversations contain 508 official Q/A bindings. No paraphrases or newly invented questions are used.

TTFT measures the first nonempty final answer, not total completion time or a routing probe. Requests with no visible output have missing visible TTFT and remain included in F1/exact match. The CSV reports visible/missing timing counts and valid-pass counts. Actual KV failures remain unavailable.

Different eligible questions at different N change task difficulty; original64/new/type/abstention groups are exported separately. F1 is lexical overlap, not a semantic correctness judgment. WISE uses its recommended layer 29 and hyperparameters; AlphaEdit uses its own configured layers and solver. This compares complete systems rather than isolating a single routing change.

Official WISE source: https://github.com/zjunlp/EasyEdit/tree/4c109870955a4522ac3d7cf10ad00f34de8e4f0d/easyeditor/models/wise . Unmodified source is vendored with its MIT license and per-file SHA-256 manifest.

No measured write-time comparison is claimed. Disk bank measurements exclude common frozen base weights and training-only restart state.
