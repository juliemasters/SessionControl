# Memory-write time complexity — original, non-versioned method

This is an algorithmic calculation, not a runtime benchmark. No elapsed seconds, estimated seconds or speedup factors are inferred from these formulas. It describes the existing compact one-key session writer and the Q/A-key dense RAG comparator. Accuracy and answer F1 are measured separately.

## Symbols

| Symbol | Meaning |
|---|---|
| N | Stored sessions; checkpoints 8, 16, 32, 64, 128, 256, 512. |
| T | Value-optimization steps per edit; configured maximum 50. |
| e | Edited FFN layers; 3 in this experiment. |
| h | FFN output width / LLM hidden size. |
| m | FFN input width of each edited down-projection matrix. |
| D | RAG embedding dimension. |
| B | Bytes of the new conversation and Q/A record. |
| p_j | Cached prefix length for optimization prompt j (grounded, ungrounded or KL prompt). |
| r_j | Optimized suffix length for prompt j, starting at the lookup token. |
| C_prefill(p) | One forward pass to cache an unchanged prefix. |
| C_tail(p,r) | One suffix forward/backward step attending to a detached cached prefix. |
| C_value | Sum_j C_prefill(p_j) + T Sum_j C_tail(p_j,r_j). The number of prompt variants is fixed here. |
| C_fwd(ell_q) | Cost of an LLM forward pass used to obtain query keys or residuals. |
| C_enc(ell_q) | Cost of encoding the stored question with the RAG encoder. |

Token lengths and model dimensions are explicit inputs; costs are only constant in N when these other quantities are held fixed. The LLM optimization reads the supplied conversation. The matched RAG baseline embeds the stored question while retaining the full conversation and supplied Q/A as text. These are different computations.

The actual grounded optimizer caches each unchanged causal prefix once, crops the appended KV entries after a suffix pass and detaches the retained prefix. It does not repeat a full conversation prefill at every optimization step. With a Transformer layers and vocabulary V:

- C_prefill(p) = Theta(a[p hm + p h^2 + p^2 h]). This uses the model backbone without a full prefix vocabulary-logit projection.
- C_tail(p,r) = Theta(a[r hm + r h^2 + r(p+r)h] + r hV), up to forward/backward constants and the relevant gradient path. The suffix attends to the cached prefix, so its attention term is r(p+r), not a new p^2 prefill per step. The KL variant has its own short suffix and logit positions.
- An encoder with dimensions a_e, h_e and m_e has C_enc(ell_q) = Theta(a_e[ell_q h_e m_e + ell_q h_e^2 + ell_q^2 h_e]); it has no vocabulary-logit term here.

These are arithmetic-work expressions, not hardware runtime predictions. Dense-attention kernels can change memory use without removing the quadratic attention arithmetic in a full prefill. Without prefix caching, the optimizer would instead repeat full-sequence work T times; that is not the implemented writer being analyzed here.

## Per newly stored session

| Method | Write work, after one-time setup |
|---|---|
| Our independent compact delta, one key | Theta(B + C_value + e C_fwd(ell_q) + e(m^2 + hm) + e(h+m)). |
| Plain AlphaEdit, actual dense solver | Theta(B + C_value + e C_fwd(ell_q) + e(m^3 + m^2 h)). |
| RAG, flat vector append | Theta(C_enc(ell_q) + B + D), with amortized append storage. |

**Compact delta derivation.** Pay each prompt-prefix prefill once, then optimize the value target for up to T suffix steps (C_value above). At each edited layer, the writer computes keys/residuals, multiplies the precomputed dense projector by one key (m-by-m times m: Theta(m^2)), and forms/applies an h-by-m outer-product update (Theta(hm)). The single-key Sherman–Morrison expression avoids a dense matrix solve. Restoring the selected FFN weights also costs Theta(e hm), already covered by the update term. Saving two factor vectors per layer costs Theta(e(h+m)) in payload size and sequential serialization work. Filesystem fsync latency is a system cost that this asymptotic calculation cannot predict.

**Dense AlphaEdit derivation.** The original implementation forms a dense m-by-m system and solves for an m-by-h update. Dense matrix formation/solving contributes Theta(m^3 + m^2 h) per edited layer. Its preservation accumulator is carried across shared-model edits. The accumulator has fixed m-by-m shape, so this particular implementation does not introduce an additional explicit N factor per edit merely because more sessions have been edited. Its state and numerical conditioning do change.

**RAG derivation.** Encode the new stored question once, append D vector components, and retain B bytes of text/Q&A. Exact search is used at recall; scanning N vectors is a retrieval cost, not an ingestion cost. For an index other than append-only flat storage, use the general formula Theta(C_enc(ell_q) + B) + I(N,D), where I is that index's insertion cost.

If a different RAG design embeds a conversation as c chunks, replace C_enc(ell_q) by the sum of the c chunk-encoding costs and D by cD. The single-question-vector formula describes the matched comparator, not every possible RAG ingestion policy.

## Scaling with session count

Holding model dimensions, token lengths, optimization budget and per-session content size fixed:

| Quantity | Independent session deltas | Flat-append RAG |
|---|---|---|
| One additional memory write, as a function of N | Theta(1) | Amortized Theta(1) |
| Build N memories | Theta(N [B + C_value + e C_fwd + e(m^2+hm)]) | Theta(N [C_enc + B + D]) |

Both total build costs are linear in N under these assumptions. For variable session lengths, sum each session's individual write cost rather than substituting one common length. Big-O alone does not establish a measured speed ratio. The substantive difference is the work inside each write: cached-prefix LLM target optimization and weight updates versus one smaller encoder pass and index storage. Neither formula includes probing every delta, answer generation or reference-answer verification.

One-time model loading and null-space projector construction are outside the per-write costs. If included in a full experiment budget, add their setup cost once. The projector is precomputed in the existing experiment.

## Important storage implementation distinction

If a RAG implementation rewrites **the entire existing text/vector bank after each insert**, insertion additionally costs Theta(B_total(N) + ND). With bounded average record size this is Theta(N(B+D)) per insert and Theta(N^2(B+D)) cumulative serialization over N inserts. This overhead belongs to that file-persistence strategy, not to RAG in general. Batch index construction or amortized append storage gives the linear total build cost above.

The previously started timing runner uses full-bundle persistence. Its measured write timings will not be used in the requested comparison; the report uses this theoretical analysis instead.

## Implementation evidence

- `Desktop/draft_1/compact_session_memory.py`: `one_key_factors` and `CompactSessionMemory.write` define the compact one-key solve, updates and factor saving.
- `Desktop/draft_1/AlphaEdit/AlphaEdit_main.py`: `apply_AlphaEdit_to_model` defines the dense solve and carried preservation accumulator.
- `Desktop/draft_1/AlphaEdit/grounded_value.py`: target optimization and grounded prompts.
- `Desktop/draft_1/memory_comparison_core.py`: `embed` and exact vector search.
- `Desktop/draft_1/experiment_real512_compact.py`: batched question-vector index construction and saved conversation/Q&A records.

These formulas are derived from those local implementations. They do not claim that every AlphaEdit implementation uses a dense solver or that every RAG system uses the same index or embedding input.
