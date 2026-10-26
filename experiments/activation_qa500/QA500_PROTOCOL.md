# Corrected Q/A memory experiment through 500 stored conversations

This protocol supersedes the evidence-capsule variant for the requested answer-writing comparison. It preserves that older run as a separate artifact; its scores are not mixed into these results.

## Write contract

Both AlphaEdit conditions receive, on every new write:

1. The **complete original conversation** as grounding context.
2. The **exact official question**, keyed as `Q: {question}\nA:`.
3. The **complete official reference answer**, followed by EOS, as the actual optimization target.

There is no capsule extraction, summarization, answer clipping, or evidence-fusion reader. The request construction is shared by both writers. Unit tests deliberately insert a conflicting capsule field and check that the full reference answer is still used. The first new executed requests are retained in each construction directory for inspection.

One independent delta represents one conversation. A conversation associated with two official questions has a joint rank-two update. For multi-session questions, the complete supervised answer is explicitly provided to every associated conversation's write. This is an exact-question memory-storage experiment; it does **not** test whether a model can derive an unseen answer by combining partial facts from multiple conversations.

## Reuse and checkpoints

The first 64 conversations retain their original order. Their contexts, questions and answers were checked against the official release. Reuse the exact saved compact answer-delta files, shared AlphaEdit patches at N=8,16,32,64, and the saved post-update preservation keys. The shared accumulator must reconstruct exactly before additional editing begins. Legacy compact deltas are previously validated approximations of rounded dense updates, not globally lossless compression.

Append 436 new original conversations in deterministic complete-support question groups. Stop at **500 stored conversations**, as requested. Do not reuse the 940 evidence-only deltas or the 448 derived quote-completion questions.

| Stored conversations | Eligible official questions per pass |
|---:|---:|
| 8 | 8 |
| 16 | 16 |
| 32 | 32 |
| 64 | 64 |
| 128 | 98 |
| 256 | 162 |
| 500 | 283 |

An official question becomes eligible only when all of its annotated supporting/related conversations have been stored. Every eligible question is evaluated at every checkpoint. Therefore 500 stored conversations do not imply 500 evaluated questions. The full release contains 500 questions spanning 940 supporting/related conversations; the latest instruction stops the bank at 500.

## Four conditions

- **SessionControl:** capture base-model question activations at FFN down-projection inputs in layers 13–15. Score all first-N deltas by normalized update energy and choose the argmax, with a fixed lexical session-ID tie rule. Restore the base weights, load/apply only the selected answer delta, generate directly, and restore the base weights again. No BERT, answer-aware scorer, retrieved text or extra reader is used.
- **AlphaEdit without session control:** one continuously edited model, original dense solver and cumulative preservation accumulator. Reuse the old trajectory through 64, append the exact same new question/context/answer bindings, and directly generate from each saved checkpoint. No independent-delta summation or session router.
- **Matched Q/A RAG:** index the supplied write questions with MiniLM. Retrieve one conversation and supply its entire conversation plus associated stored questions **and complete answers** to the base LLM. Answers are not withheld from this baseline.
- **Full-bank Q/A KV cache:** retain all first-N complete conversation/Q&A records in one reusable prefix. Tokenize and check the actual required context independently on each pass. Infeasible checkpoints are explicitly unavailable. Do not truncate the bank, substitute retrieval, or invent F1/TTFT for an unavailable condition.

The source bank is selected from official supporting/related-session metadata and all evaluation questions are supplied at write time. This is a supervised, open-book write mechanism experiment, not an official blind LongMemEval score or held-out generalization test.

## Ten fresh evaluation passes

Reuse one completed frozen bank. Run **ten actual inference passes** at **every checkpoint** for every feasible condition. Do not rebuild memories ten times or copy previous measurements into new passes. Greedy decoding is deterministic; quality SD can legitimately be zero. Timing and resource measurements may vary.

Use the same question set and query order across methods within a pass. Shuffle deterministically between passes and rotate method execution order. Use a 128-token generation cap for all methods; the longest official reference has 104 Llama tokens. Use EOS/new-turn stopping without reference-guided answer extraction. Two unrelated one-token warmups precede measurement. No OS cache flush is claimed.

The N=8 pilot is pass 1 at that checkpoint, not an eleventh pass. It verifies that the reused answer deltas reproduce their original token sequences before launching longer writes. Later execution skips that completed pilot cell and remeasures every remaining pass/checkpoint. Each completed cell is stored with its own raw outputs, timings, resources and code provenance.

## Metrics

Report **mean ± sample SD across ten per-pass metrics**:

- Whole-answer custom normalized word-token F1, normalized exact match, reference containment and generation-cap rate.
- Source hit@1, correct-source rank, MRR, and support recall/completeness@6 for activation and RAG. Source metrics are undefined for the shared model and KV; abstention items are excluded from answer-bearing source metrics.
- Mean visible TTFT, mean of per-pass median TTFT, p95 visible TTFT, first-generated-token timing and full-query latency. Count missing visible timestamps instead of imputing them.
- Extra deployed bank bytes, process peak host RSS, retained factor/index/cache payload, and CUDA peak allocated/reserved bytes. Keep storage tiers separate and identify the common base-model weight payload.

TTFT includes routing, selected-delta loading/application, tokenization, prefill and generation up to the first nonempty answer. It excludes model startup, memory writing and already-built KV prefix setup; cold-prefix first-request timing and actual cache payload are separate. Full-query latency includes base restoration. CUDA peaks include per-checkpoint bank setup. Construction occurs in separate processes so projector and preservation-state memory do not inflate inference memory.

Report all eligible questions, original64 and new-item groups, question categories, abstentions, and questions newly eligible at each checkpoint. The original64 group is a fixed retention panel from N=64 onward. Different eligible sets change difficulty, so the aggregate growth curve alone does not isolate bank-size effects. Lexical F1 is not a semantic factuality judge. Ten inference passes are not ten independent model-training replications.

Graphs are created only after the complete feasible ten-pass grid is audited. Include separate quality/routing, TTFT/latency and storage/RAM/GPU graphs. Mark KV context failures unavailable. The 100% quality line is an aspirational target; calculated compact tensor payload is an explicitly labeled storage estimate, not measured GPU use.

## Execution and storage

Only physical GPU 1 is permitted. The detached driver and progress monitor survive the interactive tool session. A restarted worker reuses committed deltas and complete evaluation cells. Shared-model restarts reconstruct the preservation accumulator from exact saved post-update keys.

The compact bank is approximately 107 MiB of tensor payload, with a strict 512 KiB per-session serialization limit. There is one dense patch per checkpoint plus a rolling atomic restart patch, not one dense file per conversation. The conservative storage budget is approximately 8.1 GiB and a 2 GiB free-space reserve is enforced. Existing results remain intact. Write-time comparisons remain theoretical in `WRITE_COMPLEXITY.md`.

On October 22 the independent writer stopped after 110 committed conversations: tokenizer decode cleanup removed a space before a quotation in the next complete answer, shortening its teacher-forcing sequence. The correction disables punctuation cleanup during decoding. A full 500-conversation audit verified exact prompt-plus-answer token alignment after the fix and identical teacher inputs for all 110 committed memories. Those deltas and actual pilot measurements are retained. Original executed source snapshots remain under `code_revisions/before-alignment-fix`; `token_alignment_audit.json` and `alignment_fix_resume.json` record the compatibility check and explicit source revision. No reference answer, question, context or evaluation protocol changes.
