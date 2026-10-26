# SessionControl: latest measured method

This source release contains the activation-routing method used in the completed **500-conversation, ten-pass experiment**, `qa500-checkpoints-tenpass-v2`.

**Full context + exact question + complete answer → independent compact AlphaEdit deltas → base-model question activations → normalized delta-response scores → selected delta → final generated answer.** No BERT scorer or per-delta candidate generation is used.

Read [the method, file roles and experiment commands](experiments/activation_qa500/README.md). Run `python verify_latest_source.py` to check the 17 execution-tracked source files against the completed run's saved hashes. The verification is CPU-only and starts no experiment.

## Key implementation files

- [Memory write](session_write.py) and [grounded answer optimization](AlphaEdit/grounded_value.py).
- [Question activation capture and rank-one scoring](compact_session_memory.py).
- [General low-rank delta response and compact write solver](experiments/activation_qa500/full500_core.py).
- [Current routing and final answer inference](experiments/activation_qa500/qa500_eval.py).
- [Answer-boundary generation](answer_only_generation.py) and [TTFT instrumentation](ttft_timing.py).
- [Experiment protocol](experiments/activation_qa500/QA500_PROTOCOL.md), [ten-pass driver](experiments/activation_qa500/qa500_remeasure.py), [report generator](experiments/activation_qa500/qa500_report.py) and [focused memory chart](experiments/activation_qa500/add_focused_memory_report.py).

The full experiment scripts are preserved with their original workspace paths. External model weights, official dataset, null-space projector and stored delta bank are required and excluded from this source archive. See the experiment README before running on a different machine. The archive is source code, not a trained model checkpoint.

The reference answers are visible at write time and the same official questions are asked at recall. This is a supervised memory mechanism test. Accuracy uses only the complete final generated answer; ten inference repetitions share one trained frozen bank.
