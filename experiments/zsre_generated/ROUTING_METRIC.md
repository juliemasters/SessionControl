# Routing accuracy extension

The running benchmark already logs `selected`, `source_hit` and `correct_rank`
for every SessionControl query. Routing accuracy counts selected delta IDs that
match the original question's written memory ID, divided by **all N questions**
at each checkpoint: 8, 16, 32, 64, 128, 256, 500 and 1,000. It does not depend on
generated-answer text, F1, exact match, EOS or whether generation reaches the cap.

Report the mean and sample standard deviation over ten actual inference passes
on one fixed trained bank. Those repetitions do not estimate training-seed
uncertainty. No new generation or routing runs are required for this addition.

The frozen running sources and input manifest remain unchanged. Start the CPU
report companion alongside the existing driver:

```sh
python -m experiments.zsre_generated.routing_report launch \
  --out runs/zsre-generated-sixmethod-1000-v3
```

The companion waits for **successful full benchmark completion**, audits the
saved routing IDs, flags, ranks, question order and source provenance, then writes:

- `routing/accuracy.csv` and `.json`: checkpoint means, sample SD and denominators.
- `routing/per_pass.csv`: correct counts and accuracy for all 80 measurement cells.
- `routing/decisions.csv`: all 20,040 decisions, including expected/selected IDs.
- `routing/accuracy.png`, `.svg`, `.pdf`: routing accuracy versus bank size.
- `routing/report.html` and `.md`: a standalone routing report.

It also appends the table and graph to the main report, preserves the original
report and manifest under `routing/`, and updates the artifact checksums. The
original comparison CSV retains its answer, timing and memory measurements;
routing metrics have their own CSV. No plots are generated from an unfinished or
failed benchmark. `routing_status.json` records companion progress independently
of the GPU driver. Code hashes for the companion and original input manifest are
recorded in `routing_extension.json`.
