# Published artifact provenance

`results/memory-comparison-v1/` and `results/memory-scaling-v1/` contain copies from completed GPU 1 experiments. They are historical results; packaging did not execute another model experiment.

The export includes measured outputs, summaries, completion records, standalone HTML reports, and unchanged PNG/SVG graphs. It excludes weights, dense deltas, projector tensors, embedding indexes, raw dataset records, full-session banks, process logs, and resource sampling streams.

Machine-specific absolute experiment paths, temporary projector paths, and GPU UUIDs are replaced with placeholders in text artifacts. Numerical measurements, question/answer excerpts, and generated outputs are preserved. Physical GPU index 1 remains recorded.

[export-manifest.json](results/export-manifest.json) records both the original and released SHA256 digest for each artifact. Hashes inside historical completion/provenance/review records still refer to original files. They must not be treated as the digest of a normalized released file. Standalone exported reports can be viewed without the models; regenerating reports requires a full local completed run and its original integrity artifacts.

Run `python scripts/verify_release.py` to verify released-file hashes and independently reproduce available-condition sample counts and answer/source aggregates. This checks the exported evidence, not the omitted tensors or original GPU execution. CPU checks do not certify the new bootstrap entry point's complete GPU path.

The existing 24-session report labels 100% as an aspirational target. Its memory-scaling curves are calculated expectations. The later 24–64 report contains actual points at each completed scale, plus calculated KV payload at scales that exceeded context capacity. No missing baseline is assigned zero accuracy, and no unmeasured accuracy is presented as a forecast.
