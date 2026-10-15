# Aggregate result provenance

The public release contains final code, aggregate measurements, standalone aggregate HTML reports, architecture diagrams and PNG/SVG/PDF graphs. Per-item questions, reference answers, generated outputs, token IDs, session identifiers, raw conversations, factors, models and sampling logs remain local. Earlier reports are not duplicated in the current tree.

`results/export-manifest.json` records original and released SHA256 digests. Public HTML and validation/completion summaries omit per-item content. Embedded source hashes refer to the original local artifacts, while released hashes identify the exported copies. Graph bytes and aggregate measurements are unchanged. The public reports were rebuilt from aggregate tables and graphs, without another neural experiment.

Run `python scripts/verify_release.py` for artifact hashes, fixed accuracy/timing denominators, aggregate consistency, memory tiers and per-item exclusion checks. The local detailed evidence was checked before export, but public aggregates cannot independently establish per-answer correctness or original GPU execution.
