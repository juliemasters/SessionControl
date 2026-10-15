#!/usr/bin/env python3
"""Check final aggregate compact512/TTFT evidence without publishing per-item records without loading models or requiring CUDA.

Original hashes attest to the recorded source artifacts. Only released_sha256 is
checked against the normalized copies in this checkout; the two are not aliases.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys


class ReleaseError(ValueError):
    """The public bundle is incomplete or inconsistent with its evidence."""


def require(condition, message):
    if not condition:
        raise ReleaseError(message)


def normalize(text):
    """Historical metric: casefold, word tokens, collapse whitespace."""
    return " ".join(re.findall(r"\w+", text.casefold()))


def close(actual, expected, label):
    require(isinstance(actual, (float, int)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isclose(actual, expected,
                                                       rel_tol=1e-10, abs_tol=1e-10),
            f"{label}: recorded {actual!r}, recomputed {expected!r}")


def exact_flags(trial):
    final = trial["final"]
    text, emitted, answer = final["text"], final["emitted_text"], trial["answer"]
    value, target = normalize(text).split(), normalize(answer).split()
    sources = trial.get("retrieved_sids") or []
    selected, reference = trial.get("selected_sid"), trial["reference_source_sid"]
    return dict(
        returned_exact=normalize(text) == normalize(answer),
        emitted_exact=normalize(emitted) == normalize(answer),
        verbatim_exact=text == answer,
        contains_reference=bool(target) and any(
            value[i:i + len(target)] == target for i in range(len(value) - len(target) + 1)),
        hit_cap=final["hit_cap"], eos=final["finish_reason"] == "eos",
        retrieval_at1=(sources[0] == reference) if sources else
                      (selected == reference if selected else None),
        retrieval_at3=(reference in sources[:3]) if sources else
                      (selected == reference if selected else None),
    )


def token_f1(trial):
    actual = normalize(trial["final"]["text"]).split()
    gold = normalize(trial["answer"]).split()
    if not actual or not gold:
        return float(actual == gold)
    overlap = sum((collections.Counter(actual) & collections.Counter(gold)).values())
    return 2 * overlap / (len(actual) + len(gold))


def verify_summary(summary, trials, label):
    require(summary.get("n") == len(trials) and bool(trials), f"{label}: sample count mismatch")
    for key in ("returned_exact", "emitted_exact", "verbatim_exact", "contains_reference",
                "hit_cap", "eos", "retrieval_at1", "retrieval_at3"):
        values = [exact_flags(t)[key] for t in trials if exact_flags(t)[key] is not None]
        metric = summary.get(key)
        if not values:
            require(metric is None, f"{label}/{key}: undefined metric must be null")
            continue
        require(isinstance(metric, dict), f"{label}/{key}: missing metric")
        count = sum(values)
        require(metric.get("count") == count, f"{label}/{key}: count mismatch")
        close(metric.get("rate"), count / len(values), f"{label}/{key}/rate")
        bounds = metric.get("ci95")
        require(isinstance(bounds, list) and len(bounds) == 2 and
                -1e-10 <= bounds[0] <= count / len(values) + 1e-10 and
                count / len(values) - 1e-10 <= bounds[1] <= 1 + 1e-10,
                f"{label}/{key}: invalid confidence interval")
    close(summary.get("mean_token_f1"), sum(token_f1(t) for t in trials) / len(trials),
          f"{label}/mean_token_f1")
    if "mean_rank" in summary:
        ranks = [t["rank"] for t in trials]
        require(all(isinstance(rank, int) and rank >= 1 for rank in ranks),
                f"{label}: invalid ranks")
        close(summary["mean_rank"], sum(ranks) / len(ranks), f"{label}/mean_rank")
        close(summary["mrr"], sum(1 / rank for rank in ranks) / len(ranks), f"{label}/mrr")






def verify_manifest(root):
    manifest_path = root / "docs/results/export-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("schema_version") == 1 and manifest.get("normalization"),
            "Export manifest must state its schema and normalization")
    files = manifest.get("files", [])
    require(bool(files), "Export manifest has no files")
    paths = set()
    for entry in files:
        relative = Path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"Unsafe exported path: {relative}")
        require(str(relative) not in paths, f"Duplicate export manifest path: {relative}")
        paths.add(str(relative))
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"Missing exported file: {relative}")
        data = path.read_bytes()
        require(len(data) == entry["bytes"], f"Export size mismatch: {relative}")
        require(hashlib.sha256(data).hexdigest() == entry["released_sha256"],
                f"Released digest mismatch: {relative}")
        require(re.fullmatch(r"[a-f0-9]{64}", entry["original_sha256"]) is not None,
                f"Invalid original historical digest: {relative}")
        require(entry["normalized"] == (entry["original_sha256"] != entry["released_sha256"]),
                f"Normalization flag contradicts digests: {relative}")
    actual = {str(p.relative_to(root)) for p in (root / "docs/results").rglob("*")
              if p.is_file() and p != manifest_path}
    require(actual == paths, "Export manifest does not cover exactly the released result files")
    return len(files)


def candidate_files(root):
    """Select public files from this checkout or a standalone extracted archive.

    Git must identify this exact root. An enclosing unrelated repository must
    not determine which files an extracted release verifies.
    """
    root = Path(root).resolve()
    try:
        probe = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root,
                               capture_output=True, check=False)
        if probe.returncode == 0 and Path(probe.stdout.decode().strip()).resolve() == root:
            result = subprocess.run(
                ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                cwd=root, capture_output=True, check=True)
            return [root / name.decode() for name in result.stdout.split(b"\0") if name]
    except FileNotFoundError:
        pass  # Extracted releases can be verified without installing Git.

    excluded_dirs = {".git", "__pycache__", ".cache", ".pytest_cache"}
    local_root_dirs = {"runs", "models", "checkpoints", "data", "results",
                       "session_deltas", "session_memory"}
    selected = []

    def visit(folder):
        for path in sorted(folder.iterdir()):
            # Never follow symlinks; include them so the public-file check rejects them.
            if path.is_symlink():
                selected.append(path)
            elif path.is_dir():
                if path.name not in excluded_dirs and not path.name.startswith((".venv", "venv")) \
                        and not (folder == root and path.name in local_root_dirs):
                    visit(path)
            elif path.suffix not in {".pyc", ".pyo", ".log"} and path.name != ".DS_Store":
                selected.append(path)

    visit(root)
    return selected


def verify_public_files(root):
    forbidden = {".pt", ".pth", ".safetensors", ".bin", ".ckpt", ".log", ".pyc"}
    secrets = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|"
                         r"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
    private_paths = re.compile(r"/ho" r"me/[^\s\"'<>]+|/tmp/ju" r"lie[^\s\"'<>]*|"
                               r"GPU-[0-9a-f]{8}-[0-9a-f-]{27,}")
    count = 0
    for path in candidate_files(root):
        require(path.exists() and not path.is_symlink(), f"Public file missing or symlinked: {path.name}")
        relative = path.relative_to(root)
        require(path.suffix.lower() not in forbidden and not path.name.startswith(".env")
                and not {"models", "checkpoints", ".venv", "runs", "session_deltas"}.intersection(relative.parts),
                f"Local weight, credential, log, or run artifact included: {relative}")
        require(path.stat().st_size < 100 * 1024 * 1024, f"GitHub file limit exceeded: {relative}")
        if path.suffix.lower() in {".py", ".md", ".json", ".jsonl", ".html", ".yml", ".yaml", ".txt", ".svg"}:
            text = path.read_text()
            require(secrets.search(text) is None, f"Possible credential found in {relative}")
            require(private_paths.search(text) is None, f"Private machine path or GPU UUID in {relative}")
        count += 1
    return count












def verify_aggregate_tables(root):
    from fixed_query_scaling_core import SIZES
    folder = root/'docs/results'
    compact = json.loads((folder/'scaling-real512-compact-v1/comparison_table.json').read_text())
    timed = json.loads((folder/'scaling-real512-ttft-v1/comparison_table.json').read_text())
    accounting = json.loads((folder/'scaling-real512-compact-v1/memory_accounting.json').read_text())
    rss = json.loads((folder/'scaling-real512-compact-v1/sampled_cpu_summary.json').read_text())['conditions']
    resources = json.loads((folder/'scaling-real512-compact-v1/resource_summary.json').read_text())['measurements']
    names = {'Compact AlphaEdit':'our_delta','Matched RAG':'rag_qa','Full-bank KV':'kv_full_qa'}
    expected = {(n,m) for n in SIZES for m in names.values()}
    require(compact['accuracy_denominator'] == 8, 'Accuracy denominator must remain eight')
    require(len(compact['rows']) == len(timed['rows']) == len(expected), 'Missing aggregate rows')
    mapped = {(r['n'],names[r['method']]):r for r in compact['rows']}
    timing = {(r['n'],r['method']):r for r in timed['rows']}
    require(set(mapped) == set(timing) == expected, 'Missing or duplicate size/method rows')
    measured = 0
    for (n,method), row in mapped.items():
        t = timing[(n,method)]
        available = method != 'kv_full_qa' or n <= 32
        require(row['available'] is available and t['status'] == ('measured' if available else 'context_limit'),
                'Incorrect method availability')
        if not available:
            for key in ['source_top1','exact','contains','retained_gib','sampled_peak_host_rss_bytes',
                        'peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','median_seconds','p95_seconds']:
                require(row[key] is None, f'Unavailable {key} must be null')
            for key in ['routing_correct','exact','mean_token_f1','ttft_median_ms','ttft_p95_ms',
                        'total_median_ms','kv_cold_first_request_ms','kv_prefix_setup_ms']:
                require(t[key] is None, f'Unavailable {key} must be null')
            require(t['accuracy_queries'] == t['warm_timing_samples'] == 0, 'Unavailable samples must be zero')
            continue
        measured += 1
        require(t['accuracy_queries'] == 8 and t['warm_timing_samples'] == 16,
                'Timing repeats must not inflate accuracy denominator')
        require(isinstance(row['exact'],int) and 0 <= row['exact'] <= 8 and row['exact'] == t['exact'],
                'Inconsistent exact counts')
        require(row['source_top1'] == t['routing_correct'], 'Inconsistent source selection counts')
        require(row['source_top1'] is None if method == 'kv_full_qa' else
                isinstance(row['source_top1'],int) and 0 <= row['source_top1'] <= 8,
                'Invalid source selection count')
        require(0 <= t['mean_token_f1'] <= 1, 'F1 must be a proportion')
        require(0 <= t['ttft_median_ms'] <= t['ttft_p95_ms'] and
                t['total_median_ms'] >= t['ttft_median_ms'], 'Invalid TTFT bounds')
        require(0 <= row['median_seconds'] <= row['p95_seconds'], 'Invalid latency bounds')
        memory = accounting['per_bank'][str(n)][method]
        close(row['retained_gib'],memory['retained_payload_bytes']/2**30,'bank storage')
        require(row['tier'] == memory['tier'], 'Memory tier mismatch')
        resource = resources[method+'_'+str(n)]
        for field,sourcefield in [('peak_gpu_allocated_bytes','peak_allocated_bytes'),
                                  ('peak_gpu_reserved_bytes','peak_reserved_bytes')]:
            require(row[field] == max(resource[sourcefield],resource.get('setup_'+sourcefield,0)),
                    'GPU memory mismatch')
        sample = rss[str(n)+'/'+method]
        require(row['sampled_peak_host_rss_bytes'] == sample['sampled_peak_rss_bytes']
                and row['host_rss_samples'] == sample['samples'], 'RSS sample mismatch')
    return measured*2


def verify_aggregate_privacy(root):
    forbidden = {'question','answer','reference','output','text','token_ids','trials',
                 'references','exact_mismatches','own_question_pairs','routing_checks'}
    def check(value):
        if isinstance(value,dict):
            require(not forbidden.intersection(value), 'Per-item records must remain local')
            for child in value.values():check(child)
        elif isinstance(value,list):
            for child in value:check(child)
    for path in (root/'docs/results').rglob('*.json'):
        check(json.loads(path.read_text()))


def verify_release(root):
    root = Path(root).resolve()
    exports = verify_manifest(root)
    verify_aggregate_privacy(root)
    protocol = json.loads((root/'docs/results/aggregate_protocol.json').read_text())
    require(protocol['stored_sessions'] == 512 and protocol['unique_evaluation_queries'] == 8
            and protocol['repeats'] == 3 and protocol['warm_samples_per_condition'] == 16,
            'Changed experiment scope')
    for name in ['scaling-real512-compact-v1','scaling-real512-ttft-v1']:
        folder = root/'docs/results'/name
        completed = json.loads((folder/'completion.json').read_text())
        require(completed['completed'] and completed['base_weights_restored']
                and completed['physical_gpu'] == 1 and completed['real_sessions'] == 512,
                'Run did not complete')
        require((folder/'report.html').is_file(), 'Missing aggregate HTML report')
        for png in folder.glob('*.png'):
            require(png.with_suffix('.svg').is_file() and png.with_suffix('.pdf').is_file(),
                    'Missing vector counterpart')
    conditions = verify_aggregate_tables(root)
    return dict(exported_artifacts=exports, verified_measured_condition_rows=conditions,
                unique_original_questions=8, public_files=verify_public_files(root))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    sys.path.insert(0, str(args.root.resolve()))
    try:
        result = verify_release(args.root)
    except (ReleaseError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"Release verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
