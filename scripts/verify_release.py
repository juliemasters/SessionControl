#!/usr/bin/env python3
"""Check released historical evidence without loading models or requiring CUDA.

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


def verify_condition(condition, label, expected_n):
    if condition.get("available") is False:
        require(not condition.get("trials") and condition.get("summary") is None
                and condition.get("resources") is None,
                f"{label}: unavailable condition must have no measured result")
        require(condition.get("reason") == "context_limit", f"{label}: unexpected unavailable reason")
        plan = condition["plan"]
        require(plan["required_tokens"] > plan["context_limit"],
                f"{label}: claimed context overflow does not exceed capacity")
        require(plan["expected_payload_bytes"] == plan["prefix_tokens"] * plan["bytes_per_token"],
                f"{label}: calculated cache payload does not match formula")
        return 0
    trials = condition.get("trials", [])
    require(len(trials) == expected_n, f"{label}: expected {expected_n} trials, found {len(trials)}")
    require(len({t["qid"] for t in trials}) == len(trials), f"{label}: duplicate questions")
    for trial in trials:
        for key, value in exact_flags(trial).items():
            require(trial.get(key) == value, f"{label}/{trial['qid']}/{key}: inconsistent trial metric")
        close(trial["token_f1"], token_f1(trial), f"{label}/{trial['qid']}/token_f1")
        if "routing_correct" in trial:
            require(trial["routing_correct"] ==
                    (trial["selected_sid"] == trial["reference_source_sid"]),
                    f"{label}/{trial['qid']}: inconsistent routing correctness")
    verify_summary(condition["summary"], trials, label)
    if "anchor24_summary" in condition:
        anchor = [t for t in trials if t.get("history") == "previous_test"]
        verify_summary(condition["anchor24_summary"], anchor, label + "/anchor24")
    if "history_summaries" in condition:
        histories = {t["history"] for t in trials}
        require(set(condition["history_summaries"]) == histories, f"{label}: missing history strata")
        for history, summary in condition["history_summaries"].items():
            verify_summary(summary, [t for t in trials if t["history"] == history],
                           f"{label}/{history}")
    resources = condition["resources"]
    for key, value in resources.items():
        if key.endswith("_bytes"):
            require(isinstance(value, int) and value >= 0, f"{label}/{key}: invalid memory measurement")
    return len(trials)


def verify_evaluation(evaluation, completion, label):
    require(completion.get("completed") is True, f"{label}: experiment did not complete")
    total = 0
    expected_methods = {"our_delta", "rag_qa", "rag_context_only", "kv_full_qa"}
    if "nested_bank_sizes" in evaluation:
        sizes = evaluation["nested_bank_sizes"]
        require(sizes == completion["nested_bank_sizes"], f"{label}: inconsistent bank sizes")
        require(set(evaluation["results"]) == {str(n) for n in sizes}, f"{label}: missing bank sizes")
        previous = None
        for n in sizes:
            methods = evaluation["results"][str(n)]
            require(set(methods) == expected_methods,
                    f"{label}/{n}: incomplete method set")
            current = [t["qid"] for t in methods["our_delta"]["trials"]]
            require(previous is None or current[:len(previous)] == previous,
                    f"{label}: banks are not nested in recorded order")
            previous = current
            for method, condition in methods.items():
                total += verify_condition(condition, f"{label}/{n}/{method}", n)
                if condition.get("available", True):
                    require([t["qid"] for t in condition["trials"]] == current,
                            f"{label}/{n}/{method}: method question order differs")
    else:
        require(set(evaluation["results"]) == expected_methods,
                f"{label}: incomplete method set")
        require(len(evaluation["results"]) == completion["methods_completed"],
                f"{label}: completed method count mismatch")
        for method, condition in evaluation["results"].items():
            total += verify_condition(condition, f"{label}/{method}", completion["answers_per_method"])
    return total


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


def verify_release(root):
    root = Path(root).resolve()
    exports = verify_manifest(root)
    trials = 0
    for name in ("memory-comparison-v1", "memory-scaling-v1"):
        folder = root / "docs/results" / name
        evaluation = json.loads((folder / "evaluation.json").read_text())
        completion = json.loads((folder / "completion.json").read_text())
        trials += verify_evaluation(evaluation, completion, name)
        require((folder / "report.html").is_file(), f"{name}: missing standalone report")
        pngs = list(folder.glob("*.png"))
        require(bool(pngs), f"{name}: missing graphs")
        for png in pngs:
            require(png.with_suffix(".svg").is_file(), f"{name}: missing SVG counterpart for {png.name}")
    files = verify_public_files(root)
    return dict(exported_artifacts=exports, verified_final_answers=trials, public_files=files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        result = verify_release(args.root)
    except (ReleaseError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"Release verification failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
