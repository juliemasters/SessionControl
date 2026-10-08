"""Run the frozen64-session protocol from a clean checkout on physical GPU1.

Reproduction first writes40 real source deltas, then the unchanged scaling
runner writes the other24 and compares nested24/32/48/64 banks. No historical
validation or comparison metrics are synthesized. A source completion marker
means actual writes and base-weight restoration, not a new validation study.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.prepare_experiment import CORPUS_SHA256, DATASET_SHA256, sha256, write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_model_link(target, expected):
    """Honor supplied local model paths without changing another checkpoint."""
    target, expected = Path(target).resolve(), Path(expected)
    if not target.is_dir() or not (target / "config.json").is_file():
        raise ValueError("Expected a downloaded local model directory: " + str(target))
    if expected.exists() or expected.is_symlink():
        if expected.resolve() != target:
            raise FileExistsError("Existing runtime model differs: " + str(expected))
        return
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.symlink_to(target, target_is_directory=True)


def check_preparation(prepared):
    prepared = Path(prepared).resolve()
    metadata = read(prepared / "preparation.json")
    if metadata["dataset_sha256"] != DATASET_SHA256:
        raise ValueError("Wrong dataset fingerprint")
    if sha256(prepared / "items.json") != metadata["items_sha256"]:
        raise ValueError("Prepared cohort changed")
    if sha256(prepared / "distractors.jsonl") != CORPUS_SHA256:
        raise ValueError("Prepared preservation corpus changed")
    items = read(prepared / "items.json")
    if len(items) != 64 or len({r["sid"] for r in items}) != 64:
        raise ValueError("Incomplete or repeated64-session bank")
    return metadata, items


def check_projection(projection):
    projection = Path(projection).resolve()
    metadata = read(projection.with_suffix(".json"))
    if metadata["layers"] != [13, 14, 15] or metadata["corpus_sha256"] != CORPUS_SHA256 or metadata["threshold"] != .02:
        raise ValueError("Projection must use the original distractor corpus, layers, and threshold")
    if not projection.is_file():
        raise FileNotFoundError(projection)
    return projection, metadata


def verify_source(source):
    complete = read(source / "completion.json")
    if not complete.get("completed") or not complete.get("base_weights_restored"):
        raise ValueError("Actual successful source writes are required")
    if complete.get("selected_writer") != "answer_plus_eos" or complete.get("physical_gpu") != 1:
        raise ValueError("Source must use the frozen answer-plus-EOS writer on physicalGPU1")
    hashes = complete["new_delta_sha256"]
    if len(hashes) != 40:
        raise ValueError("Bootstrap must contain40 independent source deltas")
    for sid, expected in hashes.items():
        if sha256(source / "deltas" / sid / "v1.pt") != expected:
            raise ValueError("Source delta changed: " + sid)
    return complete


def bootstrap(root, prepared, projection, out):
    _, items = check_preparation(prepared)
    source = out / "source-baseline"
    if source.exists():
        raise FileExistsError("Use --run-only after a successful bootstrap; failed writes require a new output")
    cohort = [r for r in items if r["history"] in {"previous_test", "writer_development"}]
    if len(cohort) != 40:
        raise ValueError("Expected40 original repair-writer items")
    required = len(cohort) * 704644657 + 3 * 1024 ** 3
    if shutil.disk_usage(out).free < required:
        raise RuntimeError("Need at least29.25GiB free for40 source delta files and write headroom")
    from zsre_routing_experiment import require_gpu1
    gpu = require_gpu1()  # Before every torch/model import.
    busy = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True, check=True)
    if busy.stdout.strip():
        raise RuntimeError("PhysicalGPU1 is busy")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    import torch
    from routing_ladder_models import AlphaEditAdapter
    from session_qa_routing_core import write_request
    from answer_only_generation import generate_answer_batch, reference_tokens_present
    from question_key_k1_core import normalize
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Only physicalGPU1 may be visible")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    source.mkdir()
    write_json(source / "items.json", cohort)
    write_json(source / "design.json", dict(
        kind="fresh_source_writer_bootstrap", physical_gpu=1, layers=[13, 14, 15],
        projection=str(projection), projection_sha256=sha256(projection),
        writer="Frozen answer_plus_eos;50steps,L2=.1; full source session and original supplied Q/A",
        role="Actual40 independent source writes only; no recreated validation, calibration, or historical evaluation",
        validation_performed=False, writer_selection_performed=False,
    ))
    write_json(source / "completion.json", dict(completed=False, stage="writing", physical_gpu=1))
    adapter, writes = None, []
    try:
        adapter = AlphaEditAdapter(root, source / "deltas", layers=[13, 14, 15], projection_path=projection)
        eos = adapter.tok.eos_token
        eos_id = adapter.tok.convert_tokens_to_ids(eos)
        generation_eos = adapter.model.generation_config.eos_token_id
        if eos_id not in (generation_eos if isinstance(generation_eos, list) else [generation_eos]):
            raise ValueError("Write EOS is not a generation stop token")
        write_json(source / "writer_config.json", dict(hparams=vars(adapter.hp), eos_token=eos, eos_id=eos_id))
        requests = []
        for index, row in enumerate(cohort):
            request = write_request(row, index)
            request["target_new"]["str"] = row["answer"] + eos
            if adapter.tok.encode(" " + request["target_new"]["str"], add_special_tokens=False)[-1] != eos_id:
                raise ValueError("EOS missing from write target")
            grounded = request["value_context_templates"][0][1].format(request["prompt"]).format(request["subject"])
            if len(adapter.tok.encode(grounded + " " + request["target_new"]["str"])) + 64 > adapter.model.config.max_position_embeddings:
                raise ValueError("Complete source write exceeds context capacity")
            requests.append(request)
        for index, (row, request) in enumerate(zip(cohort, requests)):
            started = time.monotonic()
            adapter.store.write_session(row["sid"], adapter.tok, [request], adapter.hp,
                                        torch.zeros_like(adapter.P), adapter.P, 1)
            final = adapter.store.run(row["sid"], 1, lambda model: generate_answer_batch(
                model, adapter.tok, [row["question"]], stop_at_turn=True))[0]
            delta = source / "deltas" / row["sid"] / "v1.pt"
            writes.append(dict(qid=row["qid"], sid=row["sid"], sha256=sha256(delta), bytes=delta.stat().st_size,
                               elapsed_seconds=time.monotonic() - started, immediate=final,
                               immediate_exact=normalize(final["text"]) == normalize(row["answer"]),
                               immediate_contains=reference_tokens_present(final["text"], row["answer"])))
            write_json(source / "writes.json", writes)
            print("BOOTSTRAP", index + 1, 40, row["qid"], "exact", writes[-1]["immediate_exact"], flush=True)
        adapter.store.deactivate()
        for name, base in adapter.store._base.items():
            if not torch.equal(adapter.model.get_parameter(name).detach().cpu(), base):
                raise ValueError("Source writer did not restore base weights")
        hashes = {row["sid"]: sha256(source / "deltas" / row["sid"] / "v1.pt") for row in cohort}
        if any(hashes[row["sid"]] != row["sha256"] for row in writes):
            raise ValueError("Source delta changed during bootstrap")
        write_json(source / "completion.json", dict(
            completed=True, stage="actual_source_writes_completed", physical_gpu=1, gpu_uuid=gpu,
            selected_writer="answer_plus_eos", writer_selection_performed=False, validation_performed=False,
            writes=len(writes), base_weights_restored=True, new_delta_sha256=hashes,
            scope="Completed refers solely to40 actual frozen-writer writes and restoration checks; no calibration/evaluation recreated",
        ))
    except BaseException as error:
        write_json(source / "completion.json", dict(completed=False, stage="failed", error=repr(error), physical_gpu=1))
        raise
    finally:
        if adapter is not None:
            adapter.store.deactivate()
        adapter = None
        gc.collect()
        torch.cuda.empty_cache()
    return source


def prepare_scaling(root, prepared, projection, out):
    """Create the honest metadata contract for the unchanged scaling.run."""
    metadata, items = check_preparation(prepared)
    source = out / "source-baseline"
    complete = verify_source(source)
    directory = out / "scaling"
    if directory.exists():
        raise FileExistsError("Scaling output already exists; completed and failed runs are immutable")
    reusable = [r for r in items if r["sid"] in complete["new_delta_sha256"]]
    new = [r for r in items if r["sid"] not in complete["new_delta_sha256"]]
    if len(reusable) != 40 or len(new) != 24:
        raise ValueError("Source writers do not match the published manifest")
    if shutil.disk_usage(out).free < len(new) * 704644657 + 3 * 1024 ** 3:
        raise RuntimeError("Insufficient disk for24 additional dense deltas")
    design = dict(
        physical_gpu=1, seed=0, nested_bank_sizes=[24, 32, 48, 64],
        item_order="Published question/session identifiers preserve original nested cohort order",
        source=str(source), delta_root=str(directory / "deltas"), layers=[13, 14, 15],
        model=str(root / "models/Llama-3.1-8B-Instruct"), embedding_model=str(root / "models/all-MiniLM-L6-v2"),
        dtype="float32", batch_size=1, projection=str(projection), projection_sha256=sha256(projection),
        dataset_sha256=metadata["dataset_sha256"],
        source_hashes={name: sha256(source / name) for name in ["items.json", "design.json", "completion.json", "writes.json"]},
        previous_comparison=None, previous_comparison_evaluated=False, comparison_hashes={},
        reused_delta_sids=[r["sid"] for r in reusable], new_delta_sids=[r["sid"] for r in new],
        writer="Frozen answer-plus-EOS AlphaEdit objective; full original source session and supplied Q/A;50steps,L2=.1",
        router="Normalized delta activation energy across13/14/15; exact question only; lexical session tie break",
        rag="MiniLM384-dimensional normalized original question-key dense index; top1 full session+supplied Q/A",
        secondary="Context-only MiniLM chunks192wordpieces,32overlap; top3chunks; no supplied Q/A",
        kv="Uncompressed float32 reusable full-bank prefix; chunk256; original query suffix; no truncation",
        kv_capacity="Context/GPU capacity limits reported as unavailable; never zero accuracy",
        allocator="expandable_segments:True in all conditions", recall="Exact original questions; greedy64token cap; EOS/next-turn stopping",
        supervision="All primary methods receive full source sessions plus supplied original question/answer at memory creation",
        scope="Clean-checkout reproduction on the previously inspected64-session exploratory cohort; not a new held-out/generalization experiment",
        capacity_scope="One exact supplied question per session; does not prove preservation of every fact",
        measures="Normalized delivered/emitted/verbatim exact, bounded containment, tokenF1, source ranks; lexical metrics are not semantic truth",
        memory="Separate retained bytes and storage tier from measured setup/recall GPU peaks; writes reported separately",
        latency="Bulk activation and query generation reported separately; not a serving throughput benchmark",
        accuracy_target="100% aspirational, not forecast", new_write_expected_bytes=len(new) * 704644657,
        fresh_bootstrap=True, historical_validation_reproduced=False,
        references=dict(kv="https://huggingface.co/docs/transformers/kv_cache",
                        embedding="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
                        rag="https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html"),
    )
    directory.mkdir()
    write_json(directory / "items.json", items)
    write_json(directory / "design.json", design)
    write_json(directory / "manifest_sha256.json", {name: sha256(directory / name) for name in ["items.json", "design.json"]})
    write_json(directory / "completion.json", dict(completed=False, stage="prepared"))
    return directory


def reexec_for_scaling(args):
    """Drop the bootstrap CUDA context before the runner's idle-GPU guard.

    Clearing tensors/cache leaves the CUDA process registered with nvidia-smi.
    Replacing this process releases that context; scaling starts in a clean
    interpreter and keeps the original runner's external-process safety check.
    """
    argv = [sys.executable, str(Path(__file__).resolve()),
            "--prepared", str(args.prepared.resolve()),
            "--projection", str(args.projection.resolve()),
            "--model", str(args.model.resolve()),
            "--embedding-model", str(args.embedding_model.resolve()),
            "--out", str(args.out.resolve()), "--run-only"]
    os.execv(sys.executable, argv)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "runs/prepared")
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "models/Llama-3.1-8B-Instruct")
    parser.add_argument("--embedding-model", type=Path, default=ROOT / "models/all-MiniLM-L6-v2")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/reproduction-v1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--bootstrap-only", action="store_true", help="Write40 true source deltas; defer scaling")
    mode.add_argument("--run-only", action="store_true", help="Use completed local bootstrap, then write24 and compare all banks")
    args = parser.parse_args()
    args.prepared, args.out = args.prepared.resolve(), args.out.resolve()
    runs = (ROOT / "runs").resolve()
    if not args.out.is_relative_to(runs) or not args.prepared.is_relative_to(runs):
        raise ValueError("Fresh artifacts must remain under the ignored repository runs directory")
    check_preparation(args.prepared)
    projection, _ = check_projection(args.projection)
    local_model_link(args.model, ROOT / "models/Llama-3.1-8B-Instruct")
    local_model_link(args.embedding_model, ROOT / "models/all-MiniLM-L6-v2")
    args.out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)  # Upstream util.globals expects globals.yml in cwd.
    if not args.run_only:
        bootstrap(ROOT, args.prepared, projection, args.out)
    if args.bootstrap_only:
        print("40 source writes completed; run --run-only for the scaling experiment", flush=True)
        return
    if not args.run_only:
        print("Bootstrap completed; restarting the interpreter to release its GPU context", flush=True)
        reexec_for_scaling(args)
        return  # Also prevents same-context execution when execv is mocked.
    directory = prepare_scaling(ROOT, args.prepared, projection, args.out)
    from experiment_memory_scaling import run
    run(ROOT, directory)
    print("Completed fresh comparison:", directory, flush=True)


if __name__ == "__main__":
    main()
