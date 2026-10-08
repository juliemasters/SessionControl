"""Small ZsRE/AlphaEdit probe-routing experiment. Real model: GPU 1 ONLY.

prepare uses only the standard library; run masks CUDA before importing torch.
The manifest's targets and session labels are used only by evaluation.
"""
import argparse
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def prepare(dataset, output, count=8, seed=42):
    raw = json.loads(Path(dataset).read_text(encoding="utf-8"))
    indices = list(range(len(raw)))
    random.Random(seed).shuffle(indices)
    sessions, seen_questions, seen_subjects = [], set(), set()
    for index in indices:
        row = raw[index]
        source, query, subject = row["src"], row["rephrase"], row["subject"]
        answers = row["answers"]
        if not answers or not subject or subject not in source or not query.strip():
            continue
        if normalize(source) == normalize(query):
            continue
        key = normalize(query)
        if key in seen_questions or normalize(subject) in seen_subjects:
            continue
        seen_questions.add(key)
        seen_subjects.add(normalize(subject))
        sessions.append({"session_id": f"zsre-{index:06d}", "version": 1,
                         "case_id": index, "source": source, "question": query,
                         "answers": answers,
                         "request": {"case_id": index,
                                     "prompt": source.replace(subject, "{}"),
                                     "subject": subject,
                                     "target_new": {"str": answers[0]},
                                     "target_true": {"str": "<|endoftext|>"}}})
        if len(sessions) == count:
            break
    if len(sessions) != count:
        raise ValueError(f"Only {len(sessions)} eligible unique records; requested {count}")
    dump(output, {"dataset": str(Path(dataset).resolve()), "seed": seed,
                  "sessions": sessions})
    return sessions


def require_gpu1():
    # Resolve physical nvidia-smi index 1 to UUID, avoiding ordinal ambiguity.
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                            check=True, capture_output=True, text=True)
    devices = dict(line.strip().split(", ", 1) for line in result.stdout.splitlines() if line.strip())
    if "1" not in devices:
        raise RuntimeError("Physical GPU 1 is unavailable. Refusing GPU 0 or CPU fallback.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = devices["1"]
    return devices["1"]


def run(args):
    gpu_uuid = require_gpu1()  # Must precede every torch/transformers import.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from AlphaEditSessionStore import SessionFFNMemory
    from DeltaProbeText import DeltaProbeText

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device: physical GPU 1")
    torch.cuda.set_device(0)  # Logical 0 is the GPU-1 UUID selected above.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    sessions = manifest["sessions"]
    if len(sessions) < 2 or len({s["session_id"] for s in sessions}) != len(sessions):
        raise ValueError("Need at least two distinct session IDs")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    hparams_data = json.loads(Path(args.hparams).read_text(encoding="utf-8"))
    names = [hparams_data["rewrite_module_tmp"].format(layer) + ".weight"
             for layer in hparams_data["layers"]]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to("cuda:0")
    model.eval()
    store = SessionFFNMemory(model, names, args.deltas)
    if args.write_deltas:
        if any(Path(args.deltas).iterdir()):
            raise ValueError("Writing requires an empty delta directory")
        sys.path.insert(0, str(Path(args.alphaedit_repo).resolve()))
        from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
        hp = AlphaEditHyperParams.from_json(args.hparams)
        # P must be the genuine AlphaEdit null-space projection for this model/layers.
        projection = torch.load(args.projection, map_location="cpu", weights_only=True)
        for row in sessions:
            cache = torch.zeros_like(projection)
            store.write_session(row["session_id"], tokenizer, [row["request"]], hp,
                                cache, projection, row["version"])
    expected = sorted((s["session_id"], s["version"]) for s in sessions)
    if store.latest_sessions() != expected:
        raise ValueError("Delta directory must contain exactly the manifest's sessions/versions")
    text = DeltaProbeText(model, tokenizer, probe_tokens=args.probe_tokens)
    # Routing sees only the question and candidates. Gold labels remain below.
    rows = []
    query_order = list(sessions)
    random.Random(args.seed).shuffle(query_order)
    with torch.inference_mode():
        for item in query_order:
            question = item["question"]
            result = store.route_and_run(question, text.probe, text.score)
            record = {"case_id": item["case_id"], "question": question,
                      "expected_session": item["session_id"],
                      "selected_session": result.selected.session_id,
                      "routing_correct": result.selected.session_id == item["session_id"],
                      "tied": result.tied,
                      "probes": [vars(p) for p in result.probes]}
            rows.append(record)
            with (output / "queries.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{len(rows)}/{len(sessions)} correct={record['routing_correct']} tie={result.tied}", flush=True)
    correct = sum(row["routing_correct"] for row in rows)
    summary = {"n_sessions": len(sessions), "n_questions": len(rows),
               "correct_sessions": correct, "routing_accuracy": correct / len(rows),
               "physical_gpu": 1, "gpu_uuid": gpu_uuid,
               "config": vars(args), "manifest": manifest,
               "torch_version": torch.__version__}
    dump(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"manifest", "config"}}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--dataset", required=True, help="AlphaEdit zsre_mend_eval.json")
    p.add_argument("--output", required=True)
    p.add_argument("--sessions", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("run")
    for name in ["manifest", "model", "hparams", "deltas", "output"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--probe-tokens", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--write-deltas", action="store_true")
    p.add_argument("--alphaedit-repo")
    p.add_argument("--projection")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.sessions < 2:
            parser.error("At least two sessions are required")
        prepare(args.dataset, args.output, args.sessions, args.seed)
    else:
        if args.write_deltas and not (args.alphaedit_repo and args.projection):
            parser.error("--write-deltas requires --alphaedit-repo and --projection")
        if args.probe_tokens < 1:
            parser.error("Token limits must be positive")
        run(args)


if __name__ == "__main__":
    main()
