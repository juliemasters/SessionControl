"""Rebuild the exact-question cohort and preservation corpus from local data.

This CPU-only step never copies model weights or runs an experiment. The
published manifest contains identifiers and history, not conversation text.
Generated conversations and distractors stay under the ignored runs directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random


DATASET_SHA256 = "08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894"
CORPUS_SHA256 = "e23dff571defe17a41c5a1fbe936efefea65029db2bcdd8a78cd063e70134525"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reconstruct_items(records, manifest):
    """Use the literal question, reference, and entire source session."""
    entries = manifest if isinstance(manifest, list) else manifest["sessions"]
    by_question = {str(row["question_id"]): row for row in records}
    if len(by_question) != len(records):
        raise ValueError("Dataset has duplicate question identifiers")
    items, questions, sessions = [], set(), set()
    for entry in entries:
        qid = str(entry["qid"])
        if qid in questions:
            raise ValueError("Manifest repeats a question")
        row = by_question[qid]
        answer_sessions = row.get("answer_session_ids") or []
        if row.get("question_type") != "single-session-user" or len(answer_sessions) != 1 or qid.endswith("_abs"):
            raise ValueError("Manifest must use non-abstention single-session-user items")
        source = answer_sessions[0]
        if source != entry["source_session_id"] or source in sessions:
            raise ValueError("Source session mismatch or repeated source session")
        expected_sid = "qa-" + hashlib.sha256(qid.encode()).hexdigest()[:16]
        if entry["sid"] != expected_sid:
            raise ValueError("Delta identifier does not match question")
        conversation = dict(zip(row["haystack_session_ids"], row["haystack_sessions"]))[source]
        if not conversation or not isinstance(row["answer"], str) or not row["answer"].strip():
            raise ValueError("Missing complete answer session or answer")
        items.append(dict(qid=qid, sid=expected_sid, source_session_id=source,
                          answer_session_ids=answer_sessions, history=entry["history"],
                          question=row["question"], answer=row["answer"], context=conversation))
        questions.add(qid)
        sessions.add(source)
    return items


def distractor_records(records, count=200, seed=0):
    """Exact original corpus algorithm, including first-occurrence provenance."""
    excluded = {sid for row in records for sid in row.get("answer_session_ids", [])}
    pool, seen = [], set()
    for row in records:
        for sid, session in zip(row["haystack_session_ids"], row["haystack_sessions"]):
            if sid in excluded:
                continue
            for turn_index, turn in enumerate(session):
                text = turn.get("content", "")
                if turn.get("role") == "user" and 10 <= len(text.split()) <= 40 and text not in seen:
                    seen.add(text)
                    pool.append(dict(text=text, source_session_id=sid, turn=turn_index))
    if len(pool) < count:
        raise ValueError("Insufficient eligible distractor turns")
    random.Random(seed).shuffle(pool)
    return pool[:count], len(pool), excluded


def prepare(data, manifest_path, out):
    data, manifest_path, out = Path(data).resolve(), Path(manifest_path).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Use a new preparation directory")
    if sha256(data) != DATASET_SHA256:
        raise ValueError("Dataset bytes differ from the measured original longmemeval_s.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(manifest, dict) and manifest.get("dataset_sha256", DATASET_SHA256) != DATASET_SHA256:
        raise ValueError("Manifest dataset fingerprint differs")
    records = json.loads(data.read_text(encoding="utf-8"))
    items = reconstruct_items(records, manifest)
    if len(items) != 64 or sum(r["history"] in {"previous_test", "writer_development"} for r in items) != 40:
        raise ValueError("Expected the published64-session cohort and40 bootstrap writers")
    if any(r["history"] != "previous_test" for r in items[:24]):
        raise ValueError("First24 manifest entries must be the original test anchors")
    corpus, pool_size, excluded = distractor_records(records)
    corpus_bytes = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in corpus).encode("utf-8")
    if hashlib.sha256(corpus_bytes).hexdigest() != CORPUS_SHA256:
        raise ValueError("Rebuilt distractor corpus does not reproduce the original bytes")
    out.mkdir(parents=True)
    write_json(out / "items.json", items)
    (out / "distractors.jsonl").write_bytes(corpus_bytes)
    write_json(out / "preparation.json", dict(
        stage="prepared_only", experiment_completed=False,
        dataset=str(data), dataset_sha256=DATASET_SHA256,
        manifest=str(manifest_path), manifest_sha256=sha256(manifest_path),
        items_sha256=sha256(out / "items.json"), corpus_sha256=CORPUS_SHA256,
        distractor_pool_size=pool_size, distractor_turns=200, excluded_answer_sessions=len(excluded),
        source="Literal local official dataset; no generation or paraphrasing",
        bootstrap_cohort="40 historical repair-writer items; cohorts record past usage, not new held-out splits",
    ))
    return out


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=root / "config/session_manifest.json")
    parser.add_argument("--out", type=Path, default=root / "runs/prepared")
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to((root / "runs").resolve()):
        raise ValueError("Generated source data must remain under the ignored repository runs directory")
    print("Prepared", prepare(args.data, args.manifest, args.out))
    print("No model execution occurred. Build the projector before bootstrapping.")


if __name__ == "__main__":
    main()
