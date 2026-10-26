"""Reconstruct local reference inputs from the official data and published hashes.

The bundled aggregate baseline scores were measured previously. This command
does not train or evaluate those baselines and never presents them as new runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from .core import HERE,read,save,digest,hashed


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def bootstrap(data,model,out):
    bundle=HERE/'reference';meta=read(bundle/'metadata.json')
    if out.exists():raise FileExistsError('Use a fresh local reference-input directory')
    if digest(data)!=meta['dataset_sha256']:raise ValueError('Official dataset checksum mismatch')
    for f,key in [('config.json','model_config_sha256'),('tokenizer.json','tokenizer_sha256')]:
        if digest(model/f)!=meta[key]:raise ValueError('Base model/tokenizer identity mismatch')
    for f,h in meta['results'].items():
        if digest(bundle/f)!=h:raise ValueError('Published reference aggregate changed')
    official=read(data);seed=read(HERE/'config/seed_session_ids.json');records={};queries=[]
    for row in official:
        qid=str(row['question_id']);sessions=dict(zip(row['haystack_session_ids'],row['haystack_sessions']))
        dates=dict(zip(row['haystack_session_ids'],row['haystack_dates']))
        queries.append(dict(qid=qid,question=row['question'],answer=str(row['answer']),question_date=row['question_date'],
            question_type=row['question_type'],support_sids=row['answer_session_ids'],abstention=qid.endswith('_abs'),
            original64=bool(set(seed)&set(row['answer_session_ids']))))
        for sid in row['answer_session_ids']:
            record=records.setdefault(sid,dict(sid=sid,context=sessions[sid],bindings=[]))
            if record['context']!=sessions[sid]:raise ValueError('Inconsistent source context')
            record['bindings'].append(dict(qid=qid,question=row['question'],answer=str(row['answer']),date=dates[sid]))
    ids=list(seed);seen=set(ids)
    for q in sorted(queries,key=lambda q:hashed(q['qid'])):
        for sid in q['support_sids']:
            if sid not in seen:ids.append(sid);seen.add(sid)
    bank=[records[sid] for sid in ids[:500]]
    if canonical(bank)!=meta['sessions_canonical_sha256'] or canonical(queries)!=meta['queries_canonical_sha256']:
        raise ValueError('Reconstructed inputs differ from the measured baseline experiment')
    out.mkdir(parents=True)
    save(out/'sessions.json',bank);save(out/'queries.json',queries)
    save(out/'design.json',dict(model=str(model.resolve()),dataset_sha256=meta['dataset_sha256'],
        source_reference_run=meta['source_run'],reconstructed_input_bundle=True))
    save(out/'completion.json',dict(completed=True,passes=10,finished_utc=meta['finished_utc'],
        note='Completion refers to previously measured baseline results. This command only reconstructs matching inputs.'))
    for name in meta['results']:shutil.copy2(bundle/name,out/name)
    save(out/'report_manifest.json',meta['results']);save(out/'published_reference_metadata.json',meta)
    print('Verified local reference inputs reconstructed; existing baseline aggregates copied, not remeasured.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    bootstrap(args.data.resolve(),args.model.resolve(),args.out.resolve())
