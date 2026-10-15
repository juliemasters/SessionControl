"""Shared IO, GPU selection, and evaluation for the final experiment.

Evaluation helpers receive references only after inference returns.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
from question_key import normalize, interval
from answer_only_generation import reference_tokens_present
from memory_comparison_core import token_f1

def read(path):
    return json.loads(Path(path).read_text())


def save(out, name, value):
    path = out/name
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

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

def document(record):
    return record['record_id']+'\nConversation:\n'+record['conversation']+'\nStored question: '+record['stored_question']+'\nStored answer: '+record['stored_answer']


def metrics(trials):
    n=len(trials);summary=dict(n=n)
    for key in ['returned_exact','emitted_exact','verbatim_exact','contains_reference','hit_cap','eos']:
        count=sum(t[key] for t in trials);summary[key]=dict(count=count,rate=count/n,ci95=list(interval(count,n)))
    for key in ['retrieval_at1','retrieval_at3']:
        valid=[t[key] for t in trials if t[key] is not None]
        summary[key]=None if not valid else dict(count=sum(valid),rate=sum(valid)/len(valid),ci95=list(interval(sum(valid),len(valid))))
    summary.update(mean_token_f1=sum(t['token_f1'] for t in trials)/n,
                   mean_output_tokens=sum(t['final']['tokens'] for t in trials)/n,
                   mean_query_seconds=sum(t['query_seconds'] for t in trials)/n,
                   max_query_cache_payload_bytes=max(t['final'].get('max_query_cache_payload_bytes',0) for t in trials))
    return summary


def evaluated(item,final,seconds,*,sources=None,selected_sid=None):
    # Reference values and source labels enter only after inference returns.
    sources=sources or []
    return dict(qid=item['qid'],question=item['question'],answer=item['answer'],final=final,
                query_seconds=seconds,selected_sid=selected_sid,reference_source_sid=item['sid'],retrieved_sids=sources,
                retrieval_at1=(sources[0]==item['sid']) if sources else (selected_sid==item['sid'] if selected_sid else None),
                retrieval_at3=(item['sid'] in sources[:3]) if sources else (selected_sid==item['sid'] if selected_sid else None),
                returned_exact=normalize(final['text'])==normalize(item['answer']),
                emitted_exact=normalize(final['emitted_text'])==normalize(item['answer']),
                verbatim_exact=final['text']==item['answer'],token_f1=token_f1(final['text'],item['answer']),
                contains_reference=reference_tokens_present(final['text'],item['answer']),
                hit_cap=final['hit_cap'],eos=final['finish_reason']=='eos')
