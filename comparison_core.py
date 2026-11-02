"""Definitions shared by the non-versioned comparison and its report."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

SIZES=(8,16,32,64,128,256,512)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)

def durable_save_torch(torch,path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    temporary.replace(path)
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def durable_existing(path):
    path=Path(path)
    with path.open('rb') as f:os.fsync(f.fileno())
    fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def interval_for(index):
    return next(n for n in SIZES if index<n)

def paired_sample_indices():
    """Eight deterministic spread-out records from each disjoint growth block."""
    result=[];previous=0
    for n in SIZES:
        width=n-previous
        result.extend(previous+round(j*(width-1)/7) for j in range(8))
        previous=n
    if len(set(result))!=56:raise ValueError('Duplicate timing sample')
    return tuple(result)

def timing_summary(values):
    if not values:return dict(n=0,mean_seconds=None,median_seconds=None,p95_seconds=None)
    values=sorted(values);p=.95*(len(values)-1);i=math.floor(p)
    return dict(n=len(values),mean_seconds=statistics.mean(values),median_seconds=statistics.median(values),
        p95_seconds=values[i]+(values[min(i+1,len(values)-1)]-values[i])*(p-i))

def scores(rows):
    if not rows:raise ValueError('Empty accuracy cell')
    if len({r['qid'] for r in rows})!=len(rows):raise ValueError('Repeated accuracy question')
    return dict(questions=len(rows),exact=sum(r['returned_exact'] for r in rows),
        accuracy=sum(r['returned_exact'] for r in rows)/len(rows),mean_answer_f1=statistics.mean(r['token_f1'] for r in rows))
