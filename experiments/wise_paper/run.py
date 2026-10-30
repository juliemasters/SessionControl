"""Detached server driver: acquire model, run native WISE, audit and report."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .native import ROOT
from .prepare import save


def now():return datetime.now(timezone.utc).isoformat()
def status(out,stage,**fields):save(out/'driver_status.json',dict(stage=stage,updated_utc=now(),**fields))


def child(out,mode,limit):
    while True:
        r=subprocess.run(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        if not r.stdout.strip():break
        status(out,'waiting_for_gpu1',variant=mode);time.sleep(30)
    cmd=[sys.executable,'-u','-m','experiments.wise_paper.worker','--out',str(out),'--mode',mode,'--limit',str(limit)]
    p=subprocess.Popen(cmd,cwd=ROOT)
    status(out,'running',variant=mode,limit=limit,worker_pid=p.pid)
    if p.wait():raise RuntimeError('Native '+mode+' worker failed; see driver.log')


def drive(out):
    d=json.loads((out/'design.json').read_text())
    save(out/'completion.json',dict(completed=False,stage='running',physical_gpu=1))
    try:
        status(out,'model_download')
        # The Hub downloader resumes partial files and honors existing file locks.
        from huggingface_hub import snapshot_download
        snapshot_download(d['model_repository'],revision=d['model_revision'],local_dir=d['model'],max_workers=2,
            allow_patterns=['config.json','generation_config.json','model.safetensors.index.json','model-*.safetensors',
                            'tokenizer.json','tokenizer.model','tokenizer_config.json','special_tokens_map.json'])
        model=Path(d['model']);index=json.loads((model/'model.safetensors.index.json').read_text())
        required=sorted(set(index['weight_map'].values()))+['config.json','tokenizer_config.json']
        hashes={}
        for name in required:
            status(out,'model_integrity',file=name);h=hashlib.sha256()
            with (model/name).open('rb') as f:
                for block in iter(lambda:f.read(16*2**20),b''):h.update(block)
            hashes[name]=h.hexdigest()
        save(out/'model_manifest.json',dict(repository=d['model_repository'],revision=d['model_revision'],sha256=hashes))
        for mode in d['variants']:
            # The pilot checks execution and data/metric contracts, not an accuracy
            # target. No test-set tuning or automatic hyperparameter modification.
            child(out,mode,10)
            pilot=json.loads((out/mode/'checkpoint-10.json').read_text())
            if not pilot.get('completed') or len(pilot['rows'])!=10:raise ValueError('Incomplete native pilot')
            save(out/mode/'pilot_validation.json',dict(functional=True,edits=10,summary=pilot['summary'],
                note='Functional check only. No score threshold or hyperparameter tuning. Pilot is part of the same sequential trajectory.'))
            child(out,mode,1000)
        status(out,'audit_and_report')
        from .report import build
        build(out);status(out,'completed')
    except BaseException as e:
        save(out/'completion.json',dict(completed=False,stage='failed',error=repr(e),physical_gpu=1,updated_utc=now()))
        status(out,'failed',error=repr(e));raise


def launch(out):
    complete=json.loads((out/'completion.json').read_text())
    if complete.get('completed'):raise RuntimeError('Reproduction already completed')
    meta=out/'detached_launch.json'
    if meta.exists():
        prior=json.loads(meta.read_text());p=Path('/proc')/str(prior['pid'])/'cmdline'
        if p.exists() and b'wise_paper.run' in p.read_bytes():raise RuntimeError('Driver already running')
    command=[sys.executable,'-u','-m','experiments.wise_paper.run','drive','--out',str(out)]
    with (out/'driver.log').open('a') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           close_fds=True,start_new_session=True)
    save(meta,dict(pid=p.pid,command=command,physical_gpu=1,detached=True,started_utc=now()))
    print('Detached native WISE server driver',p.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['drive','launch']);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.out.resolve())
