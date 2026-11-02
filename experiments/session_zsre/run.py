"""Detached GPU-1 driver with a functional pilot, deterministic resume and final audit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from experiments.wise_paper.native import ROOT
from experiments.wise_paper.prepare import save
from .prepare import check
from .worker import now


def status(out,stage,**fields):save(out/'driver_status.json',dict(stage=stage,updated_utc=now(),**fields))


def child(out,limit):
    while True:
        gpu=subprocess.run(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        if not gpu.stdout.strip():break
        status(out,'waiting_for_gpu1');time.sleep(30)
    command=[sys.executable,'-u','-m','experiments.session_zsre.worker','--out',str(out),'--limit',str(limit)]
    process=subprocess.Popen(command,cwd=ROOT)
    status(out,'running',limit=limit,worker_pid=process.pid)
    if process.wait():raise RuntimeError('Session ZsRE worker failed; see driver.log')


def drive(out):
    save(out/'completion.json',dict(completed=False,stage='running',physical_gpu=1))
    try:
        d=check(out)
        # Verify already downloaded weights against the exact model manifest from
        # the completed WISE run. No model substitution or fresh download.
        manifest=json.loads((Path(d['reference_run'])/'model_manifest.json').read_text())
        for name,want in manifest['sha256'].items():
            status(out,'model_integrity',file=name);h=hashlib.sha256()
            with (Path(d['model'])/name).open('rb') as f:
                for block in iter(lambda:f.read(16*2**20),b''):h.update(block)
            if h.hexdigest()!=want:raise ValueError('Base-model checksum mismatch: '+name)
        save(out/'model_validation.json',dict(exact=True,model_revision=d['model_revision'],sha256=manifest['sha256']))
        child(out,10)
        pilot=json.loads((out/'checkpoint-10.json').read_text())
        if not pilot.get('completed') or len(pilot['rows'])!=10:raise ValueError('Incomplete functional pilot')
        save(out/'pilot_validation.json',dict(functional=True,edits=10,summary=pilot['summary'],
            note='Execution/metric contract check only. No performance threshold or tuning. Resume same committed independent deltas.'))
        child(out,1000)
        status(out,'audit_and_report')
        from .report import build
        build(out);status(out,'completed')
    except BaseException as e:
        save(out/'completion.json',dict(completed=False,stage='failed',error=repr(e),physical_gpu=1,updated_utc=now()))
        status(out,'failed',error=repr(e));raise


def launch(out):
    check(out)
    if json.loads((out/'completion.json').read_text()).get('completed'):raise RuntimeError('Already completed')
    meta=out/'detached_launch.json'
    if meta.exists():
        prior=json.loads(meta.read_text());process=Path('/proc')/str(prior['pid'])/'cmdline'
        if process.exists() and b'session_zsre.run' in process.read_bytes():raise RuntimeError('Driver already running')
    command=[sys.executable,'-u','-m','experiments.session_zsre.run','drive','--out',str(out)]
    with (out/'driver.log').open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 close_fds=True,start_new_session=True)
    save(meta,dict(pid=process.pid,command=command,physical_gpu=1,detached=True,started_utc=now()))
    print('DETACHED_SERVER_DRIVER',process.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','drive']);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.out.resolve())
