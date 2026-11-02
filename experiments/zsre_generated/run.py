"""Detached GPU-1 driver: functional pilots, full banks, ten passes and final report."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from .core import ROOT,METHODS,SIZES,read,save,status,now,check


def child(out,label,args):
    while True:
        busy=subprocess.run(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True)
        if not busy.stdout.strip():break
        status(out,'waiting_for_gpu1',job=label);time.sleep(30)
    p=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT)
    save(out/'driver_status.json',dict(stage=label,worker_pid=p.pid,started_utc=now()));status(out,'running',job=label)
    if p.wait():raise RuntimeError(label+' failed; see driver.log')


def drive(out):
    check(out);save(out/'completion.json',dict(completed=False,stage='running',physical_gpu=1))
    try:
        d=check(out)
        for name,want in d['model_manifest']['sha256'].items():
            status(out,'model_integrity',file=name);h=hashlib.sha256()
            with (Path(d['model'])/name).open('rb') as f:
                for block in iter(lambda:f.read(16*2**20),b''):h.update(block)
            if h.hexdigest()!=want:raise ValueError('Model file changed: '+name)
        for method in ['plain_alphaedit','wise_merge','wise_retrieve']:
            child(out,'pilot-write-'+method,['-m','experiments.zsre_generated.build','--out',str(out),'--method',method,'--until','8'])
        for method in METHODS:
            child(out,'pilot-measure-'+method,['-m','experiments.zsre_generated.evaluate','--out',str(out),'--method',method,'--repeat','0','--sizes','8'])
            cell=read(out/method/'run-00/bank-8.json')
            if not cell.get('completed') or not cell['available'] or len(cell['trials'])!=8:raise ValueError('Incomplete functional pilot: '+method)
        save(out/'pilot_validation.json',dict(functional=True,methods=METHODS,n=8,
            note='First cells of the same ten passes; no accuracy threshold, test-set tuning, route forcing or restarts selected by quality.'))
        for method in ['plain_alphaedit','wise_merge','wise_retrieve']:
            child(out,'full-write-'+method,['-m','experiments.zsre_generated.build','--out',str(out),'--method',method,'--until','1000'])
        for repeat in range(10):
            offset=repeat%len(METHODS)
            for method in METHODS[offset:]+METHODS[:offset]:
                child(out,f'pass-{repeat+1}-{method}',['-m','experiments.zsre_generated.evaluate','--out',str(out),'--method',method,'--repeat',str(repeat)])
        status(out,'audit_and_report')
        from .report import build
        build(out);status(out,'completed');save(out/'driver_status.json',dict(stage='completed',finished_utc=now()))
    except BaseException as error:
        save(out/'completion.json',dict(completed=False,stage='failed',physical_gpu=1,error=repr(error),updated_utc=now()))
        status(out,'failed',error=repr(error));raise


def launch(out):
    check(out)
    if read(out/'completion.json').get('completed'):raise RuntimeError('Already completed')
    meta=out/'detached_launch.json'
    if meta.exists():
        old=read(meta);p=Path('/proc')/str(old['pid'])/'cmdline'
        if p.exists() and b'zsre_generated.run' in p.read_bytes():raise RuntimeError('Driver already running')
    command=[sys.executable,'-u','-m','experiments.zsre_generated.run','drive','--out',str(out)]
    with (out/'driver.log').open('a') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
    save(meta,dict(pid=p.pid,command=command,physical_gpu=1,detached=True,started_utc=now()))
    print('DETACHED_SIX_METHOD_ZSRE_DRIVER',p.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','drive']);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.out.resolve())
