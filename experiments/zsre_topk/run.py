"""Detached, serialized GPU-1 driver for both matched dataset experiments."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from .core import ROOT,METHODS,read,save,check,digest,status,now


def child(root,out,label,args):
    while True:
        busy=subprocess.run(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True)
        if not busy.stdout.strip():break
        status(root,'waiting_for_gpu1',job=label);time.sleep(30)
    p=subprocess.Popen([sys.executable,'-u','-m','experiments.zsre_topk.worker',*args,'--out',str(out)],cwd=ROOT)
    save(root/'driver_status.json',dict(stage=label,worker_pid=p.pid,started_utc=now()));status(root,'running',job=label)
    if p.wait():raise RuntimeError(label+' failed; see driver.log')


def audit_pilot(out):
    d=check(out);source=Path(d['source_run'])
    old=read(source/'our_delta/run-00/bank-8.json')
    by_old={t.get('case_id',t.get('qid')):t for t in old['trials']}
    baseline=read(out/'full_scan/run-00/bank-8.json')
    mismatch=[]
    for t in baseline['trials']:
        previous=by_old[t['case_id']]
        if t['selected']!=previous.get('selected',previous.get('selected_sid')) or t['final']['token_ids']!=previous['final']['token_ids']:
            mismatch.append(t['case_id'])
    if mismatch:raise ValueError('Fresh baseline pilot differs from original outputs: '+repr(mismatch))
    by_base={t['case_id']:t for t in baseline['trials']}
    for method in ['topk_8','topk_16','topk_32']:
        for t in read(out/method/'run-00/bank-8.json')['trials']:
            b=by_base[t['case_id']]
            if (t['selected'],t['final']['token_ids'])!=(b['selected'],b['final']['token_ids']):raise ValueError('All-candidate pilot parity failed')
    save(out/'pilot_validation.json',dict(completed=True,original_baseline_token_parity=True,
        allcandidate_token_parity=True,questions=8,note='Functional readiness only; no accuracy-based tuning or arm selection.'))


def drive(root):
    save(root/'completion.json',dict(completed=False,stage='running',physical_gpu=1,total_planned_requests=133350))
    try:
        outs=[root/name for name in ['zsre','longmemeval']]
        for out in outs:
            d=check(out)
            for name,want in d['model_manifest']['sha256'].items():
                status(root,'model_integrity',dataset=d['dataset'],file=name)
                if digest(Path(d['model'])/name)!=want:raise ValueError('Model weights changed')
            child(root,out,d['dataset']+'-index',['index'])
            child(root,out,d['dataset']+'-routing-preflight',['preflight'])
            for method in METHODS:
                child(root,out,d['dataset']+'-pilot-'+method,['evaluate','--method',method,'--repeat','0','--sizes','8'])
            audit_pilot(out)
        for repeat in range(10):
            for out in outs:
                offset=repeat%len(METHODS)
                for method in METHODS[offset:]+METHODS[:offset]:
                    child(root,out,f'{out.name}-pass-{repeat+1}-{method}',['evaluate','--method',method,'--repeat',str(repeat)])
        from .report import build
        total=0
        for out in outs:
            status(root,'audit_and_report',dataset=out.name);build(out)
            total+=read(out/'completion.json')['verified_requests']
        if total!=133350:raise ValueError('Incorrect combined measurement count')
        (root/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Activation-key top-k: two datasets</title>'
            '<h1>SessionControl activation-key top-k experiment</h1><p>Both datasets completed ten fresh passes.</p>'
            '<p><a href="zsre/report.html">ZsRE: 1,000 Q/A memories</a></p>'
            '<p><a href="longmemeval/report.html">LongMemEval: 500 conversation memories</a></p>')
        save(root/'completion.json',dict(completed=True,physical_gpu=1,verified_requests=total,finished_utc=now(),report='report.html'))
        status(root,'completed');save(root/'driver_status.json',dict(stage='completed',finished_utc=now()))
    except BaseException as error:
        save(root/'completion.json',dict(completed=False,stage='failed',error=repr(error),updated_utc=now()));status(root,'failed',error=repr(error));raise


def launch(root):
    for dataset,want in read(root/'datasets.json').items():
        if digest(root/dataset/'design.json')!=want:raise ValueError('Prepared experiment changed')
        check(root/dataset)
    pfile=root/'detached_launch.json'
    if pfile.exists():
        old=Path('/proc')/str(read(pfile)['pid'])/'cmdline'
        if old.exists() and b'zsre_topk.run' in old.read_bytes():raise RuntimeError('Driver already running')
    if read(root/'completion.json').get('completed'):raise RuntimeError('Experiment already finished')
    cmd=[sys.executable,'-u','-m','experiments.zsre_topk.run','drive','--out',str(root)]
    with (root/'driver.log').open('a') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
    save(pfile,dict(pid=p.pid,command=cmd,physical_gpu=1,detached=True,started_utc=now()))
    print('DETACHED_TWO_DATASET_TOPK_DRIVER',p.pid,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','drive']);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();globals()[a.action](a.out.resolve())
