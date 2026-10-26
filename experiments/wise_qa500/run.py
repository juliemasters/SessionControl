"""Prepare, detach, monitor, resume and export the WISE comparison."""
import argparse
from datetime import datetime
import html
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from .core import ROOT,SIZES,read,save,digest,check,prepare,status,now


def child(out,name,args,gpu=True):
    if gpu:
        while True:
            result=subprocess.run(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],check=True,capture_output=True,text=True)
            if not result.stdout.strip():break
            status(out,'waiting_for_gpu1',job=name);time.sleep(30)
    p=subprocess.Popen([sys.executable,'-u',*args],cwd=ROOT)
    save(out/'driver_status.json',dict(stage=name,pid=p.pid,started_utc=now()))
    status(out,'running',job=name)
    result=p.wait()
    if result:raise RuntimeError(f'{name} exited with code {result}')


def drive(out):
    check(out);save(out/'completion.json',dict(completed=False,stage='running',physical_gpu=1))
    worker=['-m','experiments.wise_qa500.worker']
    try:
        child(out,'write-pilot-eight',[*worker,'build','--out',str(out),'--until','8'])
        child(out,'evaluate-pilot-pass-one',[*worker,'eval','--out',str(out),'--repeat','0','--sizes','8'])
        first=read(out/'wise/run-00/bank-8.json')
        if not first.get('completed') or len(first['trials'])!=8:raise ValueError('Incomplete real WISE pilot')
        save(out/'pilot_validation.json',dict(functional=True,questions=8,metrics=first['groups']['all'],
             full_write_contract=read(out/'first_write_contract.json'),note='No accuracy threshold or tuning on evaluation questions. Pilot is the first of ten actual passes at N=8.'))
        child(out,'write-all-500',[*worker,'build','--out',str(out),'--until','500'])
        for repeat in range(10):
            child(out,f'evaluate-pass-{repeat+1}',[*worker,'eval','--out',str(out),'--repeat',str(repeat)])
        child(out,'audit-and-report',['-m','experiments.wise_qa500.report','--out',str(out)],gpu=False)
        export(out)
        status(out,'completed');save(out/'driver_status.json',dict(stage='completed',finished_utc=now()))
    except BaseException as error:
        save(out/'completion.json',dict(completed=False,stage='failed',error=repr(error),physical_gpu=1))
        status(out,'failed',error=repr(error));raise


def progress(out):
    complete=read(out/'completion.json');state='completed' if complete.get('completed') else complete.get('stage','running')
    ledger=read(out/'ledger.json');driver=read(out/'driver_status.json') if (out/'driver_status.json').exists() else {}
    recent=[r['internal_seconds'] for r in ledger[-20:]]
    estimate=f'Approximate remaining write time: {(500-len(ledger))*statistics.mean(recent)/3600:.1f} hours.' if recent and len(ledger)<500 else ''
    rows=[]
    for n in SIZES:
        cells=[]
        for repeat in range(10):
            file=out/'wise'/f'run-{repeat:02d}'/f'bank-{n}.json'
            if file.exists():
                c=read(file)
                if c.get('completed'):cells.append(c)
        means=[c['groups']['all']['ttft_mean_ms'] for c in cells if c['groups']['all']['ttft_mean_ms'] is not None]
        f1=[c['groups']['all']['mean_f1'] for c in cells]
        ttft='—' if not means else f'{statistics.mean(means):.2f}'
        if len(means)>1:ttft+=f' ± {statistics.stdev(means):.2f}'
        quality='—' if not f1 else f'{100*statistics.mean(f1):.2f}'
        rows.append(f'<tr><td>{n}</td><td>{len(cells)}/10</td><td>{ttft}</td><td>{quality}</td></tr>')
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
    page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>WISE experiment progress</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px}td,th{border:1px solid #ccc;padding:10px}table{border-collapse:collapse}</style>'
    page+=f'<h1>WISE-Retrieve: {html.escape(state)}</h1><p>Updated {stamp}. Physical GPU 1. Saved conversation writes: {len(ledger)}/500.</p>'
    page+='<p>Full source conversation + literal official question + complete reference answer at write time. Native WISE routing. Final-answer F1 and exact match only.</p>'
    page+=f'<p>Current job: {html.escape(driver.get("stage","preparing"))}. {estimate}</p>'
    page+='<h2>TTFT first</h2><p>Partial results are provisional until all ten passes complete. Mean ± sample SD across completed per-pass question means; one pass has no SD.</p>'
    page+='<table><tr><th>Conversations</th><th>Passes complete</th><th>Mean visible TTFT (ms)</th><th>Final-answer F1 (%)</th></tr>'+''.join(rows)+'</table>'
    page+='<p>Prior baselines are reused from the previous completed ten-pass experiment, not rerun or pooled into WISE scores.</p><p><a href="report.html">Full report after completion</a></p>'
    if state=='failed':page+='<p>Stopped: '+html.escape(str(complete.get('error')))+'</p>'
    temp=out/'progress.html.tmp';temp.write_text(page);temp.replace(out/'progress.html')
    return complete.get('completed') or state=='failed'


def monitor(out):
    while True:
        try:
            if progress(out):return
        except (OSError,ValueError,KeyError):pass
        time.sleep(30)


def launch(out):
    check(out)
    if read(out/'completion.json').get('completed'):raise RuntimeError('This WISE experiment is already complete')
    meta=out/'detached_launch.json'
    if meta.exists():
        prior=read(meta);cmd=Path('/proc')/str(prior['pid'])/'cmdline'
        if cmd.exists() and b'experiments.wise_qa500.run' in cmd.read_bytes():raise RuntimeError('WISE driver already running')
    command=[sys.executable,'-u','-m','experiments.wise_qa500.run','drive','--out',str(out)]
    with (out/'driver.log').open('a') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    with (out/'monitor.log').open('a') as log:
        m=subprocess.Popen([sys.executable,'-u','-m','experiments.wise_qa500.run','monitor','--out',str(out)],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    save(meta,dict(pid=p.pid,monitor_pid=m.pid,command=command,started_utc=now(),physical_gpu=1,detached=True))
    print('Detached WISE driver PID',p.pid,'monitor',m.pid,flush=True)


def export(out):
    """Stage only aggregate results and figures for GitHub; never raw data/weights."""
    done=read(out/'completion.json')
    if not done.get('completed'):raise ValueError('Cannot export incomplete experiment as results')
    dest=ROOT/'docs/results/wise-qa500-v1';dest.mkdir(parents=True,exist_ok=True)
    files=['report.html','report.md','comparison.csv','comparison.json','wise_per_pass.csv',
        *[f'{name}.{ext}' for name in ['ttft','accuracy_final_answer','memory'] for ext in ['png','svg','pdf']]]
    for file in files:shutil.copy2(out/file,dest/file)
    save(dest/'completion.json',done)
    save(dest/'manifest.json',{f.name:digest(f) for f in dest.iterdir() if f.name!='manifest.json'})
    save(out/'github_export.json',dict(staged=True,published=False,path=str(dest),
        note='Aggregate results staged locally for GitHub. Code publication is recorded separately. No raw conversations, final answer transcripts, model weights or restart states exported.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','drive','launch','monitor','export'])
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--data',type=Path);parser.add_argument('--model',type=Path);parser.add_argument('--reference-run',type=Path)
    a=parser.parse_args();out=a.out.resolve()
    if a.action=='prepare':
        if not all([a.data,a.model,a.reference_run]):parser.error('prepare requires --data, --model and --reference-run')
        prepare(out,a.data,a.model,a.reference_run)
    else:globals()[a.action](out)
