"""Ten NEW inference passes over the frozen, completed full-Q/A bank.

Only bank artifacts and inputs are reused. No evaluation cell, timing or score
from the preceding ten-pass run is copied into this run.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time

from comparison_core import read,save,digest
from qa500_core import ROOT,METHODS,check,status,now

SOURCE=ROOT/'runs/qa500-checkpoints-tenpass-v1'
OUT=ROOT/'runs/qa500-checkpoints-tenpass-v2'


def prepare(out):
    if out.exists():raise FileExistsError('New measurement passes require a fresh result directory')
    prior=check(SOURCE);complete=read(SOURCE/'completion.json')
    if not complete.get('completed') or complete['passes']!=10:
        raise ValueError('Source experiment has not completed its audited ten passes')
    if shutil.disk_usage(out.parent).free<4*2**30:
        raise RuntimeError('Reserve at least 4 GiB for new measurements and the existing 2 GiB safety margin')
    for phase in ['independent','shared']:
        if not read(SOURCE/phase/'completion.json').get('completed'):
            raise ValueError('Frozen memory construction is incomplete')
        if len(read(SOURCE/phase/'ledger.json'))!=500:
            raise ValueError('Expected a 500-conversation bank')
    for row in read(SOURCE/'independent/ledger.json'):
        if digest(SOURCE/'independent/factors'/row['file'])!=row['sha256']:
            raise ValueError('Stored answer delta changed')
    for n,h in read(SOURCE/'shared/completion.json')['patch_sha256'].items():
        if digest(SOURCE/'shared'/f'weights-{n}.pt')!=h:
            raise ValueError('Stored shared checkpoint changed')
    for file,h in read(SOURCE/'rag/index_manifest.json').items():
        if digest(SOURCE/'rag'/file)!=h:raise ValueError('Stored Q/A retrieval index changed')
    out.mkdir()
    for name in ['sessions.json','queries.json','token_alignment_audit.json','alignment_fix_resume.json','reuse_audit.json']:
        shutil.copy2(SOURCE/name,out/name)
    for phase in ['independent','shared']:(out/phase).symlink_to(SOURCE/phase,target_is_directory=True)
    for method in METHODS:(out/method).mkdir()
    for n in prior['sizes']:
        (out/'rag'/f'index-{n}').symlink_to(SOURCE/'rag'/f'index-{n}',target_is_directory=True)
    shutil.copy2(SOURCE/'rag/index_manifest.json',out/'rag/index_manifest.json')
    parent_files=['design.json','completion.json','independent/ledger.json','shared/completion.json','rag/index_manifest.json']
    d=dict(prior,name=out.name)
    d['remeasurement']=dict(source=str(SOURCE),source_artifacts={f:digest(SOURCE/f) for f in parent_files},
        new_measurement_passes=10,previous_measurement_passes=10,aggregation='NEW ten passes only; not a pooled twenty-pass average.',
        reuse='Frozen factor bank, shared checkpoints and stored Q/A retrieval indices only. No prior inference results copied.',
        measurement_protocol='Same questions, checkpoints, greedy decoding, matched per-pass query order and metrics. Method rotation offset by one.',
        prepared_utc=now())
    save(out/'design.json',d)
    save(out/'manifest.json',{f:digest(out/f) for f in ['design.json','sessions.json','queries.json']})
    save(out/'completion.json',dict(completed=False,stage='prepared',physical_gpu=1,additional_passes=10))
    save(out/'driver_status.json',dict(stage='prepared',started_utc=now()))
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Ten additional measurement passes</title>'
        '<h1>Measurements pending</h1><p>Ten fresh passes at every checkpoint on GPU 1. '
        'Previous scores are not copied or pooled. The audited mean and SD will appear after completion.</p>'
        '<p><a href="progress.html">Live progress</a></p>')
    if list(out.glob('*/run-*')):raise ValueError('A previous measurement folder was copied accidentally')
    save(out/'control_source.json',dict(files={str(Path(__file__).resolve()):digest(__file__),
        str(ROOT/'qa500_report.py'):digest(ROOT/'qa500_report.py')},prepared_utc=now()))
    shutil.copy2(__file__,out/'driver_source.py')
    print('Prepared a fresh ten-pass grid; reused bank checksums verified; zero evaluation cells copied.',flush=True)


def verify_parent(out):
    d=check(out)
    for f,h in d['remeasurement']['source_artifacts'].items():
        if digest(Path(d['remeasurement']['source'])/f)!=h:
            raise ValueError('Frozen source metadata changed: '+f)
    return d


def drive(out):
    from qa500_run import child
    d=verify_parent(out)
    save(out/'completion.json',dict(completed=False,stage='running',physical_gpu=1,additional_passes=10))
    try:
        for repeat in range(10):
            methods=list(METHODS);offset=(repeat+1)%len(methods)
            for method in methods[offset:]+methods[:offset]:
                child(out,f'remeasure-{method}-pass-{repeat+1}',[sys.executable,'-u',str(ROOT/'qa500_eval.py'),
                    '--out',str(out),'--method',method,'--repeat',str(repeat)])
        verify_parent(out)
        child(out,'report-new-ten-pass-means',[sys.executable,'-u',str(ROOT/'qa500_report.py'),'--out',str(out)],gpu=False)
        save(out/'driver_status.json',dict(stage='completed',finished_utc=now()))
        status(out,'completed')
    except BaseException as error:
        save(out/'completion.json',dict(completed=False,stage='failed',error=repr(error),physical_gpu=1))
        status(out,'failed',error=repr(error));raise


def monitor(out):
    import qa500_progress as progress
    progress.OUT=out
    while True:
        try:
            if progress.render() in ['completed','failed']:return
        except (OSError,ValueError,KeyError):pass
        time.sleep(30)


def launch(out):
    verify_parent(out)
    if read(out/'completion.json').get('completed'):raise RuntimeError('These ten passes are already completed')
    meta=out/'detached_launch.json'
    if meta.exists():
        previous=read(meta)
        cmd=Path('/proc')/str(previous['pid'])/'cmdline'
        if cmd.exists() and b'qa500_remeasure.py' in cmd.read_bytes():raise RuntimeError('Remeasurement already running')
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'drive','--out',str(out)]
    with (out/'driver.log').open('a') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True,close_fds=True)
    with (out/'progress.log').open('a') as log:
        m=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'monitor','--out',str(out)],cwd=ROOT,
                           stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    save(meta,dict(pid=p.pid,monitor_pid=m.pid,command=command,started_utc=now(),physical_gpu=1,
                  additional_passes=10,aggregation='NEW ten passes only',detached=True))
    print('Detached remeasurement PID',p.pid,'monitor PID',m.pid,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','launch','drive','monitor'])
    parser.add_argument('--out',type=Path,default=OUT);args=parser.parse_args()
    globals()[args.action](args.out.resolve())
