"""Persistent corrected-Q/A pilot, extension, ten fresh passes and report."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from comparison_core import read,save
from qa500_core import ROOT,METHODS,check,status,now,verify_reused_accumulator


def wait_gpu(out,job):
    while True:
        r=subprocess.run(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader'],
                         check=True,capture_output=True,text=True)
        if not r.stdout.strip():return
        status(out,'waiting_for_gpu1',job=job)
        time.sleep(30)


def child(out,job,command,gpu=True):
    if gpu:wait_gpu(out,job)
    status(out,'running',job=job)
    p=subprocess.Popen(command,cwd=ROOT)
    save(out/'driver_status.json',dict(stage=job,pid=p.pid,started_utc=now()))
    code=p.wait()
    if code:raise RuntimeError(f'{job} exited with code {code}')


def main(out):
    d=check(out)
    save(out/'completion.json',dict(completed=False,stage='running',physical_gpu=1))
    try:
        if not (out/'shared/reuse_validation.json').exists():verify_reused_accumulator(out)
        # A real small pass catches protocol/inference errors before long writes.
        for method in METHODS:
            child(out,'pilot-'+method,[sys.executable,'-u',str(ROOT/'qa500_eval.py'),'--out',str(out),
                                      '--method',method,'--repeat','0','--sizes','8'])
        current=read(out/'our_delta/run-00/bank-8.json')['trials']
        old=read(ROOT/'runs/official64-fourmethod-v1/our_delta/evaluation.json')['results']['8']['trials']
        expected={t['qid']:t['final']['token_ids'] for t in old}
        parity=all(t['final']['token_ids']==expected[t['qid']] for t in current)
        save(out/'pilot_validation.json',dict(saved_answer_delta_token_parity=parity,questions=8,
                                             all_write_inputs='full context + exact question + complete answer'))
        if not parity:raise ValueError('Saved-answer pilot output parity failed; refusing to launch long writes')
        for phase in ['independent','shared']:
            child(out,'write-'+phase,[sys.executable,'-u',str(ROOT/'qa500_build.py'),'--out',str(out),'--phase',phase])
        for repeat in range(d['repeats']):
            methods=list(METHODS);offset=repeat%len(methods);methods=methods[offset:]+methods[:offset]
            for method in methods:
                child(out,f'evaluate-{method}-pass-{repeat+1}',[sys.executable,'-u',str(ROOT/'qa500_eval.py'),
                      '--out',str(out),'--method',method,'--repeat',str(repeat)])
        child(out,'report',[sys.executable,'-u',str(ROOT/'qa500_report.py'),'--out',str(out)],gpu=False)
        save(out/'driver_status.json',dict(stage='completed',finished_utc=now()))
        status(out,'completed')
    except BaseException as error:
        save(out/'completion.json',dict(completed=False,stage='failed',error=repr(error),physical_gpu=1))
        status(out,'failed',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out.resolve())
