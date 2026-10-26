"""Detach the corrected, storage-bounded experiment from the tool session."""
from pathlib import Path
import subprocess
import sys

from comparison_core import read,save
from qa500_core import ROOT,storage_check,now


def main():
    out=ROOT/'runs/qa500-checkpoints-tenpass-v1'
    metadata=out/'detached_launch.json'
    if metadata.exists():
        previous=read(metadata);cmd=Path('/proc')/str(previous['pid'])/'cmdline'
        if cmd.exists() and b'qa500_run.py' in cmd.read_bytes():raise RuntimeError('Experiment already running')
    storage_check(out)
    command=[sys.executable,'-u',str(ROOT/'qa500_run.py'),'--out',str(out)]
    with (out/'driver.log').open('a') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                           start_new_session=True,close_fds=True)
    with (out/'progress.log').open('a') as log:
        monitor=subprocess.Popen([sys.executable,'-u',str(ROOT/'qa500_progress.py')],cwd=ROOT,stdin=subprocess.DEVNULL,
                                  stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    save(metadata,dict(pid=p.pid,monitor_pid=monitor.pid,command=command,started_utc=now(),physical_gpu=1,
                       checkpoint_policy='8,16,32,64,128,256,500; ten actual measurement passes',detached=True))
    print('Detached experiment PID',p.pid,'progress monitor',monitor.pid,flush=True)


if __name__=='__main__':main()
