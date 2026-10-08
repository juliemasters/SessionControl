"""Sample actual GPU/RAM/storage usage; produces no graphs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--pid', type=int, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--delta-root', type=Path, help='Read-only existing delta store, when reused')
p.add_argument('--status-path', type=Path, help='Status of an independently running training/evaluation process')
a = p.parse_args()
a.out.mkdir(parents=True,exist_ok=True)
start = time.monotonic()
with (a.out/'resource_usage.jsonl').open('a', buffering=1) as f:
    while (Path('/proc')/str(a.pid)).exists():
        row = dict(utc=datetime.now(timezone.utc).isoformat(), sample_elapsed_seconds=time.monotonic()-start,
                   pid=a.pid, measurement_scope='Sampling began at process launch; GPU board usage, experiment process RSS/HWM')
        try:
            values = subprocess.run(['nvidia-smi','-i','1','--query-gpu=memory.used,memory.total,utilization.gpu',
                                     '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout.strip().split(',')
            row.update(gpu_used_mib=float(values[0]), gpu_total_mib=float(values[1]), gpu_utilization_percent=float(values[2]))
            status = (Path('/proc')/str(a.pid)/'status').read_text()
            for line in status.splitlines():
                if line.startswith('VmRSS:'):
                    row['host_rss_mib'] = int(line.split()[1])/1024
                if line.startswith('VmHWM:'):
                    row['host_peak_rss_mib'] = int(line.split()[1])/1024
            status_path = a.status_path or a.out/'status.json'
            row['stage'] = json.loads(status_path.read_text()).get('stage') if status_path.exists() else 'starting'
            row['delta_storage_bytes'] = sum(p.stat().st_size for p in (a.delta_root or a.out/'deltas').rglob('*.pt'))
        except Exception as exc:
            row['error'] = repr(exc)
        f.write(json.dumps(row)+'\n')
        time.sleep(10)
