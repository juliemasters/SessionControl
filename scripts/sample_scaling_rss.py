"""Read-only RSS sampling sidecar; sampling interval and start are explicit."""
import argparse
import json
from pathlib import Path
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--pid',type=int,required=True)
p.add_argument('--out',type=Path,required=True)
args=p.parse_args();process=Path('/proc')/str(args.pid);begin=time.monotonic()
identity=(process/'cmdline').read_bytes()
with (args.out/'resource_usage_cpu.jsonl').open('a',buffering=1) as output:
    while process.exists():
        try:
            if (process/'cmdline').read_bytes()!=identity:break
            try:
                state=json.loads((args.out/'status.json').read_text())
            except FileNotFoundError:
                time.sleep(.5)
                continue
            row=dict(sample_seconds=time.monotonic()-begin,interval_seconds=.5,
                stage=state['stage'],n=state.get('n'),repeat=state.get('repeat'),pid=args.pid)
            measurements_path=args.out/'measurements.json'
            if measurements_path.exists():
                completed=json.loads(measurements_path.read_text())
                phase=state['stage']+(('_'+str(state['n'])) if state.get('n') is not None else '')
                if phase in completed:
                    row.update(recorded_stage=row['stage'],recorded_n=row['n'],
                               stage='between_conditions',n=None)
            for line in (process/'status').read_text().splitlines():
                if line.startswith('VmRSS:'):row['host_rss_bytes']=int(line.split()[1])*1024
                if line.startswith('VmHWM:'):row['lifetime_high_water_bytes']=int(line.split()[1])*1024
            output.write(json.dumps(row)+'\n')
        except (FileNotFoundError,ProcessLookupError):break
        time.sleep(.5)
