"""Publish measured TTFT immediately; never label partial passes as final.

CPU-only reporting. No inference settings, worker sources or timings change.
The previous ten-pass run is separate from the newly collected measurements.
"""
import argparse
import csv
from datetime import datetime
import html
from pathlib import Path
import statistics
import time
from zoneinfo import ZoneInfo

from comparison_core import read,save,digest
from qa500_core import ROOT,METHODS,SIZES,now

LABELS={'our_delta':'SessionControl','plain_alphaedit':'AlphaEdit without session control',
        'rag':'RAG','kv_full':'Full-bank KV cache'}
PRIOR=ROOT/'runs/qa500-checkpoints-tenpass-v1'
DEFAULT=ROOT/'runs/qa500-checkpoints-tenpass-v2'


def collect(root):
    rows=[]
    for n in SIZES:
        for method in METHODS:
            cells=[]
            for repeat in range(10):
                path=root/method/f'run-{repeat:02d}'/f'bank-{n}.json'
                if path.exists():
                    c=read(path)
                    if c.get('completed'):
                        if (c['n'],c['method'],c['repeat'],c['physical_gpu'])!=(n,method,repeat,1):
                            raise ValueError('Incorrect timing-cell identity')
                        cells.append(c)
            available=[c for c in cells if c['available']]
            unavailable=[c for c in cells if not c['available']]
            if available and unavailable:raise ValueError('Mixed timing availability requires investigation')
            means=[];medians=[];p95s=[];samples=0;missing=0
            for c in available:
                trials=c['trials']
                if len(trials)!=c['planned_questions']:raise ValueError('Incomplete timing query grid')
                values=sorted(t['ttft_seconds']*1000 for t in trials if t['ttft_seconds'] is not None)
                for t in trials:
                    v=t['ttft_seconds'];g=t['first_generated_token_seconds']
                    if v is not None and not 0<=g<=v<=t['query_seconds']:
                        raise ValueError('Invalid final-answer TTFT timestamp')
                if values:
                    means.append(statistics.mean(values));medians.append(statistics.median(values))
                    import math
                    p95s.append(values[math.ceil(.95*len(values))-1])
                samples+=len(values);missing+=len(trials)-len(values)
            row=dict(method=method,n=n,completed_passes=len(cells),measured_passes=len(available),
                     valid_mean_passes=len(means),visible_timing_samples=samples,missing_visible_samples=missing,
                     unavailable_passes=len(unavailable),
                     status=('final' if len(cells)==10 else 'provisional') if available else
                            ('unavailable' if unavailable else 'pending'),
                     unavailable_reason=unavailable[0]['reason'] if unavailable else None,
                     mean_ttft_ms=statistics.mean(means) if means else None,
                     sd_of_pass_mean_ms=statistics.stdev(means) if len(means)>1 else None,
                     mean_of_pass_median_ms=statistics.mean(medians) if medians else None,
                     mean_of_pass_p95_ms=statistics.mean(p95s) if p95s else None)
            rows.append(row)
    return rows


def table(rows):
    result='<table><tr><th>Sessions</th><th>Method</th><th>Passes complete</th><th>Mean visible TTFT ± SD (ms)</th><th>Mean per-pass median (ms)</th><th>Mean per-pass p95 (ms)</th><th>Status</th></tr>'
    for r in rows:
        def fmt(key):return '—' if r[key] is None else f'{r[key]:.2f}'
        mean=fmt('mean_ttft_ms')
        if r['mean_ttft_ms'] is not None:
            mean+=' ± '+fmt('sd_of_pass_mean_ms')
        state=r['status']+((': '+r['unavailable_reason']) if r['unavailable_reason'] else '')
        result+=f"<tr><td>{r['n']}</td><td>{html.escape(LABELS[r['method']])}</td><td>{r['completed_passes']}/10</td><td>{mean}</td><td>{fmt('mean_of_pass_median_ms')}</td><td>{fmt('mean_of_pass_p95_ms')}</td><td>{html.escape(state)}</td></tr>"
    return result+'</table>'


def render(out):
    current=collect(out);prior=collect(PRIOR)
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
    page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>TTFT first: ten-pass comparison</title><style>body{font:16px system-ui;max-width:1400px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;margin:20px 0;font-size:14px}td,th{padding:9px;border:1px solid #ccc}th{background:#eee}</style>'
    page+='<h1>TTFT first: time until the final answer starts</h1>'
    page+='<p>TTFT runs from query arrival through routing/retrieval, selected-delta application, tokenization and prefill to the first nonempty final-answer output. '
    page+='It excludes model startup, memory writing and an already prepared reusable KV prefix. It is not the time to finish the complete answer.</p>'
    page+=f'<p>Updated {stamp}. Physical GPU 1. Reporting priority only: full answer generation and the measurement protocol remain unchanged.</p>'
    page+='<h2>Ten additional passes: current measurements</h2><p>Partial results are provisional. Each row shows its own completed-pass count. '
    page+='Mean ± sample SD is across per-pass question-mean TTFT values. A single pass has no sample SD. Missing visible-token timings are counted in the CSV. '
    page+='Unavailable KV measurements have no TTFT. Previous timings are not pooled with these new measurements.</p>'+table(current)
    page+='<p><a href="ttft_priority.csv">Download new TTFT measurements</a> · <a href="progress.html">Run progress</a> · <a href="report.html">Full final report when completed</a></p>'
    page+='<h2>Previous ten completed passes: separate reference</h2><p>These are the earlier completed measurements, available now for comparison. They are not new passes.</p>'+table(prior)
    path=out/'ttft_priority.html';temp=path.with_suffix('.html.tmp');temp.write_text(page);temp.replace(path)
    for name,rows in [('ttft_priority.csv',current),('ttft_previous_ten.csv',prior)]:
        temp=out/(name+'.tmp')
        with temp.open('w',newline='') as handle:
            w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        temp.replace(out/name)
    save(out/'ttft_priority.json',dict(updated_utc=now(),new_passes=current,previous_passes=prior,
         aggregation='Separate runs; no pooling. Mean and sample SD across completed per-pass means.',
         source_report_complete=read(PRIOR/'completion.json')['completed']))
    return read(out/'completion.json').get('completed') or read(out/'completion.json').get('stage')=='failed'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        try:
            if render(args.out.resolve()) or not args.watch:break
        except (OSError,ValueError,KeyError):
            if not args.watch:raise
        time.sleep(30)
