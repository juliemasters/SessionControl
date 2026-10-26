"""Live progress for the corrected experiment; no provisional mean-score claims."""
from datetime import datetime
import html
from pathlib import Path
import statistics
import time
from zoneinfo import ZoneInfo

from comparison_core import read
from qa500_core import ROOT,METHODS,SIZES

OUT=ROOT/'runs/qa500-checkpoints-tenpass-v1'


def render():
    done=read(OUT/'completion.json');driver=read(OUT/'driver_status.json') if (OUT/'driver_status.json').exists() else {}
    state='completed' if done.get('completed') else 'failed' if done.get('stage')=='failed' else 'running'
    rows=[]
    for phase in ['independent','shared']:
        ledger=read(OUT/phase/'ledger.json');s=read(OUT/phase/'status.json') if (OUT/phase/'status.json').exists() else {}
        times=[r['internal_seconds'] for r in ledger[-20:] if 'internal_seconds' in r]
        estimate=f'{(500-len(ledger))*statistics.mean(times)/3600:.1f} h remaining in this write phase' if times and len(ledger)<500 else ''
        rows.append(f'<tr><td>{phase} answer writing</td><td>{len(ledger)}/500 (64 reused)</td><td>{html.escape(s.get("stage","queued"))}</td><td>{estimate}</td></tr>')
    table='<table><tr><th>Method</th>'+''.join(f'<th>N={n}</th>' for n in SIZES)+'</tr>'
    for method in METHODS:
        table+='<tr><td>'+method+'</td>'
        for n in SIZES:
            complete=0;unavailable=0
            for repeat in range(10):
                path=OUT/method/f'run-{repeat:02d}'/f'bank-{n}.json'
                if path.exists():
                    cell=read(path)
                    if cell.get('completed'):complete+=1;unavailable+=not cell['available']
            table+=f'<td>{complete}/10'+(f' ({unavailable} unavailable)' if unavailable else '')+'</td>'
        table+='</tr>'
    table+='</table>'
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S KST')
    page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="60"><title>Corrected Q/A run progress</title><style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;margin:20px 0}td,th{padding:10px;border:1px solid #ccc}</style>'
    page+=f'<h1>Corrected full-Q/A experiment: {state}</h1><p>Updated {stamp}. Physical GPU 1. Current job: {html.escape(str(driver.get("stage","preparing")))}</p>'
    page+='<p><b>Every write uses the full context, exact question and complete reference answer. Activation argmax routing. No evidence capsules.</b></p>'
    page+='<table><tr><th>Construction</th><th>Saved</th><th>Status</th><th>Estimate</th></tr>'+''.join(rows)+'</table>'+table
    page+='<p>All checkpoints are freshly evaluated. Ten passes reuse one frozen bank; final mean ± SD appears after all feasible passes complete.</p><p><a href="report.html">Final report</a> · <a href="design.json">Frozen protocol</a> · <a href="reuse_audit.json">Reuse audit</a></p>'
    if state=='failed':page+='<p>Stopped: '+html.escape(str(done.get('error')))+'</p>'
    path=OUT/'progress.html';temp=path.with_suffix('.html.tmp');temp.write_text(page);temp.replace(path)
    return state


if __name__=='__main__':
    while True:
        try:
            if render() in ['completed','failed']:break
        except (OSError,ValueError,KeyError):pass
        time.sleep(30)
