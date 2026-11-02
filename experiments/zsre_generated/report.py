"""Completion-gated audit of all six conditions and ten real inference passes."""
import argparse
import csv
import html
import math
from pathlib import Path
import statistics
from .core import SIZES,METHODS,LABELS,check,read,save,digest,order,summary,now

METRICS=['mean_f1','exact_rate','ttft_mean_ms','ttft_median_ms','ttft_p95_ms','first_generated_mean_ms',
         'query_latency_mean_ms','output_tokens_mean','hit_cap_rate']
MEMORY=['extra_disk_bank_bytes','host_peak_rss_bytes','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes']


def csvwrite(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)


def aggregate(values):
    if len(values)!=10:raise ValueError('Require exactly ten actual inference passes')
    good=[v for v in values if v is not None]
    return dict(mean=statistics.mean(good) if good else None,sd=statistics.stdev(good) if len(good)>1 else None,valid_passes=len(good))


def build(out):
    d=check(out);requests=read(out/'requests.json');rows=[];perpass=[];answers=[];verified=0;unavailable=0
    from question_key import normalize
    from memory_comparison_core import token_f1
    for method in ['plain_alphaedit','wise_merge','wise_retrieve']:
        done=read(out/method/'construction.json')
        if not done.get('completed') or done['count']!=1000:raise ValueError('Incomplete memory construction')
    for method in METHODS:
        for n in SIZES:
            cells=[]
            for repeat in range(10):
                folder=out/method/f'run-{repeat:02d}';prov=read(folder/'provenance.json')
                if prov['physical_gpu']!=1 or prov['code_sha256']!=d['code_sha256']:raise ValueError('Execution provenance mismatch')
                c=read(folder/f'bank-{n}.json')
                if not c.get('completed') or (c['method'],c['repeat'],c['n'])!=(method,repeat,n):raise ValueError('Incomplete/incorrect measurement cell')
                cells.append(c)
                if not c['available']:
                    if method!='kv_full' or c['reason'] not in ['context_limit','gpu_limit'] or c['trials']:raise ValueError('Unexpected unavailable condition')
                    unavailable+=n;continue
                if [t['case_id'] for t in c['trials']]!=order(n,repeat):raise ValueError('Unmatched question cohort/order')
                for t in c['trials']:
                    request=requests[t['case_id']]
                    if (t['question'],t['reference'])!=(request['prompt'],request['target_new']):raise ValueError('Official Q/A mismatch')
                    if abs(t['token_f1']-token_f1(t['final']['text'],request['target_new']))>1e-12:raise ValueError('Final-answer F1 mismatch')
                    if t['exact']!=(normalize(t['final']['text'])==normalize(request['target_new'])):raise ValueError('Exact-match mismatch')
                    g=t['first_generated_token_seconds'];v=t['ttft_seconds']
                    if g is None or not 0<=g<=t['query_seconds']:raise ValueError('Invalid generated-token timestamp')
                    if v is not None and not g<=v<=t['query_seconds']:raise ValueError('Invalid visible timestamp')
                    verified+=1
                    answers.append(dict(method=method,n=n,repeat=repeat,case_id=t['case_id'],question=t['question'],reference=t['reference'],
                        final_answer=t['final']['text'],token_f1=t['token_f1'],exact=t['exact'],ttft_seconds=v,
                        query_seconds=t['query_seconds'],finish_reason=t['final']['finish_reason']))
                if summary(c['trials'])!=c['groups']['all']:raise ValueError('Per-pass summary mismatch')
                perpass.append(dict(method=method,n=n,repeat=repeat,**c['groups']['all'],**{key:c['resources'][key] for key in MEMORY}))
            if len({c['available'] for c in cells})!=1:raise ValueError('Inconsistent availability across ten passes; investigate separately')
            row=dict(method=method,n=n,group='all',passes=10,questions=n if cells[0]['available'] else 0,
                     status='measured' if cells[0]['available'] else cells[0]['reason'])
            if cells[0]['available']:
                for key in METRICS:
                    for suffix,value in aggregate([c['groups']['all'][key] for c in cells]).items():row[key+'_'+suffix]=value
                for key in MEMORY:
                    for suffix,value in aggregate([c['resources'][key] for c in cells]).items():row[key+'_'+suffix]=value
                row['visible_ttft_samples']=sum(c['groups']['all']['visible_ttft_samples'] for c in cells)
                row['missing_visible_ttft']=sum(c['groups']['all']['missing_visible_ttft'] for c in cells)
            rows.append(row)
    planned=len(METHODS)*10*sum(SIZES)
    if verified+unavailable!=planned:raise ValueError('Incorrect experiment denominator')
    csvwrite(out/'comparison.csv',rows);csvwrite(out/'per_pass.csv',perpass);csvwrite(out/'question_outputs.csv',answers)
    save(out/'comparison.json',dict(completed=True,summary=rows,verified_requests=verified,unavailable_requests=unavailable,
         planned_requests=planned,baseline_metrics_reused=False,metrics=d['metrics'],repetition=d['repetition']))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    styles={'our_delta':('#0077B6','o','-'),'plain_alphaedit':('#222222','s','-'),'rag':('#666666','^','-'),
            'kv_full':('#999999','D','-'),'wise_merge':('#444444','v','--'),'wise_retrieve':('#777777','x',':')}
    def lines(ax,key,scale=1):
        for method in METHODS:
            rs=[r for r in rows if r['method']==method and r['status']=='measured' and r.get(key+'_mean') is not None]
            if rs:
                color,marker,ls=styles[method]
                ax.errorbar([r['n'] for r in rs],[r[key+'_mean']*scale for r in rs],
                    yerr=[(r[key+'_sd'] or 0)*scale for r in rs],color=color,marker=marker,ls=ls,capsize=2,label=LABELS[method])
        ax.set_xscale('log',base=2);ax.set_xticks(SIZES,labels=SIZES);ax.tick_params(axis='x',labelsize=8)
        ax.set_xlabel('Stored ZsRE Q/A items');ax.grid(alpha=.2)
    def export(fig,name):
        for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,key,title in zip(axes,['mean_f1','exact_rate'],['Complete final-answer word-token F1','Complete final-answer exact match']):
        lines(ax,key,100);ax.set_ylim(0,103);ax.set_ylabel('Percent');ax.set_title(title)
    axes[0].legend(fontsize=8);fig.suptitle('ZsRE free generation · mean ± sample SD over ten inference passes');fig.tight_layout();export(fig,'accuracy')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,key,title in zip(axes,['ttft_mean_ms','ttft_p95_ms','query_latency_mean_ms'],['Mean visible TTFT','Mean per-pass p95 TTFT','Mean total query latency']):
        lines(ax,key);ax.set_ylabel('Milliseconds');ax.set_yscale('log');ax.set_title(title)
    axes[0].legend(fontsize=7);fig.suptitle('Matched fresh measurements · query routing and prefill included');fig.tight_layout();export(fig,'latency')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,key,scale,title in zip(axes,MEMORY[:3],[1/2**20,1/2**30,1/2**30],['Retained bank disk (MiB)','Peak process RAM (GiB)','Peak allocated GPU (GiB)']):
        lines(ax,key,scale);ax.set_title(title)
    axes[0].legend(fontsize=7);fig.suptitle('Memory tiers remain separate · GPU includes common model weights');fig.tight_layout();export(fig,'memory')
    def fmt(row,key,scale=1):
        if row.get(key+'_mean') is None:return '—'
        a=row[key+'_mean']*scale;b=row.get(key+'_sd')
        return f'{a:.2f}' if b is None else f'{a:.2f} ± {b*scale:.2f}'
    text=['# ZsRE generated-answer comparison through 1,000 memories','',
          f'Completed ten NEW inference passes per method/checkpoint. Audited {verified:,} measured requests; '
          f'{unavailable:,} planned KV requests were unavailable because of measured capacity limits. No previous quality or timing scores were copied.',
          '', '**Accuracy is scored on the complete final generated answer. No teacher forcing, reference containment or gold-assisted answer extraction.**','',
          '| Items | Method | F1 (%) | Exact (%) | Mean TTFT (ms) | Bank MiB | Peak GPU GiB |',
          '|---:|---|---:|---:|---:|---:|---:|']
    for n in SIZES:
        for method in METHODS:
            row=next(r for r in rows if r['method']==method and r['n']==n)
            if row['status']=='measured':text.append(f"| {n} | {LABELS[method]} | {fmt(row,'mean_f1',100)} | {fmt(row,'exact_rate',100)} | {fmt(row,'ttft_mean_ms')} | {fmt(row,'extra_disk_bank_bytes',1/2**20)} | {fmt(row,'peak_gpu_allocated_bytes',1/2**30)} |")
            else:text.append(f"| {n} | {LABELS[method]} | Unavailable: {row['status']} | — | — | — | — |")
    text+=['','## Protocol and practical limits','']
    for key in ['unit','write','reuse','plain_solver','evaluation','routing','generation','metrics','timing','memory','wise_merge','repetition','limits']:
        text.append('- '+d[key])
    text+=['','Mistral-7B-v0.1, pinned revision '+d['model_revision']+', FP32, physical GPU 1. '
           'The original ZsRE teacher-forced reliability/generalization/locality results remain a separate benchmark. '
           'This report uses the LongMemEval study’s generated-answer metrics on the ZsRE exact-question memory task.', '',
           'Every raw answer, timing, capacity decision, checkpoint hash and execution source hash remains available locally. '
           'Ten deterministic inference repetitions do not estimate variance across independently trained models. '
           'F1 is lexical overlap and does not establish semantic factual correctness.']
    text='\n'.join(text)+'\n';(out/'report.md').write_text(text)
    svgs=''.join('<svg'+(out/f'{name}.svg').read_text().split('<svg',1)[1] for name in ['accuracy','latency','memory'])
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>ZsRE · six-method generated-answer comparison</title>'
        '<style>body{font:16px/1.6 system-ui;max-width:1500px;margin:40px auto;padding:20px}svg{width:100%;height:auto}pre{white-space:pre-wrap}</style>'
        '<h1>ZsRE · generated-answer accuracy, TTFT and memory</h1>'+svgs+'<pre>'+html.escape(text)+'</pre>')
    files=['comparison.csv','comparison.json','per_pass.csv','question_outputs.csv','report.md','report.html']+[f'{name}.{ext}' for name in ['accuracy','latency','memory'] for ext in ['png','svg','pdf']]
    save(out/'report_manifest.json',{name:digest(out/name) for name in files})
    save(out/'completion.json',dict(completed=True,physical_gpu=1,methods=METHODS,sizes=SIZES,passes=10,stored_items=1000,
        verified_requests=verified,unavailable_requests=unavailable,planned_requests=planned,finished_utc=now(),report='report.html'))
    print('FINAL_REPORT_READY',out/'report.html',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);build(p.parse_args().out.resolve())
