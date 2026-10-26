"""Audit all ten WISE passes before reporting full final-answer comparisons."""
import csv
import html
from pathlib import Path
import statistics

from .core import ROOT,HERE,VENDOR,SIZES,read,save,digest,check,order,summary,now

LABELS={'our_delta':'SessionControl','plain_alphaedit':'AlphaEdit without session control','wise':'WISE-Retrieve','rag':'RAG','kv_full':'Full-bank KV cache'}
COLORS={'our_delta':'#0077b6','plain_alphaedit':'#555555','wise':'#c45b00','rag':'#999999','kv_full':'#222222'}
METRICS=['mean_f1','exact_rate','ttft_mean_ms','ttft_median_ms','ttft_p95_ms','first_generated_mean_ms',
         'query_latency_mean_ms','output_tokens_mean','hit_cap_rate']
MEMORY=['extra_disk_bank_bytes','host_peak_rss_bytes','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes','side_gpu_payload_bytes']


def aggregate(values):
    if len(values)!=10:raise ValueError('Ten actual passes are required')
    good=[x for x in values if x is not None]
    return dict(mean=statistics.mean(good) if good else None,
                sd=statistics.stdev(good) if len(good)>1 else None,valid_passes=len(good))


def write_csv(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)


def build(out):
    d=check(out)
    if not read(out/'construction.json').get('completed'):raise ValueError('WISE writing is incomplete')
    for n,h in read(out/'construction.json')['checkpoint_sha256'].items():
        if digest(out/'checkpoints'/f'wise-{n}.pt')!=h:raise ValueError('WISE memory checkpoint changed')
    queries={q['qid']:q for q in read(out/'queries.json')}
    from question_key import normalize
    from memory_comparison_core import token_f1
    cells={};passrows=[];answers=[];count=0
    for repeat in range(10):
        folder=out/'wise'/f'run-{repeat:02d}';prov=read(folder/'provenance.json')
        if prov['physical_gpu']!=1:raise ValueError('Wrong physical GPU')
        for file,h in prov['code_sha256'].items():
            if digest(folder/'code_snapshot'/file)!=h:raise ValueError('Executed WISE source snapshot changed')
        for plan in d['plan']:
            n=plan['n'];cell=read(folder/f'bank-{n}.json')
            if not cell.get('completed') or (cell['method'],cell['repeat'],cell['n'],cell['physical_gpu'])!=('wise',repeat,n,1):
                raise ValueError('Incorrect WISE trial-cell identity')
            trials=cell['trials']
            if [t['qid'] for t in trials]!=order(plan['qids'],repeat,n):raise ValueError('Unmatched query grid/order')
            if len(trials)!=plan['questions']:raise ValueError('Incomplete WISE trial grid')
            for t in trials:
                q=queries[t['qid']]
                if (t['question'],t['reference'])!=(q['question'],q['answer']):raise ValueError('Official Q/A changed')
                if abs(t['token_f1']-token_f1(t['final']['text'],q['answer']))>1e-12:raise ValueError('Whole-answer F1 mismatch')
                if t['exact']!=(normalize(t['final']['text'])==normalize(q['answer'])):raise ValueError('Whole-answer exact-match mismatch')
                g=t['first_generated_token_seconds'];v=t['ttft_seconds']
                if g is None or not 0<=g<=t['query_seconds']:raise ValueError('Missing/invalid generated-token timestamp')
                if v is not None and not g<=v<=t['query_seconds']:raise ValueError('Invalid first-visible timestamp')
                answers.append(dict(method='wise',repeat=repeat,n=n,qid=t['qid'],question=t['question'],reference=t['reference'],
                    final_answer=t['final']['text'],question_type=q['question_type'],original64=q['original64'],
                    token_f1=t['token_f1'],exact=t['exact'],ttft_seconds=v,first_generated_seconds=g,
                    query_seconds=t['query_seconds'],finish_reason=t['final']['finish_reason']))
            if summary(trials)!=cell['groups']['all']:raise ValueError('Recorded WISE metric summary mismatch')
            count+=len(trials);cells[n,repeat]=cell
            for group,metrics in cell['groups'].items():passrows.append(dict(method='wise',repeat=repeat,n=n,group=group,**metrics))
    if count!=sum(p['questions'] for p in d['plan'])*10:raise ValueError('Incorrect WISE request denominator')
    rows=[]
    for n in SIZES:
        cs=[cells[n,r] for r in range(10)]
        groups=set(cs[0]['groups'])
        if any(set(c['groups'])!=groups for c in cs):raise ValueError('Cohort summaries differ across passes')
        for group in sorted(groups):
            row=dict(method='wise',n=n,group=group,status='measured',passes=10,questions=cs[0]['groups'][group]['questions'])
            for key in METRICS:
                a=aggregate([c['groups'][group][key] for c in cs])
                row.update({key+'_mean':a['mean'],key+'_sd':a['sd'],key+'_valid_passes':a['valid_passes']})
            for key in MEMORY:
                a=aggregate([c['resources'][key] for c in cs]);row.update({key+'_mean':a['mean'],key+'_sd':a['sd']})
            row['visible_ttft_samples']=sum(c['groups'][group]['visible_ttft_samples'] for c in cs)
            row['missing_visible_ttft']=sum(c['groups'][group]['missing_visible_ttft'] for c in cs)
            rows.append(row)
    baseline=read(out/'reference/comparison.json')['summary']
    allowed=set(METRICS+MEMORY)
    for r in baseline:
        row={k:v for k,v in r.items() if k in ['method','n','group','status','passes','questions','visible_ttft_samples','missing_visible_ttft'] or
             any(k==metric+suffix for metric in allowed for suffix in ['_mean','_sd','_valid_passes'])}
        rows.append(row)
    write_csv(out/'comparison.csv',rows);write_csv(out/'wise_per_pass.csv',passrows);write_csv(out/'wise_question_outputs.csv',answers)
    save(out/'comparison.json',dict(summary=rows,wise_verified_requests=count,wise_passes=10,baseline_passes=10,
         accuracy='Complete final generated answer only: word-token F1 and normalized whole-answer exact match.',
         baseline_source_run=Path(d['source_reference']).name,baseline_timing_note='Prior measurements; not interleaved with WISE.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    allrows=[r for r in rows if r['group']=='all']
    def lines(ax,key,scale=1):
        for method in LABELS:
            rs=sorted([r for r in allrows if r['method']==method and r['status']=='measured' and r.get(key+'_mean') is not None],key=lambda r:r['n'])
            if rs:ax.errorbar([r['n'] for r in rs],[r[key+'_mean']*scale for r in rs],
                yerr=[(r.get(key+'_sd') or 0)*scale for r in rs],marker='o',capsize=3,color=COLORS[method],label=LABELS[method])
        ax.set_xscale('log',base=2);ax.set_xticks(SIZES,labels=SIZES);ax.set_xlabel('Stored conversations');ax.grid(alpha=.2)
    def export(fig,name):
        for ext in ['svg','png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=170,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,key,title in zip(axes,['mean_f1','exact_rate'],['Complete final-answer word-token F1','Complete final-answer normalized exact match']):
        lines(ax,key,100);ax.set_title(title,fontsize=11);ax.set_ylim(0,105);ax.set_ylabel('Percent');ax.axhline(100,color='#bbb',ls='--',lw=1)
    axes[0].legend(fontsize=8,loc='lower left');fig.suptitle('WISE-Retrieve versus the completed session-memory comparison')
    fig.tight_layout(rect=(0,0,1,.93));export(fig,'accuracy_final_answer')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,key,title in zip(axes,['ttft_mean_ms','ttft_median_ms','ttft_p95_ms'],['Mean visible TTFT','Mean per-pass median TTFT','Mean per-pass p95 TTFT']):
        lines(ax,key);ax.set_title(title);ax.set_ylabel('Milliseconds');ax.set_yscale('log')
    axes[0].legend(fontsize=7);fig.suptitle('Mean ± sample SD across ten inference passes; baseline timings measured earlier')
    fig.tight_layout(rect=(0,0,1,.93));export(fig,'ttft')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,key,title,scale in zip(axes,['extra_disk_bank_bytes','host_peak_rss_bytes','peak_gpu_allocated_bytes'],
        ['Deployed bank storage (MiB)','Peak process host RAM (GiB)','Peak CUDA allocated (GiB)'],[1/2**20,1/2**30,1/2**30]):
        lines(ax,key,scale);ax.set_title(title)
    axes[0].legend(fontsize=7);fig.suptitle('Separate memory tiers; GPU includes common model weights; KV capacity failures omitted')
    fig.tight_layout(rect=(0,0,1,.93));export(fig,'memory')
    def value(r,key,scale=1):
        a=r.get(key+'_mean');b=r.get(key+'_sd')
        return '—' if a is None else f'{a*scale:.2f} ± {b*scale:.2f}' if b is not None else f'{a*scale:.2f}'
    text=['# WISE-Retrieve: full-context, exact-question, complete-answer memory comparison','',
        'Completed 500 sequential conversation writes and ten actual inference passes at every checkpoint on physical GPU 1. '
        'WISE uses its own native activation router. Accuracy is measured only on the complete final generated answer: token F1 and normalized exact match.', '',
        'The four baseline conditions are the previous audited ten-pass run. Their scores and timings are explicitly reused; '
        'only WISE was newly trained and evaluated. These are ten inference repetitions on one trained trajectory, not ten independently trained models.','',
        '| Sessions | Questions/pass | Method | F1 (%) | Exact (%) | Mean visible TTFT (ms) | Bank (MiB) | Peak GPU (GiB) |',
        '|---:|---:|---|---:|---:|---:|---:|---:|']
    for n in SIZES:
        for method in LABELS:
            r=next(r for r in allrows if r['n']==n and r['method']==method)
            if r['status']=='measured':text.append(f"| {n} | {r['questions']} | {LABELS[method]} | {value(r,'mean_f1',100)} | {value(r,'exact_rate',100)} | {value(r,'ttft_mean_ms')} | {value(r,'extra_disk_bank_bytes',1/2**20)} | {value(r,'peak_gpu_allocated_bytes',1/2**30)} |")
            else:text.append(f"| {n} | — | {LABELS[method]} | Unavailable: {r['status']} | — | — | — | — |")
    text+=['','## Protocol and limits','',
        d['writer'],d['inputs'],d['adaptation'],d['boundaries'],d['evaluation'],d['timing'],d['memory'],d['limitations'],'',
        'At checkpoints 8,16,32,64,128,256,500 the query counts are 8,16,32,64,98,162,283. '
        'The final 500 conversations contain 508 official Q/A bindings. No paraphrases or newly invented questions are used.', '',
        'TTFT measures the first nonempty final answer, not total completion time or a routing probe. '
        'Requests with no visible output have missing visible TTFT and remain included in F1/exact match. '
        'The CSV reports visible/missing timing counts and valid-pass counts. Actual KV failures remain unavailable.', '',
        'Different eligible questions at different N change task difficulty; original64/new/type/abstention groups are exported separately. '
        'F1 is lexical overlap, not a semantic correctness judgment. WISE uses its recommended layer 29 and hyperparameters; '
        'AlphaEdit uses its own configured layers and solver. This compares complete systems rather than isolating a single routing change.', '',
        f"Official WISE source: https://github.com/zjunlp/EasyEdit/tree/{d['upstream']['commit']}/easyeditor/models/wise . "
        'Unmodified source is vendored with its MIT license and per-file SHA-256 manifest.', '',
        'No measured write-time comparison is claimed. Disk bank measurements exclude common frozen base weights and training-only restart state.']
    (out/'report.md').write_text('\n'.join(text)+'\n')
    svgs=''.join('<svg'+(out/f'{name}.svg').read_text().split('<svg',1)[1] for name in ['ttft','accuracy_final_answer','memory'])
    page='<!doctype html><meta charset="utf-8"><title>WISE full-Q/A comparison</title><style>body{font:16px system-ui;max-width:1450px;margin:40px auto;padding:0 20px}svg{width:100%;height:auto}pre{white-space:pre-wrap}</style>'
    page+='<h1>WISE-Retrieve: ten-pass final-answer comparison</h1><p>TTFT, final-answer accuracy and separate memory tiers. Prior baseline measurements are reused and remain distinct from new WISE measurements.</p>'+svgs+'<pre>'+html.escape('\n'.join(text))+'</pre>'
    (out/'report.html').write_text(page)
    artifacts=['comparison.csv','comparison.json','wise_per_pass.csv','wise_question_outputs.csv','report.html','report.md',
               *[f'{name}.{ext}' for name in ['ttft','accuracy_final_answer','memory'] for ext in ['svg','png','pdf']]]
    save(out/'report_manifest.json',{f:digest(out/f) for f in artifacts})
    save(out/'completion.json',dict(completed=True,physical_gpu=1,conversations=500,bindings=508,passes=10,
        checkpoints=list(SIZES),final_eligible_questions=283,wise_verified_requests=count,baseline_metrics_reused=True,finished_utc=now()))
    print('Completed WISE comparison:',count,'verified WISE requests.',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.out.resolve())
