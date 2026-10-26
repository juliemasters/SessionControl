"""Completion-gated, audited ten-pass means and SD. No pooled-run fiction."""
import csv
import html
import math
from pathlib import Path
import statistics
import sys

from comparison_core import read,save,digest
from qa500_core import ROOT,CODE,SIZES,METHODS,check,mean_sd,now,order
from qa500_eval import checkpoint_metrics

LABELS={'our_delta':'SessionControl: answer deltas','plain_alphaedit':'AlphaEdit: no session control',
        'rag':'RAG: stored Q/A','kv_full':'Full-bank KV: stored Q/A'}
COLORS={'our_delta':'#0077b6','plain_alphaedit':'#444444','rag':'#888888','kv_full':'#bbbbbb'}
METRICS=['mean_f1','exact_rate','reference_containment_rate','source_hit1','mean_rank','mrr','support_recall6',
         'complete_support6','ttft_mean_ms','ttft_median_ms','ttft_p95_ms','first_generated_mean_ms',
         'query_latency_mean_ms','query_latency_p95_ms','output_tokens_mean','hit_cap_rate']


def summarize(values):
    if len(values)!=10:raise ValueError('Ten independently measured passes required')
    good=[v for v in values if v is not None]
    return dict(mean=statistics.mean(good) if good else None,
                sd=statistics.stdev(good) if len(good)>1 else (0. if good else None),valid_passes=len(good))


def verify_provenance(folder):
    prov=read(folder/'provenance.json')
    if prov['physical_gpu']!=1:raise ValueError('Wrong physical GPU')
    for f,h in prov['code_sha256'].items():
        if digest(folder/prov['code_snapshots'][f])!=h:raise ValueError('Executed source snapshot changed')
    for revision in prov.get('previous_executions',[]):
        verify_provenance(folder/Path(revision['provenance']).parent)
        if digest(folder/revision['token_alignment_audit'])!=revision['audit_sha256']:
            raise ValueError('Alignment compatibility audit changed')
    return prov


def build(out):
    d=check(out);sessions=read(out/'sessions.json');queries={q['qid']:q for q in read(out/'queries.json')}
    scope=read(out/'reporting_scope.json') if (out/'reporting_scope.json').exists() else {}
    if (out/'reporting_scope_manifest.json').exists():
        for f,h in read(out/'reporting_scope_manifest.json').items():
            if digest(out/f)!=h:raise ValueError('Final-answer reporting scope changed')
        if digest(__file__)!=digest(out/'report_source.py'):
            raise ValueError('Executed reporter differs from its saved scope snapshot')
    answer_only=scope.get('accuracy_target')=='complete_final_generated_answer'
    excluded={'reference_containment_rate','source_hit1','mean_rank','mrr','support_recall6','complete_support6'} if answer_only else set()
    metrics=[m for m in METRICS if m not in excluded]
    sys.path.insert(0,str(CODE))
    from question_key import normalize
    from memory_comparison_core import token_f1
    from answer_only_generation import reference_tokens_present
    for phase in ['independent','shared']:
        if not read(out/phase/'completion.json').get('completed'):raise ValueError('Construction incomplete')
        verify_provenance(out/phase)
    for row in read(out/'independent/ledger.json'):
        if digest(out/'independent/factors'/row['file'])!=row['sha256']:raise ValueError('Answer factor changed')
    for n,h in read(out/'shared/completion.json')['patch_sha256'].items():
        if digest(out/'shared'/f'weights-{n}.pt')!=h:raise ValueError('Shared Q/A snapshot changed')
    for file,h in d['source_hashes'].items():
        if digest(file)!=h:raise ValueError('Reused source artifact changed')
    for file,h in read(out/'rag/index_manifest.json').items():
        if digest(out/'rag'/file)!=h:raise ValueError('Stored Q/A RAG record/index changed')
    rows=[];passrows=[];answers=[];capacity=[];groupset={};count=0
    for method in METHODS:
        cells={}
        for repeat in range(10):
            folder=out/method/f'run-{repeat:02d}'
            verify_provenance(folder)
            for plan in d['plan']:
                n=plan['n'];cell=read(folder/f'bank-{n}.json');cells[n,repeat]=cell
                if not cell.get('completed') or cell['repeat']!=repeat or cell['method']!=method or cell['n']!=n:
                    raise ValueError('Incorrect method/pass/checkpoint identity')
                if cell['available']:
                    ts=cell['trials']
                    if len(ts)!=plan['questions'] or {t['qid'] for t in ts}!=set(plan['qids']):raise ValueError('Incomplete query grid')
                    if [t['qid'] for t in ts]!=order(plan['qids'],repeat,n):raise ValueError('Method query order differs')
                    for t in ts:
                        q=queries[t['qid']]
                        if (t['question'],t['reference'])!=(q['question'],q['answer']):raise ValueError('Question/reference changed')
                        if abs(t['token_f1']-token_f1(t['final']['text'],q['answer']))>1e-12:raise ValueError('F1 mismatch')
                        if t['exact']!=(normalize(t['final']['text'])==normalize(q['answer'])):raise ValueError('Exact mismatch')
                        if t['contains_reference']!=reference_tokens_present(t['final']['text'],q['answer']):raise ValueError('Containment mismatch')
                        if method in ['our_delta','rag'] and not q['abstention']:
                            if t['source_hit1']!=(t['selected_sid'] in q['support_sids']):raise ValueError('Source score mismatch')
                        if method in ['plain_alphaedit','kv_full'] and t['source_hit1'] is not None:raise ValueError('Undefined source accuracy assigned')
                        visible=t['ttft_seconds'];generated=t['first_generated_token_seconds']
                        if generated is None or not 0<=generated<=t['query_seconds']:raise ValueError('Invalid first token timestamp')
                        if visible is not None and not generated<=visible<=t['query_seconds']:raise ValueError('Invalid visible TTFT')
                        answers.append(dict(method=method,repeat=repeat,n=n,qid=t['qid'],question=t['question'],reference=q['answer'],
                                            answer=t['final']['text'],question_type=q['question_type'],original64=q['original64'],
                                            first_checkpoint=t['first_checkpoint'],exact=t['exact'],token_f1=t['token_f1'],source_hit1=t['source_hit1'],
                                            ttft_seconds=visible,first_generated_seconds=generated,query_seconds=t['query_seconds'],
                                            selected_sid=t['selected_sid'],finish_reason=t['final']['finish_reason']))
                        if answer_only:
                            answers[-1].pop('source_hit1');answers[-1].pop('selected_sid')
                    if checkpoint_metrics(ts)!=cell['groups']['all']:raise ValueError('Metric summary mismatch')
                    count+=len(ts)
                    for group,s in cell['groups'].items():
                        passrows.append(dict(method=method,repeat=repeat,n=n,group=group,
                                             **{k:v for k,v in s.items() if k not in excluded}))
                elif method!='kv_full' or cell['reason'] not in ['context_limit','gpu_limit'] or cell['trials']:
                    raise ValueError('Unexpected unavailable condition')
                if method=='kv_full':capacity.append(dict(repeat=repeat,n=n,available=cell['available'],reason=cell['reason'],**cell['resources']['capacity']))
        for n in d['sizes']:
            cs=[cells[n,r] for r in range(10)];availability={c['available'] for c in cs}
            if len(availability)!=1:raise ValueError('Mixed capacity across passes requires explicit investigation')
            if not cs[0]['available']:
                rows.append(dict(method=method,n=n,group='all',questions=0,passes=10,status=cs[0]['reason']))
                continue
            groups=set(cs[0]['groups'])
            if any(set(c['groups'])!=groups for c in cs):raise ValueError('Unmatched cohort summaries across passes')
            for group in sorted(groups):
                outrow=dict(method=method,n=n,group=group,questions=cs[0]['groups'][group]['questions'],passes=10,status='measured')
                for metric in metrics:
                    s=summarize([c['groups'][group][metric] for c in cs])
                    outrow.update({metric+'_mean':s['mean'],metric+'_sd':s['sd'],metric+'_valid_passes':s['valid_passes']})
                for metric in ['extra_disk_bank_bytes','host_peak_rss_bytes','peak_gpu_allocated_bytes','peak_gpu_reserved_bytes',
                               'retained_cpu_payload_bytes','retained_gpu_prefix_bytes']:
                    s=summarize([c['resources'][metric] for c in cs])
                    outrow.update({metric+'_mean':s['mean'],metric+'_sd':s['sd']})
                outrow['visible_ttft_samples']=sum(c['groups'][group]['ttft_samples'] for c in cs)
                outrow['missing_visible_ttft']=sum(c['groups'][group]['missing_visible_ttft'] for c in cs)
                rows.append(outrow)
    if count!=sum(r['questions']*10 for r in rows if r['group']=='all' and r['status']=='measured'):
        raise ValueError('Verified request denominator mismatch')
    for name,data in [('comparison.csv',rows),('per_pass.csv',passrows),('question_outputs.csv',answers),('kv_capacity.csv',capacity)]:
        fields=list(dict.fromkeys(k for r in data for k in r))
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(data)
    save(out/'comparison.json',dict(summary=rows,verified_requests=count,passes=10,
                                   **({'accuracy_scope':scope} if scope else {})))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    allrows=[r for r in rows if r['group']=='all']
    def lines(ax,metric,scale=1):
        for method in METHODS:
            rs=[r for r in allrows if r['method']==method and r['status']=='measured' and r.get(metric+'_mean') is not None]
            if rs:ax.errorbar([r['n'] for r in rs],[scale*r[metric+'_mean'] for r in rs],
                             yerr=[scale*r[metric+'_sd'] for r in rs],marker='o',capsize=3,label=LABELS[method],color=COLORS[method])
        ax.set_xscale('log',base=2);ax.set_xticks(d['sizes'],labels=d['sizes']);ax.set_xlabel('Stored conversations');ax.grid(alpha=.2)
    def export(fig,name):
        for ext in ['png','svg','pdf']:fig.savefig(out/(name+'.'+ext),dpi=170,bbox_inches='tight')
        plt.close(fig)
    quality=[('mean_f1','Complete final-answer word-token F1'),('exact_rate','Complete final-answer exact match')]
    if not answer_only:quality.append(('source_hit1','Source selection @1 (answerable)'))
    quality_name='accuracy_final_answer' if answer_only else 'accuracy_f1_routing'
    fig,axes=plt.subplots(1,len(quality),figsize=(5*len(quality),4.6))
    for ax,(key,title) in zip(axes,quality):
        lines(ax,key,100);ax.set_title(title);ax.set_ylim(0,105);ax.set_ylabel('Percent')
        ax.axhline(100,color='#bbbbbb',ls='--',lw=1)
    axes[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Full context + question + answer: ten fresh passes per checkpoint')
    fig.text(.5,.01,'Error bars: sample SD across ten inference passes on reused weights. Dashed line: aspirational 100% target.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94));export(fig,quality_name)
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for ax,key,title in zip(axes,['ttft_mean_ms','ttft_median_ms','query_latency_mean_ms'],
                           ['Mean visible TTFT','Mean of per-pass median TTFT','Mean complete-query latency']):
        lines(ax,key);ax.set_title(title);ax.set_ylabel('Milliseconds');ax.set_yscale('log')
    axes[0].legend(fontsize=8)
    fig.suptitle('Measured end-to-end inference time; mean ± sample SD across ten passes')
    fig.tight_layout(rect=(0,0,1,.93));export(fig,'ttft_latency')
    fig,axes=plt.subplots(1,3,figsize=(15,4.6))
    for ax,key,title,scale in zip(axes,['extra_disk_bank_bytes','host_peak_rss_bytes','peak_gpu_allocated_bytes'],
                                 ['Extra bank storage (MiB)','Peak process host RAM (GiB)','Peak CUDA allocated (GiB)'],[1/2**20,1/2**30,1/2**30]):
        lines(ax,key,scale);ax.set_title(title)
    config=read(Path(d['model'])/'config.json')
    tensor_payload=[sum(len(r['bindings']) for r in sessions[:n])*len(d['layers'])*(config['hidden_size']+config['intermediate_size'])*4/2**20 for n in d['sizes']]
    axes[0].plot(d['sizes'],tensor_payload,'--',color='#0077b6',alpha=.5,label='Calculated factor tensor payload')
    axes[0].legend(fontsize=7)
    fig.suptitle('Separate storage tiers; GPU peaks include common base-model weights')
    fig.text(.5,.01,'KV checkpoints beyond capacity are unavailable, not zero memory. Disk payload estimate excludes serialization/metadata.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94));export(fig,'memory')
    def val(row,key,scale=1):
        x=row.get(key+'_mean');s=row.get(key+'_sd')
        return '—' if x is None else f'{x*scale:.2f} ± {s*scale:.2f}'
    text=['# Corrected full-Q/A checkpoint experiment','',
          'All writes used the FULL conversation, exact official question and COMPLETE reference answer. '
          'Activation routing selects one independent answer delta, followed by direct answer generation. '
          'No evidence capsules or extra reader are used. RAG and KV retain the same supplied Q/A information.', '',
          'Each checkpoint was actually remeasured ten times. Values below are means ± sample SD across ten per-pass metrics. '
          'This measures inference repeatability on one frozen reused bank, not ten independently trained models.', '',
          '| Sessions | Eligible questions per pass | Method | F1 (%) | Exact (%) | Mean TTFT (ms) | Bank (MiB) | Peak GPU (GiB) |',
          '|---:|---:|---|---:|---:|---:|---:|---:|']
    if d.get('remeasurement'):
        text[2:2]=['This is a separate repeat experiment: ten ADDITIONAL, freshly measured inference passes over the saved bank. '
                   'The mean and sample SD use only these new ten passes. Previous measurements are neither copied nor pooled. '
                   'The source run is recorded in design.json.','']
    if answer_only:
        text[2:2]=['Accuracy is scored ONLY against the complete final generated answer: normalized word-token F1 and '
                   'whole-answer normalized exact match. No answer extraction, substring-containment credit or routing accuracy '
                   'is used in these accuracy summaries, graphs or exported metric tables. TTFT and memory remain measured.','']
    for n in d['sizes']:
        for method in METHODS:
            r=next(r for r in allrows if r['n']==n and r['method']==method)
            if r['status']=='measured':
                text.append(f"| {n} | {r['questions']} | {LABELS[method]} | {val(r,'mean_f1',100)} | {val(r,'exact_rate',100)} | {val(r,'ttft_mean_ms')} | {val(r,'extra_disk_bank_bytes',1/2**20)} | {val(r,'peak_gpu_allocated_bytes',1/2**30)} |")
            else:text.append(f"| {n} | — | {LABELS[method]} | Unavailable: {r['status']} | — | — | — | — |")
    text+=['','## Protocol and interpretation','',
           f"Reused 64 compatible answer deltas and shared patches at N=8,16,32,64; wrote 436 additional conversation memories. "
           f"The final 500-session prefix has {d['eligible_final_questions']} completely supported official questions. "
           'No derived quote-completion annotations are mixed into the study.', '',
           'The same exact question and answer are supplied at write time and queried at recall. For multi-session items the full official answer '
           'is explicitly taught to each associated session delta; this does not test combining unknown facts or held-out reasoning. '
           'The source bank is selected using official support metadata. Report this as a supervised memory mechanism experiment, '
           'not an official blind LongMemEval score or an unseen-question generalization result.', '',
           'All eligible questions are evaluated at every N. The eligible set changes with N; compare the original64 group from N=64 onward '
           'for fixed-cohort retention. The new_at_checkpoint group separately measures only questions becoming eligible at that checkpoint. '
           'Ten greedy passes may have zero quality SD. Repeated answers are not independent training replicates.', '',
           'Full per-pass means/medians/p95, generated versus visible TTFT, complete query latency, '
           +('' if answer_only else 'source ranks/MRR, containment, ')+
           'cap rate, question categories, original64/new cohorts and memory tiers are in the CSV/JSON files. '
           'Missing visible-token timestamps are counted, not imputed. KV capacity checks are actual per-pass checks; calculated required KV payload '
           'is not a measured allocation. Common model weights are included in GPU peaks. Model startup and writing are excluded from query timing.', '',
           'Reference-answer lexical overlap is not a semantic factuality judge. Preference rubrics and explanatory abstention references '
           'are intentionally used as supervised targets in this mechanism check. No measured write-time comparison is added.']
    (out/'report.md').write_text('\n'.join(text)+'\n')
    svgs=''.join('<svg'+(out/(name+'.svg')).read_text().split('<svg',1)[1] for name in [quality_name,'ttft_latency','memory'])
    page='<!doctype html><meta charset="utf-8"><title>Corrected Q/A memory checkpoint study</title><style>body{font:16px system-ui;max-width:1400px;margin:40px auto;padding:0 20px}svg{width:100%;height:auto}pre{white-space:pre-wrap}</style>'
    page+='<h1>Corrected full-context, question and answer experiment</h1><p>Completed ten fresh measurement passes at every checkpoint through 500 stored conversations. Reused compatible answer memories; no evidence-only deltas.</p>'+svgs+'<pre>'+html.escape('\n'.join(text))+'</pre>'
    if d.get('remeasurement'):
        page=page.replace('<h1>Corrected full-context, question and answer experiment</h1>',
             '<h1>Ten additional measurement passes over the frozen Q/A memory bank</h1>'
             '<p><b>Means and sample SD below use only the new ten passes; earlier results are not pooled.</b></p>')
    if answer_only:
        page=page.replace('<p>Completed ten fresh measurement passes',
             '<p><b>Answer accuracy uses only the complete final generated answer: F1 and exact match.</b></p>'
             '<p>Completed ten fresh measurement passes')
    (out/'report.html').write_text(page)
    artifacts=['report.html','report.md','comparison.csv','comparison.json','per_pass.csv','question_outputs.csv','kv_capacity.csv',
               *[name+'.'+ext for name in [quality_name,'ttft_latency','memory'] for ext in ['png','svg','pdf']]]
    artifacts += [f for f in ['reporting_scope.json','reporting_scope_manifest.json','report_source.py'] if (out/f).exists()]
    save(out/'report_manifest.json',{f:digest(out/f) for f in artifacts})
    index=ROOT/'README.md';original=index.read_text()
    start=original.find('## Running: corrected full-Q/A experiment, ten passes through 500 sessions')
    end=original.find('## Completed: all 500 official questions with conversation-level memories',start)
    if start>=0 and end>start:
        section='## Completed: corrected full-Q/A experiment, ten passes through 500 sessions\n\n'
        section+='All writes used the full conversation, exact official question and complete reference answer. '
        section+='The run reused 64 compatible answer deltas and four shared-model checkpoints, added 436 conversations and remeasured every checkpoint '
        section+='8,16,32,64,128,256,500 in ten actual inference passes on physical GPU 1. '
        section+=f"The final prefix has {d['eligible_final_questions']} completely supported official questions. "
        section+='Values below are mean ± sample SD across ten inference passes on one frozen bank, not ten separately trained models.\n\n'
        section+='| Method at N=500 | F1 (%) | Mean TTFT (ms) | Extra bank (MiB) |\n|---|---:|---:|---:|\n'
        for method in METHODS:
            r=next(r for r in allrows if r['n']==500 and r['method']==method)
            if r['status']=='measured':
                section+=f"| {LABELS[method]} | {val(r,'mean_f1',100)} | {val(r,'ttft_mean_ms')} | {val(r,'extra_disk_bank_bytes',1/2**20)} |\n"
            else:section+=f"| {LABELS[method]} | Unavailable: {r['status']} | — | — |\n"
        section+='\n[Full report](runs/qa500-checkpoints-tenpass-v1/report.html) · [Quality/routing](runs/qa500-checkpoints-tenpass-v1/accuracy_f1_routing.png) · [TTFT/latency](runs/qa500-checkpoints-tenpass-v1/ttft_latency.png) · [Memory](runs/qa500-checkpoints-tenpass-v1/memory.png) · [Protocol](QA500_PROTOCOL.md).\n\n'
        index.write_text(original[:start]+section+original[end:])
    save(out/'completion.json',dict(completed=True,physical_gpu=1,passes=10,checkpoints=d['sizes'],
                                   stored_conversations=500,final_eligible_questions=d['eligible_final_questions'],
                                   verified_requests=count,finished_utc=now(),
                                   **({'accuracy_scope':scope} if scope else {}),
                                   **({'remeasurement':d['remeasurement']} if d.get('remeasurement') else {})))
    print('\n'.join(text[:12]),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.out.resolve())
