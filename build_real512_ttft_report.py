"""Publish measured TTFT/F1 only after completed GPU1 replay and integrity checks."""
import argparse
import base64
import csv
import html
import json
from pathlib import Path

from experiment_utils import read,save,sha256
from fixed_query_scaling_core import SIZES,check_complete
from build_real512_report import verify_outcomes
from ttft_timing import ttft_summary


def build(out):
    completed=read(out/'completion.json')
    required=['completed','source_unchanged','base_weights_restored','code_unchanged',
              'protocol_unchanged','all_outputs_and_routes_replayed']
    if not all(completed.get(k) for k in required) or completed.get('physical_gpu')!=1:
        raise ValueError('No graphs: require completed verified GPU1 TTFT replay')
    design=read(out/'design.json');source=Path(design['source']);items=read(source/'items.json')
    for name,digest in design['source_sha256'].items():
        if sha256(source/name)!=digest:raise ValueError('Source artifacts changed')
    for sid,digest in design['factor_sha256'].items():
        if sha256(source/'factors'/(sid+'.pt'))!=digest:raise ValueError('Source factors changed')
    root=Path(__file__).resolve().parent
    for name,digest in completed['code_sha256'].items():
        if sha256(root/name)!=digest:raise ValueError('Replay source code changed')
    results=read(out/'evaluation.json')['results']
    verify_outcomes(results,design['panel_ids'],design['repeats'],{r['qid']:r for r in items})
    previous=read(source/'evaluation.json')['results'];rows=[]
    for n in SIZES:
        for method,cell in results[str(n)].items():
            if not cell['available']:
                rows.append(dict(n=n,method=method,status=cell['reason'],routing_correct=None,exact=None,
                    mean_token_f1=None,ttft_median_ms=None,ttft_p95_ms=None,total_median_ms=None,
                    kv_cold_first_request_ms=None,kv_prefix_setup_ms=None,accuracy_queries=0,warm_timing_samples=0))
                continue
            summary=ttft_summary(cell['trials'])
            if summary!=cell['ttft']:raise ValueError('TTFT summary mismatch')
            for t in cell['trials']:
                old=next(p for p in previous[str(n)][method]['trials'] if p['qid']==t['qid'] and p['repeat']==t['repeat'])
                if t['final']['token_ids']!=old['final']['token_ids'] or t['retrieved_sids']!=old['retrieved_sids'] or t['selected_sid']!=old['selected_sid']:
                    raise ValueError('Output/source replay changed')
            s=cell['summary'];cold=cell.get('cold_first_request')
            rows.append(dict(n=n,method=method,status='measured',
                routing_correct=None if s['retrieval_at1'] is None else s['retrieval_at1']['count'],
                exact=s['returned_exact']['count'],mean_token_f1=s['mean_token_f1'],
                ttft_median_ms=summary['median_seconds']*1000,ttft_p95_ms=summary['p95_seconds']*1000,
                total_median_ms=cell['latency']['warm_median_seconds']*1000,
                kv_cold_first_request_ms=None if cold is None else cold['ttft_seconds']*1000,
                kv_prefix_setup_ms=None if cold is None else cold['prefix_setup_seconds']*1000,
                accuracy_queries=8,warm_timing_samples=summary['warm_n']))
    save(out,'comparison_table.json',dict(rows=rows,ttft=design['ttft'],f1=design['f1']))
    with (out/'comparison_table.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'real512-ttft'})
    colors={'our_delta':'#007A74','rag_qa':'#555555','kv_full_qa':'#939393'}
    names={'our_delta':'Compact deltas','rag_qa':'Matched RAG','kv_full_qa':'Full-bank KV'}
    figures=[]
    def axis(ax,label):
        ax.set_xscale('log',base=2);ax.set_xticks(SIZES,labels=SIZES);ax.set_xlabel('Stored real sessions, N')
        ax.set_ylabel(label);ax.grid(alpha=.18)
    def plot(ax,method,field,**kwargs):
        valid=[r for r in rows if r['method']==method and r[field] is not None]
        ax.plot([r['n'] for r in valid],[r[field] for r in valid],color=colors[method],**kwargs)
    def publish(fig,name,caption):
        for ext in ['png','svg','pdf']:
            metadata={'Date':None} if ext=='svg' else {'CreationDate':None,'ModDate':None} if ext=='pdf' else None
            fig.savefig(out/(name+'.'+ext),dpi=180,bbox_inches='tight',metadata=metadata)
        plt.close(fig);figures.append(dict(name=name,caption=caption))
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    for method in names:
        marker={'our_delta':'o','rag_qa':'s','kv_full_qa':'^'}[method]
        plot(axes[0],method,'ttft_median_ms',marker=marker,markersize=8 if method=='our_delta' else 5,
             markerfacecolor='none' if method=='our_delta' else colors[method],label=names[method])
        plot(axes[0],method,'ttft_p95_ms',linestyle=':',alpha=.75)
        plot(axes[1],method,'total_median_ms',marker=marker,label=names[method])
    axis(axes[0],'Milliseconds');axes[0].set_title('Query to first visible answer: median; dotted=p95');axes[0].set_ylim(bottom=0);axes[0].legend(fontsize=8,loc='center left')
    axis(axes[1],'Milliseconds');axes[1].set_title('Complete query work: warm median');axes[1].set_ylim(bottom=0);axes[1].legend(fontsize=8,loc='center right')
    plot(axes[2],'kv_full_qa','kv_cold_first_request_ms',marker='^',label='One cold-cache first request per N')
    plot(axes[2],'kv_full_qa','kv_prefix_setup_ms',linestyle=':',label='Prefix setup portion')
    axis(axes[2],'Milliseconds');axes[2].set_title('KV cold-prefix TTFT; model already resident');axes[2].set_ylim(bottom=0);axes[2].legend(fontsize=8)
    publish(fig,'ttft_scaling','Direct first-token timestamps from eight fixed original questions, with 16 later-pass timings per measured condition. TTFT includes all query routing/retrieval, loading/application, query tokenization and prefill. No network/queue or model startup is measured. Warm KV assumes its memory-prefix cache exists. Cold KV is one contiguous measured prefix-construction-to-first-answer span per N, not a p95 estimate. Total query work additionally includes completion and cleanup/cache integrity checks. KV above 32 has no measured timings. OS caches not flushed; methods run sequentially.')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for method in names:
        marker={'our_delta':'o','rag_qa':'s','kv_full_qa':'^'}[method]
        for ax,field in zip(axes,['mean_token_f1','exact']):
            valid=[r for r in rows if r['method']==method and r[field] is not None]
            ys=[r[field]*100 if field=='mean_token_f1' else r[field]/8*100 for r in valid]
            ax.plot([r['n'] for r in valid],ys,marker+'-',color=colors[method],label=names[method],
                    markersize=9 if method=='our_delta' else 5,markerfacecolor='none' if method=='our_delta' else colors[method])
    for ax in axes:
        axis(ax,'Percent');ax.set_ylim(0,110);ax.axhline(100,color='black',ls='--',lw=1,label='100% aspirational target');ax.legend(fontsize=8,loc='lower left')
    axes[0].set_title('Mean custom normalized word-token F1');axes[1].set_title('Whole returned answer exact match')
    publish(fig,'f1_scaling','Accuracy uses only eight unique first-pass queries per N. The timing observer preserved emitted tokens and selected sources for every corresponding prior trial. F1 retains articles and averages per-question token-overlap scores; it does not establish factual correctness or preservation of every session fact. Exact match penalizes sentence framing; inspect outputs in evaluation.json.')
    headers=['N','Method','Status','Source correct /8','Exact /8','Mean F1 %','Warm TTFT median ms','Warm TTFT p95 ms','Warm samples','Cold KV first-request ms']
    def show(v,digits=2):return 'N/A' if v is None else f'{v:.{digits}f}'
    table='<div class="scroll"><table><tr>'+''.join('<th>'+h+'</th>' for h in headers)+'</tr>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (r['n'],names[r['method']],r['status'],
         r['routing_correct'] if r['routing_correct'] is not None else 'N/A',r['exact'] if r['exact'] is not None else 'N/A',
         show(None if r['mean_token_f1'] is None else r['mean_token_f1']*100),show(r['ttft_median_ms']),show(r['ttft_p95_ms']),
         r['warm_timing_samples'],show(r['kv_cold_first_request_ms'])))+'</tr>' for r in rows)+'</table></div>'
    sections=''.join('<section><h2>'+f['name'].replace('_',' ')+'</h2><img alt="'+f['name']+'" src="data:image/svg+xml;base64,'+
        base64.b64encode((out/(f['name']+'.svg')).read_bytes()).decode('ascii')+'"><p>'+html.escape(f['caption'])+'</p></section>' for f in figures)
    kv_containment='; '.join(f"N = {n}: {results[str(n)]['kv_full_qa']['summary']['contains_reference']['count']}/8"
                            for n in SIZES if results[str(n)]['kv_full_qa']['available'])
    sections='<p><strong>Interpreting KV answer scores:</strong> Reference-token containment in measured conditions was '+html.escape(kv_containment)+'. Word-token F1 and exact match penalize extra sentence framing as well as missing or wrong content. Containing the reference does not validate every statement in an output.</p>'+sections
    report='<!doctype html><html lang="en"><meta charset="utf-8"><title>Measured TTFT and F1: real512</title><style>body{font:16px system-ui;color:#222;max-width:1800px;margin:auto;padding:24px}img{width:100%}section{margin:36px 0}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse;width:100%}.scroll{overflow-x:auto}</style><h1>Measured TTFT and token F1 through 512 stored real sessions</h1><p>Completed replay on physical GPU 1, Llama-3.1-8B-Instruct float32, batch 1, layers 13–15, greedy 64-token cap and EOS/next-turn stopping. The existing compact factor bank was reused read-only; no memories were rewritten.</p><p>'+html.escape(design['scope'])+'</p><p>'+html.escape(design['annotation'])+'</p><p>'+html.escape(design['ttft'])+'</p><p>'+html.escape(design['repeats_note'])+'</p><p>'+html.escape(design['cold_kv'])+'</p><p>'+html.escape(design['f1'])+'</p><p>HF prompt-stream events are excluded. Cached decoding uses the same greedy operations with a token observer; every output and source selection was checked against the prior corresponding trial. End-to-end TTFT stops at the first nonempty decoded answer text, including tiny local decoding/observer overhead; this is not an HTTP/serving benchmark. Model-only TTFT is not substituted for pipeline TTFT. One-token outputs can have TTFT close to complete-answer time.</p>'+table+sections+'<p>Numeric data: comparison_table.csv, comparison_table.json, evaluation.json, measurements.json, completion.json. Infeasible KV conditions have no invented F1 or TTFT. The 512 bank consists of 64 original annotated sessions plus 448 additional real sessions with new extractive annotations; evaluation uses eight previously inspected original queries, not 512 independent questions.</p></html>'
    (out/'report.html').write_text(report)
    artifacts=[p for p in out.iterdir() if p.suffix in ['.html','.svg','.png','.pdf']]+[out/'comparison_table.json',out/'comparison_table.csv']
    save(out,'report_manifest.json',dict(builder_sha256=sha256(Path(__file__)),
        input_sha256={n:sha256(out/n) for n in ['design.json','completion.json','evaluation.json','measurements.json']},
        artifact_sha256={p.name:sha256(p) for p in artifacts}))
    print('REPORT',out/'report.html')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    build(p.parse_args().out.resolve())
