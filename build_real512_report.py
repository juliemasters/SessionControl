"""Generate plots/HTML only from a completed, integrity-checked real512 run."""
import argparse
import base64
import html
import json
from pathlib import Path
import statistics

from experiment_utils import read,save,sha256
from experiment_utils import metrics
from fixed_query_scaling_core import SIZES,check_complete
from three_tier_memory import recall_rss
from scripts.verify_release import exact_flags, token_f1, close, verify_summary


def verify_outcomes(results, panel_ids, repeats, references=None):
    """Reproduce recorded lexical/source measures before any plot is created."""
    check_complete(results, panel_ids, repeats)
    for n, cells in results.items():
        for method, cell in cells.items():
            if not cell['available']:
                continue
            for trial in cell['trials']:
                if references is not None:
                    reference = references[trial['qid']]
                    for field in ('question', 'answer'):
                        if trial[field] != reference[field]:
                            raise ValueError(f'{n}/{method}: changed reference {field}')
                    if trial['reference_source_sid'] != reference['sid']:
                        raise ValueError(f'{n}/{method}: changed reference source')
                for key, expected in exact_flags(trial).items():
                    if trial.get(key) != expected:
                        raise ValueError(f'{n}/{method}/{trial["qid"]}: inconsistent {key}')
                close(trial['token_f1'], token_f1(trial), f'{n}/{method}/token_f1')
            verify_summary(cell['summary'], [t for t in cell['trials'] if t['repeat'] == 0],
                           f'{n}/{method}')


def build(out):
    done=read(out/'completion.json')
    required=('completed','compact_validation_passed','base_weights_restored','historical_deltas_unchanged',
        'written_factors_unchanged','protocol_unchanged','code_unchanged')
    if not all(done.get(k) for k in required) or done.get('real_sessions')!=512:
        raise ValueError('No graphs: full512 experiment and integrity checks must pass')
    design=read(out/'design.json');evaluation=read(out/'evaluation.json');results=evaluation['results']
    if len(design['panel_ids']) != 8 or len(done['factor_sha256']) != 512:
        raise ValueError('Require eight fixed questions and all512 stored factors')
    frozen_items=read(out/'items.json')
    if len(frozen_items) != 512 or set(done['factor_sha256']) != {r['sid'] for r in frozen_items}:
        raise ValueError('Stored factor IDs must match the frozen512-session bank')
    verify_outcomes(results,design['panel_ids'],design['repeats'],
                    {r['qid']:r for r in frozen_items})
    validation=read(out/'compact_validation.json')
    if not validation['passed']:raise ValueError('Compact validation did not pass')
    for name,expected in read(out/'manifest_sha256.json').items():
        if sha256(out/name)!=expected:raise ValueError('Frozen protocol changed')
    for sid,expected in done['factor_sha256'].items():
        if sha256(out/'factors'/(sid+'.pt'))!=expected:raise ValueError('Factor bank changed')
    writes=read(out/'writes.json')
    if len(writes) != 448:
        raise ValueError('Require all448 own-question write diagnostics')
    write_summary=metrics(writes)
    verify_summary(write_summary,writes,'new448 write checks')
    write_failures=[dict(qid=t['qid'],reference=t['answer'],output=t['final']['text'])
                    for t in writes if not t['returned_exact']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'real512-compact'})
    colors={'our_delta':'#007A74','rag_qa':'#555555','kv_full_qa':'#939393'}
    names={'our_delta':'Compact AlphaEdit','rag_qa':'Matched RAG','kv_full_qa':'Full-bank KV'}
    figures=[]
    def publish(fig,name,caption):
        fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(out/(name+'.svg'),bbox_inches='tight',metadata={'Date':None})
        fig.savefig(out/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
        plt.close(fig);figures.append(dict(name=name,caption=caption))
    def axis(ax,ylabel):
        ax.set_xscale('log',base=2);ax.set_xticks(SIZES,labels=list(SIZES))
        ax.set_xlabel('Stored real conversation sessions, N');ax.set_ylabel(ylabel)
        ax.grid(alpha=.18)
    def series(method,field):
        xs=[n for n in SIZES if results[str(n)][method]['available']]
        return xs,[field(results[str(n)][method]) for n in xs]
    def draw(ax,method,field):
        xs,ys=series(method,field)
        marker={'our_delta':'o','rag_qa':'s','kv_full_qa':'^'}[method]
        ax.plot(xs,ys,marker+'-',color=colors[method],label=names[method],lw=2,
                markersize=9 if method=='our_delta' else 5,
                markerfacecolor='none' if method=='our_delta' else colors[method])
    fig,axes=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
    for ax,key,title in zip(axes,['retrieval_at1','returned_exact','contains_reference'],
        ['Source selection','Whole returned answer exact match','Reference answer contained']):
        for method in colors:
            if method=='kv_full_qa' and key=='retrieval_at1':continue
            draw(ax,method,lambda c:c['summary'][key]['rate']*100)
        ax.axhline(100,color='black',ls='--',lw=1,label='100% target (aspirational)')
        ax.set_ylim(-4,110);axis(ax,'Percent of the same eight questions');ax.set_title(title)
        ax.legend(fontsize=8,loc='lower left')
    k1=read(out/'historical_k1.json')
    axes[1].axhline(k1['exact_match_rate']*100,color='#999',ls=':',label='Historical K1 exact (different protocol)')
    axes[1].legend(fontsize=8,loc='lower left')
    publish(fig,'accuracy_scaling','Actual first-pass outcomes on the same eight original questions at every N, including 512. Repeats do not increase accuracy sample size. The KV baseline stops where capacity makes it unavailable; no zero accuracy is assigned. K1 is historical context with different items/writer/token cap, not a paired baseline or forecast. No semantic correctness claim from lexical measures.')
    resources=read(out/'resource_summary.json');items=read(out/'items.json');bank=read(out/'bank.json')
    cpu_path=out/'resource_usage_cpu.jsonl'
    cpu_samples=[json.loads(line) for line in cpu_path.read_text().splitlines()] if cpu_path.exists() else []
    boundary_path=out/'cpu_recall_end.json'
    cpu_boundary=read(boundary_path) if boundary_path.exists() else None
    if cpu_boundary is not None:
        count=cpu_boundary['cpu_samples_through_final_recall']
        if not 0<count<=len(cpu_samples):raise ValueError('Invalid observed recall RSS boundary')
        cpu_samples=cpu_samples[:count]
    cpu_by_condition=recall_rss(cpu_samples,SIZES)
    save(out,'sampled_cpu_summary.json',dict(interval_seconds=.5,conditions=cpu_by_condition,
        final_recall_boundary=cpu_boundary,
        note='Sampled recall/setup phase process RSS, not lifetime ru_maxrss. N is recorded. Includes shared model CPU allocations, all 512 metadata, allocator retention and backup even during baselines. This is a sampled maximum, not an exact or incremental per-method peak.'))
    expected={};dense_actual={}
    source=Path(design['source'])
    for n in SIZES:
        cell=results[str(n)];folder=out/('bank-'+str(n))
        dense_actual[str(n)]=sum((source/'deltas'/r['sid']/'v1.pt').stat().st_size for r in items[:n]) if n<=64 else None
        expected[str(n)]=dict(compact_raw_factor_bytes=n*3*(4096+14336)*4,
            rag_corpus_plus_raw_vector_bytes=(folder/'bank.json').stat().st_size+n*384*4,
            dense_counterfactual_bytes=n*resources['dense_counterfactual_bytes_per_session'],
            kv_required_prefix_bytes=cell['kv_full_qa']['plan']['calculated_prefix_payload_bytes'],
            kv_context_feasible=cell['kv_full_qa']['plan']['required_tokens']<=cell['kv_full_qa']['plan']['model_context_limit'])
    save(out,'expected_memory.json',dict(values=expected,
        note='All actual banks512 were evaluated by compact AlphaEdit and RAG. Dense beyond64 was not stored; it is a counterfactual. KV required payload is calculated from actual encoded banks and is not an allocation where infeasible. Compact raw factors exclude serialization headers; RAG expected vectors exclude tensor serialization headers.'))
    save(out,'memory_accounting.json',dict(common_llm_parameter_bytes=resources['base_model_parameter_bytes'],
        compact_cpu_base_backup_bytes=resources['cpu_base_backup_bytes'],
        rag_additional_gpu_parameter_bytes=resources['embedding_parameter_bytes'],
        rag_cpu_vector_bytes=resources['measurements']['rag_index_setup']['vector_payload_bytes'],
        rag_live_gpu_vector_bytes=0,
        per_bank={str(n):{m:dict(retained_payload_bytes=c.get('resources',{}).get('persistent_payload_bytes'),
            tier=c.get('resources',{}).get('persistent_tier'),
            max_decode_cache_bytes=max((t['final'].get('max_query_cache_payload_bytes',0) for t in c['trials']),default=None),
            cache_kind='configuration estimate' if m=='our_delta' else 'actual tensor payload',
            available=c['available']) for m,c in results[str(n)].items()} for n in SIZES},
        note='Retained representation chart measures the active-prefix file/tensor payload, not total experiment-directory storage. It excludes common LLM and reports auxiliary model parameters/base backup separately here. RAG serialized text/vectors are on disk; the full 512 vector array remains live on CPU at smaller N and is included in host RSS. The encoder and Llama run on GPU. CUDA peaks include model parameters. CPU sampled RSS includes harness metadata and backup even during baseline stages. This is not isolated deployment-memory accounting. Transient decode cache is distinct from persistent bank storage. Projection is write-only and released before recall measurements.'))
    fig,axes=plt.subplots(1,3,figsize=(18,5.6),layout='constrained')
    ax=axes[0]
    for method in colors:
        draw(ax,method,lambda c:c['resources']['persistent_payload_bytes']/2**30)
    ax.plot(SIZES,[expected[str(n)]['compact_raw_factor_bytes']/2**30 for n in SIZES],':',color=colors['our_delta'],label='Compact raw factor expectation')
    ax.plot(SIZES,[expected[str(n)]['rag_corpus_plus_raw_vector_bytes']/2**30 for n in SIZES],':',color=colors['rag_qa'],label='RAG corpus + raw vector expectation')
    ax.plot(SIZES,[expected[str(n)]['kv_required_prefix_bytes']/2**30 for n in SIZES],'--',color=colors['kv_full_qa'],label='KV required (calculated; may not fit)')
    ax.plot(SIZES[:4],[dense_actual[str(n)]/2**30 for n in SIZES[:4]],'s-',color='black',label='Legacy dense delta files (existing, actual)')
    ax.plot(SIZES[3:],[expected[str(n)]['dense_counterfactual_bytes']/2**30 for n in SIZES[3:]],'--',color='black',label='Dense>64 counterfactual (not stored)')
    ax.set_yscale('log');axis(ax,'Retained representation payload (GiB, log scale)');ax.set_title('Storage tiers differ: factors=disk, RAG=CPU/disk, KV=GPU')
    fig.legend(*ax.get_legend_handles_labels(),fontsize=8,loc='outside lower center',ncol=4)
    ax=axes[1]
    for method in colors:
        xs=[n for n in SIZES if results[str(n)][method]['available'] and f'{n}/{method}' in cpu_by_condition]
        ax.plot(xs,[cpu_by_condition[f'{n}/{method}']['sampled_peak_rss_bytes']/2**30 for n in xs],
            'o-',color=colors[method],label=names[method])
    host_peaks=[s['sampled_peak_rss_bytes']/2**30 for s in cpu_by_condition.values()]
    ax.set_ylim(0,max(host_peaks,default=1)*1.15)
    axis(ax,'Sampled peak host RSS (GiB)');ax.set_title('CPU process recall RSS (0.5s samples)');ax.legend(fontsize=8,loc='lower left')
    ax=axes[2]
    for method in colors:
        draw(ax,method,lambda c:max(c['resources']['peak_allocated_bytes'],c['resources'].get('setup_peak_allocated_bytes',0))/2**30)
        xs,ys=series(method,lambda c:max(c['resources']['peak_reserved_bytes'],c['resources'].get('setup_peak_reserved_bytes',0))/2**30)
        ax.plot(xs,ys,':',color=colors[method],alpha=.8)
    ax.axhline(resources['base_model_parameter_bytes']/2**30,color='black',ls='--',label='Common base model parameter payload')
    axis(ax,'Peak CUDA GPU memory (GiB)');ax.set_title('Actual inference/setup GPU peaks; dotted=reserved');ax.legend(fontsize=8)
    publish(fig,'memory_scaling','Three separate tiers, never a summed total: retained bank and physical tier; recall-phase sampled host RSS; CUDA allocation/reservation. RSS is a 0.5 s sampled maximum and includes shared allocations/full manifest/allocator retention/base backup; it is not lifetime HWM or an exact incremental peak. Solid measured storage points and actual GPU peaks are distinct from dashed/dotted calculated expectations. Dense 512 counterfactual is not the compact implementation. Retained payload excludes common model weights; GPU peaks include them. RAG text/index grows with N. Infeasible KV required memory is not measured allocated memory.')
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    for method in colors:
        draw(axes[0],method,lambda c:c['latency']['warm_median_seconds'])
        xs,ys=series(method,lambda c:c['latency']['warm_p95_seconds'])
        axes[0].plot(xs,ys,':',color=colors[method],alpha=.7)
    axis(axes[0],'Seconds/query');axes[0].set_title('End-to-end: median; dotted=p95');axes[0].legend(fontsize=8)
    for method in ('our_delta','rag_qa'):
        draw(axes[1],method,lambda c:statistics.median(t['route_seconds'] if method=='our_delta' else t['embedding_seconds']+t['search_seconds'] for t in c['trials'] if t['repeat']>0))
    axis(axes[1],'Seconds/query');axes[1].set_title('Routing or embedding + search');axes[1].legend(fontsize=8)
    draw(axes[2],'kv_full_qa',lambda c:c['resources']['setup_seconds'])
    axis(axes[2],'One-time setup seconds');axes[2].set_title('Full-bank KV prefix prefill');axes[2].legend(fontsize=8)
    publish(fig,'latency_scaling','Actual serial batch-one timing at every feasible N. Compact routing includes a fresh base-model key forward plus reading/scoring every factor; no precomputed query score matrix. Warm medians/p95 use16repeat timings after the first pass; OS file cache was not flushed. RAG includes query embedding, search, tokenization, prefix prefill and decode. KV query latency excludes separately plotted reusable prefix setup. Sequential method order is a limitation, not a serving throughput benchmark.')
    write_summary.pop('retrieval_at1',None);write_summary.pop('retrieval_at3',None)
    save(out,'write_quality.json',dict(new448_own_question_diagnostics=write_summary,
        exact_mismatches=write_failures,
        legacy64_identical_token_pairs=sum(r['tokens_identical'] for r in validation['own_question_pairs']),
        note='New extractive own-question diagnostics are not the eight original routing test. Grounded labels are literal user quotes, not semantic annotations for all session facts.'))
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    errors=[max(r['relative_frobenius_errors'].values()) for r in validation['own_question_pairs']]
    axes[0].hist(errors,bins=12,color=colors['our_delta']);axes[0].axvline(.002,color='black',ls='--',label='Frozen reconstruction gate')
    axes[0].set_xlabel('Maximum relative Frobenius error per legacy delta');axes[0].set_ylabel('Number of legacy deltas');axes[0].set_title('Approximate 64-delta compression');axes[0].legend(fontsize=8)
    axes[1].bar(['Whole exact','Contains quote'],[100*write_summary[k]['rate'] for k in ('returned_exact','contains_reference')],color=[colors['our_delta'],'#999'])
    axes[1].axhline(100,color='black',ls='--');axes[1].set_ylim(0,110);axes[1].set_ylabel('Percent of 448 new own-question checks');axes[1].set_title('Write retention diagnostics (different question type)')
    publish(fig,'compact_validation','The 64 legacy emitted token sequences and 32 panel top1 routes must agree before expansion. Frobenius error measures approximation, not global losslessness. The 448 extractive write checks test their own keys separately from the original eight-question scaling panel.')
    table=[]
    for n in SIZES:
        for method in colors:
            c=results[str(n)][method]
            table.append(dict(n=n,method=names[method],available=c['available'],reason=c.get('reason'),
                source_top1=None if not c['available'] or c['summary']['retrieval_at1'] is None else c['summary']['retrieval_at1']['count'],
                exact=None if not c['available'] else c['summary']['returned_exact']['count'],
                contains=None if not c['available'] else c['summary']['contains_reference']['count'],
                retained_gib=None if not c['available'] else c['resources']['persistent_payload_bytes']/2**30,
                tier=None if not c['available'] else c['resources']['persistent_tier'],
                sampled_peak_host_rss_bytes=cpu_by_condition.get(f'{n}/{method}',{}).get('sampled_peak_rss_bytes') if c['available'] else None,
                host_rss_samples=cpu_by_condition.get(f'{n}/{method}',{}).get('samples') if c['available'] else None,
                peak_gpu_allocated_bytes=None if not c['available'] else max(c['resources']['peak_allocated_bytes'],c['resources'].get('setup_peak_allocated_bytes',0)),
                peak_gpu_reserved_bytes=None if not c['available'] else max(c['resources']['peak_reserved_bytes'],c['resources'].get('setup_peak_reserved_bytes',0)),
                median_seconds=None if not c['available'] else c['latency']['warm_median_seconds'],
                p95_seconds=None if not c['available'] else c['latency']['warm_p95_seconds']))
    save(out,'comparison_table.json',dict(rows=table,accuracy_denominator=8))
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in (
        r['n'],r['method'],r['reason'] or 'measured',r['source_top1'] if r['source_top1'] is not None else 'N/A',
        f"{r['exact']}/8" if r['exact'] is not None else 'N/A',f"{r['retained_gib']:.6f}" if r['retained_gib'] is not None else 'N/A',
        r['tier'] or 'N/A',
        f"{r['sampled_peak_host_rss_bytes']/2**30:.3f}" if r['sampled_peak_host_rss_bytes'] is not None else 'Unmeasured',
        f"{r['peak_gpu_allocated_bytes']/2**30:.3f}" if r['peak_gpu_allocated_bytes'] is not None else 'N/A',
        f"{r['peak_gpu_reserved_bytes']/2**30:.3f}" if r['peak_gpu_reserved_bytes'] is not None else 'N/A',
        f"{r['median_seconds']:.3f}" if r['median_seconds'] is not None else 'N/A'))+'</tr>' for r in table)
    sections=''.join('<section><h2>'+html.escape(f['name'].replace('_',' '))+'</h2><img alt="'+html.escape(f['name'].replace('_',' '))+'" src="data:image/svg+xml;base64,'+base64.b64encode((out/(f['name']+'.svg')).read_bytes()).decode('ascii')+'"><p>'+html.escape(f['caption'])+'</p></section>' for f in figures)
    panel_failures=[dict(method=names[method],question=t['question'],reference=t['answer'],
                        output=t['final']['text'],source_correct=t['retrieval_at1'])
                    for method,c in results['512'].items() if c['available']
                    for t in c['trials'] if t['repeat']==0 and not t['returned_exact']]
    if panel_failures:
        sections+='<section><h2>Original-question exact mismatches at N = 512</h2><p>Exact mismatch alone does not establish factual error. Inspect missing details, wording and unsupported claims separately. Source correctness is the router/retriever result.</p><table><tr><th>Method</th><th>Question</th><th>Reference</th><th>Returned answer</th><th>Correct source</th></tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(t[k]))+'</td>' for k in ('method','question','reference','output','source_correct'))+'</tr>'
            for t in panel_failures)+'</table></section>'
    kv_examples={}
    for n in SIZES:
        cell=results[str(n)]['kv_full_qa']
        if not cell['available']:continue
        for t in cell['trials']:
            if t['repeat']==0 and not t['returned_exact']:
                key=(t['question'],t['answer'],t['final']['text'])
                kv_examples.setdefault(key,[]).append(n)
    if kv_examples:
        sections+='<section><h2>KV exact-match failures in measured conditions</h2><p>Whole-answer exact match penalizes sentence framing as well as wrong values. Reference containment is shown separately in the accuracy graph; it does not certify every output statement.</p><table><tr><th>Measured N</th><th>Question</th><th>Reference</th><th>Returned answer</th></tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (', '.join(map(str,ns)),*key))+'</tr>'
            for key,ns in kv_examples.items())+'</table></section>'
    if write_failures:
        sections+='<section><h2>New-memory write mismatches</h2><p>These are own-question quote-retention diagnostics, separate from the eight original routing queries. No failed write was removed from the bank.</p><table><tr><th>Question ID</th><th>Reference quote</th><th>Generated output</th></tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(t[k])+'</td>' for k in ('qid','reference','output'))+'</tr>'
            for t in write_failures)+'</table></section>'
    report='<!doctype html><html><meta charset="utf-8"><title>Real512 compact scaling</title><style>body{font-family:system-ui;max-width:1800px;margin:auto;padding:24px;color:#222}img{width:100%}section{margin:32px 0}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}table{border-collapse:collapse;width:100%}.scroll{overflow-x:auto}</style><h1>Measured real-session scaling through 512</h1><p>'+html.escape(design['scope'])+'</p><p>'+html.escape(design['annotation'])+'</p><p>'+html.escape(design['compact'])+'</p><p>All computation: physical GPU 1, Llama-3.1-8B float32. Accuracy uses eight fixed original questions at every N; repeated trials measure timing and do not create additional test items. Primary methods receive full sessions plus supplied Q/A at memory creation. 100% is an aspiration, not a forecast.</p><p>Memory is reported in three noninterchangeable tiers: retained bank with its physical location, sampled peak host RSS during recall, and peak GPU allocated/reserved memory. These columns are never summed. Host RSS excludes lifetime loading/write high-water marks and includes shared harness allocations; 0.5 s samples can miss brief peaks. Missing measurements are explicit. GPU peaks include the common model; bank payload excludes it.</p><div class="scroll"><table><tr><th>N</th><th>Method</th><th>Status</th><th>Source correct/8</th><th>Exact</th><th>Retained bank GiB</th><th>Bank tier</th><th>Sampled peak host RSS GiB</th><th>Peak GPU allocated GiB</th><th>Peak GPU reserved GiB</th><th>Warm median s</th></tr>'+rows+'</table></div>'+sections+'<p>Full measured details: comparison_table.json, evaluation.json, resource_summary.json, expected_memory.json, compact_validation.json, sampled_cpu_summary.json, memory_accounting.json. New annotation/write quality is reported separately. Routing receives query text and stored memory, without oracle source labels. RAG and KV read the stored Q/A as authorized memory.</p></html>'
    (out/'report.html').write_text(report,encoding='utf-8')
    input_names=['completion.json','evaluation.json','resource_summary.json','compact_validation.json','writes.json']
    if cpu_path.exists():input_names.append('resource_usage_cpu.jsonl')
    if boundary_path.exists():input_names.append('cpu_recall_end.json')
    save(out,'report_manifest.json',dict(builder_sha256=sha256(Path(__file__)),input_sha256={n:sha256(out/n) for n in input_names},
        artifact_sha256={p.name:sha256(p) for p in out.iterdir() if p.suffix in ('.svg','.png','.pdf','.html')}))
    print('REPORT',out/'report.html')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    build(parser.parse_args().out.resolve())
