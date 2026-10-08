"""Standalone scaling figures only after a complete, verified GPU1 run."""
import argparse
import base64
import html
from pathlib import Path
import statistics
from experiment_larger_routing import read,sha256,save
from experiment_memory_comparison import metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('results/memory-scaling-v1'))
    out=p.parse_args().out.resolve();done=read(out/'completion.json')
    for key in ['completed','source_artifacts_unchanged','delta_hashes_unchanged','base_weights_restored','protocol_unchanged']:
        if done.get(key) is not True:raise ValueError('Successful verified run required before graph creation')
    if done['physical_gpu']!=1:raise ValueError('WrongGPU')
    design=read(out/'design.json');evaluation=read(out/'evaluation.json');resource=read(out/'resource_summary.json')
    sizes=design['nested_bank_sizes'];cells=evaluation['results'];items=read(out/'items.json')
    fresh_bootstrap=design.get('fresh_bootstrap',False)
    for name,value in done['frozen_manifest'].items():
        if sha256(out/name)!=value:raise ValueError('Frozen protocol changed')
    for k in sizes:
        for method,cell in cells[str(k)].items():
            if not cell['available']:
                if method!='kv_full_qa' or cell['reason'] not in ['context_limit','gpu_limit']:raise ValueError('Unexpected unavailable baseline')
                continue
            if len(cell['trials'])!=k or [t['qid'] for t in cell['trials']]!=[r['qid'] for r in items[:k]]:raise ValueError('Incomplete/unpaired trials')
            summary=metrics(cell['trials'])
            if any(cell['summary'][key]!=value for key,value in summary.items()):raise ValueError('Measures do not reproduce')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    primary=['our_delta','rag_qa','kv_full_qa'];colors=dict(our_delta='#178578',rag_qa='#477eb1',kv_full_qa='#ad7843')
    labels=dict(our_delta='Our dense deltas',rag_qa='Dense RAG + supplied Q/A',kv_full_qa='Full-bank KV + supplied Q/A')
    markers=dict(our_delta='s',rag_qa='o',kv_full_qa='^')
    gib=2**30;mib=2**20;figures=[]
    def publish(fig,name,caption):
        fig.savefig(out/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig)
        figures.append((name,caption))
    def available(method):return [(k,cells[str(k)][method]) for k in sizes if cells[str(k)][method]['available']]
    def rate(ax,method,key,scope='summary'):
        data=available(method);x=[k for k,c in data];vals=[c[scope][key]['rate']*100 for k,c in data]
        low=[c[scope][key]['ci95'][0]*100 for k,c in data];high=[c[scope][key]['ci95'][1]*100 for k,c in data]
        ax.errorbar(x,vals,yerr=[np.maximum(0,np.array(vals)-low),np.maximum(0,np.array(high)-vals)],color=colors[method],marker=markers[method],markersize=10 if method=='our_delta' else 6,markerfacecolor='none' if method=='our_delta' else colors[method],capsize=4,label=labels[method])
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for method in primary:
        rate(axes[0,0],method,'returned_exact')
        rate(axes[0,1],method,'returned_exact','anchor24_summary')
        if method!='kv_full_qa':rate(axes[1,0],method,'retrieval_at1')
        data=available(method)
        axes[1,1].plot([k for k,c in data],[c['summary']['mean_token_f1']*100 for k,c in data],color=colors[method],marker=markers[method],markersize=10 if method=='our_delta' else 6,markerfacecolor='none' if method=='our_delta' else colors[method],label=labels[method])
    for ax,title in zip(axes.flat,['Delivered answer exact match: all bank questions','Delivered exact: same 24 anchor questions','Correct source top1: selector methods only','Mean normalized token F1: all bank questions']):
        ax.set_title(title);ax.axhline(100,color='#555',ls='--',lw=1,label='100% aspiration');ax.set_xticks(sizes);ax.set_ylim(-3,116);ax.set_xlabel('Stored sessions');ax.set_ylabel('Percent');ax.legend(fontsize=8,loc='lower left')
    for ax in axes[0]:ax.legend(fontsize=8,loc='lower right')
    fig.suptitle('Exact original questions · fixed writer · same model · physical GPU 1',fontsize=15,weight='bold')
    fig.tight_layout(rect=[0,.055,1,.95])
    fig.text(.01,.01,'Exact/source rates: Wilson 95% intervals. KV has no selector, so source accuracy is undefined. Missing KV points mean capacity exceeded.\nLexical scores do not establish factual truth. Nested banks contain previously inspected test/development/pilot items; this is exploratory.',fontsize=9)
    publish(fig,'accuracy_scaling','The same 24 anchor questions isolate the effect of added competing sessions. All-question scores also change the question mix. No accuracy is assigned to an unavailable KV baseline.')
    def peaks(k,method):
        c=cells[str(k)][method];r=c['resources'];a=r['peak_allocated_bytes'];b=r['peak_reserved_bytes']
        if method=='kv_full_qa':a=max(a,r['setup_peak_allocated_bytes']);b=max(b,r['setup_peak_reserved_bytes'])
        if method=='rag_qa':
            setup=resource['measurements']['rag_index_creation'];a=max(a,setup['peak_allocated_bytes']);b=max(b,setup['peak_reserved_bytes'])
        if method=='our_delta':
            setup=resource['measurements']['activation_matrix'];a=max(a,setup['peak_allocated_bytes']);b=max(b,setup['peak_reserved_bytes'])
        return a,b
    fig,axes=plt.subplots(1,2,figsize=(13,5.7))
    for method in primary:
        data=available(method)
        axes[0].plot([k for k,c in data],[c['resources']['persistent_payload_bytes']/mib for k,c in data],color=colors[method],marker='o',label=labels[method]+': measured')
        axes[1].plot([k for k,c in data],[peaks(k,method)[0]/gib for k,c in data],color=colors[method],marker='o',label=labels[method])
    axes[0].plot(sizes,[cells[str(k)]['kv_full_qa']['plan']['expected_payload_bytes']/mib for k in sizes],color=colors['kv_full_qa'],ls='--',label='KV payload: config-derived expectation')
    axes[0].set_yscale('log');axes[0].set_ylabel('Retained memory bank MiB (log scale)');axes[0].set_title('Bank representation size across different storage tiers')
    axes[1].axhline(resource['base_model_parameter_bytes']/gib,color='#555',ls='--',label='Common LLM parameters')
    axes[1].set_ylabel('GiB on GPU');axes[1].set_title('Measured peak live GPU tensors: setup + recall');axes[1].set_ylim(0,None)
    for method,offset in [('our_delta',-3),('rag_qa',2),('kv_full_qa',2)]:
        candidates=available(method)
        if not candidates:continue
        k,c=candidates[-1];value=peaks(k,method)[0]/gib
        axes[1].annotate(f'{value:.2f} GiB',(k,value),xytext=(k,value+offset),ha='right' if k==max(sizes) else 'center',color=colors[method],fontsize=9)
    axes[1].set_ylim(0,max(peaks(k,method)[0]/gib for method in primary for k,c in available(method))*1.15)
    for ax in axes:ax.set_xticks(sizes);ax.set_xlabel('Stored sessions');ax.legend(fontsize=8)
    fig.tight_layout(rect=[0,.14,1,1]);fig.text(.01,.01,'Deltas are on disk; RAG corpus/index is on CPU/disk; full-bank KV is GPU-resident. Common model weights are excluded from bank size.\nGPU peaks include all common weights and routing/index/prefill setup. AlphaEdit writing is excluded. The64-delta activation matrix\nwas computed once and its measured setup peak is shared conservatively across bank sizes; this is not four separate routing benchmarks.',fontsize=9)
    publish(fig,'memory_scaling','Solid points are actual measurements. Dashed KV payload is calculated even where the complete prefix cannot run. A calculated payload does not imply the baseline fits the context window or GPU.')
    fig,axes=plt.subplots(1,2,figsize=(13,5.6))
    for method in primary:
        data=available(method)
        axes[0].plot([k for k,c in data],[peaks(k,method)[1]/gib for k,c in data],marker='o',color=colors[method],label=labels[method])
    axes[0].set_xticks(sizes);axes[0].set_xlabel('Stored sessions');axes[0].set_ylabel('GiB reserved by PyTorch');axes[0].set_title('Measured GPU allocator reservation');axes[0].legend(fontsize=8);axes[0].set_ylim(0,None)
    required=[cells[str(k)]['kv_full_qa']['plan']['required_tokens'] for k in sizes];limit=cells[str(sizes[0])]['kv_full_qa']['plan']['context_limit']
    axes[1].bar(range(len(sizes)),required,color=['#ad7843' if cells[str(k)]['kv_full_qa']['available'] else '#bd6363' for k in sizes])
    axes[1].axhline(limit,color='#333',ls='--',label=f'Model context limit: {limit:,} tokens')
    for i,(k,value) in enumerate(zip(sizes,required)):
        reason='ran' if cells[str(k)]['kv_full_qa']['available'] else cells[str(k)]['kv_full_qa']['reason'].replace('_',' ')
        axes[1].text(i,value+limit*.02,f'{value:,}\n{reason}',ha='center',fontsize=9)
    axes[1].set_xticks(range(len(sizes)),sizes);axes[1].set_xlabel('Stored sessions');axes[1].set_ylabel('Prefix + query + answer-cap tokens');axes[1].set_title('Full-bank KV capacity with every session preserved');axes[1].set_ylim(0,max(required)*1.23);axes[1].legend(fontsize=8)
    fig.tight_layout(rect=[0,.11,1,1]);fig.text(.01,.01,'Allocator reservation can contain unused blocks and differs from live tensor allocation. All new conditions use expandable segments.\nContext-limit cases are preflight capacity results; their accuracy and actual KV GPU peak are unavailable, not zero.',fontsize=9)
    publish(fig,'kv_capacity','The fresh24control uses the same allocator setting as the larger runs. This run does not test quantized/offloaded/compressed caches or session retrieval before KV prefill.')
    fig,axes=plt.subplots(1,2,figsize=(13,5.2))
    for key,color,label in [('returned_exact','#647a91','Delivered exact'),('contains_reference','#91a0ae','Reference containment'),('retrieval_at1','#8c609b','Correct source top1'),('retrieval_at3','#b28cbd','Correct source in top3chunks')]:
        xs=sizes;ys=[cells[str(k)]['rag_context_only']['summary'][key]['rate']*100 for k in sizes]
        axes[0].plot(xs,ys,marker='o',color=color,label=label)
    axes[0].set_ylabel('Percent');axes[0].set_ylim(0,110);axes[0].set_xticks(sizes);axes[0].set_xlabel('Stored sessions');axes[0].set_title('Secondary baseline: context-only dense RAG');axes[0].legend(fontsize=8)
    for method in ['our_delta','rag_qa']:
        axes[1].plot(sizes,[cells[str(k)][method]['summary']['returned_exact']['rate']*100 for k in sizes],color=colors[method],marker='o',label=labels[method])
    axes[1].plot(sizes,[cells[str(k)]['rag_context_only']['summary']['returned_exact']['rate']*100 for k in sizes],color='#647a91',marker='o',label='Context-only RAG (different information)')
    axes[1].set_xticks(sizes);axes[1].set_ylim(0,110);axes[1].set_xlabel('Stored sessions');axes[1].set_ylabel('Delivered exact percent');axes[1].set_title('Write-time Q/A supervision changes the task');axes[1].legend(fontsize=8)
    fig.tight_layout(rect=[0,.07,1,1]);fig.text(.01,.01,'Context-only RAG has no supplied Q/A annotations. Its score is not a matched-information comparison with supervised delta writes.',fontsize=9)
    publish(fig,'context_rag_scaling','Correct source among retrieved chunks does not guarantee that the answer-bearing excerpt was retrieved or that generation used it correctly.')
    esc=lambda x:html.escape(str(x))
    def table(heads,rows):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in heads)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    def score(c,key='returned_exact',scope='summary'):
        return 'Unavailable: '+c['reason'] if not c['available'] else f"{c[scope][key]['count']}/{c[scope]['n']}"
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Memory scaling comparison</title>',
          '<style>body{font-family:system-ui;background:#f3f6fa;color:#173247;max-width:1200px;margin:auto;padding:28px}section{background:white;border:1px solid #dbe4ed;border-radius:12px;padding:22px;margin:20px 0}p{line-height:1.6}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dce5ee;text-align:left;vertical-align:top}th{background:#edf3f8}.scroll{overflow:auto}.muted{color:#526a7e;font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3f8;padding:12px}a{color:#156caa}</style>',
          '<h1>Memory and accuracy at 24, 32, 48, and 64 sessions</h1>',
          '<p>Completed on physical GPU 1. Same exact original LongMemEval questions, Llama-3.1-8B-Instruct float32, fixed answer-plus-EOS writer, and activation router. No BERT, calibration, paraphrases, compression, or test-based tuning.</p>',
          '<p>All primary methods receive the full source sessions and supplied original questions/answers at memory creation. Our method recalls from the bare question key through a selected delta; RAG retrieves a full record using its supplied question-key index, then the unedited model generates; KV reads all records from its reusable prefix. Only the original question is supplied as the new recall input. Since questions are identical at write and recall, this measures the supervised exact-question mechanism rather than unseen-query retrieval.</p>',
          '<section><h2>Measured primary results</h2>']
    data=[]
    for k in sizes:
        for method in primary:
            c=cells[str(k)][method]
            if c['available']:
                r=c['resources'];a,b=peaks(k,method);s=c['summary'];source='N/A: whole bank' if s['retrieval_at1'] is None else score(c,'retrieval_at1')
                data.append([k,labels[method],score(c),score(c,scope='anchor24_summary'),source,f"{s['mean_token_f1']:.3f}",f"{r['persistent_payload_bytes']/mib:.3f} MiB",r['persistent_tier'],f'{a/gib:.2f} / {b/gib:.2f} GiB'])
            else:data.append([k,labels[method],score(c),'Unavailable','N/A','Unavailable',f"Expected only: {c['plan']['expected_payload_bytes']/gib:.2f} GiB",'Would be GPU-resident','Unavailable'])
    page.append(table(['Sessions','Method','Delivered exact','Same24anchor exact','Correct source top1','Token F1','Bank payload','Storage tier','Peak live / reserved GPU'],data))
    page.append('<p>Exact match normalizes case and punctuation. Token F1 and bounded containment are lexical proxies. They can penalize faithful sentences and can reward text containing the reference alongside a contradiction. Inspect complete outputs before calling an exact-match failure a factual error.</p></section>')
    largest=cells[str(sizes[-1])];own=largest['our_delta'];rag=largest['rag_qa']
    page.append('<section><h2>Memory interpretation</h2><p>At64sessions, our dense delta bank occupies '+esc(round(own['resources']['persistent_payload_bytes']/gib,2))+' GiB, while matched RAG text/index occupies '+esc(round(rag['resources']['persistent_payload_bytes']/mib,3))+' MiB. This does not support a universal memory-efficiency advantage over RAG. Deltas are loaded individually; retained delta storage and actual GPU use are different quantities.</p><p>All methods use an ordinary decoding KV cache. The full-bank KV baseline additionally retains every memory record as KV across questions; our method and RAG retain their banks separately and build short-lived query caches.</p><p>Common LLM parameter bytes are '+esc(round(resource['base_model_parameter_bytes']/gib,2))+' GiB. RAG additionally uses '+esc(round(resource['embedding_parameter_bytes']/mib,2))+' MiB of encoder parameters, included in its measured GPU peaks. RAG retains full session text; full-bank KV represents every prefix token. Only one supplied question per session is evaluated, so the delta results do not prove full-session fact retention.</p></section>')
    for name,caption in figures:
        data=base64.b64encode((out/(name+'.png')).read_bytes()).decode()
        page.append('<section><img alt="'+esc(name)+'" src="data:image/png;base64,'+data+'"><p class="muted">'+esc(caption)+'</p></section>')
    anchors=[]
    for method in primary:
        base=cells['24'][method]
        if not base['available']:continue
        old={r['qid']:r for r in base['trials']}
        for k in sizes:
            c=cells[str(k)][method]
            if not c['available']:continue
            rows=[r for r in c['trials'] if r['qid'] in old]
            regress=sum(old[r['qid']]['returned_exact'] and not r['returned_exact'] for r in rows)
            improve=sum(not old[r['qid']]['returned_exact'] and r['returned_exact'] for r in rows)
            source_switch=sum(old[r['qid']]['selected_sid']!=r['selected_sid'] for r in rows) if method=='our_delta' else 'N/A'
            anchors.append([k,labels[method],score(c,scope='anchor24_summary'),regress,improve,source_switch])
    page.append('<section><h2>Paired anchor checks</h2>'+table(['Sessions','Method','Anchor exact','Regressed vsfresh24','Improved vsfresh24','Activation source switches'],anchors)+'<p>These compare exactly the same24questions at each scale. Different all-question rates can reflect newly added questions as well as routing competition.</p></section>')
    strata=[]
    for method in primary:
        c=largest[method]
        if not c['available']:continue
        for history,s in c['history_summaries'].items():strata.append([labels[method],history,s['n'],f"{s['returned_exact']['count']}/{s['n']}", 'N/A' if s['retrieval_at1'] is None else f"{s['retrieval_at1']['count']}/{s['n']}"])
    page.append('<section><h2>Prior exposure and scope</h2>'+table(['Method','Earlier role','Questions','Delivered exact','Correct source'],strata)+'<p>The64sessions include earlier pilot, writer-development, calibration, training, and test items. The writer/router were frozen before this run and no new parameter was tuned. This is a capacity stress test under exact-question supervision, not an independent held-out accuracy estimate. The additional24deltas are new EOS memory writes; their underlying questions are not necessarily newly unseen.</p></section>')
    writing=resource['measurements']['additional_writes'];writes=read(out/'writes.json')
    historical_path=Path(design['source'])/'writes.json'
    immediate={r['qid']:dict(r,check_origin='Fresh bootstrap write-time check' if fresh_bootstrap else 'Historical write-time check') for r in read(historical_path)}
    immediate.update({r['qid']:dict(r,check_origin='New write-time check') for r in writes})
    breakdown=[]
    reviewed_notes={
        '6f9b354f':'Partial wording/detail: gray is conveyed, but lighter shade is not explicit.',
        '3d86fd0a':'Partial detail: coffee shop is conveyed, but in the city is omitted.',
        '15745da0':'Wrong information: supplies a start year instead of the reference duration of three months.',
        '1e043500':'Missing answer: gives a generic inability-to-access-Spotify response and reaches the token cap instead of naming Summer Vibes.',
        '8550ddae':'Wrong subtype: classic gin fizz replaces the reference lavender gin fizz.',
        'f8c5f88b':'Reference location conveyed with uncertainty/hedging rather than the exact short answer.'}
    if fresh_bootstrap:reviewed_notes={}  # Historical judgments do not grade fresh outputs.
    for t in largest['our_delta']['trials']:
        if t['returned_exact']:continue
        check=immediate[t['qid']]
        breakdown.append(dict(qid=t['qid'],question=t['question'],reference=t['answer'],selected_answer=t['final']['text'],
                              correct_delta_selected=t['routing_correct'],correct_delta_rank=t['rank'],
                              own_immediate_answer=check['immediate']['text'],own_immediate_exact=check['immediate_exact'],
                              immediate_check_origin=check['check_origin'],
                              post_hoc_review=reviewed_notes.get(t['qid'],'Requires inspection; no semantic judgment assigned.')))
    save(out,'nonexact_diagnostics.json',dict(evaluation_sha256=sha256(out/'evaluation.json'),historical_writes_sha256=sha256(historical_path),
                                             note='Write-time checks diagnose independent own-delta recall. Bootstrap checks are fresh local writes.' if fresh_bootstrap else 'Write-time checks diagnose independent own-delta recall. Historical checks were not regenerated in this run. Nonexact does not necessarily mean a factual error.',cases=breakdown))
    source_note='Fresh local bootstrap checks are separate from this scaling run.' if fresh_bootstrap else 'Historical checks are shown as diagnostics, not fresh measurements.'
    page.append('<section><h2>Nonexact delta answers at64sessions</h2><p>The new24write-time checks returned '+esc(sum(r['immediate_exact'] for r in writes))+'/24normalized exact answers. Source-bank write-time checks for the reused40deltas returned '+esc(sum(immediate[r['qid']]['immediate_exact'] for r in items if r['sid'] in design['reused_delta_sids']))+'/40. '+source_note+'</p>')
    page.append(table(['Question','Reference','Selected final','Correct delta selected','Correct delta rank','Own write-time output','Post hoc descriptive review'],[[r['question'],r['reference'],r['selected_answer'],r['correct_delta_selected'],r['correct_delta_rank'],r['own_immediate_answer'],r['post_hoc_review']] for r in breakdown]))
    review_note='Fresh nonexact answers require inspection; no historical semantic judgments are assigned.' if fresh_bootstrap else 'The case notes are a post hoc, unblinded review by Codex, not independent semantic grading. Wording differences, missing details, and uncertainty are distinguished from clearly wrong/missing answers.'
    page.append('<p>If the correct delta was selected, the nonexact result comes from its output rather than a routing failure. If another delta was selected, routing competition is involved. '+review_note+'</p></section>')
    page.append('<section><h2>Execution and costs</h2><p>'+esc(done['reused_deltas'])+' previously verified deltas were reused through symlinks; '+esc(done['new_writes'])+' additional deltas were written. New-write time was '+esc(round(writing['elapsed_seconds']/60,2))+' minutes and write-phase peak live GPU use was '+esc(round(writing['peak_allocated_bytes']/gib,2))+' GiB. These write costs are reported separately from recall graphs.</p><p>A fresh64×64activation matrix contains4096question–delta energy scores. It is not4096generated answer probes. Each nested bank routes using only its own columns, then generates fresh batch1 answers. The three complete conditions each produced '+esc(done['own_rag_secondary_answers_each'])+' answers across the four sizes. Baseline KV completed where feasible, and its capacity failures are explicitly reported.</p><p>Total runtime '+esc(round(resource['elapsed_seconds']/60,2))+' minutes. Source artifacts, all reused/new delta hashes, the projection, code, and base weights were checked unchanged/restored. No old delta files were deleted.</p></section>')
    if fresh_bootstrap:
        bootstrap_seconds=sum(r['elapsed_seconds'] for r in read(historical_path))
        page.append('<section><h2>Fresh bootstrap scope</h2><p>The runtime above covers the scaling phase only. Forty source deltas were written freshly before that phase; their summed write/check time was '+esc(round(bootstrap_seconds/60,2))+' minutes. Projector construction and model-loading overhead for that bootstrap are additional. Original historical validation, writer selection, and the earlier comparison were not rerun. No historical semantic case notes or KV content-review counts grade these fresh outputs.</p></section>')
    page.append('<section><h2>All64outputs at the largest bank</h2>')
    for i,r in enumerate(items):
        page.append('<details><summary>'+esc(r['question'])+'</summary><p>Reference: '+esc(r['answer'])+'</p>')
        page.append(table(['Method','Delivered answer','Exact','Source hit top1','Finish'],[[labels.get(method,method),largest[method]['trials'][i]['final']['text'],largest[method]['trials'][i]['returned_exact'],largest[method]['trials'][i]['retrieval_at1'],largest[method]['trials'][i]['final']['finish_reason']] for method in primary+['rag_context_only'] if largest[method]['available']]))
        page.append('</details>')
    page.append('</section><section><h2>Full KV outputs at the largest feasible tested bank</h2>')
    feasible=[k for k in sizes if cells[str(k)]['kv_full_qa']['available']]
    if feasible:
        k=max(feasible)
        page.append('<p>Complete outputs at '+esc(k)+'sessions; longer sentences can fail exact match while conveying the requested fact. This does not establish that every extra statement is correct.</p>')
        if k==32 and not fresh_bootstrap:
            page.append('<p>A post hoc, unblinded Codex review of these32outputs found the requested reference fact conveyed without contradiction in31/32responses. The dog-breed response denies knowing the answer and fails this criterion. This supplementary descriptive review is not independent human evaluation and does not verify every additional statement.</p>')
        page.append(table(['Question','Reference','Complete delivered output','Exact','Reference contained','Finish'],[[t['question'],t['answer'],t['final']['text'],t['returned_exact'],t['contains_reference'],t['final']['finish_reason']] for t in cells[str(k)]['kv_full_qa']['trials']]))
    page.append('</section><section><h2>Reproducibility</h2><p>References: <a href="'+esc(design['references']['embedding'])+'">MiniLM official model card</a>, <a href="'+esc(design['references']['kv'])+'">Transformers KV-cache documentation</a>, <a href="'+esc(design['references']['rag'])+'">Sentence Transformers retrieval documentation</a>.</p>')
    for n in ['design.json','resource_summary.json','completion.json']:
        import json
        page.append('<details><summary>'+esc(n)+'</summary><pre>'+esc(json.dumps(read(out/n),indent=2))+'</pre></details>')
    page.append('</section></html>');(out/'report.html').write_text('\n'.join(page))
    controls=None
    if design.get('previous_comparison'):
        old=read(Path(design['previous_comparison'])/'evaluation.json')['results']
        controls={method:len(old[method]['trials'])==len(cells['24'][method]['trials']) and all(a['qid']==b['qid'] and a['final']['text']==b['final']['text'] for a,b in zip(old[method]['trials'],cells['24'][method]['trials'])) for method in primary if cells['24'][method]['available']}
    save(out,'report_verification.json',dict(completed=True,evaluation_sha256=sha256(out/'evaluation.json'),report_sha256=sha256(out/'report.html'),previous24_outputs_reproduced=controls,figures=[n for n,_ in figures]))
    print('REPORT',out/'report.html')


if __name__=='__main__':main()
