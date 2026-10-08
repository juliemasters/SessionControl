"""Accuracy and memory comparison graphs, gated on a successful complete run."""
import argparse
import base64
import html
import json
from pathlib import Path

from experiment_larger_routing import read,sha256
from experiment_memory_comparison import metrics
from question_key_k1_core import interval


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('results/memory-comparison-v1'))
    out=p.parse_args().out.resolve();done=read(out/'completion.json')
    if any(done.get(k) is not True for k in ['completed','sources_unchanged','delta_hashes_unchanged','base_weights_restored',
                                            'kv_payload_matches_formula','kv_prefix_is_reused','cache_tail_cropped_between_queries']):
        raise ValueError('Complete successful GPU1 experiment required before figures')
    if done['physical_gpu']!=1 or done['methods_completed']!=4 or done['answers_per_method']!=24:raise ValueError('Incomplete comparison')
    for n,v in done['frozen_manifest'].items():
        if sha256(out/n)!=v:raise ValueError('Protocol changed')
    design=read(out/'design.json');evaluation=read(out/'evaluation.json');res=read(out/'resource_summary.json')
    cells=evaluation['results'];order=['our_delta','rag_qa','kv_full_qa'];secondary=cells['rag_context_only']
    for key,cell in cells.items():
        if len(cell['trials'])!=24 or metrics(cell['trials'])!=cell['summary']:raise ValueError('Measures do not reproduce')
    primary_qids=[t['qid'] for t in cells['our_delta']['trials']]
    for key in cells:
        if [t['qid'] for t in cells[key]['trials']]!=primary_qids:raise ValueError('Unpaired question sets')
    source=Path(design['source'])
    for n,v in design['source_hashes'].items():
        if sha256(source/n)!=v:raise ValueError('Historical source changed')
    review=read(out/'answer_content_review.json')
    if review['evaluation_sha256']!=sha256(out/'evaluation.json'):raise ValueError('Content review refers to different outputs')
    for method in order:
        if [r['qid'] for r in review['methods'][method]]!=primary_qids:raise ValueError('Unpaired content review')
    # No plot imports before successful-completion checks.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    names=['Our method\nDense AlphaEdit deltas','Dense RAG\nSession + supplied Q/A','Full-bank KV cache\nSessions + supplied Q/A']
    colors=['#178578','#477eb1','#ad7843'];gib=1024**3;figures=[]
    def figure(fig,name,caption):
        fig.savefig(out/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight')
        plt.close(fig);figures.append((name,caption))
    fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,(key,title) in zip(axes.flat,[('returned_exact','Delivered answer exact match'),('emitted_exact','Complete emitted text exact match'),
                                       ('contains_reference','Token-bounded reference containment'),('mean_token_f1','Mean normalized token F1')]):
        if key=='mean_token_f1':
            values=[cells[k]['summary'][key]*100 for k in order];lows=[];highs=[]
            rng=np.random.default_rng(0)
            for method in order:
                data=np.array([t['token_f1'] for t in cells[method]['trials']])
                boot=data[rng.integers(0,len(data),size=(2000,len(data)))].mean(1)*100
                lo,hi=np.percentile(boot,[2.5,97.5]);lows.append(lo);highs.append(hi)
        else:
            values=[cells[k]['summary'][key]['rate']*100 for k in order]
            lows=[cells[k]['summary'][key]['ci95'][0]*100 for k in order];highs=[cells[k]['summary'][key]['ci95'][1]*100 for k in order]
        ax.bar(range(3),values,color=colors,yerr=[np.maximum(0,np.array(values)-lows),np.maximum(0,np.array(highs)-values)],capsize=5)
        ax.axhline(100,color='#555',ls='--',label='Requested 100% target');ax.set_ylim(-2,116)
        ax.set_xticks(range(3),names,fontsize=9);ax.set_ylabel('Percent');ax.set_title(title)
        for i,value in enumerate(values):
            text=f'{value:.1f}%' if key=='mean_token_f1' else str(cells[order[i]]['summary'][key]['count'])+'/24'
            inside=value>55;y=max(8,lows[i]-7) if inside else highs[i]+3
            ax.text(i,y,text,ha='center',weight='bold',color='white' if inside else '#223548')
        ax.legend(loc='upper left',fontsize=8)
    fig.suptitle('Matched write-time information · 24 exact original questions · GPU 1',fontsize=15,weight='bold')
    fig.tight_layout(rect=[0,.045,1,.95])
    fig.text(.01,.012,'All three receive the full sessions and supplied Q/A at memory creation. Greedy Llama-3.1-8B, float32.\nExact/containment: Wilson 95% intervals. Token F1: question bootstrap. Targets are aspirations, not forecasts.',fontsize=9)
    figure(fig,'accuracy_matched','Exact match uses case/punctuation normalization. Token F1 and containment are lexical measures, not a semantic truth judge. No references are injected from evaluation at recall.')
    content_counts=[sum(r['reference_fact_correct_without_contradiction'] for r in review['methods'][k]) for k in order]
    content_intervals=[interval(count,24) for count in content_counts]
    content_values=np.array(content_counts)/24*100
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(range(3),content_values,color=colors,yerr=[content_values-np.array([v[0] for v in content_intervals])*100,np.array([v[1] for v in content_intervals])*100-content_values],capsize=5)
    ax.axhline(100,color='#555',ls='--',label='Requested 100% target')
    for i,count in enumerate(content_counts):ax.text(i,72,f'{count}/24',ha='center',color='white',weight='bold')
    ax.set_ylim(0,114);ax.set_xticks(range(3),names,fontsize=9);ax.set_ylabel('Percent');ax.set_title('Exploratory answer-content review: reference fact without contradiction')
    ax.legend(loc='upper left',fontsize=8);fig.tight_layout(rect=[0,.11,1,1])
    fig.text(.01,.01,'Post hoc, unblinded review by Codex against the reference answer; 95% Wilson intervals. Different wording can pass.\nA response that names the answer while explicitly denying knowledge of that fact fails. This is not independent human evaluation.',fontsize=9)
    figure(fig,'answer_content_review','This supplementary review checks whether the requested reference fact is correctly conveyed without contradiction. It does not establish that every additional statement is true. Review criteria were added after inspecting outputs; this is exploratory, not a preregistered metric.')
    def peaks(method):
        row=cells[method]['resources'];a=row['peak_allocated_bytes'];r=row['peak_reserved_bytes']
        if method=='kv_full_qa':
            build=res['methods']['kv_full_qa_build'];a=max(a,build['peak_allocated_bytes']);r=max(r,build['peak_reserved_bytes'])
        if method=='rag_qa':
            build=res['methods']['rag_index_creation'];a=max(a,build['peak_allocated_bytes']);r=max(r,build['peak_reserved_bytes'])
        return a/gib,r/gib
    fig,axes=plt.subplots(1,2,figsize=(13,5.4))
    allocated,reserved=zip(*(peaks(k) for k in order));x=np.arange(3)
    axes[0].bar(x-.18,allocated,.36,color=colors,label='Peak live tensor allocation')
    axes[0].bar(x+.18,reserved,.36,color=colors,alpha=.35,label='Peak allocator reservation')
    for i,(a,r) in enumerate(zip(allocated,reserved)):
        axes[0].text(i,max(a,r)+.6,f'{a:.2f} / {r:.2f}',ha='center',fontsize=9)
    axes[0].axhline(res['base_model_parameter_bytes']/gib,color='#555',ls='--',label='Common LLM parameter minimum')
    axes[0].set_ylabel('GiB on GPU');axes[0].set_ylim(0,max(reserved)*1.2);axes[0].set_title('Measured GPU peak: setup + recall')
    axes[0].legend(loc='upper left',fontsize=8)
    payload=[cells[k]['resources']['persistent_payload_bytes']/1024**2 for k in order]
    axes[1].bar(x,payload,color=colors);axes[1].set_yscale('log');axes[1].set_ylim(min(payload)/4,max(payload)*5)
    axes[1].set_ylabel('MiB of retained bank representation (log scale)');axes[1].set_title('Retained 24-session memory payload')
    for i,value in enumerate(payload):
        formatted=f'{value/1024:.2f} GiB' if value>=1024 else f'{value:.3f} MiB'
        axes[1].text(i,value*1.25,formatted,ha='center',fontsize=10)
    for ax in axes:ax.set_xticks(x,names,fontsize=9)
    fig.tight_layout(rect=[0,.10,1,1])
    fig.text(.01,.01,'Payload storage tiers differ: deltas on disk, RAG text/index on CPU/disk, reusable KV on GPU. Common LLM weights are excluded from payload.\nRAG retriever weights are included in GPU peaks and reported separately. No new AlphaEdit write cost is included.',fontsize=9)
    figure(fig,'memory_matched','GPU peaks are measured PyTorch allocation/reservation, not bank file size. KV prefill and RAG index creation are included. Retained payload is storage/representation size; it must not be interpreted as GPU residency for all methods.')
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    kv=[cells[k]['summary']['max_query_cache_payload_bytes']/1024**2 for k in order]
    axes[0].bar(range(3),kv,color=colors);axes[0].set_yscale('log');axes[0].set_ylim(min(kv)/4,max(kv)*4)
    axes[0].set_ylabel('MiB (log scale)');axes[0].set_title('Maximum decoding KV payload per query')
    for i,value in enumerate(kv):axes[0].text(i,value*1.2,f'{value:.2f} MiB',ha='center',fontsize=9)
    axes[0].set_xticks(range(3),names,fontsize=9)
    scale=read(out/'expected_memory_scaling.json')
    for method,key,color,label in [('our_delta','expected_dense_delta_bytes',colors[0],'Dense deltas: expected'),
                                   ('rag_qa','expected_rag_payload_bytes',colors[1],'RAG text/index: expected'),
                                   ('kv_full_qa','expected_kv_bytes',colors[2],'Full-bank KV: expected')]:
        axes[1].plot([r['sessions'] for r in scale],[r[key]/1024**2 for r in scale],marker='o',color=color,label=label)
        axes[1].scatter([24],[cells[method]['resources']['persistent_payload_bytes']/1024**2],s=95,facecolors='none',edgecolors=color,linewidths=2)
    axes[1].set_yscale('log');axes[1].set_xlabel('Stored sessions');axes[1].set_ylabel('Retained payload MiB (log scale)')
    axes[1].set_xticks([1,8,16,24]);axes[1].set_title('Calculated memory scaling; measured point at K=24');axes[1].legend(fontsize=8)
    fig.tight_layout(rect=[0,.11,1,1])
    fig.text(.01,.01,'KV/RAG query cache payloads are measured from tensors; our transient decoding cache is calculated from config and token count.\nScaling is a calculation from actual text/token counts and model dimensions. Accuracy was measured only at 24 sessions.',fontsize=9)
    figure(fig,'cache_and_scaling','All methods use a decoding KV cache. Our method retains information in delta files; the full-prefix KV baseline retains the whole bank in its decoding cache. Hollow points are measured K=24 payloads.')
    # Secondary experiment is visually separated because it has less write-time supervision.
    fig,ax=plt.subplots(figsize=(9,4.6));keys=['returned_exact','contains_reference','retrieval_at1','retrieval_at3']
    labels=['Delivered answer\nexact match','Reference\ncontainment','Correct session in\nfirst chunk','Correct session in\ntop 3 chunks']
    vals=[secondary['summary'][k]['rate']*100 for k in keys]
    lows=[secondary['summary'][k]['ci95'][0]*100 for k in keys];highs=[secondary['summary'][k]['ci95'][1]*100 for k in keys]
    ax.bar(range(4),vals,color='#87929d',yerr=[np.maximum(0,np.array(vals)-lows),np.maximum(0,np.array(highs)-vals)],capsize=5)
    for i,value in enumerate(vals):ax.text(i,max(7,lows[i]-6) if value>55 else highs[i]+3,str(secondary['summary'][keys[i]]['count'])+'/24',ha='center',color='white' if value>55 else '#223548',weight='bold')
    ax.set_xticks(range(4),labels);ax.set_ylim(0,115);ax.set_ylabel('Percent');ax.set_title('Secondary baseline: context-only dense RAG (different write information)')
    fig.tight_layout()
    figure(fig,'rag_context_only','This baseline indexes conversation excerpts without the supplied Q/A pairs. It retrieves three chunks and is not a matched-information competitor to the supervised delta writes.')
    esc=lambda v:html.escape(str(v))
    def table(heads,data):
        return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in heads)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in data)+'</tbody></table></div>'
    own=cells['our_delta']['summary'];rag=cells['rag_qa']['summary'];full=cells['kv_full_qa']['summary']
    ratio=cells['our_delta']['resources']['persistent_payload_bytes']/cells['rag_qa']['resources']['persistent_payload_bytes']
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Delta vs RAG vs KV memory comparison</title>',
          '<style>body{font-family:system-ui,sans-serif;background:#f3f6fa;color:#172c3e;max-width:1200px;margin:auto;padding:28px}section{background:white;border:1px solid #dbe4ed;border-radius:12px;padding:22px;margin:20px 0}h1{font-size:30px}h2{font-size:22px}p{line-height:1.6}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #dce5ee;text-align:left;vertical-align:top}th{background:#edf3f8}.scroll{overflow-x:auto}img{width:100%;height:auto}.muted{color:#526a7e;font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf3f8;padding:12px}a{color:#156caa}</style>',
          '<h1>Our memory method vs dense RAG vs full-bank KV cache</h1>',
          '<p>Measured on physical GPU 1 using the same 24 original questions, Llama-3.1-8B-Instruct, float32, and greedy decoding. No paraphrased questions or new memory writes.</p>',
          '<section><h2>What the comparison shows</h2><p>Delivered answer exact match: our method <strong>'+esc(own['returned_exact']['count'])+'/24</strong>, matched dense RAG <strong>'+esc(rag['returned_exact']['count'])+'/24</strong>, full-bank KV <strong>'+esc(full['returned_exact']['count'])+'/24</strong>.</p>',
          '<p>In a separate, post hoc answer-content review by Codex, the requested reference fact was conveyed without contradiction in '+esc(content_counts[0])+'/24 delta outputs, '+esc(content_counts[1])+'/24 matched RAG outputs, and '+esc(content_counts[2])+'/24 KV outputs. Most KV exact-match failures are wording differences. The dog-breed response both denies having information and names the correct breed, so it fails this content criterion despite containing the reference.</p>',
          '<p>The dense delta bank occupies '+esc(round(cells['our_delta']['resources']['persistent_payload_bytes']/gib,2))+' GiB. The RAG text/index occupies '+esc(round(cells['rag_qa']['resources']['persistent_payload_bytes']/1024**2,3))+' MiB, making the current dense delta files about '+esc(round(ratio))+' times larger. The full-bank reusable KV payload occupies '+esc(round(cells['kv_full_qa']['resources']['persistent_payload_bytes']/gib,2))+' GiB. These are different storage tiers; the GPU peak chart measures their actual runtime effect.</p>',
          '<p>This experiment does not support a claim that dense deltas are universally more memory-efficient than RAG. It compares these specific implementations under exact-question, supplied-Q/A supervision.</p></section>',
          '<section><h2>Matched information and measurement rules</h2><p>All three primary methods receive the same full answer-bearing sessions and supplied questions/answers at memory creation. Our verified deltas were already written using that information. Dense RAG indexes the question keys and retains the full sessions plus Q/A records, then the unedited Llama generates from the top retrieved record. Full-bank KV prefills all records into one reusable cache and answers without oracle session selection.</p>',
          '<p>Our inference sees only the bare original question key. RAG and KV use an instruction/chat prompt appropriate to reading stored records; neither receives the current evaluation answer or correct-session label as an extra input. Answer references and source IDs are used only after generation to compute metrics. The different prompt formats are necessary to preserve the delta write key and provide explicit stored text to the baselines.</p>',
          '<p>The supplied question is identical at write and recall time, so the RAG index can match an exact stored question. This is a fair mechanism control for the delta setup, not a retrieval-generalization benchmark. The 24 test sessions were inspected previously; the results are exploratory. The context-only RAG condition receives no supplied Q/A and is presented separately.</p>',
          '<p>Only one supplied question per session is tested. Feeding a full session to AlphaEdit at write time does not establish that its delta retains every fact in that session. RAG keeps the session text and KV represents the whole prefix; the payload comparison measures these implementations on the tested task, not equal verified capacity for all session facts.</p>',
          '<p>All methods use transient decoding KV states. The KV baseline additionally retains the complete bank as KV across questions. Common LLM parameter bytes are '+esc(round(res['base_model_parameter_bytes']/gib,2))+' GiB. RAG adds '+esc(round(res['embedding_parameter_bytes']/1024**2,2))+' MiB of embedding-model parameters, included in measured GPU peaks but excluded from the data-only bank payload.</p></section>',
          '<section><h2>Measured accuracy and memory</h2>']
    data=[]
    for key,label in zip(order,['Our dense delta method','Dense RAG, session + Q/A','Full-bank KV, sessions + Q/A']):
        s=cells[key]['summary'];r=cells[key]['resources'];a,b=peaks(key)
        retrieval=s['retrieval_at1'];retrieve='Not applicable: whole bank' if retrieval is None else f"{retrieval['count']}/24"
        data.append([label,f"{s['returned_exact']['count']}/24",f"{s['emitted_exact']['count']}/24",f"{s['verbatim_exact']['count']}/24",f"{content_counts[order.index(key)]}/24",f"{s['mean_token_f1']:.3f}",retrieve,
                     f"{r['persistent_payload_bytes']/1024**2:.3f} MiB",r['persistent_tier'],f'{a:.2f} / {b:.2f} GiB'])
    page.append(table(['Method','Delivered exact','Emitted exact','Verbatim exact','Content review (post hoc)','Token F1','Correct source top1','Bank payload','Payload tier','Peak allocated / reserved GPU'],data))
    page.append('<p>Exact match uses the same case/punctuation normalization as the previous experiments. Verbatim exact compares delivered strings directly. Token F1 and containment are lexical proxies, not a semantic truth judge. Full-bank KV has no session selector, so its source-retrieval accuracy is undefined; supplying all sessions is not counted as 100% routing.</p></section>')
    for name,caption in figures:
        image=base64.b64encode((out/(name+'.png')).read_bytes()).decode()
        page.append('<section><img alt="'+esc(name)+'" src="data:image/png;base64,'+image+'"><p class="muted">'+esc(caption)+'</p></section>')
    page.append('<section><h2>Persistent storage, serving memory, and setup costs</h2><p>Our bank payload counts only the 24 active delta files, excluding previous development banks. Deltas are loaded one at a time from disk; the runtime also holds a CPU backup of the edited base matrices. RAG payload includes its serialized text corpus and vector index. KV payload counts the actual float32 key and value tensors resident on the GPU; it is not a disk file measurement.</p><p>GPU peaks include the common model, transient decoding cache, activation/attention workspace, and the RAG encoder where applicable. KV prefix prefill and RAG index creation are included in the plotted method peaks. AlphaEdit write cost was incurred in the previous experiment and is not included in this inference comparison; the earlier approximately 37 GiB write-inclusive peak must not be used as an inference-only comparison.</p>')
    kvplan=read(out/'kv_plan.json')
    page.append('<p>The full-bank prefix has '+esc(kvplan['prefix_tokens'])+' tokens. With '+esc(kvplan['bytes_per_token'])+' bytes per cached token, the expected payload is '+esc(round(kvplan['expected_payload_bytes']/gib,3))+' GiB; the actual tensors matched this formula exactly. Chunked prefill preserves all tokens. Prefix cache samples were checked after every query, and query tails were cropped before reuse.</p>')
    page.append('<p>Allocator reservation includes reusable blocks that may not contain live tensors; it is not another live-memory estimate. Accuracy was not run at the smaller scaling points. Those memory curves are calculations using actual corpus token counts and model dimensions. The current implementation saves dense three-layer deltas; low-rank, quantized deltas, compressed/offloaded KV, and alternative RAG models are outside this comparison.</p></section>')
    page.append('<section><h2>Per-question outputs and errors</h2>')
    for i,qid in enumerate(primary_qids):
        reference=cells['our_delta']['trials'][i]
        page.append('<details><summary>'+esc(reference['question'])+'</summary><p>Reference: '+esc(reference['answer'])+'</p>')
        page.append(table(['Method','Delivered answer','Exact','Source hit top1','Finish'],[[key,cells[key]['trials'][i]['final']['text'],cells[key]['trials'][i]['returned_exact'],cells[key]['trials'][i]['retrieval_at1'],cells[key]['trials'][i]['final']['finish_reason']] for key in order+['rag_context_only']]))
        page.append('</details>')
    page.append('</section><section><h2>Reproducibility and sources</h2><p>All four conditions completed on GPU 1. Source datasets, previous results, and delta hashes were verified unchanged; base FFN weights were restored. Cache formula, chunked-prefill equivalence, prefix reuse, and retrieval/number-scoring checks passed.</p><p>Runtime '+esc(round(res['elapsed_seconds']/60,2))+' minutes. Retrieval encoder: <a href="'+esc(design['references']['embedding'])+'">all-MiniLM-L6-v2 official model card</a>. Method references: <a href="'+esc(design['references']['kv'])+'">Transformers KV-cache documentation</a> and <a href="'+esc(design['references']['rag'])+'">Sentence Transformers retrieval documentation</a>.</p>')
    for n in ['design.json','resource_summary.json','kv_plan.json','completion.json','answer_content_review.json']:
        page.append('<details><summary>'+esc(n)+'</summary><pre>'+esc(json.dumps(read(out/n),indent=2))+'</pre></details>')
    page.append('</section></html>');(out/'report.html').write_text('\n'.join(page))
    print(json.dumps(dict(report=str(out/'report.html'),figures=[n for n,_ in figures],completed=True)))


if __name__=='__main__':main()
