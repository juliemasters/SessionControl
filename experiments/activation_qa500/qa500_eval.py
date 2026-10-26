"""Ten genuinely remeasured checkpoint passes with full Q/A targets."""
import argparse
import gc
import math
from pathlib import Path
import statistics
import time

from comparison_core import read,save,digest
from qa500_core import ROOT,CODE,METHODS,SIZES,load,check,status,now,canonical,score,document,order


def checkpoint_metrics(trials):
    def average(key):
        values=[r[key] for r in trials if r[key] is not None]
        return statistics.mean(values) if values else None
    visible=sorted(r['ttft_seconds'] for r in trials if r['ttft_seconds'] is not None)
    latency=sorted(r['query_seconds'] for r in trials)
    return dict(questions=len(trials),mean_f1=average('token_f1'),exact_rate=average('exact'),
                reference_containment_rate=average('contains_reference'),source_hit1=average('source_hit1'),
                mean_rank=average('correct_rank'),mrr=average('reciprocal_rank'),
                support_recall6=average('support_recall6'),complete_support6=average('complete_support6'),
                ttft_mean_ms=statistics.mean(visible)*1000 if visible else None,
                ttft_median_ms=statistics.median(visible)*1000 if visible else None,
                ttft_p95_ms=visible[math.ceil(.95*len(visible))-1]*1000 if visible else None,
                first_generated_mean_ms=(average('first_generated_token_seconds')*1000
                                         if average('first_generated_token_seconds') is not None else None),
                query_latency_mean_ms=statistics.mean(latency)*1000,
                query_latency_p95_ms=latency[math.ceil(.95*len(latency))-1]*1000,
                output_tokens_mean=average('output_tokens'),hit_cap_rate=average('hit_cap'),
                ttft_samples=len(visible),missing_visible_ttft=len(trials)-len(visible))


def run(out,method,repeat,sizes=None):
    folder=out/method/f'run-{repeat:02d}'
    d=check(out);sizes=sizes or d['sizes']
    if all((folder/f'bank-{n}.json').exists() and read(folder/f'bank-{n}.json').get('completed') for n in sizes):return
    d,torch,tok,model=load(out,folder)
    torch.manual_seed(repeat);torch.cuda.manual_seed_all(repeat)
    from official64_core import HostPeak
    from compact_session_memory import capture_keys
    from answer_only_generation import generate_answer_batch,reference_tokens_present
    from memory_comparison_core import bank_prefix,query_suffix,prefill,embed,cache_payload_bytes,kv_bytes_per_token,token_f1
    from ttft_timing import FirstAnswerClock,observe_generate,decode_cached_observed
    from question_key import normalize
    from full500_core import restore
    sessions=read(out/'sessions.json');questions={q['qid']:q for q in read(out/'queries.json')}
    born={qid:p['n'] for p in reversed(d['plan']) for qid in p['qids']}
    names=[f'model.layers.{l}.mlp.down_proj.weight' for l in d['layers']]
    base={n:model.get_parameter(n).detach().cpu().clone() for n in names} if method=='our_delta' else None
    base_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
    encoder_bytes=0;encoder=None;etok=None
    def clean():gc.collect();torch.cuda.empty_cache()
    def sync():torch.cuda.synchronize()
    if method=='rag':
        from transformers import AutoTokenizer,AutoModel
        etok=AutoTokenizer.from_pretrained(d['embedding_model'],local_files_only=True)
        encoder=AutoModel.from_pretrained(d['embedding_model'],local_files_only=True).to(model.device).float().eval()
        encoder_bytes=sum(p.numel()*p.element_size() for p in encoder.parameters())
        root=out/'rag';index_manifest=root/'index_manifest.json'
        if not index_manifest.exists():
            bindings=[dict(sid=r['sid'],question=b['question']) for r in sessions for b in r['bindings']]
            vectors=embed(encoder,etok,[b['question'] for b in bindings]).cpu()
            hashes={}
            for n in d['sizes']:
                bf=root/f'index-{n}';bf.mkdir(exist_ok=True)
                count=sum(len(r['bindings']) for r in sessions[:n])
                save(bf/'records.json',sessions[:n]);save(bf/'bindings.json',bindings[:count])
                torch.save(vectors[:count].clone().contiguous(),bf/'vectors.pt')
                hashes.update({str(f.relative_to(root)):digest(f) for f in bf.iterdir()})
            save(index_manifest,hashes);del vectors,bindings
        for file,h in read(index_manifest).items():
            if digest(root/file)!=h:raise ValueError('Stored Q/A RAG index changed')
        embed(encoder,etok,['Unrelated encoder warmup.'])
    for _ in range(2):generate_answer_batch(model,tok,['Answer briefly.'],cap=1)
    clean();sync()
    ledger=read(out/'independent/ledger.json') if method=='our_delta' else None
    try:
        for n in sizes:
            path=folder/f'bank-{n}.json'
            if path.exists() and read(path).get('completed'):continue
            qids=next(p['qids'] for p in d['plan'] if p['n']==n)
            expected=order(qids,repeat,n)
            cache=None;payloads=None;vectors=None;bank=None;bindings=None
            extra={};available=True;reason=None;trials=[]
            clean();sync();torch.cuda.reset_peak_memory_stats()
            status(folder,'bank_setup',n=n,repeat=repeat,completed=0,total=len(qids))
            with HostPeak() as host:
                if method=='our_delta':
                    if len(ledger)<n:raise ValueError('Requested answer-delta prefix has not been written')
                    payloads={};files={};bank_disk=0
                    for row in ledger[:n]:
                        f=out/'independent/factors'/row['file']
                        if digest(f)!=row['sha256']:raise ValueError('Answer factor changed')
                        payloads[row['sid']]=canonical(torch.load(f,map_location='cpu',weights_only=True))
                        files[row['sid']]=f;bank_disk+=f.stat().st_size
                    cpu_payload=sum(t.numel()*t.element_size() for p in payloads.values() for pair in p.values() for t in pair)
                    gpu_payload=0
                    extra['base_restore_cpu_bytes']=sum(v.numel()*v.element_size() for v in base.values())
                elif method=='plain_alphaedit':
                    f=out/'shared'/f'weights-{n}.pt'
                    if not f.exists():raise ValueError('Requested cumulative checkpoint not written')
                    if n<=64 and digest(f)!=d['source_hashes'][str(f.resolve())]:raise ValueError('Reused shared patch changed')
                    cp=torch.load(f,map_location='cpu',weights_only=True)
                    if cp['count']!=n:raise ValueError('Wrong cumulative checkpoint')
                    restore(model,cp['weights']);del cp;clean()
                    bank_disk=f.stat().st_size;cpu_payload=0;gpu_payload=0
                elif method=='rag':
                    bf=out/'rag'/f'index-{n}'
                    bank=read(bf/'records.json');bindings=read(bf/'bindings.json')
                    vectors=torch.load(bf/'vectors.pt',map_location='cpu',weights_only=True)
                    bank_disk=sum((bf/f).stat().st_size for f in ['records.json','bindings.json','vectors.pt'])
                    cpu_payload=vectors.numel()*vectors.element_size();gpu_payload=0
                    bank_by_sid={r['sid']:r for r in bank}
                else:
                    sync();cold_arrival=time.monotonic()
                    prefix=tok.encode(bank_prefix(tok,[document(r) for r in sessions[:n]]),add_special_tokens=False)
                    required=len(prefix)+max(len(tok.encode(query_suffix(tok,questions[q]['question']),add_special_tokens=False)) for q in qids)+d['answer_cap']
                    capacity=dict(prefix_tokens=len(prefix),required_tokens=required,context_limit=model.config.max_position_embeddings,
                                  calculated_prefix_payload_bytes=len(prefix)*kv_bytes_per_token(model.config))
                    extra['capacity']=capacity;bank_disk=0;cpu_payload=0;gpu_payload=None
                    if required>model.config.max_position_embeddings:
                        available=False;reason='context_limit'
                    else:
                        status(folder,'kv_prefill',n=n,repeat=repeat,tokens=len(prefix))
                        try:
                            cache=prefill(model,prefix,chunk_size=d['prefill_chunk'])
                            sync();gpu_payload=cache_payload_bytes(cache)
                            if gpu_payload!=capacity['calculated_prefix_payload_bytes']:raise ValueError('Incorrect KV payload')
                            extra['prefix_setup_seconds']=time.monotonic()-cold_arrival
                        except torch.OutOfMemoryError:
                            available=False;reason='gpu_limit';cache=None;clean()
                    del prefix
                if available:
                    for qi,qid in enumerate(expected):
                        # Only the literal question enters the parametric models.
                        question=questions[qid]['question']
                        status(folder,'query',n=n,repeat=repeat,completed=qi,total=len(expected),qid=qid)
                        sync();arrival=time.monotonic();observer=FirstAnswerClock(tok,arrival)
                        selected=None;ranking=None
                        if method=='our_delta':
                            keys=capture_keys(model,tok,names,[question])
                            ranking=sorted([(score(p,keys),sid) for sid,p in payloads.items()],key=lambda x:(-x[0],x[1]))
                            selected=ranking[0][1]
                            restore(model,base)
                            p=canonical(torch.load(files[selected],map_location='cpu',weights_only=True))
                            try:
                                with torch.no_grad():
                                    for name,(u,v) in p.items():
                                        if u.shape[1]==1:
                                            delta=u[:,0].to(model.device)[:,None]*v[:,0].to(model.device)[None,:]
                                        else:delta=u.to(model.device)@v.to(model.device).T
                                        model.get_parameter(name).add_(delta)
                                    del delta
                                with observe_generate(model,observer):final=generate_answer_batch(model,tok,[question],cap=d['answer_cap'])[0]
                            finally:restore(model,base)
                            del p
                        elif method=='plain_alphaedit':
                            with observe_generate(model,observer):final=generate_answer_batch(model,tok,[question],cap=d['answer_cap'])[0]
                        elif method=='rag':
                            query=embed(encoder,etok,[question])[0].cpu()
                            scores=(vectors@query).tolist()
                            indices=sorted(range(len(scores)),key=lambda i:(-scores[i],bindings[i]['sid']))
                            ranked_sids=[]
                            for i in indices:
                                sid=bindings[i]['sid']
                                if sid not in ranked_sids:ranked_sids.append(sid)
                            selected=ranked_sids[0];ranking=[(0,s) for s in ranked_sids]
                            context=tok.encode(bank_prefix(tok,[document(bank_by_sid[selected])]),add_special_tokens=False)
                            suffix=tok.encode(query_suffix(tok,question),add_special_tokens=False)
                            if len(context)+len(suffix)+d['answer_cap']>model.config.max_position_embeddings:raise ValueError('RAG context limit')
                            query_cache=prefill(model,context,chunk_size=d['prefill_chunk'])
                            final,query_cache=decode_cached_observed(model,tok,query_cache,suffix,observer,cap=d['answer_cap'])
                            query_cache=None
                        else:
                            suffix=tok.encode(query_suffix(tok,question),add_special_tokens=False)
                            final,cache=decode_cached_observed(model,tok,cache,suffix,observer,cap=d['answer_cap'])
                        sync();seconds=time.monotonic()-arrival
                        # Evaluation labels are consulted only AFTER the measured request.
                        label=questions[qid];target=label['answer'];support=set(label['support_sids'])
                        ranked=[s for _,s in ranking] if ranking is not None else None
                        rank=min((i+1 for i,s in enumerate(ranked) if s in support),default=None) if ranked is not None else None
                        source_hit=(selected in support) if selected is not None and not label['abstention'] else None
                        top=set(ranked[:6]) if ranked is not None else None
                        trial=dict(qid=qid,question=question,reference=target,question_type=label['question_type'],
                                   original64=label['original64'],abstention=label['abstention'],final=final,
                                   first_checkpoint=born[qid],
                                   query_seconds=seconds,repeat=repeat,n=n,selected_sid=selected,
                                   source_hit1=source_hit,correct_rank=rank if source_hit is not None else None,
                                   reciprocal_rank=1/rank if rank and source_hit is not None else None,
                                   support_recall6=len(top&support)/len(support) if top is not None and not label['abstention'] else None,
                                   complete_support6=support<=top if top is not None and not label['abstention'] else None,
                                   exact=normalize(final['text'])==normalize(target),token_f1=token_f1(final['text'],target),
                                   contains_reference=reference_tokens_present(final['text'],target),
                                   output_tokens=final['tokens'],hit_cap=final['hit_cap'],**observer.result())
                        trials.append(trial)
                        if method=='kv_full' and qi==0:
                            extra['cold_prefix_first_visible_seconds']=(observer.first_visible_at-cold_arrival
                                                                      if observer.first_visible_at is not None else None)
                        save(folder/f'bank-{n}.partial.json',dict(trials=trials,completed=False))
                    if len(trials)!=len(qids) or {t['qid'] for t in trials}!=set(qids):raise ValueError('Incomplete checkpoint query grid')
                sync()
                resource=dict(extra_disk_bank_bytes=bank_disk,retained_cpu_payload_bytes=cpu_payload,retained_gpu_prefix_bytes=gpu_payload,
                              base_model_parameter_bytes=base_bytes,encoder_parameter_bytes=encoder_bytes,
                              peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),**extra)
            resource.update(host.result())
            groups={}
            if trials:
                for group in ['all','original64','new','new_at_checkpoint','answerable','abstention',*sorted({t['question_type'] for t in trials})]:
                    ts=[t for t in trials if group=='all' or (group=='original64' and t['original64']) or
                        (group=='new' and not t['original64']) or (group=='answerable' and not t['abstention']) or
                        (group=='new_at_checkpoint' and t['first_checkpoint']==n) or
                        (group=='abstention' and t['abstention']) or group==t['question_type']]
                    if ts:groups[group]=checkpoint_metrics(ts)
            save(path,dict(completed=True,available=available,reason=reason,physical_gpu=1,method=method,repeat=repeat,n=n,
                           planned_questions=len(qids),trials=trials,resources=resource,groups=groups,finished_utc=now()))
            partial=folder/f'bank-{n}.partial.json'
            if partial.exists():partial.unlink()
            print('QA_CHECKPOINT',method,repeat,n,'questions',len(trials),'F1',groups.get('all',{}).get('mean_f1'),reason,flush=True)
            cache=None;payloads=None;vectors=None;bank=None;bindings=None;clean()
        check(out)
        if all((folder/f'bank-{n}.json').exists() for n in d['sizes']):
            save(folder/'completion.json',dict(completed=True,physical_gpu=1,repeat=repeat,method=method,finished_utc=now()))
        status(folder,'completed_requested_checkpoints',repeat=repeat,sizes=sizes)
    finally:
        if base is not None:restore(model,base)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--repeat',type=int,required=True);p.add_argument('--sizes',type=int,nargs='+')
    a=p.parse_args();run(a.out.resolve(),a.method,a.repeat,a.sizes)
