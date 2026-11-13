"""GPU-1 index construction, routing preflight and genuine generated-answer passes."""
import argparse
import gc
from pathlib import Path
import time
from .core import (METHODS,read,save,digest,durable_save_torch,check,load,now,status,capture,generate,
                   normalized_key,ActivationRouter,read_bank,scorer,apply_payload,routing_summary,question_order,summary)


def names_for(d):return [f'model.layers.{i}.mlp.down_proj.weight' for i in d['layers']]


def index(out):
    done=out/'index/build.json'
    if done.exists() and read(done).get('completed'):return
    d,torch,tok,model=load(out,out/'index_build');names=names_for(d)
    bindings=read(out/'bindings.json');vectors={};began=time.monotonic()
    for i,b in enumerate(bindings):
        if b['prompt'] not in vectors:
            status(out/'index_build','capturing_write_key',completed=i,total=len(bindings))
            vectors[b['prompt']]=normalized_key(capture(d,model,tok,names,b['prompt']),names)
        if (i+1)%50==0:print('INDEX',d['dataset'],i+1,'/',len(bindings),flush=True)
    matrix=torch.stack([vectors[b['prompt']] for b in bindings]);del vectors
    manifest={}
    for n in d['sizes']:
        bank=set(d['bank_sids'][:n]);rows=[i for i,b in enumerate(bindings) if b['sid'] in bank]
        subset=matrix[rows].clone().contiguous();entries=[bindings[i] for i in rows]
        file=out/'index'/f'bank-{n}.pt'
        durable_save_torch(torch,file,dict(n=n,vectors=subset,index_sids=[b['sid'] for b in entries]))
        meta=dict(n=n,rows=len(rows),dimensions=subset.shape[1],sha256=digest(file),
                  file_bytes=file.stat().st_size,payload_bytes=subset.numel()*subset.element_size())
        save(file.with_suffix('.json'),meta);manifest[str(file.relative_to(out))]=meta['sha256']
        del subset
    save(out/'index/manifest.json',manifest)
    save(done,dict(completed=True,memories=len(d['bank_sids']),binding_keys=len(bindings),
                   distinct_questions=len({b['prompt'] for b in bindings}),elapsed_seconds=time.monotonic()-began,
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),finished_utc=now()))
    print('INDEX_READY',d['dataset'],read(done),flush=True)


def load_index(out,torch,n):
    file=out/'index'/f'bank-{n}.pt';meta=read(file.with_suffix('.json'))
    if digest(file)!=meta['sha256']:raise ValueError('Activation index changed')
    state=torch.load(file,map_location='cpu',weights_only=True)
    if state['n']!=n:raise ValueError('Wrong activation index prefix')
    return state,meta


def warmup(d,model,tok):
    # Match the original two one-token warmups without changing measured caps.
    warm=dict(d,answer_cap=1)
    for _ in range(2):generate(warm,model,tok,'Answer briefly.')


def preflight(out):
    done=out/'preflight/routing.json'
    if done.exists() and read(done).get('completed'):return
    d,torch,tok,model=load(out,out/'preflight');names=names_for(d);n=d['sizes'][-1]
    bank,_,_=read_bank(d,torch,model,names,n);state,_=load_index(out,torch,n)
    router=ActivationRouter(bank,names,state['vectors'],d['bank_sids'][:n],index_sids=state['index_sids'],scorer=scorer(d))
    requests={r['case_id']:r for r in read(out/'requests.json')};ids=question_order(d,n,0)
    rows=[];warmup(d,model,tok)
    for i,case_id in enumerate(ids):
        status(out/'preflight','routing_query',completed=i,total=len(ids),case_id=case_id)
        keys=capture(d,model,tok,names,requests[case_id]['prompt'])
        for k in d['ks']:
            result=router.route(keys,k)
            label=requests[case_id];support=set(label['answer_sids']);valid=not label['abstention']
            rows.append(dict(case_id=case_id,k=k,selected=result['selected'],
                source_hit=result['selected'] in support if valid else None,
                shortlist_hit=bool(support&set(result['shortlist'])) if valid else None,
                coarse_rank=min((j+1 for j,sid in enumerate(result['coarse_order']) if sid in support),default=None) if valid else None))
        if (i+1)%25==0:
            save(out/'preflight/routing.partial.json',dict(completed=False,trials=rows))
            print('PREFLIGHT',d['dataset'],i+1,'/',len(ids),flush=True)
    results={}
    for k in d['ks']:
        ts=[r for r in rows if r['k']==k];valid=[r for r in ts if r['source_hit'] is not None]
        results[str(k)]=dict(questions=len(ts),routing_questions=len(valid),
            correct=sum(r['source_hit'] for r in valid),routing_accuracy=sum(r['source_hit'] for r in valid)/len(valid),
            shortlist_recall=sum(r['shortlist_hit'] for r in valid)/len(valid))
    save(done,dict(completed=True,preliminary=True,n=n,summary=results,trials=rows,
        note='One fresh query-activation pass per question, reused over predeclared k values. Routing only; no answer F1 or TTFT measured here. Exact stored questions.',finished_utc=now()))
    (out/'preflight/routing.partial.json').unlink(missing_ok=True)
    print('PREFLIGHT_READY',d['dataset'],results,flush=True)


def evaluate(out,method,repeat,sizes=None):
    d=check(out);sizes=sizes or d['sizes'];folder=out/method/f'run-{repeat:02d}'
    if all((folder/f'bank-{n}.json').exists() and read(folder/f'bank-{n}.json')['completed'] for n in sizes):return
    d,torch,tok,model=load(out,folder);torch.manual_seed(repeat);torch.cuda.manual_seed_all(repeat)
    from experiments.wise_qa500.core import HostPeak
    from memory_comparison_core import token_f1
    from question_key import normalize
    from ttft_timing import FirstAnswerClock,observe_generate
    names=names_for(d);base={name:model.get_parameter(name).detach().cpu().clone() for name in names}
    def restore():
        with torch.no_grad():
            for name,value in base.items():model.get_parameter(name).copy_(value.to(model.device))
    requests={r['case_id']:r for r in read(out/'requests.json')};warmup(d,model,tok)
    k=None if method=='full_scan' else int(method.split('_')[1])
    try:
        for n in sizes:
            file=folder/f'bank-{n}.json'
            if file.exists() and read(file).get('completed'):continue
            bank=None;state=None;router=None;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            ids=question_order(d,n,repeat);status(folder,'bank_setup',n=n,repeat=repeat,completed=0,total=len(ids))
            trials=[]
            with HostPeak() as host:
                bank,factor_disk,factor_bytes=read_bank(d,torch,model,names,n)
                index_disk=0;index_bytes=0
                if k is not None:
                    state,meta=load_index(out,torch,n);index_disk=meta['file_bytes'];index_bytes=meta['payload_bytes']
                router=ActivationRouter(bank,names,None if state is None else state['vectors'],d['bank_sids'][:n],
                    index_sids=None if state is None else state['index_sids'],scorer=scorer(d))
                files={r['sid']:Path(r['path']) for r in d['factor_files'][:n]}
                gc.collect();torch.cuda.empty_cache()
                for qi,case_id in enumerate(ids):
                    question=requests[case_id]['prompt']
                    status(folder,'query',n=n,repeat=repeat,completed=qi,total=len(ids),case_id=case_id)
                    torch.cuda.synchronize();arrival=time.monotonic();observer=FirstAnswerClock(tok,arrival)
                    # Preserve original restore placement for each dataset.
                    if d['dataset']=='zsre':restore()
                    keys=capture(d,model,tok,names,question);result=router.route(keys,k)
                    routed=time.monotonic()-arrival
                    if d['dataset']=='longmemeval':restore()
                    payload=torch.load(files[result['selected']],map_location='cpu',weights_only=True)
                    try:
                        apply_payload(torch,model,payload)
                        with observe_generate(model,observer):final=generate(d,model,tok,question)
                    finally:restore()
                    torch.cuda.synchronize();elapsed=time.monotonic()-arrival;del payload
                    # Expected IDs/answers enter only after routing and generation.
                    label=requests[case_id];support=set(label['answer_sids']);valid=not label['abstention']
                    source_hit=result['selected'] in support if valid else None
                    shortlist_hit=bool(set(result['shortlist'])&support) if valid and result['shortlist'] is not None else None
                    coarse=min((j+1 for j,s in enumerate(result['coarse_order'] or []) if s in support),default=None) if valid else None
                    fine=min((j+1 for j,s in enumerate(result['fine_order']) if s in support),default=None) if valid else None
                    trials.append(dict(case_id=case_id,question=question,reference=label['target_new'],n=n,repeat=repeat,
                        final=final,token_f1=token_f1(final['text'],label['target_new']),exact=normalize(final['text'])==normalize(label['target_new']),
                        source_hit=source_hit,selected=result['selected'],shortlist_hit=shortlist_hit,coarse_rank=coarse,fine_rank=fine,
                        shortlisted_sids=result['shortlist'],selected_score=result['score'],fine_margin=result['margin'],
                        query_seconds=elapsed,routing_seconds=routed,output_tokens=final['tokens'],hit_cap=final['hit_cap'],**observer.result()))
                    if (qi+1)%10==0 or qi+1==len(ids):save(folder/f'bank-{n}.partial.json',dict(completed=False,trials=trials))
                resources=dict(extra_disk_bank_bytes=factor_disk+index_disk,factor_bank_disk_bytes=factor_disk,key_index_disk_bytes=index_disk,
                    cpu_factor_payload_bytes=factor_bytes,cpu_key_index_payload_bytes=index_bytes,
                    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
            resources.update(host.result())
            save(file,dict(completed=True,available=True,physical_gpu=1,method=method,repeat=repeat,n=n,trials=trials,
                groups={'all':summary(trials)},routing=routing_summary(trials),resources=resources,finished_utc=now()))
            (folder/f'bank-{n}.partial.json').unlink(missing_ok=True)
            print('TOPK_CHECKPOINT',d['dataset'],method,repeat,n,'questions',len(ids),'routing',routing_summary(trials),'F1',summary(trials)['mean_f1'],flush=True)
            bank=None;state=None;router=None;gc.collect();torch.cuda.empty_cache()
    finally:restore()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['index','preflight','evaluate']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--method',choices=METHODS);p.add_argument('--repeat',type=int,default=0);p.add_argument('--sizes',nargs='+',type=int)
    a=p.parse_args();out=a.out.resolve()
    if a.action=='evaluate':evaluate(out,a.method,a.repeat,a.sizes)
    else:globals()[a.action](out)
