"""Six matched free-generation conditions with the original timing/memory metrics."""
import argparse
import gc
from pathlib import Path
import time
from .core import (SIZES,METHODS,VENDOR,read,save,digest,status,now,check,load,modules,install_wise,
                   document,prefix,suffix,generate_raw,order,summary)


def evaluate(out,method,repeat,sizes=None):
    folder=out/method/f'run-{repeat:02d}';d=check(out);sizes=sizes or d['sizes']
    if all((folder/f'bank-{n}.json').exists() and read(folder/f'bank-{n}.json').get('completed') for n in sizes):return
    d,torch,tok,model=load(out,folder);torch.manual_seed(repeat);torch.cuda.manual_seed_all(repeat)
    from compact_session_memory import CompactSessionMemory
    from experiments.session_zsre.core import capture_query_keys,select_candidate
    from experiments.wise_qa500.core import HostPeak
    from memory_comparison_core import embed,prefill,cache_payload_bytes,kv_bytes_per_token,token_f1
    from question_key import normalize
    from ttft_timing import FirstAnswerClock,observe_generate,decode_cached_observed
    requests=read(out/'requests.json');ledger=read(out/'factor_ledger.json')
    names=[f'model.layers.{layer}.mlp.down_proj.weight' for layer in d['layers']]
    memory=CompactSessionMemory(model,names,out/'factors') if method=='our_delta' else None
    editor=None;encoder=None;etok=None
    if method.startswith('wise_'):
        native=modules();hp=native['wise_hparams'].WISEHyperParams.from_hparams(str(VENDOR/'hparams/WISE/mistral-7b.yaml'))
        hp.model_name=d['model'];hp.device=0;hp.retrieve=method=='wise_retrieve'
        editor=native['WISE'].WISE(hp,model,'cuda:0')
    if method=='rag':
        from transformers import AutoTokenizer,AutoModel
        etok=AutoTokenizer.from_pretrained(d['embedding_model'],local_files_only=True)
        encoder=AutoModel.from_pretrained(d['embedding_model'],local_files_only=True).to(model.device).float().eval()
        manifest=out/'rag/index_manifest.json'
        if not manifest.exists():
            vectors=embed(encoder,etok,[r['prompt'] for r in requests]).cpu();hashes={}
            for n in SIZES:
                path=out/'rag'/f'index-{n}';path.mkdir(exist_ok=True)
                records=[{k:r[k] for k in ['case_id','prompt','target_new']} for r in requests[:n]]
                save(path/'records.json',records);torch.save(vectors[:n].clone().contiguous(),path/'vectors.pt')
                for file in path.iterdir():hashes[str(file.relative_to(out/'rag'))]=digest(file)
            save(manifest,hashes);del vectors
        for name,want in read(manifest).items():
            if digest(out/'rag'/name)!=want:raise ValueError('RAG index changed')
        embed(encoder,etok,['Unrelated encoder warmup.'])
    for _ in range(2):generate_raw(model,tok,'Answer briefly.',1)
    for n in sizes:
        path=folder/f'bank-{n}.json'
        if path.exists() and read(path).get('completed'):continue
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        status(folder,'bank_setup',n=n,repeat=repeat,completed=0,total=n)
        cache=None;bank=None;vectors=None;records=None;available=True;reason=None;extra={};trials=[]
        with HostPeak() as host:
            if method=='our_delta':
                bank={}
                for row in ledger[:n]:
                    f=out/'factors'/(row['sid']+'.pt')
                    if digest(f)!=row['sha256']:raise ValueError('Saved factor changed')
                    bank[row['sid']]=memory.load(row['sid'])
                disk=sum((out/'factors'/(row['sid']+'.pt')).stat().st_size for row in ledger[:n])
            elif method=='plain_alphaedit':
                cp=out/method/f'weights-{n}.pt';meta=read(cp.with_suffix('.json'))
                if digest(cp)!=meta['sha256']:raise ValueError('Shared snapshot changed')
                state=torch.load(cp,map_location='cpu',weights_only=True)
                if state['count']!=n:raise ValueError('Wrong shared prefix')
                with torch.no_grad():
                    for name,value in state['weights'].items():model.get_parameter(name).copy_(value.to(model.device))
                disk=cp.stat().st_size;del state
            elif method.startswith('wise_'):
                cp=out/method/f'wise-{n}.pt';meta=read(cp.with_suffix('.json'))
                if digest(cp)!=meta['sha256']:raise ValueError('WISE snapshot changed')
                state=torch.load(cp,map_location='cpu',weights_only=True)
                if state['n']!=n:raise ValueError('Wrong WISE prefix')
                install_wise(editor,state);disk=cp.stat().st_size;del state
            elif method=='rag':
                root=out/'rag'/f'index-{n}';records=read(root/'records.json')
                vectors=torch.load(root/'vectors.pt',map_location='cpu',weights_only=True)
                disk=sum((root/f).stat().st_size for f in ['records.json','vectors.pt'])
            else:
                disk=0;start=time.monotonic()
                ids=tok.encode(prefix([document(r) for r in requests[:n]]))
                maxsuffix=max(len(tok.encode(suffix(r['prompt']),add_special_tokens=False)) for r in requests[:n])
                required=len(ids)+maxsuffix+d['answer_cap']
                extra['capacity']=dict(prefix_tokens=len(ids),required_tokens=required,context_limit=model.config.max_position_embeddings,
                    sliding_window=model.config.sliding_window,calculated_payload_bytes=len(ids)*kv_bytes_per_token(model.config))
                if required>model.config.max_position_embeddings:available=False;reason='context_limit'
                else:
                    try:
                        cache=prefill(model,ids,chunk_size=1024);torch.cuda.synchronize()
                        actual=cache_payload_bytes(cache)
                        if actual!=len(ids)*kv_bytes_per_token(model.config):raise ValueError('KV payload mismatch')
                        extra.update(prefix_payload_bytes=actual,prefix_setup_seconds=time.monotonic()-start)
                    except torch.OutOfMemoryError:
                        available=False;reason='gpu_limit';cache=None;gc.collect();torch.cuda.empty_cache()
                del ids
            gc.collect();torch.cuda.empty_cache()
            if available:
                for index,case_id in enumerate(order(n,repeat)):
                    query=requests[case_id]['prompt']
                    status(folder,'query',n=n,repeat=repeat,completed=index,total=n,case_id=case_id)
                    torch.cuda.synchronize();arrival=time.monotonic();observer=FirstAnswerClock(tok,arrival);selected=None;rank=None
                    if method=='our_delta':
                        memory.restore();keys=capture_query_keys(model,tok,names,query)
                        selected,_,_,ranks=select_candidate(bank,keys)
                        try:
                            # Retain the prior query-time factor read/application.
                            memory.apply(memory.load(selected))
                            with observe_generate(model,observer):final=generate_raw(model,tok,query,d['answer_cap'])
                        finally:memory.restore()
                        rank=ranks[f'case_{case_id:04d}']
                    elif method in ['plain_alphaedit','wise_merge','wise_retrieve']:
                        with observe_generate(model,observer):final=generate_raw(model,tok,query,d['answer_cap'])
                    elif method=='rag':
                        vector=embed(encoder,etok,[query])[0].cpu();scores=(vectors@vector).tolist()
                        chosen=min(range(n),key=lambda i:(-scores[i],i));selected=f'case_{chosen:04d}'
                        rank=sorted(range(n),key=lambda i:(-scores[i],i)).index(case_id)+1
                        ids=tok.encode(prefix([document(records[chosen])]))
                        tail=tok.encode(suffix(query),add_special_tokens=False)
                        query_cache=prefill(model,ids,chunk_size=1024)
                        final,query_cache=decode_cached_observed(model,tok,query_cache,tail,observer,cap=d['answer_cap']);query_cache=None
                    else:
                        tail=tok.encode(suffix(query),add_special_tokens=False)
                        final,cache=decode_cached_observed(model,tok,cache,tail,observer,cap=d['answer_cap'])
                    torch.cuda.synchronize();elapsed=time.monotonic()-arrival
                    # Only after inference/timing is the expected answer accessed.
                    reference=requests[case_id]['target_new']
                    trials.append(dict(case_id=case_id,question=query,reference=reference,repeat=repeat,n=n,final=final,
                        token_f1=token_f1(final['text'],reference),exact=normalize(final['text'])==normalize(reference),
                        output_tokens=final['tokens'],hit_cap=final['hit_cap'],query_seconds=elapsed,selected=selected,
                        correct_rank=rank,source_hit=None if selected is None else selected==f'case_{case_id:04d}',**observer.result()))
                    if (index+1)%10==0 or index+1==n:save(folder/f'bank-{n}.partial.json',dict(completed=False,trials=trials))
            resources=dict(extra_disk_bank_bytes=disk,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),**extra)
        resources.update(host.result())
        save(path,dict(completed=True,available=available,reason=reason,method=method,n=n,repeat=repeat,physical_gpu=1,
            trials=trials,groups={'all':summary(trials)} if available else {},resources=resources,finished_utc=now()))
        (folder/f'bank-{n}.partial.json').unlink(missing_ok=True)
        print('CHECKPOINT',method,repeat,n,summary(trials) if available else reason,flush=True)
        cache=None;bank=None;vectors=None;records=None;gc.collect();torch.cuda.empty_cache()
    status(folder,'completed_requested_checkpoints',repeat=repeat,sizes=sizes)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--repeat',type=int,required=True);p.add_argument('--sizes',nargs='+',type=int)
    a=p.parse_args();evaluate(a.out.resolve(),a.method,a.repeat,a.sizes)
