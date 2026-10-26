"""GPU 1 only: sequential official WISE writes, then ten matched inference passes."""
import argparse
import gc
from pathlib import Path
import statistics
import time

from .core import (ROOT,SIZES,load,read,save,digest,status,now,check,tokens_for_binding,
                   atomic_torch,export_state,install_state,rng_state,restore_rng,order,summary,HostPeak,tensor_leaves,vendor_modules)


def build(out,until=500):
    construction=out/'construction.json'
    if construction.exists() and read(construction).get('conversations',0)>=until:return
    d,torch,tok,editor=load(out,out/'writing')
    sessions=read(out/'sessions.json');locality=read(out/'locality.json');ledger=read(out/'ledger.json')
    start=0
    if (out/'restart.pt').exists():
        cp=torch.load(out/'restart.pt',map_location='cpu',weights_only=True)
        install_state(editor,cp['state']);a=editor.get_adapter_layer();a.weight_mask=cp['mask'].to(a.new_weight.device,dtype=torch.int64)
        official,_=vendor_modules();official.edit_history=cp['edit_history'];official.merge_group_edit_history=cp['merge_history']
        restore_rng(torch,cp['rng']);start=cp['state']['conversations'];ledger=ledger[:start];save(out/'ledger.json',ledger);del cp
    for index in range(start,until):
        record=sessions[index];status(out/'writing','writing',completed=index,total=500,sid=record['sid'])
        begin=time.monotonic();bindings=[]
        for j,binding in enumerate(record['bindings']):
            loc=locality[(index*2+j)%len(locality)]
            tokens,act_mask,deact_mask=tokens_for_binding(tok,record,binding,loc['text'],editor.device)
            if tokens['input_ids'].shape[1]>editor.model.config.max_position_embeddings:
                raise ValueError('Full write exceeds capacity; refuse truncation')
            target=tok.encode(' '+binding['answer']+tok.eos_token,add_special_tokens=False)
            for row in tokens['labels'][:2]:
                if row[row!=-100].tolist()!=target:raise ValueError('Complete official answer was not preserved')
            if index==0:save(out/'first_write_contract.json',dict(qid=binding['qid'],context_sha256=__import__('hashlib').sha256(__import__('json').dumps(record['context'],ensure_ascii=False).encode()).hexdigest(),
                target_token_count=len(target),target_sha256=__import__('hashlib').sha256(binding['answer'].encode()).hexdigest(),complete_context=True,complete_reference_answer=True,
                rows=int(tokens['input_ids'].shape[0]),sequence_tokens=int(tokens['input_ids'].shape[1])))
            editor.edit(editor.config,tokens,act_mask=act_mask,deact_mask=deact_mask)
            if not torch.isfinite(editor.get_adapter_layer().new_weight).all():raise ValueError('Nonfinite WISE memory')
            bindings.append(dict(qid=binding['qid'],input_tokens=int(tokens['input_ids'].shape[1]),target_tokens=len(target),
                                 ft_loss=editor.last_ft_loss,locality_id=loc['id']))
            del tokens,act_mask,deact_mask
        a=editor.get_adapter_layer()
        for name in ['original_layer_output','new_weight_layer_output']:
            if hasattr(a,name):delattr(a,name)
        n=index+1;ledger.append(dict(index=index,sid=record['sid'],bindings=bindings,internal_seconds=time.monotonic()-begin,
                                   editing_total_cnt=a.editing_total_cnt,memory_shards=len(a.memory_weight)))
        save(out/'ledger.json',ledger)
        if n in SIZES or n%16==0 or n==until:
            state=export_state(editor,n)
            if n in SIZES:
                path=out/'checkpoints'/f'wise-{n}.pt';atomic_torch(torch,path,state)
                snapshot=dict(conversations=n,bindings=state['bindings'],file=path.name,sha256=digest(path),
                              extra_disk_bank_bytes=path.stat().st_size,created_utc=now())
                save(out/'checkpoints'/f'wise-{n}.json',snapshot)
            official,_=vendor_modules()
            restart=dict(state=state,mask=a.weight_mask.detach().cpu().bool(),rng=rng_state(torch),
                         edit_history=official.edit_history,merge_history=official.merge_group_edit_history)
            atomic_torch(torch,out/'restart.pt',restart);del state,restart
        print('WISE_WRITE',n,'/500',round(ledger[-1]['internal_seconds'],2),'seconds','bindings',a.editing_total_cnt,flush=True)
        gc.collect();torch.cuda.empty_cache()
    save(construction,dict(conversations=until,completed=until==500,bindings=editor.get_adapter_layer().editing_total_cnt,
                            physical_gpu=1,checkpoint_sha256={str(n):read(out/'checkpoints'/f'wise-{n}.json')['sha256'] for n in SIZES if n<=until},finished_utc=now()))
    status(out/'writing','completed_requested_prefix',completed=until,total=500)


def evaluate(out,repeat,sizes=None):
    d=check(out);sizes=sizes or d['sizes'];folder=out/'wise'/f'run-{repeat:02d}'
    if all((folder/f'bank-{n}.json').exists() and read(folder/f'bank-{n}.json').get('completed') for n in sizes):return
    d,torch,tok,editor=load(out,folder)
    from answer_only_generation import generate_answer_batch
    from question_key import normalize
    from memory_comparison_core import token_f1
    from ttft_timing import FirstAnswerClock,observe_generate
    torch.manual_seed(repeat);torch.cuda.manual_seed_all(repeat)
    queries={q['qid']:q for q in read(out/'queries.json')}
    born={qid:p['n'] for p in reversed(d['plan']) for qid in p['qids']}
    for _ in range(2):generate_answer_batch(editor.model,tok,['Answer briefly.'],cap=1)
    base_bytes=sum(p.numel()*p.element_size() for p in editor.model.parameters())
    for n in sizes:
        path=folder/f'bank-{n}.json'
        if path.exists() and read(path).get('completed'):continue
        qids=next(p['qids'] for p in d['plan'] if p['n']==n);expected=order(qids,repeat,n)
        gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        status(folder,'checkpoint_setup',n=n,repeat=repeat,total=len(qids),completed=0)
        checkpoint=out/'checkpoints'/f'wise-{n}.pt';meta=read(out/'checkpoints'/f'wise-{n}.json')
        if digest(checkpoint)!=meta['sha256']:raise ValueError('Saved WISE checkpoint changed')
        trials=[]
        with HostPeak() as host:
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            if state['conversations']!=n:raise ValueError('Incorrect WISE prefix')
            install_state(editor,state);del state;gc.collect();torch.cuda.empty_cache()
            a=editor.get_adapter_layer()
            side_payload=sum(w.numel()*w.element_size() for w in [a.new_weight,*a.memory_weight])
            for index,qid in enumerate(expected):
                question=queries[qid]['question']
                status(folder,'query',n=n,repeat=repeat,completed=index,total=len(expected),qid=qid)
                torch.cuda.synchronize();start=time.monotonic();observer=FirstAnswerClock(tok,start)
                with observe_generate(editor.model,observer):final=generate_answer_batch(editor.model,tok,[question],cap=d['generation_cap'])[0]
                torch.cuda.synchronize();seconds=time.monotonic()-start
                # Ground truth is accessed only after final generation and timing.
                label=queries[qid]
                trials.append(dict(qid=qid,question=question,reference=label['answer'],repeat=repeat,n=n,final=final,
                    question_type=label['question_type'],original64=label['original64'],abstention=label['abstention'],first_checkpoint=born[qid],
                    token_f1=token_f1(final['text'],label['answer']),exact=normalize(final['text'])==normalize(label['answer']),
                    output_tokens=final['tokens'],hit_cap=final['hit_cap'],query_seconds=seconds,**observer.result()))
                save(folder/f'bank-{n}.partial.json',dict(completed=False,trials=trials))
            resources=dict(extra_disk_bank_bytes=checkpoint.stat().st_size,side_gpu_payload_bytes=side_payload,
                           peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                           base_and_registered_adapter_parameter_bytes=base_bytes,memory_shards=len(a.memory_weight),bindings=a.editing_total_cnt)
        resources.update(host.result())
        groups={}
        for group in ['all','original64','new','new_at_checkpoint','answerable','abstention',*sorted({t['question_type'] for t in trials})]:
            selected=[t for t in trials if group=='all' or (group=='original64' and t['original64']) or
                (group=='new' and not t['original64']) or (group=='new_at_checkpoint' and t['first_checkpoint']==n) or
                (group=='answerable' and not t['abstention']) or (group=='abstention' and t['abstention']) or group==t['question_type']]
            if selected:groups[group]=summary(selected)
        save(path,dict(completed=True,available=True,method='wise',repeat=repeat,n=n,physical_gpu=1,planned_questions=len(expected),
                       trials=trials,groups=groups,resources=resources,finished_utc=now()))
        (folder/f'bank-{n}.partial.json').unlink(missing_ok=True)
        print('WISE_CHECKPOINT',repeat,n,groups['all'],flush=True)
    status(folder,'completed_requested_checkpoints',repeat=repeat,sizes=sizes)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['build','eval']);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--until',type=int,default=500);parser.add_argument('--repeat',type=int,default=0);parser.add_argument('--sizes',type=int,nargs='+')
    args=parser.parse_args()
    if args.phase=='build':build(args.out.resolve(),args.until)
    else:evaluate(args.out.resolve(),args.repeat,args.sizes)
