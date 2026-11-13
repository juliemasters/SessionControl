"""Extend the existing 64 full-Q/A memories; never reuse evidence-only weights."""
import argparse
import gc
from pathlib import Path
import time

from comparison_core import read,save,digest,durable_save_torch
from qa500_core import CODE,ROOT,SIZES,load,check,storage_check,require_disk,status,now,answer_requests


def build(out,phase):
    folder=out/phase
    if (folder/'completion.json').exists() and read(folder/'completion.json').get('completed'):return
    storage_check(out)
    d,torch,tok,model=load(out,folder)
    from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
    import AlphaEdit.AlphaEdit_main as writer
    from full500_core import compact_write,restore
    hp=AlphaEditHyperParams.from_json(str(CODE/'config/alphaedit-hparams.json'))
    hp.layers=d['layers'];hp.L2=d['l2'];hp.v_num_grad_steps=d['write_steps']
    writer.CONTEXT_TEMPLATES_CACHE=[['{}']]
    names=[f'model.layers.{l}.mlp.down_proj.weight' for l in d['layers']]
    P=torch.load(d['projection'],map_location='cpu',weights_only=True)
    sessions=read(out/'sessions.json');ledger=read(folder/'ledger.json')
    if phase=='independent':
        base={n:model.get_parameter(n).detach().cpu().clone() for n in names}
        start=len(ledger)
        for row in ledger:
            if digest(folder/'factors'/row['file'])!=row['sha256']:raise ValueError('Committed answer delta changed')
    else:
        accumulator=torch.zeros_like(P)
        candidates=[(n,folder/f'weights-{n}.pt') for n in SIZES if (folder/f'weights-{n}.pt').exists()]
        if (folder/'restart.pt').exists():
            restart=torch.load(folder/'restart.pt',map_location='cpu',weights_only=True)
            candidates.append((restart['count'],folder/'restart.pt'));del restart
        start,path=max(candidates,key=lambda p:p[0])
        cp=torch.load(path,map_location='cpu',weights_only=True)
        if cp['count']!=start:raise ValueError('Incorrect checkpoint count')
        restore(model,cp['weights']);del cp
        ledger=ledger[:start];save(folder/'ledger.json',ledger)
        for i in range(start):
            keys=torch.load(folder/'post_keys'/f'{i}.pt',map_location='cpu',weights_only=True)
            for layer,k in enumerate(keys):
                accumulator[layer]+=k[:,None]@k[None,:] if k.ndim==1 else k.T@k
        if float(accumulator.norm())!=ledger[-1]['cache_norm']:
            raise ValueError('Exact preservation accumulator restart check failed')
        save(folder/'resume.json',dict(start=start,cache_norm=float(accumulator.norm()),exact=True))
        captured=[];original=writer.compute_ks
        def tracked(*args,**kwargs):
            value=original(*args,**kwargs);captured.append(value.detach().cpu().clone());return value
        writer.compute_ks=tracked
    try:
        for index in range(start,500):
            record=sessions[index];status(folder,'writing',completed=index,total=500,reused=64,sid=record['sid'])
            reqs=answer_requests(record,tok,index)
            for req,b in zip(reqs,record['bindings']):
                if req['target_new']['str']!=b['answer']+tok.eos_token:raise ValueError('Actual reference answer must be the target')
            if index==64:save(folder/'first_new_write_requests.json',reqs)
            begin=time.monotonic()
            if phase=='independent':
                payload=compact_write(model,tok,reqs,hp,P,base)
                if any(u.ndim!=2 or v.ndim!=2 or u.shape[1]!=v.shape[1] or u.shape[1]>2 for u,v in payload.values()):
                    raise ValueError('Only compact rank-one/two factors may be serialized')
                size=sum(t.numel()*t.element_size() for pair in payload.values() for t in pair)
                if size>512*1024:raise ValueError('Per-session payload exceeds compact storage limit')
                require_disk(out,size+65536);file=f'{index:04d}.pt'
                durable_save_torch(torch,folder/'factors'/file,payload)
                entry=dict(index=index,sid=record['sid'],file=file,sha256=digest(folder/'factors'/file),
                           bindings=len(reqs),reused=False,internal_seconds=time.monotonic()-begin)
            else:
                captured.clear()
                model,accumulator=writer.apply_AlphaEdit_to_model(model,tok,reqs,hp,cache_c=accumulator,P=P)
                if len(captured)!=6 or not all(torch.isfinite(model.get_parameter(n)).all() for n in names):
                    raise ValueError('Invalid cumulative AlphaEdit state')
                durable_save_torch(torch,folder/'post_keys'/f'{index}.pt',captured[-3:])
                entry=dict(index=index,sid=record['sid'],bindings=len(reqs),cache_norm=float(accumulator.norm()),
                           reused=False,internal_seconds=time.monotonic()-begin)
            ledger.append(entry);save(folder/'ledger.json',ledger)
            n=index+1
            if phase=='shared' and (n%32==0 or n in SIZES):
                require_disk(out,sum(model.get_parameter(name).numel()*4 for name in names)+65536)
                state=dict(count=n,weights={name:model.get_parameter(name).detach().cpu().clone() for name in names})
                durable_save_torch(torch,folder/'restart.pt',state)
                if n in SIZES:durable_save_torch(torch,folder/f'weights-{n}.pt',state)
                del state
            print('QA_WRITE',phase,n,'/500',round(entry['internal_seconds'],2),'seconds',flush=True)
            gc.collect();torch.cuda.empty_cache()
        check(out)
        result=dict(completed=True,physical_gpu=1,conversations=500,reused=64,new=436,finished_utc=now(),
                    write_target='COMPLETE OFFICIAL ANSWER + EOS')
        if phase=='shared':result['patch_sha256']={str(n):digest(folder/f'weights-{n}.pt') for n in SIZES}
        save(folder/'completion.json',result);status(folder,'completed',completed=500,total=500,reused=64)
    except BaseException as error:
        save(folder/'completion.json',dict(completed=False,error=repr(error),physical_gpu=1));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=['independent','shared'],required=True);a=p.parse_args()
    build(a.out.resolve(),a.phase)
