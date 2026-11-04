"""Build missing cumulative AlphaEdit and native WISE checkpoint trajectories."""
import argparse
import gc
from pathlib import Path
import random
import time
from .core import (VENDOR,SIZES,read,save,digest,durable_save_torch,status,now,load,modules,cumulative_right,wise_state,finalized_merge_snapshot)


def plain(out,until):
    folder=out/'plain_alphaedit';done=folder/'construction.json'
    if done.exists() and read(done)['count']>=until:return
    d,torch,tok,model=load(out,folder/'writing')
    from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
    from AlphaEdit.compute_z import compute_z,get_module_input_output_at_words
    from AlphaEdit.compute_ks import compute_ks
    from experiments.session_zsre.core import raw_write_request
    hp=AlphaEditHyperParams(**d['alpha_hparams']);hp.model_name='Mistral-7B-v0.1'
    names=[hp.rewrite_module_tmp.format(layer)+'.weight' for layer in hp.layers]
    P=[]
    for layer in hp.layers:
        file=Path(d['projection'])/f'layer-{layer}.pt'
        if digest(file)!=d['projector']['files'][file.name]:raise ValueError('Changed projector')
        P.append(torch.load(file,map_location='cuda:0',weights_only=True))
    previous=[torch.empty((model.config.intermediate_size,0),device=model.device) for _ in hp.layers]
    start=0;ledger=read(folder/'ledger.json') if (folder/'ledger.json').exists() else []
    restart=folder/'restart.pt'
    if restart.exists():
        cp=torch.load(restart,map_location='cpu',weights_only=True);start=cp['count'];ledger=ledger[:start]
        with torch.no_grad():
            for name,value in cp['weights'].items():model.get_parameter(name).copy_(value.to(model.device))
        previous=[v.to(model.device) for v in cp['keys']];del cp
    requests=read(out/'requests.json')
    for i in range(start,until):
        status(folder/'writing','writing',completed=i,total=1000);began=time.monotonic()
        req=raw_write_request(requests[i],tok);req['target_new']['str']=' '+req['target_new']['str'].lstrip()
        target=compute_z(model,tok,req,hp,hp.layers[-1],[['{}']])
        for j,layer in enumerate(hp.layers):
            key=compute_ks(model,tok,[req],hp,layer,[['{}']])[0]
            current=get_module_input_output_at_words(model,tok,hp.layers[-1],context_templates=[req['prompt']],
                words=[req['subject']],module_template=hp.layer_module_tmp,fact_token_strategy=hp.fact_token)[1][0]
            residual=(target-current)/(len(hp.layers)-j)
            right=cumulative_right(P[j],key,previous[j],hp.L2)
            with torch.no_grad():model.get_parameter(names[j]).add_(residual.detach()[:,None]*right[None,:])
        for j,layer in enumerate(hp.layers):
            key=compute_ks(model,tok,[req],hp,layer,[['{}']])[0]
            previous[j]=torch.cat([previous[j],key[:,None]],dim=1)
        if any(not torch.isfinite(model.get_parameter(name)).all() for name in names):raise ValueError('Nonfinite shared weights')
        ledger.append(dict(case_id=i,internal_write_seconds=time.monotonic()-began));save(folder/'ledger.json',ledger)
        n=i+1
        if n in SIZES or n%25==0 or n==until:
            weights={name:model.get_parameter(name).detach().cpu().clone() for name in names}
            durable_save_torch(torch,restart,dict(count=n,weights=weights,keys=[v.cpu() for v in previous]))
            if n in SIZES:
                path=folder/f'weights-{n}.pt';durable_save_torch(torch,path,dict(count=n,weights=weights))
                save(folder/f'weights-{n}.json',dict(n=n,sha256=digest(path),bank_bytes=path.stat().st_size))
            del weights
        print('WRITE plain_alphaedit',n,'/1000',round(ledger[-1]['internal_write_seconds'],3),'seconds',flush=True)
        del target,current,key,residual,right;gc.collect();torch.cuda.empty_cache()
    save(done,dict(count=until,completed=until==1000,finished_utc=now()))


def wise(out,mode,until):
    folder=out/mode;done=folder/'construction.json'
    if done.exists() and read(done)['count']>=until:return
    d,torch,tok,model=load(out,folder/'writing');native=modules()
    if tok.padding_side!='left':raise ValueError('Native WISE must match the validated left-padding convention')
    from experiments.wise_paper.native import restore
    from experiments.wise_paper.worker import native_save,numpy_state,tuple_tree
    import numpy as np
    hp=native['wise_hparams'].WISEHyperParams.from_hparams(str(VENDOR/'hparams/WISE/mistral-7b.yaml'))
    hp.model_name=d['model'];hp.device=0;hp.retrieve=mode=='wise_retrieve'
    native['utils'].CONTEXT_TEMPLATES_CACHE=read(out/'context_templates.json')
    start=0;editor=None;restart=folder/'restart.pt';ledger=read(folder/'ledger.json') if (folder/'ledger.json').exists() else []
    if restart.exists():
        editor=native['WISE'].WISE(hp,model,'cuda:0');start=restore(editor,restart,torch);ledger=ledger[:start]
        rng=read(folder/'rng.json');random.setstate(tuple_tree(rng['python']))
        np.random.set_state((rng['numpy'][0],np.array(rng['numpy'][1],dtype='uint32'),*rng['numpy'][2:]))
        states=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True)
        torch.set_rng_state(states['cpu']);torch.cuda.set_rng_state_all(states['cuda'])
    requests=read(out/'requests.json')
    for i in range(start,until):
        status(folder/'writing','writing',completed=i,total=1000);began=time.monotonic()
        editor,_=native['wise_main'].apply_wise_to_model(model,tok,[requests[i]],hp,copy=False)
        a=editor.get_adapter_layer()
        if a.editing_total_cnt!=i+1 or not torch.isfinite(a.new_weight).all():raise ValueError('Invalid native edit state')
        ledger.append(dict(case_id=i,internal_write_seconds=time.monotonic()-began));save(folder/'ledger.json',ledger)
        n=i+1
        if n in SIZES:
            if n==500 and mode=='wise_merge':
                # Native finalization applies only to the evaluation snapshot.
                # The helper restores ordinary tensors before training resumes.
                deployment=finalized_merge_snapshot(editor,n,tok('Memory initialization.',return_tensors='pt').to(model.device))
            else:deployment=wise_state(editor,n)
            path=folder/f'wise-{n}.pt';durable_save_torch(torch,path,deployment)
            save(folder/f'wise-{n}.json',dict(n=n,sha256=digest(path),bank_bytes=path.stat().st_size))
            del deployment
        if n in SIZES or n%25==0 or n==until:
            native_save(editor,restart,torch)
            save(folder/'rng.json',dict(python=random.getstate(),numpy=numpy_state(np)))
            durable_save_torch(torch,folder/'rng.pt',dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()))
        print('WRITE',mode,n,'/1000',round(ledger[-1]['internal_write_seconds'],3),'seconds',flush=True)
        gc.collect();torch.cuda.empty_cache()
    save(done,dict(count=until,completed=until==1000,finished_utc=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--method',choices=['plain_alphaedit','wise_merge','wise_retrieve'],required=True);p.add_argument('--until',type=int,default=1000)
    a=p.parse_args()
    if a.method=='plain_alphaedit':plain(a.out.resolve(),a.until)
    else:wise(a.out.resolve(),a.method,a.until)
