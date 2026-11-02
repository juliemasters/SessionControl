"""Write independent compact memories, then evaluate query-only activation routing."""
import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import statistics
import time
from experiments.wise_paper.native import ROOT, VENDOR, digest, evaluator, modules
from experiments.wise_paper.prepare import save
from .prepare import check


def now():return datetime.now(timezone.utc).isoformat()


def status(out,stage,**fields):
    save(out/'status.json',dict(stage=stage,updated_utc=now(),**fields))


def durable_torch(torch,path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temporary)
    with temporary.open('rb') as f:os.fsync(f.fileno())
    temporary.replace(path)


def load(out):
    d=check(out)
    from experiment_utils import require_gpu1
    uuid=require_gpu1()
    import numpy as np
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    if torch.cuda.device_count()!=1:raise RuntimeError('Only physical GPU 1 may be visible')
    torch.set_num_threads(8);torch.manual_seed(d['seed']);np.random.seed(d['seed']);random.seed(d['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(d['model'],local_files_only=True)
    tok.pad_token_id=tok.eos_token_id;tok.padding_side='right'
    model=AutoModelForCausalLM.from_pretrained(d['model'],torch_dtype=torch.float32,
        attn_implementation='sdpa',local_files_only=True).eval().to('cuda:0')
    from AlphaEdit.AlphaEdit_hparams import AlphaEditHyperParams
    hp=AlphaEditHyperParams(**d['hyperparameters']);hp.model_name='Mistral-7B-v0.1'
    native=modules()
    metric_hp=native['wise_hparams'].WISEHyperParams.from_hparams(str(VENDOR/'hparams/WISE/mistral-7b.yaml'))
    metric_hp.model_name=d['model'];metric_hp.device=0
    save(out/'provenance.json',dict(physical_gpu=1,gpu_uuid=uuid,torch_version=torch.__version__,
         write_hparams=hp.__dict__,metric_hparams=metric_hp.__dict__,code_sha256=d['code_sha256'],started_utc=now()))
    return d,torch,tok,model,hp,metric_hp


def projector(out,d,torch,tok,model,names):
    """Model-specific projector, using training-locality text only, never test data."""
    folder=out/'projection';folder.mkdir(exist_ok=True)
    metadata=folder/'manifest.json'
    if metadata.exists():
        info=json.loads(metadata.read_text())
        if info['corpus_sha256']!=d['corpus_sha256']:raise ValueError('Projector corpus changed')
        for name,want in info['files'].items():
            if digest(folder/name)!=want:raise ValueError('Projector changed')
        return [torch.load(folder/f'layer-{layer}.pt',map_location='cpu',weights_only=True) for layer in d['layers']]
    from .core import nullspace
    corpus=json.loads((out/'preservation_corpus.json').read_text())
    keys={name:[] for name in names};handles=[]
    for name in names:
        def capture(module,args,name=name):keys[name].append(args[0][0].detach().float().cpu())
        handles.append(model.get_submodule(name.removesuffix('.weight')).register_forward_pre_hook(capture))
    try:
        for i,text in enumerate(corpus):
            status(out,'projection_training_tokens',completed=i,total=len(corpus))
            x=tok(text,return_tensors='pt',truncation=False).to(model.device)
            with torch.inference_mode():model.model(**x,use_cache=False)
    finally:
        for h in handles:h.remove()
    projections=[];diagnostics=[]
    for layer,name in zip(d['layers'],names):
        status(out,'projection_svd',layer=layer)
        matrix=torch.cat(keys.pop(name)).to(model.device)
        result,detail=nullspace(matrix,d['hyperparameters']['nullspace_threshold'])
        diagnostics.append(dict(layer=layer,texts=len(corpus),**detail))
        cpu=result.cpu();durable_torch(torch,folder/f'layer-{layer}.pt',cpu);projections.append(cpu)
        print('PROJECTION',diagnostics[-1],flush=True)
        del matrix,result;gc.collect();torch.cuda.empty_cache()
    save(metadata,dict(corpus_sha256=d['corpus_sha256'],files={f'layer-{l}.pt':digest(folder/f'layer-{l}.pt') for l in d['layers']},
        method='I - V_protected V_protected.T; eig=X_singular^2/token_count; threshold=.02',diagnostics=diagnostics,
        limitation='Small 200-training-prompt empirical preservation corpus, not full Wikipedia AlphaEdit statistics.'))
    return projections


def baseline_validation(out,d,metric,model,tok,metric_hp,requests):
    from experiments.wise_paper.worker import score
    pre=json.loads((Path(d['reference_run'])/'baseline.json').read_text())
    if len(pre['rows'])!=1000 or pre['requests_sha256']!=d['requests_sha256']:
        raise ValueError('Incompatible native baseline')
    # Reuse the full original baseline, checking the first ten against this model
    # process, input interface and official evaluator before writing any delta.
    keys=['reliability_token_accuracy','generalization_token_accuracy','locality_tokens']
    for i in range(10):
        status(out,'native_baseline_parity',completed=i,total=10)
        row=score(metric,model,tok,metric_hp,requests[i])
        for key in keys:
            if row[key]!=pre['rows'][i][key]:raise ValueError('Unedited baseline parity failed: '+key)
    save(out/'baseline_validation.json',dict(exact=True,checked=10,full_reused=1000,
        baseline_sha256=d['reference_files']['baseline.json'],note='Same raw prompts, tokenizer, FP32 base weights and official metric.'))
    return pre['rows']


def routed_metric(memory,bank,metric,model,tok,metric_hp,query,target,*,locality=False):
    from .core import capture_query_keys,select_candidate
    memory.restore()
    keys=capture_query_keys(model,tok,memory.names,query)
    selected,score,margin,ranks=select_candidate(bank,keys)
    # Only after target-free selection does the official evaluator receive labels.
    try:
        memory.apply(bank[selected])
        value=metric(model,tok,metric_hp,query,target,0,locality=locality)
        if len(value)!=1:raise ValueError('Unexpected native metric shape')
        return value[0],dict(selected=selected,score=score,margin=margin),ranks
    finally:memory.restore()


def checkpoint(out,d,n,memory,metric,model,tok,metric_hp,requests,pre,ledger,torch):
    target=out/f'checkpoint-{n}.json'
    hashes={r['sid']:r['sha256'] for r in ledger[:n]}
    if target.exists():
        saved=json.loads(target.read_text())
        if saved.get('completed') and saved['bank_sha256']==hashes:return
        raise ValueError('Conflicting checkpoint')
    bank={r['sid']:memory.load(r['sid']) for r in ledger[:n]}
    partial=out/f'checkpoint-{n}.partial.json'
    saved=json.loads(partial.read_text()) if partial.exists() else None
    if saved and saved['bank_sha256']!=hashes:raise ValueError('Partial evaluation bank changed')
    rows=saved['rows'] if saved else []
    if [r['case_id'] for r in rows]!=list(range(len(rows))):raise ValueError('Partial order changed')
    began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for i in range(len(rows),n):
        status(out,'checkpoint_evaluation',n=n,completed=i,total=n)
        r=requests[i]
        rel,relroute,relranks=routed_metric(memory,bank,metric,model,tok,metric_hp,r['prompt'],r['target_new'])
        gen,genroute,genranks=routed_metric(memory,bank,metric,model,tok,metric_hp,r['rephrase_prompt'],r['target_new'])
        loc,locroute,_=routed_metric(memory,bank,metric,model,tok,metric_hp,r['locality']['prompt'],r['locality']['ground_truth'],locality=True)
        old=pre[i]['locality_tokens'];sid=f'case_{i:04d}'
        if len(old)!=len(loc) or not old:raise ValueError('Locality token alignment changed')
        if not 0<=rel<=1 or not 0<=gen<=1:raise ValueError('Invalid metric')
        rows.append(dict(case_id=i,reliability_token_accuracy=float(rel),generalization_token_accuracy=float(gen),
            reliability_all_tokens_correct=rel==1.,generalization_all_tokens_correct=gen==1.,
            locality_tokens=loc,locality_token_preservation=sum(a==b for a,b in zip(old,loc))/len(old),
            locality_all_tokens_preserved=old==loc,rewrite_route=relroute,paraphrase_route=genroute,locality_route=locroute,
            rewrite_routing_hit=relroute['selected']==sid,paraphrase_routing_hit=genroute['selected']==sid,
            rewrite_correct_delta_rank=relranks[sid],paraphrase_correct_delta_rank=genranks[sid]))
        if (i+1)%10==0 or i+1==n:save(partial,dict(completed=False,bank_sha256=hashes,rows=rows))
    keys=['reliability_token_accuracy','generalization_token_accuracy','locality_token_preservation',
          'reliability_all_tokens_correct','generalization_all_tokens_correct','locality_all_tokens_preserved',
          'rewrite_routing_hit','paraphrase_routing_hit','rewrite_correct_delta_rank','paraphrase_correct_delta_rank']
    summary={key:statistics.mean(r[key] for r in rows) for key in keys}
    summary['average_token_metric']=statistics.mean(summary[k] for k in keys[:3])
    memory.restore()
    if any(not torch.equal(model.get_parameter(name).detach().cpu(),base) for name,base in memory.base.items()):
        raise ValueError('Base restoration is not exact')
    save(target,dict(completed=True,variant='session_control',n=n,questions=n,rows=rows,summary=summary,bank_sha256=hashes,
        evaluation_seconds_this_process=time.monotonic()-began,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        retained_bank_bytes=sum((out/'factors'/(r['sid']+'.pt')).stat().st_size for r in ledger[:n]),finished_utc=now(),
        metric_note='Same native teacher-forced token metrics as WISE. Routing is question-only and independent for all three query types.'))
    print('CHECKPOINT session_control',n,summary,flush=True)


def run(out,limit):
    d,torch,tok,model,hp,metric_hp=load(out)
    from compact_session_memory import CompactSessionMemory
    from .core import raw_write_request
    names=[hp.rewrite_module_tmp.format(layer)+'.weight' for layer in hp.layers]
    class AtomicMemory(CompactSessionMemory):
        # Storage-only override; all value optimization and update math unchanged.
        def save(self,sid,payload):
            path=self.root/(sid+'.pt')
            if path.exists():raise ValueError('Never overwrite committed factors')
            durable_torch(torch,path,{name:(left.detach().cpu().contiguous(),right.detach().cpu().contiguous())
                                      for name,(left,right) in payload.items()})
    memory=AtomicMemory(model,names,out/'factors')
    requests=json.loads((out/'requests.json').read_text());metric=evaluator()
    pre=baseline_validation(out,d,metric,model,tok,metric_hp,requests)
    P=projector(out,d,torch,tok,model,names)
    ledger_file=out/'ledger.json';ledger=json.loads(ledger_file.read_text()) if ledger_file.exists() else []
    if [r['case_id'] for r in ledger]!=list(range(len(ledger))):raise ValueError('Noncontiguous committed writes')
    for r in ledger:
        if digest(out/'factors'/(r['sid']+'.pt'))!=r['sha256']:raise ValueError('Committed delta changed')
    # A complete factor save can precede a crash before ledger commit. Preserve
    # that uncommitted artifact for audit, then re-execute its write deterministically.
    for path in (out/'factors').glob('case_*.pt'):
        if int(path.stem.split('_')[1])>=len(ledger):
            orphan=out/'uncommitted';orphan.mkdir(exist_ok=True)
            path.rename(orphan/(path.name+'.'+str(time.time_ns())))
    for n in d['checkpoints']:
        if n<=len(ledger):checkpoint(out,d,n,memory,metric,model,tok,metric_hp,requests,pre,ledger,torch)
    for i in range(len(ledger),limit):
        status(out,'writing',completed=i,total=1000)
        sid=f'case_{i:04d}';req=raw_write_request(requests[i],tok)
        began=time.monotonic();memory.write(sid,tok,req,hp,P);torch.cuda.synchronize()
        payload=memory.load(sid)
        if any(left.ndim!=1 or right.ndim!=1 for left,right in payload.values()):raise ValueError('Unexpected factor rank')
        path=out/'factors'/(sid+'.pt')
        ledger.append(dict(case_id=i,sid=sid,sha256=digest(path),bytes=path.stat().st_size,
                           internal_write_seconds=time.monotonic()-began))
        save(ledger_file,ledger)
        print('WRITE session_control',i+1,'/1000',round(ledger[-1]['internal_write_seconds'],3),'seconds',flush=True)
        if i+1 in d['checkpoints']:checkpoint(out,d,i+1,memory,metric,model,tok,metric_hp,requests,pre,ledger,torch)
        gc.collect();torch.cuda.empty_cache()
    check(out)
    save(out/'worker_completion.json',dict(completed=True,edits=len(ledger),full_run=len(ledger)==1000,physical_gpu=1,finished_utc=now()))
    status(out,'completed_requested_edits',completed=len(ledger),total=1000)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int,default=1000)
    a=p.parse_args();run(a.out.resolve(),a.limit)
