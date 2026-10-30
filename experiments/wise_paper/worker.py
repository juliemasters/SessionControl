"""Run unmodified native WISE on physical GPU 1, with paper-style checkpoints."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import copy
import json
import os
import random
import statistics
import time
from .native import ROOT,VENDOR,digest,modules,evaluator,restore
from .prepare import save


def now():return datetime.now(timezone.utc).isoformat()


def check(out):
    d=json.loads((out/'design.json').read_text())
    if digest(out/'requests.json')!=d['requests_sha256']:raise ValueError('Frozen requests changed')
    for name,sha in d['code_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Execution code changed: '+name)
    modules()
    return d


def status(folder,stage,**fields):save(folder/'status.json',dict(stage=stage,updated_utc=now(),**fields))


def load(out,mode):
    d=check(out)
    from experiment_utils import require_gpu1
    uuid=require_gpu1()
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if torch.cuda.device_count()!=1:raise RuntimeError('Only physical GPU 1 may be visible')
    torch.set_num_threads(8);torch.manual_seed(d['seed']);np.random.seed(d['seed']);random.seed(d['seed'])
    torch.backends.cuda.matmul.allow_tf32=False
    native=modules();hp=native['wise_hparams'].WISEHyperParams.from_hparams(str(VENDOR/'hparams/WISE/mistral-7b.yaml'))
    hp.model_name=d['model'];hp.device=0;hp.retrieve=mode=='retrieve'
    tok=AutoTokenizer.from_pretrained(d['model'],local_files_only=True);tok.pad_token_id=tok.eos_token_id;tok.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(d['model'],torch_dtype=torch.float32,attn_implementation='sdpa',local_files_only=True).eval().to('cuda:0')
    folder=out/mode
    metadata=dict(code_sha256=d['code_sha256'],official_revision=d['source_revision'],gpu_uuid=uuid,
        physical_gpu=1,torch_version=torch.__version__,variant=mode,hparams=hp.__dict__,started_utc=now(),
        note='Native WISE.apply_wise_to_model; no custom training subclass or loss/router changes.')
    save(folder/'provenance.json',metadata)
    return d,native,hp,tok,model,torch,np


def score(metric,model,tok,hp,request):
    rel=metric(model,tok,hp,request['prompt'],request['target_new'],0)
    gen=metric(model,tok,hp,request['rephrase_prompt'],request['target_new'],0)
    loc=metric(model,tok,hp,request['locality']['prompt'],request['locality']['ground_truth'],0,locality=True)
    if len(rel)!=1 or len(gen)!=1 or len(loc)!=1 or not loc[0]:raise ValueError('Unexpected official metric shape')
    r,g=float(rel[0]),float(gen[0])
    if not 0<=r<=1 or not 0<=g<=1:raise ValueError('Invalid official metric')
    return dict(case_id=request['case_id'],reliability_token_accuracy=r,generalization_token_accuracy=g,
                reliability_all_tokens_correct=r==1.,generalization_all_tokens_correct=g==1.,locality_tokens=loc[0])


def baseline(out,folder,model,tok,hp,metric,requests):
    complete=out/'baseline.json'
    if complete.exists():
        saved=json.loads(complete.read_text())
        if saved['requests_sha256']!=digest(out/'requests.json'):raise ValueError('Baseline belongs to different data')
        return saved['rows']
    partial=out/'baseline.partial.json';rows=json.loads(partial.read_text())['rows'] if partial.exists() else []
    for i in range(len(rows),len(requests)):
        status(folder,'unedited_baseline',completed=i,total=len(requests))
        rows.append(score(metric,model,tok,hp,requests[i]))
        if (i+1)%25==0 or i+1==len(requests):save(partial,dict(rows=rows))
    save(complete,dict(completed=True,requests_sha256=digest(out/'requests.json'),rows=rows))
    return rows


def checkpoint(out,folder,n,editor,tok,hp,metric,requests,pre,torch):
    target=folder/f'checkpoint-{n}.json'
    if target.exists() and json.loads(target.read_text()).get('completed'):return
    rows=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for i in range(n):
        status(folder,'checkpoint_evaluation',n=n,completed=i,total=n)
        row=score(metric,editor,tok,hp,requests[i])
        old=pre[i]['locality_tokens'];new=row['locality_tokens']
        if len(old)!=len(new):raise ValueError('Locality lengths changed')
        row['locality_token_preservation']=sum(a==b for a,b in zip(old,new))/len(old)
        row['locality_all_tokens_preserved']=old==new
        rows.append(row)
        if (i+1)%25==0 or i+1==n:save(folder/f'checkpoint-{n}.partial.json',dict(completed=False,rows=rows))
    keys=['reliability_token_accuracy','generalization_token_accuracy','locality_token_preservation',
          'reliability_all_tokens_correct','generalization_all_tokens_correct','locality_all_tokens_preserved']
    summary={key:statistics.mean(r[key] for r in rows) for key in keys}
    summary['average_token_metric']=statistics.mean(summary[k] for k in keys[:3])
    result=dict(completed=True,variant=folder.name,n=n,questions=n,rows=rows,summary=summary,
        evaluation_seconds=time.monotonic()-start,peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),finished_utc=now(),
        metric_note='Native teacher-forced target-token accuracy/preservation, not freely generated-answer EM or F1.')
    save(target,result);print('CHECKPOINT',folder.name,n,summary,flush=True)


def native_save(editor,path,torch):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.pt.tmp');editor.save(str(temp))
    with temp.open('rb') as f:os.fsync(f.fileno())
    temp.replace(path)


def run(out,mode,limit):
    folder=out/mode
    if (folder/'completion.json').exists():
        state=json.loads((folder/'completion.json').read_text())
        if state.get('completed') and state['edits']>=limit:return
    status(folder,'loading_model',limit=limit)
    d,native,hp,tok,model,torch,np=load(out,mode)
    requests=json.loads((out/'requests.json').read_text());metric=evaluator()
    pre=baseline(out,folder,model,tok,hp,metric,requests)
    # Generate exactly the upstream cached augmentation templates on the unedited
    # model. Do not replace these with LongMemEval grounding or hand-made prompts.
    template_file=out/'context_templates.json'
    if template_file.exists():native['utils'].CONTEXT_TEMPLATES_CACHE=json.loads(template_file.read_text())
    else:
        templates=native['utils'].get_context_templates(model,tok,[[5,5],[10,5]],'cuda:0')
        save(template_file,templates)
    editor=None;start=0;ledger_file=folder/'ledger.json'
    ledger=json.loads(ledger_file.read_text()) if ledger_file.exists() else []
    restart=folder/'restart.pt'
    if restart.exists():
        editor=native['WISE'].WISE(hp,model,'cuda:0');start=restore(editor,restart,torch)
        rng=json.loads((folder/'resume.json').read_text())
        random.setstate(tuple_tree(rng['python']))
        np.random.set_state((rng['numpy'][0],np.array(rng['numpy'][1],dtype='uint32'),*rng['numpy'][2:]))
        states=torch.load(folder/'rng.pt',map_location='cpu',weights_only=True)
        torch.set_rng_state(states['cpu']);torch.cuda.set_rng_state_all(states['cuda'])
        ledger=ledger[:start]
    for i in range(start,limit):
        status(folder,'writing',completed=i,total=1000)
        began=time.monotonic()
        # Native official entry point; neither algorithm nor input assembly is overridden.
        editor,_=native['wise_main'].apply_wise_to_model(model,tok,[requests[i]],hp,copy=False)
        torch.cuda.synchronize()
        a=editor.get_adapter_layer()
        if a.editing_total_cnt!=i+1:raise ValueError('Native sequential count mismatch')
        if not torch.isfinite(a.new_weight).all():raise ValueError('Nonfinite native WISE weights')
        ledger.append(dict(case_id=i,write_seconds=time.monotonic()-began,editing_total_cnt=a.editing_total_cnt,
                           memory_shards=len(a.memory_weight),threshold=a.editing_mean_act.min_act()*hp.act_ratio))
        save(ledger_file,ledger)
        n=i+1
        if n in d['checkpoints']:checkpoint(out,folder,n,editor,tok,hp,metric,requests,pre,torch)
        if n%25==0 or n in d['checkpoints'] or n==limit:
            native_save(editor,restart,torch)
            save(folder/'resume.json',dict(completed_edits=n,python=random.getstate(),numpy=numpy_state(np)))
            torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()),folder/'rng.pt')
        print('WRITE',mode,n,'/1000',round(ledger[-1]['write_seconds'],3),'seconds',flush=True)
    save(folder/'completion.json',dict(completed=True,edits=limit,full_run=limit==1000,variant=mode,physical_gpu=1,finished_utc=now()))
    status(folder,'completed_requested_edits',completed=limit,total=1000)


def tuple_tree(value):return tuple(tuple_tree(v) for v in value) if isinstance(value,list) else value
def numpy_state(np):
    state=np.random.get_state();return [state[0],state[1].tolist(),*state[2:]]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['merge','retrieve'],required=True)
    p.add_argument('--limit',type=int,default=1000);a=p.parse_args();run(a.out,a.mode,a.limit)
